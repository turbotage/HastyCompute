//! Batches of independent problems (one per voxel) and the symbolic-model objective. Layout (structure of arrays,
//! problem index fastest — coalesced on the GPU):
//!   params  [n_param][n_prob]          (in: start, out: fit)
//!   vars    [n_var][n_data]            (shared by all problems, e.g. echo times)
//!   fixed   [n_fixed][n_prob]          (per problem, e.g. a B1 map)
//!   y       [n_out][n_data][n_prob]    (measured data)
//!   w       [n_data][n_prob]           (optional weights, applied to every output)
//!   lo, hi  [n_param]                  (box bounds, shared)

use crate::lm::{lm, FitOut, LmOptions, Objective};
use crate::model::{hidx, Model};

#[derive(Clone, Copy, Debug)]
pub struct Data<'a, T> {
    pub n_prob: usize,
    pub n_data: usize,
    pub vars: &'a [T],
    pub fixed: &'a [T],
    pub y: &'a [T],
    pub w: Option<&'a [T]>,
}

/// one problem of a batch as an [`Objective`]
pub struct SymProblem<'a> {
    pub model: &'a Model,
    pub n_data: usize,
    pub vars: &'a [f64],
    /// this problem's fixed inputs [n_fixed]
    pub fixed: Vec<f64>,
    /// this problem's data [n_out][n_data]
    pub y: Vec<f64>,
    /// [n_data]
    pub w: Option<Vec<f64>>,
}

impl<'a> SymProblem<'a> {
    pub fn of(model: &'a Model, d: &Data<'a, f64>, k: usize) -> SymProblem<'a> {
        let (np, nd, nout) = (d.n_prob, d.n_data, model.n_out());
        SymProblem {
            model,
            n_data: nd,
            vars: d.vars,
            fixed: (0..model.n_fixed()).map(|f| d.fixed[f * np + k]).collect(),
            y: (0..nout * nd).map(|i| d.y[i * np + k]).collect(),
            w: d.w.map(|w| (0..nd).map(|i| w[i * np + k]).collect()),
        }
    }
    fn var_at(&self, i: usize, v: &mut [f64]) {
        for (k, x) in v.iter_mut().enumerate() {
            *x = self.vars[k * self.n_data + i];
        }
    }
}

impl Objective for SymProblem<'_> {
    fn n_param(&self) -> usize {
        self.model.n_param()
    }
    fn n_resid(&self) -> usize {
        self.n_data * self.model.n_out()
    }
    fn value(&self, p: &[f64]) -> f64 {
        let m = self.model;
        let prog = &m.prog_val;
        let mut slots = vec![0.0; prog.ops.len()];
        let mut v = vec![0.0; m.n_var()];
        let mut f = 0.0;
        for i in 0..self.n_data {
            self.var_at(i, &mut v);
            prog.run(p, &v, &self.fixed, &mut slots);
            let w = self.w.as_ref().map_or(1.0, |w| w[i]);
            for k in 0..m.n_out() {
                let r = slots[prog.outputs[k] as usize] - self.y[k * self.n_data + i];
                f += 0.5 * w * r * r;
            }
        }
        f
    }
    fn derivs(&self, p: &[f64], full: bool, g: &mut [f64], h: &mut [f64], d: &mut [f64]) -> f64 {
        let m = self.model;
        let (n, nout, nh) = (m.n_param(), m.n_out(), m.n_hes());
        let prog = if full { &m.prog_full } else { &m.prog_gn };
        let mut slots = vec![0.0; prog.ops.len()];
        let mut v = vec![0.0; m.n_var()];
        g.fill(0.0);
        h.fill(0.0);
        d.fill(0.0);
        let mut f = 0.0;
        for i in 0..self.n_data {
            self.var_at(i, &mut v);
            prog.run(p, &v, &self.fixed, &mut slots);
            let w = self.w.as_ref().map_or(1.0, |w| w[i]);
            for k in 0..nout {
                let r = slots[prog.outputs[k] as usize] - self.y[k * self.n_data + i];
                f += 0.5 * w * r * r;
                let jk = |j: usize| slots[prog.outputs[nout + k * n + j] as usize];
                for a in 0..n {
                    let ja = jk(a);
                    g[a] += w * r * ja;
                    d[a] += w * ja * ja;
                    for b in 0..=a {
                        let mut v2 = ja * jk(b);
                        if full {
                            v2 += r * slots[prog.outputs[nout + nout * n + k * nh + hidx(b, a)] as usize];
                        }
                        h[a * n + b] += w * v2;
                    }
                }
            }
        }
        for a in 0..n {
            for b in 0..a {
                h[b * n + a] = h[a * n + b];
            }
        }
        f
    }
}

/// CPU reference: fit every problem of the batch (threads over problems); `params` [n_param][n_prob] in/out
pub fn fit_cpu(model: &Model, d: &Data<f64>, params: &mut [f64], lo: &[f64], hi: &[f64], o: &LmOptions) -> Vec<FitOut> {
    let (np, nprob) = (model.n_param(), d.n_prob);
    let nt = std::thread::available_parallelism().map_or(1, |x| x.get()).min(nprob.max(1));
    let chunk = nprob.div_ceil(nt.max(1));
    let mut outs = vec![FitOut { f: 0.0, iters: 0, status: crate::lm::Status::MaxIter }; nprob];
    let mut results: Vec<(usize, Vec<f64>)> = vec![];
    std::thread::scope(|sc| {
        let handles: Vec<_> = (0..nt)
            .map(|t| {
                let pr = &*params;
                sc.spawn(move || {
                    let (a, b) = (t * chunk, ((t + 1) * chunk).min(nprob));
                    let mut res = vec![];
                    for k in a..b {
                        let prob = SymProblem::of(model, d, k);
                        let mut p: Vec<f64> = (0..np).map(|j| pr[j * nprob + k]).collect();
                        let fo = lm(&prob, &mut p, lo, hi, o);
                        res.push((k, p, fo));
                    }
                    res
                })
            })
            .collect();
        for h in handles {
            for (k, p, fo) in h.join().unwrap() {
                outs[k] = fo;
                results.push((k, p));
            }
        }
    });
    for (k, p) in results {
        for j in 0..np {
            params[j * nprob + k] = p[j];
        }
    }
    outs
}
