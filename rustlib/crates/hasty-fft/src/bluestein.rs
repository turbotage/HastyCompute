//! Bluestein (chirp-z) core for axis lengths with a prime factor > 13.
//!
//! X_k = Σ_j x_j e^{s 2πi jk/n} = c_k Σ_j (x_j c_j) b_{k−j},  c_j = e^{s iπ j²/n},  b_d = e^{−s iπ d²/n}  (s = ∓1)
//! — a linear convolution, evaluated as a cyclic one of smooth length m ≥ 2n − 1: FFT_m(a) · B̂ → IFFT_m, B̂ = FFT_m(b̃),
//! b̃ the chirp b wrapped onto 0..m. Everything happens in shared memory inside the generated axis kernel, so the
//! load / store semantics of the plain kernel (strided lines, `n_in` / `n_out` pruning, pre / post diagonals, polynomial
//! diagonals, scale, accumulate, fused convolution) are unchanged: only the transform core differs.

use std::fmt::Write;

use crate::codegen::{kern_mul, kern_type, poly_expr, prelude, sfx, AxisCfg, DiagOp};
use crate::factor::factorize;
use crate::radix::emit_butterfly;

/// Smallest supported cyclic length m ≥ 2n − 1, with its radices.
pub(crate) fn conv_length(n: usize) -> (usize, Vec<usize>) {
    let mut m = 2 * n - 1;
    loop {
        if let Some(r) = factorize(m) {
            return (m, r);
        }
        m += 1;
    }
}

/// Host tables: chirp c_j = e^{−iπ j²/n} (j < n; the inverse direction uses its conjugate) and B̂ for both directions.
pub(crate) fn tables(n: usize, m: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let pi = std::f64::consts::PI;
    let chirp: Vec<(f64, f64)> = (0..n)
        .map(|j| {
            let a = -pi * ((j * j) % (2 * n)) as f64 / n as f64;
            (a.cos(), a.sin())
        })
        .collect();
    // e^{−2πi t/m}
    let tw: Vec<(f64, f64)> = (0..m).map(|t| { let a = -2.0 * pi * t as f64 / m as f64; (a.cos(), a.sin()) }).collect();
    // b̃ for the forward transform (s = −1): b_d = e^{+iπ d²/n}; inverse: the conjugate
    let bhat = |conj: bool| -> Vec<(f64, f64)> {
        (0..m)
            .map(|k| {
                let (mut re, mut im) = (0.0, 0.0);
                for d in -(n as i64 - 1)..(n as i64) {
                    let a = pi * (((d * d) as usize) % (2 * n)) as f64 / n as f64;
                    let (br, bi) = (a.cos(), if conj { -a.sin() } else { a.sin() });
                    let t = (d.rem_euclid(m as i64) as usize * k) % m;
                    let (wr, wi) = tw[t];
                    re += br * wr - bi * wi;
                    im += br * wi + bi * wr;
                }
                (re, im)
            })
            .collect()
    };
    (chirp, bhat(false), bhat(true))
}

/// Out-of-place shared-memory Stockham passes of length m over `bufs`; returns the buffer index holding the result.
fn emit_passes(s: &mut String, radices: &[usize], m: usize, sign: i32, mut cur: usize) -> usize {
    let tw = if sign < 0 { "tw_m" } else { "tw_p" };
    let mut ns = 1usize;
    for &r in radices {
        let q = m / r;
        let (src, dst) = (cur, 1 - cur);
        writeln!(s, "  {{ const cx* src = bufs[{src}]; cx* dst = bufs[{dst}];").unwrap();
        writeln!(s, "    for (int jj = tl; jj < {q}; jj += TPL) {{\n      cx v[{r}];").unwrap();
        writeln!(s, "      #pragma unroll\n      for (int t = 0; t < {r}; ++t) v[t] = src[SIDX(jj + t * {q})];").unwrap();
        if ns > 1 {
            let tws = m / (ns * r);
            writeln!(s, "      const int k = jj % {ns};").unwrap();
            writeln!(s, "      #pragma unroll\n      for (int t = 1; t < {r}; ++t) v[t] = cmul(v[t], {tw}[t * k * {tws}]);").unwrap();
            writeln!(s, "      const int ob = (jj / {ns}) * {} + k;", ns * r).unwrap();
        } else {
            writeln!(s, "      const int ob = jj * {r};").unwrap();
        }
        writeln!(s, "      bfly{r}{}(v);", sfx(sign)).unwrap();
        writeln!(s, "      #pragma unroll\n      for (int u = 0; u < {r}; ++u) dst[SIDX(ob + u * {ns})] = v[u];").unwrap();
        writeln!(s, "    }}\n  }}\n  __syncthreads();").unwrap();
        ns *= r;
        cur = dst;
    }
    cur
}

/// One chirp-z transform of direction `sign` on bufs[cur] (already chirped and zero-padded): FFT_m → ×B̂ → IFFT_m.
/// Returns the buffer index holding m · (a ⊛ b̃).
fn emit_chirp_core(s: &mut String, radices: &[usize], m: usize, sign: i32, cur: usize) -> usize {
    let cur = emit_passes(s, radices, m, -1, cur);
    let bh = if sign < 0 { "bh_f" } else { "bh_i" };
    writeln!(s, "  for (int e = tl; e < M; e += TPL) bufs[{cur}][SIDX(e)] = cmul(bufs[{cur}][SIDX(e)], {bh}[e]);\n  __syncthreads();").unwrap();
    emit_passes(s, radices, m, 1, cur)
}

/// chirp multiply of `v` at index `e` for transform direction `sign`
fn chirp(sign: i32, v: &str) -> String {
    if sign < 0 { format!("cmul({v}, ch[e])") } else { format!("cmulc({v}, ch[e])") }
}

pub(crate) fn generate(cfg: &AxisCfg, m: usize, radices: &[usize]) -> String {
    let n = cfg.n;
    let sign1 = if cfg.forward { -1 } else { 1 };
    let mut s = prelude(cfg.ctype);
    let mut done = vec![];
    for &r in radices {
        for sg in [-1, 1] {
            if !done.contains(&(r, sg)) {
                s.push_str(&emit_butterfly(r, sg));
                done.push((r, sg));
            }
        }
    }
    writeln!(s, "#define N {n}\n#define M {m}\n#define N_IN {}\n#define N_OUT {}\n#define N_IN_TAIL {}\n#define N_OUT_TAIL {}\n#define LPB {}\n#define TPL {}", cfg.n_in, cfg.n_out, cfg.n_in_tail, cfg.n_out_tail, cfg.lpb, cfg.tpl).unwrap();
    writeln!(
        s,
        "extern \"C\" __global__ void __launch_bounds__({}) fft_axis(const cx* __restrict__ in, cx* out, const cx* __restrict__ tw_m, const cx* __restrict__ tw_p, {} __restrict__ kern, real kscale,\n  const cx* __restrict__ p0, const cx* __restrict__ p1, const cx* __restrict__ p2, const cx* __restrict__ p3,\n  const cx* __restrict__ q0, const cx* __restrict__ q1, const cx* __restrict__ q2, const cx* __restrict__ q3,\n  const real* __restrict__ ppre, int nppre, const real* __restrict__ ppost, int nppost,\n  long la_cnt, long lb_cnt, long in_la, long in_lb, long in_sn, long out_la, long out_lb, long out_sn,\n  long lb1_cnt, long lb1_hi, long lb1_gap, long lb2_hi, long lb2_gap, long in_lb2, long out_lb2,\n  const cx* __restrict__ ch, const cx* __restrict__ bh_f, const cx* __restrict__ bh_i) {{",
        cfg.block_threads(),
        kern_type(cfg.kfmt)
    )
    .unwrap();
    writeln!(s, "  extern __shared__ cx smem[];").unwrap();
    if cfg.contiguous {
        writeln!(s, "  const int tid = threadIdx.x;\n  const int li = tid / TPL, tl = tid % TPL;\n  const long line = (long)blockIdx.x * LPB + li;\n  const bool act = line < la_cnt * lb_cnt;\n  const long la = line % la_cnt, lb = line / la_cnt;\n  long lb1 = lb % lb1_cnt, lb2 = lb / lb1_cnt;\n  if (lb1 >= lb1_hi) lb1 += lb1_gap;\n  if (lb2 >= lb2_hi) lb2 += lb2_gap;\n  const long ib = la * in_la + lb1 * in_lb + lb2 * in_lb2, ob_ = la * out_la + lb1 * out_lb + lb2 * out_lb2;\n#define SIDX(n) (li * M + (n))").unwrap();
    } else {
        writeln!(s, "  const int tid = threadIdx.x;\n  const int li = tid % LPB, tl = tid / LPB;\n  const long la = (long)blockIdx.x * LPB + li;\n  const long lb = blockIdx.y;\n  const bool act = la < la_cnt;\n  long lb1 = lb % lb1_cnt, lb2 = lb / lb1_cnt;\n  if (lb1 >= lb1_hi) lb1 += lb1_gap;\n  if (lb2 >= lb2_hi) lb2 += lb2_gap;\n  const long ib = la * in_la + lb1 * in_lb + lb2 * in_lb2, ob_ = la * out_la + lb1 * out_lb + lb2 * out_lb2;\n#define SIDX(n) ((n) * LPB + li)").unwrap();
    }
    writeln!(s, "#define GIN(n) (ib + (long)(n) * in_sn)\n#define GOUT(n) (ob_ + (long)(n) * out_sn)\n  cx* bufs[2] = {{ smem, smem + M * LPB }};").unwrap();
    let pre_ptr = ["p0", "p1", "p2", "p3"];
    let post_ptr = ["q0", "q1", "q2", "q3"];
    // load: diagonals, chirp, zero padding to M
    writeln!(s, "  for (int e = tl; e < M; e += TPL) {{\n    cx x = make_cx(RL(0), RL(0));\n    if (act && (e < N_IN || (e >= N - N_IN_TAIL && e < N))) {{\n      x = in[GIN(e)];").unwrap();
    for (k, op) in cfg.pre.iter().enumerate() {
        let f = if *op == DiagOp::Mult { "cmul" } else { "cmulc" };
        writeln!(s, "      x = {f}(x, {}[GIN(e)]);", pre_ptr[k]).unwrap();
    }
    if cfg.poly_pre {
        writeln!(s, "      x = cscale(x, {});", poly_expr("ppre", "nppre", "e", "la", "lb", cfg.vox)).unwrap();
    }
    writeln!(s, "      x = {};\n    }}\n    bufs[0][SIDX(e)] = x;\n  }}\n  __syncthreads();", chirp(sign1, "x")).unwrap();
    let mut cur = emit_chirp_core(&mut s, radices, m, sign1, 0);
    let mut sign = sign1;
    if cfg.conv {
        // k-space: chirp, 1/M, kernel multiply, then the chirp of the opposite transform and zero padding (in place)
        let sign2 = -sign1;
        writeln!(s, "  for (int e = tl; e < M; e += TPL) {{\n    cx v = make_cx(RL(0), RL(0));\n    if (e < N) {{\n      v = cscale(bufs[{cur}][SIDX(e)], RL({:.17e}));\n      v = {};\n      if (act) v = {};\n      v = {};\n    }}\n    bufs[{cur}][SIDX(e)] = v;\n  }}\n  __syncthreads();", 1.0 / m as f64, chirp(sign1, "v"), kern_mul(cfg.kfmt, "v"), chirp(sign2, "v")).unwrap();
        cur = emit_chirp_core(&mut s, radices, m, sign2, cur);
        sign = sign2;
    }
    // store: chirp, 1/M, scale, diagonals, accumulate
    writeln!(s, "  for (int e = tl; e < N; e += TPL) {{\n    if (!act || !(e < N_OUT || e >= N - N_OUT_TAIL)) continue;\n    cx w = cscale(bufs[{cur}][SIDX(e)], RL({:.17e}));\n    w = {};", cfg.scale / m as f64, chirp(sign, "w")).unwrap();
    for (k, op) in cfg.post.iter().enumerate() {
        let f = if *op == DiagOp::Mult { "cmul" } else { "cmulc" };
        writeln!(s, "    w = {f}(w, {}[GOUT(e)]);", post_ptr[k]).unwrap();
    }
    if cfg.poly_post {
        writeln!(s, "    w = cscale(w, {});", poly_expr("ppost", "nppost", "e", "la", "lb", cfg.vox)).unwrap();
    }
    if cfg.accumulate {
        writeln!(s, "    out[GOUT(e)] = cadd(out[GOUT(e)], w);").unwrap();
    } else {
        writeln!(s, "    out[GOUT(e)] = w;").unwrap();
    }
    writeln!(s, "  }}\n#undef GIN\n#undef GOUT\n#undef SIDX\n}}").unwrap();
    s
}
