//! Conjugate gradients on (E^H E + λ I) x = E^H y, and a power iteration for ‖E^H E‖.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::C32;
use num_complex::Complex64;

use crate::setop::LinOp;
use crate::Error;

pub fn power_norm(op: &mut impl LinOp, st: &Arc<CudaStream>, iters: usize) -> Result<f64, Error> {
    let nv = op.len_x();
    let host: Vec<C32> = (0..nv).map(|i| C32::new(1.0 + 0.1 * (i % 7) as f32, 0.05 * (i % 5) as f32)).collect();
    let mut v = st.clone_htod(&host)?;
    let mut w = st.alloc_zeros::<C32>(nv)?;
    let mut lam = 0.0;
    for _ in 0..iters {
        let n = op.kernels().dot(st, &v, &v)?.re.sqrt();
        op.kernels().axpby(st, Complex64::new(0.0, 0.0), &w, Complex64::new(1.0 / n, 0.0), &mut v)?;
        op.normal(st, &v, &mut w)?;
        lam = op.kernels().dot(st, &v, &w)?.re;
        std::mem::swap(&mut v, &mut w);
    }
    Ok(lam)
}

/// Returns the relative residual history. `x` holds the initial guess on entry.
pub fn cg(op: &mut impl LinOp, st: &Arc<CudaStream>, b: &CudaSlice<C32>, x: &mut CudaSlice<C32>, lambda: f64, iters: usize) -> Result<Vec<f64>, Error> {
    let nv = op.len_x();
    let one = Complex64::new(1.0, 0.0);
    let mut ap = st.alloc_zeros::<C32>(nv)?;
    // r = b − A x
    op.normal(st, x, &mut ap)?;
    op.kernels().axpby(st, Complex64::new(lambda, 0.0), x, one, &mut ap)?;
    let mut r = st.clone_dtod(b)?;
    op.kernels().axpby(st, -one, &ap, one, &mut r)?;
    let mut p = st.clone_dtod(&r)?;
    let bn = op.kernels().dot(st, b, b)?.re.sqrt();
    let mut rr = op.kernels().dot(st, &r, &r)?.re;
    let mut hist = vec![rr.sqrt() / bn];
    for _ in 0..iters {
        op.normal(st, &p, &mut ap)?;
        op.kernels().axpby(st, Complex64::new(lambda, 0.0), &p, one, &mut ap)?;
        let pap = op.kernels().dot(st, &p, &ap)?.re;
        let alpha = rr / pap;
        op.kernels().axpby(st, Complex64::new(alpha, 0.0), &p, one, x)?;
        op.kernels().axpby(st, Complex64::new(-alpha, 0.0), &ap, one, &mut r)?;
        let rr_new = op.kernels().dot(st, &r, &r)?.re;
        hist.push(rr_new.sqrt() / bn);
        let beta = rr_new / rr;
        rr = rr_new;
        op.kernels().axpby(st, one, &r, Complex64::new(beta, 0.0), &mut p)?;
    }
    Ok(hist)
}

/// Preconditioned CG on (E^H E + λ I) x = E^H y with M ≈ (E^H E)⁻¹. Returns the relative residual history.
pub fn pcg(op: &mut impl LinOp, pre: &mut crate::precond::CirculantPrecond, st: &Arc<CudaStream>, b: &CudaSlice<C32>, x: &mut CudaSlice<C32>, lambda: f64, iters: usize) -> Result<Vec<f64>, Error> {
    let nv = op.len_x();
    let one = Complex64::new(1.0, 0.0);
    let mut ap = st.alloc_zeros::<C32>(nv)?;
    op.normal(st, x, &mut ap)?;
    op.kernels().axpby(st, Complex64::new(lambda, 0.0), x, one, &mut ap)?;
    let mut r = st.clone_dtod(b)?;
    op.kernels().axpby(st, -one, &ap, one, &mut r)?;
    let mut z = st.alloc_zeros::<C32>(nv)?;
    pre.apply(st, &r, &mut z)?;
    let mut p = st.clone_dtod(&z)?;
    let bn = op.kernels().dot(st, b, b)?.re.sqrt();
    let mut rz = op.kernels().dot(st, &r, &z)?.re;
    let mut hist = vec![op.kernels().dot(st, &r, &r)?.re.sqrt() / bn];
    for _ in 0..iters {
        op.normal(st, &p, &mut ap)?;
        op.kernels().axpby(st, Complex64::new(lambda, 0.0), &p, one, &mut ap)?;
        let alpha = rz / op.kernels().dot(st, &p, &ap)?.re;
        op.kernels().axpby(st, Complex64::new(alpha, 0.0), &p, one, x)?;
        op.kernels().axpby(st, Complex64::new(-alpha, 0.0), &ap, one, &mut r)?;
        hist.push(op.kernels().dot(st, &r, &r)?.re.sqrt() / bn);
        pre.apply(st, &r, &mut z)?;
        let rz_new = op.kernels().dot(st, &r, &z)?.re;
        let beta = rz_new / rz;
        rz = rz_new;
        op.kernels().axpby(st, one, &z, Complex64::new(beta, 0.0), &mut p)?;
    }
    Ok(hist)
}

/// Preconditioned CGLS (CGNR with explicit residual): never forms E^H E p. Without density compensation the spiral
/// normal operator spans ~10⁶ (every readout dwells near k = 0), so in f32 the high-frequency part of E^H E p drowns in
/// the rounding of its DC part; with the residual r = y − E x (DC fitted early) E^H r stays accurate.
/// Minimises ‖E x − y‖² + λ‖x‖² (λ enters as an image-domain shift of s). Returns ‖s‖/‖s₀‖ per iteration.
pub fn pcgls(op: &mut impl LinOp, pre: Option<&mut crate::precond::CirculantPrecond>, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>, lambda: f64, iters: usize) -> Result<Vec<f64>, Error> {
    pcgls_cb(op, pre, st, y, x, lambda, iters, &mut |_, _| {})
}

/// [`pcgls`] calling `cb(k, x)` after every iteration k = 1..=iters (e.g. to track the error against a known truth).
#[allow(clippy::too_many_arguments)]
pub fn pcgls_cb(
    op: &mut impl LinOp,
    pre: Option<&mut crate::precond::CirculantPrecond>,
    st: &Arc<CudaStream>,
    y: &CudaSlice<C32>,
    x: &mut CudaSlice<C32>,
    lambda: f64,
    iters: usize,
    cb: &mut dyn FnMut(usize, &CudaSlice<C32>),
) -> Result<Vec<f64>, Error> {
    Ok(pcgls_ctl(op, pre, st, y, x, lambda, iters, cb, None)?.0)
}

/// Resumable [`pcgls_cb`]: with `ctl = (run control, state name)` it continues from a saved state (x, r, p, γ, ‖s₀‖,
/// iteration, history), checkpoints every `ctl.every` iterations and stops cleanly on request.
#[allow(clippy::too_many_arguments)]
pub fn pcgls_ctl(
    op: &mut impl LinOp,
    pre: Option<&mut crate::precond::CirculantPrecond>,
    st: &Arc<CudaStream>,
    y: &CudaSlice<C32>,
    x: &mut CudaSlice<C32>,
    lambda: f64,
    iters: usize,
    cb: &mut dyn FnMut(usize, &CudaSlice<C32>),
    ctl: Option<(&crate::checkpoint::RunCtl, &str)>,
) -> Result<(Vec<f64>, crate::checkpoint::Outcome), Error> {
    use crate::checkpoint::{dev_value, get, get_f64, get_usize, value_dev, Outcome};
    use hasty_value::{Tensor, Value};
    let nv = op.len_x();
    let one = Complex64::new(1.0, 0.0);
    let mut pre = pre;
    let m_len = y.len();
    let mut q = st.alloc_zeros::<C32>(m_len)?;
    let mut s = st.alloc_zeros::<C32>(nv)?;
    let mut z = st.alloc_zeros::<C32>(nv)?;
    let saved = match ctl {
        Some((c, name)) => c.load(name)?,
        None => None,
    };
    let (mut r, mut p, mut gamma, s0, mut hist, it0);
    if let Some(v) = saved {
        *x = value_dev(st, get(&v, "x")?, nv)?;
        r = value_dev(st, get(&v, "r")?, m_len)?;
        p = value_dev(st, get(&v, "p")?, nv)?;
        gamma = get_f64(&v, "gamma")?;
        s0 = get_f64(&v, "s0")?;
        it0 = get_usize(&v, "it")?;
        hist = get(&v, "hist")?.as_tensor().map(|t| t.to_vec::<f64>()).transpose()?.unwrap_or_default();
        eprintln!("  resuming {} at CG iteration {it0} / {iters}", ctl.unwrap().1);
    } else {
        r = st.clone_dtod(y)?;
        op.forward(st, x, &mut q)?;
        op.kernels().axpby(st, -one, &q, one, &mut r)?;
        op.adjoint(st, &r, &mut s)?;
        op.kernels().axpby(st, Complex64::new(-lambda, 0.0), x, one, &mut s)?;
        match pre.as_deref_mut() {
            Some(p) => p.apply(st, &s, &mut z)?,
            None => st.memcpy_dtod(&s, &mut z)?,
        }
        p = st.clone_dtod(&z)?;
        gamma = op.kernels().dot(st, &s, &z)?.re;
        s0 = op.kernels().dot(st, &s, &s)?.re.sqrt();
        hist = vec![1.0];
        it0 = 0;
    }
    for it in it0..iters {
        op.forward(st, &p, &mut q)?;
        let qq = op.kernels().dot(st, &q, &q)?.re + lambda * op.kernels().dot(st, &p, &p)?.re;
        let alpha = gamma / qq;
        op.kernels().axpby(st, Complex64::new(alpha, 0.0), &p, one, x)?;
        op.kernels().axpby(st, Complex64::new(-alpha, 0.0), &q, one, &mut r)?;
        op.adjoint(st, &r, &mut s)?;
        op.kernels().axpby(st, Complex64::new(-lambda, 0.0), x, one, &mut s)?;
        match pre.as_deref_mut() {
            Some(pp) => pp.apply(st, &s, &mut z)?,
            None => st.memcpy_dtod(&s, &mut z)?,
        }
        let g_new = op.kernels().dot(st, &s, &z)?.re;
        hist.push(op.kernels().dot(st, &s, &s)?.re.sqrt() / s0);
        let beta = g_new / gamma;
        gamma = g_new;
        op.kernels().axpby(st, one, &z, Complex64::new(beta, 0.0), &mut p)?;
        cb(it + 1, x);
        if let Some((c, name)) = ctl {
            let next = it + 1;
            let stop = c.stop_requested() && next < iters;
            if stop || (c.due(next) && next < iters) {
                c.save(name, &Value::Dict(vec![
                    ("it".into(), Value::Int(next as i64)),
                    ("gamma".into(), Value::Float(gamma)),
                    ("s0".into(), Value::Float(s0)),
                    ("hist".into(), Value::Tensor(Tensor::from_slice(&[hist.len()], &hist)?)),
                    ("x".into(), dev_value(st, x)?),
                    ("r".into(), dev_value(st, &r)?),
                    ("p".into(), dev_value(st, &p)?),
                ]))?;
            }
            if stop {
                return Ok((hist, Outcome::Stopped(next)));
            }
        }
    }
    if let Some((c, name)) = ctl {
        c.clear(name);
    }
    Ok((hist, crate::checkpoint::Outcome::Finished))
}

/// [`pcgls_ctl`] on the union of sample sets that are never resident on the device together: `op` (plain SENSE) is
/// switched between the sets (`set_points`), set i has the host coordinates `coords[i]` and the host data `y_of(i)`
/// (read once, at the start), and the data-space residual r_i stays on the host. The device holds one set's samples at
/// a time, so the memory no longer grows with the pooled sample count (pooling 40 frames of a 10-min scan needs 4 × 7.5 GB
/// of data-space vectors). Each iteration evaluates E_i p twice per set (once for ‖E p‖², once for the residual update)
/// instead of storing it. Same iterates as [`pcgls_ctl`] on the concatenated sets up to rounding. Checkpoints x, p, γ
/// (the residual is recomputed from x on resume).
#[allow(clippy::too_many_arguments)]
pub fn pcgls_sets_ctl(
    op: &mut crate::op::SenseOp,
    pre: Option<&mut crate::precond::CirculantPrecond>,
    st: &Arc<CudaStream>,
    coords: &[[&[f32]; 3]],
    y_of: &mut dyn FnMut(usize) -> Vec<C32>,
    x: &mut CudaSlice<C32>,
    lambda: f64,
    iters: usize,
    cb: &mut dyn FnMut(usize, &CudaSlice<C32>),
    ctl: Option<(&crate::checkpoint::RunCtl, &str)>,
) -> Result<(Vec<f64>, crate::checkpoint::Outcome), Error> {
    use crate::checkpoint::{dev_value, get, get_f64, get_usize, value_dev, Outcome};
    use hasty_value::{Tensor, Value};
    let nv = op.len_x();
    let nc = op.nc;
    let one = Complex64::new(1.0, 0.0);
    let mut pre = pre;
    let n = coords.len();
    let mut s = st.alloc_zeros::<C32>(nv)?;
    let mut z = st.alloc_zeros::<C32>(nv)?;
    let mut tmp = st.alloc_zeros::<C32>(nv)?;
    // switch the operator to set i (coordinates uploaded per switch: 12 B per sample, small against the data)
    let select = |op: &mut crate::op::SenseOp, i: usize| -> Result<usize, Error> {
        let m = coords[i][0].len();
        let dc: Vec<CudaSlice<f32>> = coords[i].iter().map(|c| st.clone_htod(*c)).collect::<Result<_, _>>()?;
        op.set_points(st, [&dc[0], &dc[1], &dc[2]], m)?;
        Ok(m)
    };
    let saved = match ctl {
        Some((c, name)) => c.load(name)?,
        None => None,
    };
    let (mut p, mut gamma, s0, mut hist, it0);
    let resumed = saved.is_some();
    if let Some(v) = &saved {
        *x = value_dev(st, get(v, "x")?, nv)?;
        p = value_dev(st, get(v, "p")?, nv)?;
        gamma = get_f64(v, "gamma")?;
        s0 = get_f64(v, "s0")?;
        it0 = get_usize(v, "it")?;
        hist = get(v, "hist")?.as_tensor().map(|t| t.to_vec::<f64>()).transpose()?.unwrap_or_default();
        eprintln!("  resuming {} at CG iteration {it0} / {iters}", ctl.unwrap().1);
    } else {
        p = st.alloc_zeros::<C32>(nv)?;
        gamma = 0.0;
        s0 = 0.0;
        hist = vec![1.0];
        it0 = 0;
    }
    // residual r_i = y_i − E_i x on the host, s = Σ_i E_i^H r_i − λ x
    let mut r: Vec<Vec<C32>> = Vec::with_capacity(n);
    st.memset_zeros(&mut s)?;
    for i in 0..n {
        let m = select(op, i)?;
        let mut ri = st.clone_htod(&y_of(i))?;
        assert_eq!(ri.len(), nc * m, "set {i}: data length");
        let mut q = st.alloc_zeros::<C32>(nc * m)?;
        op.forward(st, x, &mut q)?;
        op.kernels().axpby(st, -one, &q, one, &mut ri)?;
        op.adjoint(st, &ri, &mut tmp)?;
        op.kernels().axpby(st, one, &tmp, one, &mut s)?;
        r.push(st.clone_dtoh(&ri)?);
    }
    op.kernels().axpby(st, Complex64::new(-lambda, 0.0), x, one, &mut s)?;
    let s0 = if resumed {
        s0
    } else {
        match pre.as_deref_mut() {
            Some(pp) => pp.apply(st, &s, &mut z)?,
            None => st.memcpy_dtod(&s, &mut z)?,
        }
        p = st.clone_dtod(&z)?;
        gamma = op.kernels().dot(st, &s, &z)?.re;
        op.kernels().dot(st, &s, &s)?.re.sqrt()
    };
    for it in it0..iters {
        // pass 1: ‖E p‖² + λ‖p‖²
        let mut qq = lambda * op.kernels().dot(st, &p, &p)?.re;
        for i in 0..n {
            let m = select(op, i)?;
            let mut q = st.alloc_zeros::<C32>(nc * m)?;
            op.forward(st, &p, &mut q)?;
            qq += op.kernels().dot(st, &q, &q)?.re;
        }
        let alpha = gamma / qq;
        op.kernels().axpby(st, Complex64::new(alpha, 0.0), &p, one, x)?;
        // pass 2: r_i −= α E_i p, s = Σ_i E_i^H r_i − λ x
        st.memset_zeros(&mut s)?;
        for (i, ri_host) in r.iter_mut().enumerate() {
            let m = select(op, i)?;
            let mut q = st.alloc_zeros::<C32>(nc * m)?;
            op.forward(st, &p, &mut q)?;
            let mut ri = st.clone_htod(ri_host)?;
            op.kernels().axpby(st, Complex64::new(-alpha, 0.0), &q, one, &mut ri)?;
            op.adjoint(st, &ri, &mut tmp)?;
            op.kernels().axpby(st, one, &tmp, one, &mut s)?;
            st.memcpy_dtoh(&ri, ri_host)?;
        }
        op.kernels().axpby(st, Complex64::new(-lambda, 0.0), x, one, &mut s)?;
        match pre.as_deref_mut() {
            Some(pp) => pp.apply(st, &s, &mut z)?,
            None => st.memcpy_dtod(&s, &mut z)?,
        }
        let g_new = op.kernels().dot(st, &s, &z)?.re;
        hist.push(op.kernels().dot(st, &s, &s)?.re.sqrt() / s0);
        let beta = g_new / gamma;
        gamma = g_new;
        op.kernels().axpby(st, one, &z, Complex64::new(beta, 0.0), &mut p)?;
        cb(it + 1, x);
        if let Some((c, name)) = ctl {
            let next = it + 1;
            let stop = c.stop_requested() && next < iters;
            if stop || (c.due(next) && next < iters) {
                c.save(name, &Value::Dict(vec![
                    ("it".into(), Value::Int(next as i64)),
                    ("gamma".into(), Value::Float(gamma)),
                    ("s0".into(), Value::Float(s0)),
                    ("hist".into(), Value::Tensor(Tensor::from_slice(&[hist.len()], &hist)?)),
                    ("x".into(), dev_value(st, x)?),
                    ("p".into(), dev_value(st, &p)?),
                ]))?;
            }
            if stop {
                return Ok((hist, Outcome::Stopped(next)));
            }
        }
    }
    if let Some((c, name)) = ctl {
        c.clear(name);
    }
    Ok((hist, Outcome::Finished))
}
