//! Joint coil × time-segment channels of the segmented NUFFT (`research/coil_segment_compression.md`).
//!
//! Today's operator spends one FFT channel and one grid interpolation per (coil c, segment l): the C·L spatial
//! functions D_cl = S_c P_l with block-diagonal mixing y_c = Σ_l a_l(τ) g_cl. Here they are replaced by R shared
//! spatial functions Φ_q with a full per-sample mixing
//!
//!   E_c(r, τ) = S_c(r) g(r, τ) ≈ Σ_q B_cq(τ) Φ_q(r)
//!   forward  y_c(j) = Σ_q B_cq(τ_j) [F(Φ_q x)](k_j)
//!   adjoint  x = Σ_q conj(Φ_q) Fᴴ[Σ_c conj(B_cq(τ_j)) r_c(j)]
//!
//! with the optimal (Eckart–Young) basis of the weighted (coil, time) Gram of E. Build ([`build_joint_basis`], GPU,
//! fp64 Gram or fp32 TSQR):
//! 1. time Gram G_τ = Σ_r w_r Σ_c |S_c(r)|² g̃_r g̃_rᴴ on the time grid (g̃ = g · √tw) → V (n_τ × K), the voxels' time
//!    functions to a relative energy residual ≤ `k_tol` (K ≈ 30–50);
//! 2. h_r = Vᴴ g̃_r for every voxel (kept on the device, fp32, K × nv) and the reduced Gram H = Σ_r w_r z_r z_rᴴ,
//!    z_r = S(r) ⊗ h_r ∈ ℂ^{C·K} over the fit voxels (fp64 cuBLAS ZHERK, chunked);
//! 3. eigh(H) (cuSOLVER ZHEEVD): λ_q, U. The weighted squared error of rank R is Σ_{q ≥ R} λ_q, so R follows from
//!    the spectrum ([`JointBasis::count_for`]);
//! 4. the time functions off the grid, v_k(τ) at the table times (the set's ADC times), by voxel-weighted least squares
//!    on a voxel subset; B_cq(τ) = Σ_k v_k(τ) U_{(c,k),q}.
//!
//! Φ_q(r) = Σ_{c,k} conj(U_{(c,k),q}) S_c(r) h_k(r) is never stored for all q: [`JointSegNufft`] generates it per
//! channel batch (kernel `joint_phi`) into the coil maps of an inner nseg = 1 [`SegNufft`] engine (static, knot-major
//! or 2-D knots: any engine type, order 0–3), runs the inner forward / adjoint, and maps the batch's channel samples to
//! the coils (`joint_mix_fwd` / `joint_mix_adj`). The per-sample corrections of the inner engine (φ0, ΔC, order-3
//! terms) act on every channel alike (§2.5), so they carry over unchanged. The inner engine's one segment carries a
//! per-voxel diagonal (e.g. the per-encode concomitant phase e^{−iθ_e}, §2.6) or ones.
//!
//! Sample layout: regular readouts, sample j = readout · na + n with the B table at the ADC times τ_n (n < na).

use std::sync::Arc;

use cudarc::cublas::{sys as cb, CudaBlas};
use cudarc::cusolver::{sys as cs, DnHandle};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, C32, C64};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use rayon::prelude::*;

use crate::plan::NufftError;
use crate::segnufft::{SegNufft, SegPoints, SegSegments};

/// channels per Φ register tile of `joint_phi`
const QT: usize = 8;
/// ADC samples per mixing block (block = NN × 64 readout lanes)
const NN: usize = 4;

fn blas_ok(s: cb::cublasStatus_t, what: &str) -> Result<(), NufftError> {
    if s == cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS { Ok(()) } else { Err(NufftError::Invalid(format!("{what}: {s:?}"))) }
}
fn solver_ok(s: cs::cusolverStatus_t, what: &str) -> Result<(), NufftError> {
    if s == cs::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS { Ok(()) } else { Err(NufftError::Invalid(format!("{what}: {s:?}"))) }
}
fn lc1(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (n.div_ceil(256).max(1) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}
#[inline]
fn c64(z: C32) -> Complex64 {
    Complex64::new(z.re as f64, z.im as f64)
}

/// Options of [`build_joint_basis`].
#[derive(Clone, Debug)]
pub struct JointBuildOpts {
    /// relative energy tail of the time basis (K: Σ_{k ≥ K} Λ_k ≤ k_tol · Σ Λ)
    pub k_tol: f64,
    pub kmax: usize,
    /// voxels per GPU chunk
    pub chunk: usize,
    /// fit voxels of the least-squares time extension (subset, strided)
    pub ls_voxels: usize,
    /// eigenvectors kept (columns of U; the largest R a caller may ask for)
    pub rmax: usize,
    /// fp32 TSQR of Zᴴ (cuSOLVER CGEQRF on stacked [R; Z_chunkᴴ]) instead of the fp64 Gram: the singular values carry
    /// the square root of the dynamic range (research/coil_segment_compression.md §6)
    pub fp32_tsqr: bool,
    /// print the build stages
    pub verbose: bool,
}

impl Default for JointBuildOpts {
    fn default() -> Self {
        Self { k_tol: 1e-13, kmax: 64, chunk: 16384, ls_voxels: 32768, rmax: 1024, fp32_tsqr: false, verbose: false }
    }
}

/// The joint coil × time basis of one ratemap / coil-map pair (see the module docs).
pub struct JointBasis {
    pub nc: usize,
    pub nv: usize,
    /// time-basis functions
    pub k: usize,
    /// eigenvalues of the reduced Gram H (descending, all C·K)
    pub lam: Vec<f64>,
    /// eigenvectors, (C·K) × ncol column-major, row index c · K + k
    pub u: Vec<Complex64>,
    pub ncol: usize,
    /// time grid, its √weights and the time basis V (n_τ × K, column-major)
    pub ts: Vec<f64>,
    pub stw: Vec<f64>,
    pub v: Vec<Complex64>,
    /// eigenvalues of the time Gram (descending)
    pub time_ev: Vec<f64>,
    /// table times and v_k(τ_n) there ([n][k])
    pub tab_times: Vec<f64>,
    pub vt: Vec<Complex64>,
    /// h_k(r) of every voxel on the device (fp32, K × nv)
    pub h: CudaSlice<C32>,
    /// (stage, seconds)
    pub timing: Vec<(String, f64)>,
}

/// One chunk of host-evaluated weighted time functions: column j = voxel `vox[j]`, rows the time grid,
/// a[i + j · ns] = g(v, t_i) √tw_i · scale_j.
fn eval_chunk(vox: &[usize], ts: &[f64], stw: &[f64], scale: &(dyn Fn(usize) -> f64 + Sync), g: &(dyn Fn(usize, f64) -> Complex64 + Sync), out: &mut [C64]) {
    let ns = ts.len();
    out.par_chunks_mut(ns).zip(vox.par_iter().enumerate()).for_each(|(col, (j, &v))| {
        let s = scale(j);
        for i in 0..ns {
            let z = if s == 0.0 { Complex64::new(0.0, 0.0) } else { g(v, ts[i]) * (stw[i] * s) };
            col[i] = C64::new(z.re, z.im);
        }
    });
}

/// descending eigen-decomposition of a Hermitian n × n fp64 device matrix (lower triangle valid), cuSOLVER ZHEEVD:
/// (eigenvalues, eigenvectors column-major)
fn zheevd(dn: &DnHandle, st: &Arc<CudaStream>, a: &mut CudaSlice<C64>, n: usize) -> Result<(Vec<f64>, Vec<Complex64>), NufftError> {
    let mut w = st.alloc_zeros::<f64>(n)?;
    let mut info = st.alloc_zeros::<i32>(1)?;
    let mut lwork: i32 = 0;
    unsafe {
        let (ap, _g0) = a.device_ptr(st);
        let (wp, _g1) = w.device_ptr(st);
        solver_ok(cs::cusolverDnZheevd_bufferSize(dn.cu(), cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, cs::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, n as i32, ap as *const _, n as i32, wp as *const _, &mut lwork), "Zheevd_bufferSize")?;
    }
    let mut work = st.alloc_zeros::<C64>(lwork.max(1) as usize)?;
    unsafe {
        let (ap, _g0) = a.device_ptr_mut(st);
        let (wp, _g1) = w.device_ptr_mut(st);
        let (kp, _g2) = work.device_ptr_mut(st);
        let (ip, _g3) = info.device_ptr_mut(st);
        solver_ok(cs::cusolverDnZheevd(dn.cu(), cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, cs::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, n as i32, ap as *mut _, n as i32, wp as *mut _, kp as *mut _, lwork, ip as *mut _), "Zheevd")?;
    }
    let inf = st.clone_dtoh(&info)?;
    if inf[0] != 0 {
        return Err(NufftError::Invalid(format!("Zheevd info {}", inf[0])));
    }
    let wv = st.clone_dtoh(&w)?;
    let av = st.clone_dtoh(&a.slice(0..n * n))?;
    // ascending → descending
    let ev: Vec<f64> = (0..n).rev().map(|i| wv[i].max(0.0)).collect();
    let mut vecs = vec![Complex64::new(0.0, 0.0); n * n];
    for (o, i) in (0..n).rev().enumerate() {
        for r in 0..n {
            let z = av[i * n + r];
            vecs[o * n + r] = Complex64::new(z.re, z.im);
        }
    }
    Ok((ev, vecs))
}

struct BuildFuncs {
    kron: CudaFunction,
    hstore: CudaFunction,
    zt32: CudaFunction,
    keep_r: CudaFunction,
}

fn module(dev: &Device, nc: usize, k: usize) -> Result<Arc<cudarc::driver::CudaModule>, NufftError> {
    let src = format!("#define NC {nc}\n#define NK {k}\n#define QT {QT}\n#define NN {NN}\n{}", include_str!("segjoint.cu"));
    Ok(dev.load_module(&format!("segjoint_c{nc}_k{k}"), &src, &CompileSettings::default())?)
}

/// Build the joint basis (module docs): `smaps` (nc × nv), the time function `g(v, t)` of the ratemap, fit voxels with
/// signal weights `voxel_w` (e.g. PD energy: the coils enter through S), the time grid `ts` / quadrature weights `tw`
/// (e.g. [`crate::seg_time_grid`]), and the table times (the set's ADC times) at which B is tabulated.
#[allow(clippy::too_many_arguments)]
pub fn build_joint_basis(
    dev: &Device,
    st: &Arc<CudaStream>,
    nv: usize,
    nc: usize,
    smaps: &[C32],
    fit_voxels: &[usize],
    voxel_w: &[f64],
    ts: &[f64],
    tw: &[f64],
    tab_times: &[f64],
    g: &(dyn Fn(usize, f64) -> Complex64 + Sync),
    opts: &JointBuildOpts,
) -> Result<JointBasis, NufftError> {
    assert_eq!(smaps.len(), nc * nv);
    assert_eq!(voxel_w.len(), fit_voxels.len());
    assert_eq!(ts.len(), tw.len());
    let t_all = std::time::Instant::now();
    let mut timing: Vec<(String, f64)> = vec![];
    let lap = |timing: &mut Vec<(String, f64)>, what: &str, t: &mut std::time::Instant| {
        let s = t.elapsed().as_secs_f64();
        if opts.verbose {
            println!("    [joint build] {what}: {s:.2} s");
        }
        timing.push((what.to_string(), s));
        *t = std::time::Instant::now();
    };
    let mut tl = std::time::Instant::now();
    let ns = ts.len();
    let stw: Vec<f64> = tw.iter().map(|&x| x.max(0.0).sqrt()).collect();
    let cen: Vec<f64> = (0..nv).into_par_iter().map(|v| (0..nc).map(|c| { let z = smaps[c * nv + v]; (z.re as f64).powi(2) + (z.im as f64).powi(2) }).sum()).collect();
    let blas = CudaBlas::new(st.clone()).map_err(|e| NufftError::Invalid(format!("cuBLAS: {e:?}")))?;
    let dn = DnHandle::new(st.clone()).map_err(|e| NufftError::Invalid(format!("cuSOLVER: {e:?}")))?;
    let chunk = opts.chunk.max(256);
    let nfit = fit_voxels.len();
    // 1. time Gram over the fit voxels (weights w_r Σ_c |S_c|²)
    let mut gt = st.alloc_zeros::<C64>(ns * ns)?;
    let mut ad = st.alloc_zeros::<C64>(ns * chunk)?;
    let mut ah = vec![C64::new(0.0, 0.0); ns * chunk];
    for f0 in (0..nfit).step_by(chunk) {
        let f1 = (f0 + chunk).min(nfit);
        let vox = &fit_voxels[f0..f1];
        eval_chunk(vox, ts, &stw, &|j| (voxel_w[f0 + j].max(0.0) * cen[vox[j]]).sqrt(), g, &mut ah[..ns * (f1 - f0)]);
        st.memcpy_htod(&ah[..ns * (f1 - f0)], &mut ad.slice_mut(0..ns * (f1 - f0)))?;
        let (one, b1) = (1.0f64, 1.0f64);
        unsafe {
            let (ap, _g0) = ad.device_ptr(st);
            let (gp, _g1) = gt.device_ptr_mut(st);
            blas_ok(cb::cublasZherk_v2(*blas.handle(), cb::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, cb::cublasOperation_t::CUBLAS_OP_N, ns as i32, (f1 - f0) as i32, &one, ap as *const _, ns as i32, &b1, gp as *mut _, ns as i32), "Zherk (time Gram)")?;
        }
    }
    let (time_ev, tvecs) = zheevd(&dn, st, &mut gt, ns)?;
    drop(gt);
    let ttot: f64 = time_ev.iter().sum::<f64>().max(1e-300);
    let mut tail = 0.0;
    let mut k = ns;
    for kk in (0..ns).rev() {
        tail += time_ev[kk];
        if tail > opts.k_tol * ttot {
            k = kk + 1;
            break;
        }
    }
    let k = k.clamp(1, opts.kmax.min(ns));
    let v: Vec<Complex64> = tvecs[..ns * k].to_vec();
    lap(&mut timing, &format!("time Gram ({nfit} fit voxels × {ns} times) + eigh → K = {k}"), &mut tl);
    // 2. h for every voxel (device table), reduced Gram over the fit voxels
    let ck = nc * k;
    let m = module(dev, nc, k)?;
    let bf = BuildFuncs { kron: m.load_function("joint_kron")?, hstore: m.load_function("joint_hstore")?, zt32: m.load_function("joint_zt32")?, keep_r: m.load_function("joint_keep_r")? };
    let vd = st.clone_htod(&v.iter().map(|z| C64::new(z.re, z.im)).collect::<Vec<_>>())?;
    let mut hd = st.alloc_zeros::<C32>(k * nv)?;
    let mut hc = st.alloc_zeros::<C64>(k * chunk)?;
    let mut fit_idx = vec![u32::MAX; nv];
    for (i, &q) in fit_voxels.iter().enumerate() {
        fit_idx[q] = i as u32;
    }
    let mut hh = if opts.fp32_tsqr { None } else { Some(st.alloc_zeros::<C64>(ck * ck)?) };
    // TSQR state: stacked [R (ck rows); Z_chunkᴴ (≤ chunk rows)] × ck, fp32, column-major
    let tsqr_rows = ck + chunk;
    let mut tq: Option<(CudaSlice<C32>, CudaSlice<C32>, CudaSlice<C32>, CudaSlice<i32>, bool)> = None;
    if opts.fp32_tsqr {
        let a = st.alloc_zeros::<C32>(tsqr_rows * ck)?;
        let tau = st.alloc_zeros::<C32>(ck)?;
        let mut lwork: i32 = 0;
        unsafe {
            let (ap, _g) = a.device_ptr(st);
            solver_ok(cs::cusolverDnCgeqrf_bufferSize(dn.cu(), tsqr_rows as i32, ck as i32, ap as *mut _, tsqr_rows as i32, &mut lwork), "Cgeqrf_bufferSize")?;
        }
        let work = st.alloc_zeros::<C32>(lwork.max(1) as usize)?;
        tq = Some((a, tau, work, st.alloc_zeros::<i32>(1)?, false));
    }
    let mut zd = st.alloc_zeros::<C64>(ck * chunk)?;
    let mut swh: Vec<C64> = vec![C64::new(0.0, 0.0); nc * chunk];
    let mut cols: Vec<i32> = vec![0; chunk];
    let mut swd = st.alloc_zeros::<C64>(nc * chunk)?;
    let mut cold = st.alloc_zeros::<i32>(chunk)?;
    let all: Vec<usize> = (0..nv).collect();
    for v0 in (0..nv).step_by(chunk) {
        let v1 = (v0 + chunk).min(nv);
        let ncol = v1 - v0;
        eval_chunk(&all[v0..v1], ts, &stw, &|_| 1.0, g, &mut ah[..ns * ncol]);
        st.memcpy_htod(&ah[..ns * ncol], &mut ad.slice_mut(0..ns * ncol))?;
        let one = cb::cuDoubleComplex { x: 1.0, y: 0.0 };
        let zero = cb::cuDoubleComplex { x: 0.0, y: 0.0 };
        unsafe {
            let (vp, _g0) = vd.device_ptr(st);
            let (ap, _g1) = ad.device_ptr(st);
            let (hp, _g2) = hc.device_ptr_mut(st);
            blas_ok(cb::cublasZgemm_v2(*blas.handle(), cb::cublasOperation_t::CUBLAS_OP_C, cb::cublasOperation_t::CUBLAS_OP_N, k as i32, ncol as i32, ns as i32, &one, vp as *const _, ns as i32, ap as *const _, ns as i32, &zero, hp as *mut _, k as i32), "Zgemm (h = Vᴴ g)")?;
        }
        let (nvl, v0l, ncl) = (nv as i64, v0 as i64, ncol as i32);
        unsafe { st.launch_builder(&bf.hstore).arg(&hc).arg(&mut hd).arg(&nvl).arg(&v0l).arg(&ncl).launch(lc1(k * ncol))? };
        // fit voxels of the chunk
        let mut nf = 0usize;
        for (j, q) in (v0..v1).enumerate() {
            let i = fit_idx[q];
            if i == u32::MAX {
                continue;
            }
            let w = voxel_w[i as usize].max(0.0).sqrt();
            if w == 0.0 {
                continue;
            }
            cols[nf] = j as i32;
            for c in 0..nc {
                let z = smaps[c * nv + q];
                swh[c + nf * nc] = C64::new(z.re as f64 * w, z.im as f64 * w);
            }
            nf += 1;
        }
        if nf == 0 {
            continue;
        }
        st.memcpy_htod(&swh[..nc * nf], &mut swd.slice_mut(0..nc * nf))?;
        st.memcpy_htod(&cols[..nf], &mut cold.slice_mut(0..nf))?;
        let nfi = nf as i32;
        unsafe { st.launch_builder(&bf.kron).arg(&swd).arg(&cold).arg(&hc).arg(&mut zd).arg(&nfi).launch(lc1(ck * nf))? };
        if let Some(h) = hh.as_mut() {
            let (one, b1) = (1.0f64, 1.0f64);
            unsafe {
                let (zp, _g0) = zd.device_ptr(st);
                let (gp, _g1) = h.device_ptr_mut(st);
                blas_ok(cb::cublasZherk_v2(*blas.handle(), cb::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, cb::cublasOperation_t::CUBLAS_OP_N, ck as i32, nf as i32, &one, zp as *const _, ck as i32, &b1, gp as *mut _, ck as i32), "Zherk (reduced Gram)")?;
            }
        } else {
            // TSQR step: rows [0, ck) hold R (upper triangle, zeros below), rows [ck, ck + nf) the chunk's Zᴴ;
            // rows beyond are zero (QR of the stacked block; the new R replaces the old)
            let (a, tau, work, info, started) = tq.as_mut().unwrap();
            if !*started {
                st.memset_zeros(a)?;
                *started = true;
            }
            let (ld, r0) = (tsqr_rows as i32, ck as i32);
            unsafe { st.launch_builder(&bf.zt32).arg(&zd).arg(&mut *a).arg(&ld).arg(&r0).arg(&nfi).launch(lc1(ck * nf))? };
            let lw = work.len() as i32;
            unsafe {
                let (ap, _g0) = a.device_ptr_mut(st);
                let (tp, _g1) = tau.device_ptr_mut(st);
                let (wp, _g2) = work.device_ptr_mut(st);
                let (ip, _g3) = info.device_ptr_mut(st);
                solver_ok(cs::cusolverDnCgeqrf(dn.cu(), tsqr_rows as i32, ck as i32, ap as *mut _, tsqr_rows as i32, tp as *mut _, wp as *mut _, lw, ip as *mut _), "Cgeqrf")?;
            }
            // keep R: zero the strictly-lower part of the top ck rows (Householder vectors) and the rows below
            let (ld, cki) = (tsqr_rows as i32, ck as i32);
            unsafe { st.launch_builder(&bf.keep_r).arg(&mut *a).arg(&ld).arg(&cki).launch(lc1(tsqr_rows * ck))? };
        }
    }
    drop(zd);
    drop(ad);
    drop(hc);
    lap(&mut timing, &format!("h for {nv} voxels + reduced {} ({ck} × {ck})", if opts.fp32_tsqr { "fp32 TSQR" } else { "fp64 Gram" }), &mut tl);
    // 3. eigen-decomposition of H (fp64), or of RᴴR with R from the TSQR (formed in fp64)
    let mut hmat = match hh {
        Some(h) => h,
        None => {
            let (a, ..) = tq.take().unwrap();
            let av = st.clone_dtoh(&a.slice(0..tsqr_rows * ck))?;
            // H = Rᴴ R in fp64 (host, R upper ck × ck)
            let r = DMatrix::<Complex64>::from_fn(ck, ck, |i, j| if i <= j { c64(av[j * tsqr_rows + i]) } else { Complex64::new(0.0, 0.0) });
            let h = r.adjoint() * &r;
            let hv: Vec<C64> = (0..ck * ck).map(|e| { let z = h[(e % ck, e / ck)]; C64::new(z.re, z.im) }).collect();
            st.clone_htod(&hv)?
        }
    };
    let (lam, uvec) = zheevd(&dn, st, &mut hmat, ck)?;
    drop(hmat);
    let ncol = opts.rmax.min(ck);
    let u: Vec<Complex64> = uvec[..ck * ncol].to_vec();
    lap(&mut timing, &format!("eigh of the reduced Gram ({ck})"), &mut tl);
    // 4. least-squares time extension at the table times over a fit-voxel subset (scaled coordinates h̃ = h / √Λ)
    let step = (nfit / opts.ls_voxels.max(1)).max(1);
    let sub: Vec<usize> = (0..nfit).step_by(step).collect();
    let sc: Vec<f64> = (0..k).map(|kk| 1.0 / time_ev[kk].max(1e-300).sqrt()).collect();
    let hsub: Vec<(usize, f64, Vec<Complex64>)> = sub
        .par_iter()
        .map(|&i| {
            let q = fit_voxels[i];
            let wt = voxel_w[i].max(0.0) * cen[q];
            let gv: Vec<Complex64> = (0..ns).map(|t| g(q, ts[t]) * stw[t]).collect();
            let hv: Vec<Complex64> = (0..k).map(|kk| (0..ns).map(|t| v[kk * ns + t].conj() * gv[t]).sum::<Complex64>() * sc[kk]).collect();
            (q, wt, hv)
        })
        .filter(|p| p.1 > 0.0)
        .collect();
    let mut mm = DMatrix::<Complex64>::zeros(k, k);
    for (_, wt, hv) in &hsub {
        for a in 0..k {
            let ca = hv[a].conj() * *wt;
            for b in 0..k {
                mm[(a, b)] += ca * hv[b];
            }
        }
    }
    let trm = (0..k).map(|a| mm[(a, a)].re).sum::<f64>() / k as f64;
    for a in 0..k {
        mm[(a, a)] += Complex64::new(1e-9 * trm, 0.0);
    }
    let chol = mm.cholesky().ok_or_else(|| NufftError::Invalid("joint basis: LS time-extension matrix not positive definite".into()))?;
    let vt: Vec<Complex64> = tab_times
        .par_iter()
        .flat_map_iter(|&t| {
            let mut rhs = DVector::<Complex64>::zeros(k);
            for (q, wt, hv) in &hsub {
                let gq = g(*q, t) * *wt;
                for a in 0..k {
                    rhs[a] += hv[a].conj() * gq;
                }
            }
            let x = chol.solve(&rhs);
            let sc = &sc;
            (0..k).map(move |a| x[a] * sc[a])
        })
        .collect();
    lap(&mut timing, &format!("least-squares time extension ({} voxels × {} table times)", hsub.len(), tab_times.len()), &mut tl);
    if opts.verbose {
        println!("    [joint build] total {:.1} s", t_all.elapsed().as_secs_f64());
    }
    Ok(JointBasis { nc, nv, k, lam, u, ncol, ts: ts.to_vec(), stw, v, time_ev, tab_times: tab_times.to_vec(), vt, h: hd, timing })
}

/// Point-image and global error of a basis at rank R ([`JointBasis::error`]).
#[derive(Clone, Debug, Default)]
pub struct JointError {
    /// relative signal-weighted error over the eval voxels and times
    pub global: f64,
    /// relative error of the single-voxel image per eval voxel (ρ(r), research/coil_segment_compression.md §2.3)
    pub rho: Vec<f64>,
}

impl JointBasis {
    /// relative weighted error √(Σ_{q ≥ r} λ_q / Σ λ) predicted on the time grid
    pub fn tail_error(&self, r: usize) -> f64 {
        let tot: f64 = self.lam.iter().sum::<f64>().max(1e-300);
        (self.lam[r.min(self.lam.len())..].iter().sum::<f64>() / tot).sqrt()
    }

    /// smallest R with [`tail_error`](Self::tail_error) ≤ tol (≤ ncol)
    pub fn count_for(&self, tol: f64) -> usize {
        (1..=self.ncol).find(|&r| self.tail_error(r) <= tol).unwrap_or(self.ncol)
    }

    /// B table at the table times for rank r: [n][q][c] (fp32)
    pub fn b_table(&self, r: usize) -> Vec<C32> {
        assert!(r <= self.ncol);
        let (nc, k, ck) = (self.nc, self.k, self.nc * self.k);
        let nt = self.tab_times.len();
        let mut out = vec![C32::new(0.0, 0.0); nt * r * nc];
        out.par_chunks_mut(r * nc).enumerate().for_each(|(n, row)| {
            let vn = &self.vt[n * k..(n + 1) * k];
            for q in 0..r {
                let uq = &self.u[q * ck..(q + 1) * ck];
                for c in 0..nc {
                    let z: Complex64 = (0..k).map(|kk| vn[kk] * uq[c * k + kk]).sum();
                    row[q * nc + c] = C32::new(z.re as f32, z.im as f32);
                }
            }
        });
        out
    }

    /// conj(U) for rank r, layout [(c · K + k) · ldu + q], columns zero-padded to ldu = ⌈r / QT⌉ · QT
    pub fn uc_table(&self, r: usize) -> (Vec<C32>, usize) {
        let ck = self.nc * self.k;
        let ldu = r.div_ceil(QT) * QT;
        let mut out = vec![C32::new(0.0, 0.0); ck * ldu];
        for i in 0..ck {
            for q in 0..r {
                let z = self.u[q * ck + i].conj();
                out[i * ldu + q] = C32::new(z.re as f32, z.im as f32);
            }
        }
        (out, ldu)
    }

    /// Achieved error at rank r on the host: E_c(r, τ_n) = S_c g(v, τ_n) against Σ_k v_k(τ_n) (P z_v)_{(c,k)}
    /// (P = the rank-r projector; the operator's B Φ up to fp32 rounding), over `eval` voxels with weights `w` and
    /// the table times `tidx` (indices into `tab_times`) with weights `tw`. Includes the time-basis truncation and the
    /// least-squares extension.
    #[allow(clippy::too_many_arguments)]
    pub fn error(&self, r: usize, smaps: &[C32], eval: &[usize], w: &[f64], tidx: &[usize], tw: &[f64], g: &(dyn Fn(usize, f64) -> Complex64 + Sync)) -> JointError {
        let (nc, k, ck, ns, nv) = (self.nc, self.k, self.nc * self.k, self.ts.len(), self.nv);
        assert!(r <= self.ncol);
        let parts: Vec<(f64, f64, f64)> = eval
            .par_iter()
            .zip(w.par_iter())
            .map(|(&q, &wq)| {
                let gv: Vec<Complex64> = (0..ns).map(|t| g(q, self.ts[t]) * self.stw[t]).collect();
                let hv: Vec<Complex64> = (0..k).map(|kk| (0..ns).map(|t| self.v[kk * ns + t].conj() * gv[t]).sum()).collect();
                let s: Vec<Complex64> = (0..nc).map(|c| c64(smaps[c * nv + q])).collect();
                let z: Vec<Complex64> = (0..ck).map(|i| s[i / k] * hv[i % k]).collect();
                let mut pz = vec![Complex64::new(0.0, 0.0); ck];
                for qq in 0..r {
                    let uq = &self.u[qq * ck..(qq + 1) * ck];
                    let a: Complex64 = (0..ck).map(|i| uq[i].conj() * z[i]).sum();
                    for i in 0..ck {
                        pz[i] += uq[i] * a;
                    }
                }
                let (mut num, mut den) = (0.0f64, 0.0f64);
                for (ti, &n) in tidx.iter().enumerate() {
                    let t = self.tab_times[n];
                    let gt = g(q, t);
                    let vn = &self.vt[n * k..(n + 1) * k];
                    for c in 0..nc {
                        let e = s[c] * gt;
                        let a: Complex64 = (0..k).map(|kk| vn[kk] * pz[c * k + kk]).sum();
                        num += tw[ti] * (a - e).norm_sqr();
                        den += tw[ti] * e.norm_sqr();
                    }
                }
                (wq * num, wq * den, if den > 0.0 { (num / den).sqrt() } else { 0.0 })
            })
            .collect();
        let (n, d) = parts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
        JointError { global: (n / d.max(1e-300)).sqrt(), rho: parts.iter().map(|p| p.2).collect() }
    }

    /// Error of this basis (rank r) on a different time function / weighting, e.g. after a ratemap update (the old
    /// basis on the new g): the subspace of the old basis, interpolators refitted to the new g by the projection
    /// (the h of the new g are formed with the old time basis V; the time-basis truncation of the new g counts).
    #[allow(clippy::too_many_arguments)]
    pub fn error_on(&self, r: usize, smaps: &[C32], eval: &[usize], w: &[f64], tidx: &[usize], tw: &[f64], g_new: &(dyn Fn(usize, f64) -> Complex64 + Sync)) -> JointError {
        self.error(r, smaps, eval, w, tidx, tw, g_new)
    }
}

struct OpFuncs {
    phi: CudaFunction,
    mixf: CudaFunction,
    mixa: CudaFunction,
    /// Khatri–Rao rows of the Φ GEMM, mixing with sample offsets (chunked sets)
    zkr: CudaFunction,
    mixf_o: CudaFunction,
    mixa_o: CudaFunction,
}

impl OpFuncs {
    fn load(m: &Arc<cudarc::driver::CudaModule>, smem_max: usize) -> Result<Self, NufftError> {
        let f = Self {
            phi: m.load_function("joint_phi")?,
            mixf: m.load_function("joint_mix_fwd")?,
            mixa: m.load_function("joint_mix_adj")?,
            zkr: m.load_function("joint_zkr")?,
            mixf_o: m.load_function("joint_mix_fwd_o")?,
            mixa_o: m.load_function("joint_mix_adj_o")?,
        };
        for fx in [&f.mixf, &f.mixa, &f.mixf_o, &f.mixa_o] {
            fx.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_max as i32)?;
        }
        Ok(f)
    }
}

/// launch shape of the mixing kernels (as `JointSegNufft::mix_cfg`) for na ADC samples, nc coils, nro readouts, nq
/// channels (`smem_max` checked by `set_rank` for the full batch)
fn mix_cfg(na: usize, nc: usize, nro: usize, nq: usize, smem_max: usize) -> (LaunchConfig, i32) {
    let (lanes, rit) = (64usize, 8usize);
    let sm = NN * (nq * nc + 1) * 8;
    debug_assert!(sm <= smem_max);
    (LaunchConfig { grid_dim: (na.div_ceil(NN) as u32, nro.div_ceil(lanes * rit).max(1) as u32, 1), block_dim: ((NN * lanes) as u32, 1, 1), shared_mem_bytes: sm as u32 }, rit as i32)
}

/// Φ generation by default as a cuBLAS CGEMM (env HASTY_JOINT_PHI=kernel: the register-tile kernel `joint_phi`)
pub fn phi_gemm_default() -> bool {
    std::env::var("HASTY_JOINT_PHI").map_or(true, |v| v != "kernel")
}

/// voxels per Khatri–Rao chunk of the Φ GEMM: Z (nvc × C·K complex) ≤ 256 MiB (env HASTY_JOINT_PHI_CHUNK = voxels)
fn phi_chunk(nv: usize, ck: usize) -> usize {
    let v = std::env::var("HASTY_JOINT_PHI_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or((256usize << 20) / (8 * ck.max(1)));
    v.clamp(256, nv.max(256)).min(nv.max(1))
}

/// Φ[q · nv + v] = Σ_{c,k} Uc[(c · K + k) · ldu + q0 + q] S[c · nv + v] h[k · nv + v] (q < nq) as GEMMs over voxel
/// chunks: Z (Khatri–Rao rows of S and h, `joint_zkr`) · Ucᵀ, cuBLAS CGEMM (fp32, default math mode)
#[allow(clippy::too_many_arguments)]
fn phi_by_gemm(st: &Arc<CudaStream>, blas: &mut CudaBlas, zkr: &CudaFunction, s: &CudaSlice<C32>, h: &CudaSlice<C32>, uc: &CudaSlice<C32>, phi: &mut CudaSlice<C32>, z: &mut CudaSlice<C32>, nv: usize, ck: usize, ldu: usize, q0: usize, nq: usize) -> Result<(), NufftError> {
    let nvc = z.len() / ck;
    assert!(nvc >= 1 && phi.len() >= nq * nv && q0 + nq <= ldu);
    unsafe { blas.set_stream(st.clone()) }.map_err(|e| NufftError::Invalid(format!("cuBLAS stream: {e:?}")))?;
    let (one, zero) = (cb::cuComplex { x: 1.0, y: 0.0 }, cb::cuComplex { x: 0.0, y: 0.0 });
    let mut v0 = 0;
    while v0 < nv {
        let n = nvc.min(nv - v0);
        let (nvl, v0l, ni) = (nv as i64, v0 as i64, n as i32);
        unsafe { st.launch_builder(zkr).arg(s).arg(h).arg(&mut *z).arg(&nvl).arg(&v0l).arg(&ni).launch(lc1(ck * n))? };
        unsafe {
            let (zp, _g0) = z.device_ptr(st);
            let (up, _g1) = uc.device_ptr(st);
            let (pp, _g2) = phi.device_ptr_mut(st);
            let up = up + (q0 * std::mem::size_of::<C32>()) as u64;
            let pp = pp + (v0 * std::mem::size_of::<C32>()) as u64;
            blas_ok(cb::cublasCgemm_v2(*blas.handle(), cb::cublasOperation_t::CUBLAS_OP_N, cb::cublasOperation_t::CUBLAS_OP_T, n as i32, nq as i32, ck as i32, &one, zp as *const _, n as i32, up as *const _, ldu as i32, &zero, pp as *mut _, nv as i32), "Cgemm (Φ = Z Ucᵀ)")?;
        }
        v0 += n;
    }
    Ok(())
}

/// The joint operator: an inner nseg = 1 [`SegNufft`] with cbj channel maps, fed Φ batches (module docs).
pub struct JointSegNufft {
    pub inner: SegNufft,
    pub nc: usize,
    pub r: usize,
    /// channels per batch (= inner.nc) and batches
    pub cbj: usize,
    pub nb: usize,
    /// ADC samples per readout (regular layout)
    pub na: usize,
    k: usize,
    nv: usize,
    ldu: usize,
    smaps: CudaSlice<C32>,
    uc: CudaSlice<C32>,
    btab: CudaSlice<C32>,
    #[allow(dead_code)]
    phi: CudaSlice<C32>,
    g: Option<CudaSlice<C32>>,
    f: OpFuncs,
    smem_max: usize,
    /// per-stage timing (ms) when `profile` is set: [Φ generation, inner engine, mixing]
    pub profile: bool,
    /// Φ generation as a cuBLAS CGEMM over voxel chunks (default, [`phi_gemm_default`]) or the `joint_phi` kernel
    pub phi_gemm: bool,
    blas: Option<CudaBlas>,
    /// Khatri–Rao chunk of the Φ GEMM (allocated on first use)
    zbuf: Option<CudaSlice<C32>>,
    /// timing-only operator ([`new_timing`](Self::new_timing)): Φ is a fixed buffer, not generated
    timing_only: bool,
    pub prof_ms: [f64; 3],
    basis: JointBasis,
}

impl JointSegNufft {
    /// channels per batch for rank r with at most `cbj_max` per batch: equal batches (waste < one per batch)
    pub fn batch_for(r: usize, cbj_max: usize) -> usize {
        let nb = r.div_ceil(cbj_max.max(1)).max(1);
        r.div_ceil(nb)
    }

    /// `inner`: an nseg = 1 engine with nc = the batch width (see [`batch_for`](Self::batch_for)); `basis` (its h moves
    /// in), coil maps (nc × nv), rank r, the ADC samples per readout `na` (= basis.tab_times.len()).
    pub fn new(dev: &Device, st: &Arc<CudaStream>, mut inner: SegNufft, basis: JointBasis, smaps: &[C32], r: usize) -> Result<Self, NufftError> {
        assert_eq!(inner.nseg, 1, "joint operator: the inner engine needs nseg = 1");
        let nv = inner.nv();
        assert_eq!(nv, basis.nv);
        assert_eq!(smaps.len(), basis.nc * nv);
        let cbj = inner.nc;
        let m = module(dev, basis.nc, basis.k)?;
        let smem_max = (dev.ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)? as usize).max(48 << 10);
        let f = OpFuncs::load(&m, smem_max)?;
        // the inner engine's maps hold the Φ batch (generated in place)
        inner.set_maps(st, &vec![C32::new(0.0, 0.0); cbj * nv])?;
        let phi = st.alloc_zeros::<C32>(1)?;
        let sm = st.clone_htod(smaps)?;
        let na = basis.tab_times.len();
        let mut op = Self { inner, nc: basis.nc, r: 0, cbj, nb: 0, na, k: basis.k, nv, ldu: 0, smaps: sm, uc: st.alloc_zeros::<C32>(1)?, btab: st.alloc_zeros::<C32>(1)?, phi, g: None, f, smem_max, profile: false, phi_gemm: phi_gemm_default(), blas: None, zbuf: None, timing_only: false, prof_ms: [0.0; 3], basis };
        op.set_rank(st, r)?;
        Ok(op)
    }

    /// Timing-only operator for grids whose coil maps and h table do not fit next to the engine (1 mm torso / head):
    /// the same inner engine passes and mixing (random B table, `na` ADC samples per readout), Φ a fixed random batch
    /// (not generated: time [`gen_phi_ms`](Self::gen_phi_ms) on a smaller grid and scale by voxels). Not an operator
    /// of any physics.
    pub fn new_timing(dev: &Device, st: &Arc<CudaStream>, mut inner: SegNufft, nc: usize, k: usize, r: usize, na: usize) -> Result<Self, NufftError> {
        assert_eq!(inner.nseg, 1);
        let nv = inner.nv();
        let cbj = inner.nc;
        let m = module(dev, nc, k)?;
        let smem_max = (dev.ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)? as usize).max(48 << 10);
        let f = OpFuncs::load(&m, smem_max)?;
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f32 / (1u64 << 53) as f32 - 0.5 };
        let ph: Vec<C32> = (0..cbj * nv).map(|_| C32::new(rnd(), rnd())).collect();
        inner.set_maps(st, &ph)?;
        drop(ph);
        let phi = st.alloc_zeros::<C32>(1)?;
        let bt: Vec<C32> = (0..na * r * nc).map(|_| C32::new(rnd(), rnd())).collect();
        let basis = JointBasis { nc, nv, k, lam: vec![1.0; r], u: vec![], ncol: r, ts: vec![], stw: vec![], v: vec![], time_ev: vec![], tab_times: vec![0.0; na], vt: vec![], h: st.alloc_zeros::<C32>(1)?, timing: vec![] };
        Ok(Self { inner, nc, r, cbj, nb: r.div_ceil(cbj), na, k, nv, ldu: 0, smaps: st.alloc_zeros::<C32>(1)?, uc: st.alloc_zeros::<C32>(1)?, btab: st.clone_htod(&bt)?, phi, g: None, f, smem_max, profile: false, phi_gemm: phi_gemm_default(), blas: None, zbuf: None, timing_only: true, prof_ms: [0.0; 3], basis })
    }

    /// time (ms, median of `reps`) of one Φ batch generation of `nq` channels on a grid of `nv` voxels with nc coils
    /// and k time functions (random inputs; the cost is nv · nq · nc · k complex MACs)
    pub fn gen_phi_ms(dev: &Device, st: &Arc<CudaStream>, nv: usize, nc: usize, k: usize, nq: usize, reps: usize) -> Result<f64, NufftError> {
        let m = module(dev, nc, k)?;
        let f = m.load_function("joint_phi")?;
        let s = st.alloc_zeros::<C32>(nc * nv)?;
        let h = st.alloc_zeros::<C32>(k * nv)?;
        let ldu = nq.div_ceil(QT) * QT;
        let uc = st.alloc_zeros::<C32>(nc * k * ldu)?;
        let mut phi = st.alloc_zeros::<C32>(nq * nv)?;
        let (nvl, ldi, q0, nqi) = (nv as i64, ldu as i32, 0i32, nq as i32);
        let mut t = vec![];
        for i in 0..=reps {
            st.synchronize()?;
            let t0 = std::time::Instant::now();
            unsafe { st.launch_builder(&f).arg(&s).arg(&h).arg(&uc).arg(&mut phi).arg(&nvl).arg(&ldi).arg(&q0).arg(&nqi).launch(LaunchConfig { grid_dim: (nv.div_ceil(128) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 })? };
            st.synchronize()?;
            if i > 0 {
                t.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(t[t.len() / 2])
    }

    /// [`gen_phi_ms`](Self::gen_phi_ms) of the CGEMM path (Khatri–Rao chunks + cuBLAS), the default Φ generation
    pub fn gen_phi_gemm_ms(dev: &Device, st: &Arc<CudaStream>, nv: usize, nc: usize, k: usize, nq: usize, reps: usize) -> Result<f64, NufftError> {
        let m = module(dev, nc, k)?;
        let zkr = m.load_function("joint_zkr")?;
        let s = st.alloc_zeros::<C32>(nc * nv)?;
        let h = st.alloc_zeros::<C32>(k * nv)?;
        let ldu = nq.div_ceil(QT) * QT;
        let uc = st.alloc_zeros::<C32>(nc * k * ldu)?;
        let mut phi = st.alloc_zeros::<C32>(nq * nv)?;
        let ck = nc * k;
        let mut z = st.alloc_zeros::<C32>(ck * phi_chunk(nv, ck))?;
        let mut blas = CudaBlas::new(st.clone()).map_err(|e| NufftError::Invalid(format!("cuBLAS: {e:?}")))?;
        let mut t = vec![];
        for i in 0..=reps {
            st.synchronize()?;
            let t0 = std::time::Instant::now();
            phi_by_gemm(st, &mut blas, &zkr, &s, &h, &uc, &mut phi, &mut z, nv, ck, ldu, 0, nq)?;
            st.synchronize()?;
            if i > 0 {
                t.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(t[t.len() / 2])
    }

    pub fn basis(&self) -> &JointBasis {
        &self.basis
    }

    /// change the rank (rebuilds the U and B tables; R ≤ basis.ncol)
    pub fn set_rank(&mut self, st: &Arc<CudaStream>, r: usize) -> Result<(), NufftError> {
        assert!(r >= 1 && r <= self.basis.ncol && !self.timing_only);
        let (uc, ldu) = self.basis.uc_table(r);
        let need = NN * (self.cbj.min(r) * self.nc + 1) * 8;
        if need > self.smem_max {
            return Err(NufftError::Invalid(format!("joint mixing: {} channels × {} coils need {need} B of shared memory per block (max {})", self.cbj, self.nc, self.smem_max)));
        }
        self.uc = st.clone_htod(&uc)?;
        self.ldu = ldu;
        self.btab = st.clone_htod(&self.basis.b_table(r))?;
        self.r = r;
        self.nb = r.div_ceil(self.cbj);
        Ok(())
    }

    /// device bytes of the joint state besides the inner engine: maps, h, U, B, Φ batch, g buffer for m samples (the
    /// largest chunk with [`forward_chunks`](Self::forward_chunks)), the Φ GEMM's Khatri–Rao chunk
    pub fn extra_bytes(&self, m: usize) -> usize {
        let ck = self.nc * self.k;
        8 * (self.nc * self.nv + self.k * self.nv + ck * self.ldu + self.na * self.r * self.nc + self.cbj * self.nv + self.cbj * m + if self.phi_gemm { ck * phi_chunk(self.nv, ck) } else { 0 })
    }

    fn gen_phi(&mut self, st: &Arc<CudaStream>, q0: usize, nq: usize) -> Result<(), NufftError> {
        if self.timing_only {
            return Ok(());
        }
        if self.phi_gemm {
            let ck = self.nc * self.k;
            if self.zbuf.is_none() {
                self.zbuf = Some(st.alloc_zeros::<C32>(ck * phi_chunk(self.nv, ck))?);
            }
            if self.blas.is_none() {
                self.blas = Some(CudaBlas::new(st.clone()).map_err(|e| NufftError::Invalid(format!("cuBLAS: {e:?}")))?);
            }
            let (nv, ldu) = (self.nv, self.ldu);
            let (blas, z) = (self.blas.as_mut().unwrap(), self.zbuf.as_mut().unwrap());
            let phi = self.inner.smaps_mut();
            phi_by_gemm(st, blas, &self.f.zkr, &self.smaps, &self.basis.h, &self.uc, phi, z, nv, ck, ldu, q0, nq)?;
            if nq < self.cbj {
                st.memset_zeros(&mut phi.slice_mut(nq * self.nv..self.cbj * self.nv))?;
            }
            return Ok(());
        }
        let (nvl, ldu, q0i, nqi) = (self.nv as i64, self.ldu as i32, q0 as i32, nq as i32);
        let phi = self.inner.smaps_mut();
        unsafe {
            st.launch_builder(&self.f.phi)
                .arg(&self.smaps)
                .arg(&self.basis.h)
                .arg(&self.uc)
                .arg(&mut *phi)
                .arg(&nvl)
                .arg(&ldu)
                .arg(&q0i)
                .arg(&nqi)
                .launch(LaunchConfig { grid_dim: (self.nv.div_ceil(128) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 })?
        };
        if nq < self.cbj {
            st.memset_zeros(&mut phi.slice_mut(nq * self.nv..self.cbj * self.nv))?;
        }
        Ok(())
    }

    fn mix_cfg(&self, nro: usize, nq: usize) -> (LaunchConfig, i32) {
        let lanes = 64usize;
        let rit = 8usize;
        let gy = nro.div_ceil(lanes * rit);
        (LaunchConfig { grid_dim: (self.na.div_ceil(NN) as u32, gy as u32, 1), block_dim: ((NN * lanes) as u32, 1, 1), shared_mem_bytes: (NN * (nq * self.nc + 1) * 8) as u32 }, rit as i32)
    }

    fn ensure_g(&mut self, st: &Arc<CudaStream>, m: usize) -> Result<(), NufftError> {
        if self.g.as_ref().is_none_or(|g| g.len() < self.cbj * m) {
            self.g = None;
            self.g = Some(st.alloc_zeros::<C32>(self.cbj * m)?);
        }
        Ok(())
    }

    fn tick(&self, st: &Arc<CudaStream>) -> Option<std::time::Instant> {
        if self.profile {
            st.synchronize().ok();
            Some(std::time::Instant::now())
        } else {
            None
        }
    }
    fn tock(&mut self, st: &Arc<CudaStream>, t: Option<std::time::Instant>, i: usize) {
        if let Some(t) = t {
            st.synchronize().ok();
            self.prof_ms[i] += t.elapsed().as_secs_f64() * 1e3;
        }
    }

    /// y (nc × m) = A x; `seg1` the inner engine's one segment (diagonal + a ≡ 1)
    pub fn forward(&mut self, st: &Arc<CudaStream>, seg1: &SegSegments, pts: &SegPoints, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        let m = pts.m;
        assert!(self.na > 0 && m % self.na == 0, "joint operator: regular readouts of {} samples expected (m = {m})", self.na);
        assert!(y.len() >= self.nc * m);
        let nro = m / self.na;
        self.ensure_g(st, m)?;
        for b in 0..self.nb {
            let q0 = b * self.cbj;
            let nq = self.cbj.min(self.r - q0);
            let t = self.tick(st);
            self.gen_phi(st, q0, nq)?;
            self.tock(st, t, 0);
            let t = self.tick(st);
            let mut g = self.g.take().unwrap();
            let res = self.inner.forward(st, seg1, pts, x, &mut g);
            self.tock(st, t, 1);
            res?;
            let t = self.tick(st);
            let (cfg, rit) = self.mix_cfg(nro, nq);
            let (ml, nai, nroi, ri, q0i, nqi, acc) = (m as i64, self.na as i32, nro as i32, self.r as i32, q0 as i32, nq as i32, (b > 0) as i32);
            unsafe { st.launch_builder(&self.f.mixf).arg(&g).arg(&self.btab).arg(&mut *y).arg(&ml).arg(&nai).arg(&nroi).arg(&ri).arg(&q0i).arg(&nqi).arg(&rit).arg(&acc).launch(cfg)? };
            self.g = Some(g);
            self.tock(st, t, 2);
        }
        Ok(())
    }

    /// x = Aᴴ y
    pub fn adjoint(&mut self, st: &Arc<CudaStream>, seg1: &SegSegments, pts: &SegPoints, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        let m = pts.m;
        assert!(self.na > 0 && m % self.na == 0, "joint operator: regular readouts of {} samples expected (m = {m})", self.na);
        assert!(y.len() >= self.nc * m);
        let nro = m / self.na;
        self.ensure_g(st, m)?;
        for b in 0..self.nb {
            let q0 = b * self.cbj;
            let nq = self.cbj.min(self.r - q0);
            let t = self.tick(st);
            self.gen_phi(st, q0, nq)?;
            self.tock(st, t, 0);
            let t = self.tick(st);
            let mut g = self.g.take().unwrap();
            let (cfg, rit) = self.mix_cfg(nro, nq);
            let (ml, nai, nroi, ri, q0i, nqi) = (m as i64, self.na as i32, nro as i32, self.r as i32, q0 as i32, nq as i32);
            unsafe { st.launch_builder(&self.f.mixa).arg(y).arg(&self.btab).arg(&mut g).arg(&ml).arg(&nai).arg(&nroi).arg(&ri).arg(&q0i).arg(&nqi).arg(&rit).launch(cfg)? };
            if nq < self.cbj {
                st.memset_zeros(&mut g.slice_mut(nq * m..self.cbj * m))?;
            }
            self.tock(st, t, 2);
            let t = self.tick(st);
            self.inner.adjoint_accumulate = b > 0;
            let res = self.inner.adjoint(st, seg1, pts, &g, x);
            self.inner.adjoint_accumulate = false;
            self.g = Some(g);
            self.tock(st, t, 1);
            res?;
        }
        Ok(())
    }

    /// Forward with the set's samples in chunks: `chunks[i]` = the inner engine's tables of samples
    /// offs[i] .. offs[i] + chunks[i].m of the set (whole readouts: offs[i] and m multiples of na; `total` = the set's
    /// M; y is nc × total). The g buffer holds one chunk (cbj × max chunk) instead of cbj × M: per Φ batch one grid
    /// transform per frame slot / knot, each chunk gathered and mixed into y right away (per knot for knot engines:
    /// the mixing is linear and B does not depend on the knot). Equal to [`forward`](Self::forward) on the whole set
    /// up to f32 summation order.
    pub fn forward_chunks(&mut self, st: &Arc<CudaStream>, seg1: &SegSegments, chunks: &[&SegPoints], offs: &[usize], total: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        self.check_chunks(chunks, offs, total)?;
        assert!(y.len() >= self.nc * total);
        let mmax = chunks.iter().map(|p| p.m).max().unwrap();
        self.ensure_g(st, mmax)?;
        st.memset_zeros(&mut y.slice_mut(0..self.nc * total))?;
        for b in 0..self.nb {
            let q0 = b * self.cbj;
            let nq = self.cbj.min(self.r - q0);
            let t = self.tick(st);
            self.gen_phi(st, q0, nq)?;
            self.tock(st, t, 0);
            let t = self.tick(st);
            let mut g = self.g.take().unwrap();
            let (na, nc, r, smem) = (self.na, self.nc, self.r, self.smem_max);
            let (mixf, btab) = (&self.f.mixf_o, &self.btab);
            let yv: &mut CudaSlice<C32> = &mut *y;
            let mut sink = |st: &Arc<CudaStream>, j: usize, c0: usize, ncb: usize, gb: &CudaSlice<C32>| -> Result<(), NufftError> {
                // rows c0 .. c0 + ncb of the chunk buffer = channels q0 + c0 .. of this batch (live: < nq)
                let nql = ncb.min(nq.saturating_sub(c0));
                if nql == 0 {
                    return Ok(());
                }
                let m = chunks[j].m;
                let nro = m / na;
                let (cfg, rit) = mix_cfg(na, nc, nro, nql, smem);
                let gv = gb.slice(c0 * m..(c0 + nql) * m);
                let (ml, mg, j0, nai, nroi, ri, q0i, nqi) = (total as i64, m as i64, offs[j] as i64, na as i32, nro as i32, r as i32, (q0 + c0) as i32, nql as i32);
                unsafe { st.launch_builder(mixf).arg(&gv).arg(btab).arg(&mut *yv).arg(&ml).arg(&mg).arg(&j0).arg(&nai).arg(&nroi).arg(&ri).arg(&q0i).arg(&nqi).arg(&rit).launch(cfg)? };
                Ok(())
            };
            let res = self.inner.forward_sink(st, seg1, chunks, x, &mut g, &mut sink);
            self.g = Some(g);
            self.tock(st, t, 1);
            res?;
        }
        Ok(())
    }

    /// Adjoint of [`forward_chunks`](Self::forward_chunks): per Φ batch, each chunk's g is mixed from y right before its
    /// scatter (per knot for knot engines), the batches accumulate into x.
    pub fn adjoint_chunks(&mut self, st: &Arc<CudaStream>, seg1: &SegSegments, chunks: &[&SegPoints], offs: &[usize], total: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        self.check_chunks(chunks, offs, total)?;
        assert!(y.len() >= self.nc * total);
        let mmax = chunks.iter().map(|p| p.m).max().unwrap();
        self.ensure_g(st, mmax)?;
        for b in 0..self.nb {
            let q0 = b * self.cbj;
            let nq = self.cbj.min(self.r - q0);
            let t = self.tick(st);
            self.gen_phi(st, q0, nq)?;
            self.tock(st, t, 0);
            let t = self.tick(st);
            let mut g = self.g.take().unwrap();
            let (na, nc, r, smem) = (self.na, self.nc, self.r, self.smem_max);
            let (mixa, btab) = (&self.f.mixa_o, &self.btab);
            let mut src = |st: &Arc<CudaStream>, j: usize, c0: usize, ncb: usize, gb: &mut CudaSlice<C32>| -> Result<(), NufftError> {
                let m = chunks[j].m;
                let nql = ncb.min(nq.saturating_sub(c0));
                if nql < ncb {
                    // channels past the rank: zero input
                    st.memset_zeros(&mut gb.slice_mut((c0 + nql) * m..(c0 + ncb) * m))?;
                }
                if nql == 0 {
                    return Ok(());
                }
                let nro = m / na;
                let (cfg, rit) = mix_cfg(na, nc, nro, nql, smem);
                let mut gv = gb.slice_mut(c0 * m..(c0 + nql) * m);
                let (ml, mg, j0, nai, nroi, ri, q0i, nqi) = (total as i64, m as i64, offs[j] as i64, na as i32, nro as i32, r as i32, (q0 + c0) as i32, nql as i32);
                unsafe { st.launch_builder(mixa).arg(y).arg(btab).arg(&mut gv).arg(&ml).arg(&mg).arg(&j0).arg(&nai).arg(&nroi).arg(&ri).arg(&q0i).arg(&nqi).arg(&rit).launch(cfg)? };
                Ok(())
            };
            self.inner.adjoint_accumulate = b > 0;
            let res = self.inner.adjoint_source(st, seg1, chunks, &mut g, &mut src, x);
            self.inner.adjoint_accumulate = false;
            self.g = Some(g);
            self.tock(st, t, 1);
            res?;
        }
        Ok(())
    }

    fn check_chunks(&self, chunks: &[&SegPoints], offs: &[usize], total: usize) -> Result<(), NufftError> {
        let ok = !chunks.is_empty()
            && chunks.len() == offs.len()
            && self.na > 0
            && total % self.na == 0
            && chunks.iter().zip(offs).all(|(p, &o)| p.m % self.na == 0 && o % self.na == 0 && o + p.m <= total);
        if ok { Ok(()) } else { Err(NufftError::Invalid(format!("joint chunks: {} chunks / {} offsets of whole readouts ({} samples) within M = {total} expected", chunks.len(), offs.len(), self.na))) }
    }

    /// the inner engine's trivial segment: diagonal `diag` (nv; None: ones) and a ≡ 1
    pub fn segment1(&self, st: &Arc<CudaStream>, diag: Option<&[C32]>) -> Result<SegSegments, NufftError> {
        let ones = vec![C32::new(1.0, 0.0); self.nv];
        self.inner.segments(st, diag.unwrap_or(&ones), &[C32::new(1.0, 0.0); 2], 0.0, 1.0)
    }

    /// release the g buffer (it is reallocated on the next call)
    pub fn release_work(&mut self) {
        self.g = None;
    }
}
