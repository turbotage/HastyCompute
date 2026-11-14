//! GPU LLR building blocks at the 2 mm size: batched Gram (cuBLAS C/Zgemm strided batched) and batched Hermitian
//! eigendecomposition (cuSOLVER XsyevBatched; and a per-matrix Xsyevd loop) of T×T Gram matrices of B³×T Casorati
//! blocks (B = 8, T = 280 by default), f32 and f64, timed, and checked against nalgebra f64 on a few blocks.
//!   eigbench [--t 280] [--rows 512] [--batch 256] [--rank 12]

use std::time::Instant;

use cudarc::cublas::sys as cb;
use cudarc::cublas::CudaBlas;
use cudarc::cusolver::{sys as cs, DnHandle};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use hasty_cuda::{Device, C32};
use hasty_testkit::Rng;
use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64;
use std::sync::Arc;

fn arg(name: &str, d: usize) -> usize {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|s| s == name).and_then(|i| a.get(i + 1)).map(|s| s.parse().unwrap()).unwrap_or(d)
}

/// (gram ms, eig ms) for one batch; returns eigenvalues (ascending per block) on the host
fn run_f32(st: &Arc<CudaStream>, blas: &CudaBlas, dn: &DnHandle, c: &CudaSlice<C32>, rows: usize, t: usize, nb: usize, looped: bool) -> (f64, f64, Vec<f32>, i32) {
    let mut g = st.alloc_zeros::<C32>(nb * t * t).unwrap();
    let mut w = st.alloc_zeros::<f32>(nb * t).unwrap();
    let mut info = st.alloc_zeros::<i32>(nb).unwrap();
    st.synchronize().unwrap();
    let t0 = Instant::now();
    unsafe {
        let one = cb::cuComplex { x: 1.0, y: 0.0 };
        let zero = cb::cuComplex { x: 0.0, y: 0.0 };
        let (cp, _g0) = c.device_ptr(st);
        let (gp, _g1) = g.device_ptr_mut(st);
        let s = cb::cublasCgemmStridedBatched(*blas.handle(), cb::cublasOperation_t::CUBLAS_OP_C, cb::cublasOperation_t::CUBLAS_OP_N, t as i32, t as i32, rows as i32, &one, cp as *const _, rows as i32, (rows * t) as i64, cp as *const _, rows as i32, (rows * t) as i64, &zero, gp as *mut _, t as i32, (t * t) as i64, nb as i32);
        assert_eq!(s, cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }
    st.synchronize().unwrap();
    let tg = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    unsafe {
        let mut params: cs::cusolverDnParams_t = std::ptr::null_mut();
        assert_eq!(cs::cusolverDnCreateParams(&mut params), cs::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
        let (gp, _g1) = g.device_ptr_mut(st);
        let (wp, _g2) = w.device_ptr_mut(st);
        let (ip, _g3) = info.device_ptr_mut(st);
        let (ct, rt) = (cs::cudaDataType::CUDA_C_32F, cs::cudaDataType::CUDA_R_32F);
        let jobz = cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
        let lo = cs::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        if !looped {
            let (mut dws, mut hws) = (0usize, 0usize);
            let s = cs::cusolverDnXsyevBatched_bufferSize(dn.cu(), params, jobz, lo, t as i64, ct, gp as *const _, t as i64, rt, wp as *const _, ct, &mut dws, &mut hws, nb as i64);
            assert_eq!(s, cs::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS, "XsyevBatched_bufferSize");
            let mut dbuf = st.alloc_zeros::<u8>(dws.max(1)).unwrap();
            let mut hbuf = vec![0u8; hws.max(1)];
            let (dp, _g4) = dbuf.device_ptr_mut(st);
            let s = cs::cusolverDnXsyevBatched(dn.cu(), params, jobz, lo, t as i64, ct, gp as *mut _, t as i64, rt, wp as *mut _, ct, dp as *mut _, dws, hbuf.as_mut_ptr() as *mut _, hws, ip as *mut _, nb as i64);
            assert_eq!(s, cs::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS, "XsyevBatched");
            eprintln!("  XsyevBatched f32 workspace: device {:.1} MB, host {:.1} MB", dws as f64 / 1e6, hws as f64 / 1e6);
        } else {
            let (mut dws, mut hws) = (0usize, 0usize);
            let s = cs::cusolverDnXsyevd_bufferSize(dn.cu(), params, jobz, lo, t as i64, ct, gp as *const _, t as i64, rt, wp as *const _, ct, &mut dws, &mut hws);
            assert_eq!(s, cs::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
            let mut dbuf = st.alloc_zeros::<u8>(dws.max(1)).unwrap();
            let mut hbuf = vec![0u8; hws.max(1)];
            let (dp, _g4) = dbuf.device_ptr_mut(st);
            for b in 0..nb {
                let s = cs::cusolverDnXsyevd(dn.cu(), params, jobz, lo, t as i64, ct, (gp + (b * t * t * 8) as u64) as *mut _, t as i64, rt, (wp + (b * t * 4) as u64) as *mut _, ct, dp as *mut _, dws, hbuf.as_mut_ptr() as *mut _, hws, (ip + (b * 4) as u64) as *mut _);
                assert_eq!(s, cs::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
            }
        }
        cs::cusolverDnDestroyParams(params);
    }
    st.synchronize().unwrap();
    let te = t1.elapsed().as_secs_f64() * 1e3;
    let inf = st.clone_dtoh(&info).unwrap();
    let bad = inf.iter().filter(|&&i| i != 0).count() as i32;
    (tg, te, st.clone_dtoh(&w).unwrap(), bad)
}

fn run_f64(st: &Arc<CudaStream>, blas: &CudaBlas, dn: &DnHandle, c: &CudaSlice<f64>, rows: usize, t: usize, nb: usize) -> (f64, f64, Vec<f64>, i32) {
    let mut g = st.alloc_zeros::<f64>(2 * nb * t * t).unwrap();
    let mut w = st.alloc_zeros::<f64>(nb * t).unwrap();
    let mut info = st.alloc_zeros::<i32>(nb).unwrap();
    st.synchronize().unwrap();
    let t0 = Instant::now();
    unsafe {
        let one = cb::cuDoubleComplex { x: 1.0, y: 0.0 };
        let zero = cb::cuDoubleComplex { x: 0.0, y: 0.0 };
        let (cp, _g0) = c.device_ptr(st);
        let (gp, _g1) = g.device_ptr_mut(st);
        let s = cb::cublasZgemmStridedBatched(*blas.handle(), cb::cublasOperation_t::CUBLAS_OP_C, cb::cublasOperation_t::CUBLAS_OP_N, t as i32, t as i32, rows as i32, &one, cp as *const _, rows as i32, (rows * t) as i64, cp as *const _, rows as i32, (rows * t) as i64, &zero, gp as *mut _, t as i32, (t * t) as i64, nb as i32);
        assert_eq!(s, cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }
    st.synchronize().unwrap();
    let tg = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    unsafe {
        let mut params: cs::cusolverDnParams_t = std::ptr::null_mut();
        cs::cusolverDnCreateParams(&mut params);
        let (gp, _g1) = g.device_ptr_mut(st);
        let (wp, _g2) = w.device_ptr_mut(st);
        let (ip, _g3) = info.device_ptr_mut(st);
        let (ct, rt) = (cs::cudaDataType::CUDA_C_64F, cs::cudaDataType::CUDA_R_64F);
        let jobz = cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
        let lo = cs::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        let (mut dws, mut hws) = (0usize, 0usize);
        let s = cs::cusolverDnXsyevBatched_bufferSize(dn.cu(), params, jobz, lo, t as i64, ct, gp as *const _, t as i64, rt, wp as *const _, ct, &mut dws, &mut hws, nb as i64);
        assert_eq!(s, cs::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
        let mut dbuf = st.alloc_zeros::<u8>(dws.max(1)).unwrap();
        let mut hbuf = vec![0u8; hws.max(1)];
        let (dp, _g4) = dbuf.device_ptr_mut(st);
        let s = cs::cusolverDnXsyevBatched(dn.cu(), params, jobz, lo, t as i64, ct, gp as *mut _, t as i64, rt, wp as *mut _, ct, dp as *mut _, dws, hbuf.as_mut_ptr() as *mut _, hws, ip as *mut _, nb as i64);
        assert_eq!(s, cs::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
        cs::cusolverDnDestroyParams(params);
    }
    st.synchronize().unwrap();
    let te = t1.elapsed().as_secs_f64() * 1e3;
    let inf = st.clone_dtoh(&info).unwrap();
    (tg, te, st.clone_dtoh(&w).unwrap(), inf.iter().filter(|&&i| i != 0).count() as i32)
}

fn main() {
    let t = arg("--t", 280);
    let rows = arg("--rows", 512);
    let nb = arg("--batch", 256);
    let rank = arg("--rank", 12);
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let blas = CudaBlas::new(st.clone()).unwrap();
    let dn = DnHandle::new(st.clone()).unwrap();
    // Casorati blocks (column-major rows × t): rank-`rank` signal (decaying singular values) + unit complex noise
    let mut r = Rng::new(3);
    let mut host = vec![C32::new(0.0, 0.0); nb * rows * t];
    let mut nrm = || { let (a, b) = (r.uniform().max(1e-300), r.uniform()); (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos() };
    for b in 0..nb {
        let blk = &mut host[b * rows * t..(b + 1) * rows * t];
        for k in 0..rank {
            let amp = 200.0 * 0.6f64.powi(k as i32);
            let u: Vec<Complex64> = (0..rows).map(|_| Complex64::new(nrm(), nrm())).collect();
            let v: Vec<Complex64> = (0..t).map(|_| Complex64::new(nrm(), nrm()) / (t as f64).sqrt()).collect();
            for j in 0..t {
                for i in 0..rows {
                    let z = u[i] * v[j] * amp;
                    { let q = &mut blk[i + rows * j]; *q = C32::new(q.re + z.re as f32, q.im + z.im as f32); }
                }
            }
        }
        for z in blk.iter_mut() {
            *z = C32::new(z.re + nrm() as f32, z.im + nrm() as f32);
        }
    }
    let c32 = st.clone_htod(&host).unwrap();
    let host64: Vec<Complex64> = host.iter().map(|z| Complex64::new(z.re as f64, z.im as f64)).collect();
    let flat64: Vec<f64> = host64.iter().flat_map(|z| [z.re, z.im]).collect();
    let c64 = st.clone_htod(&flat64).unwrap();
    println!("{nb} blocks of {rows} x {t} (rank {rank} + noise)");
    // warm-up, then timed
    let _ = run_f32(&st, &blas, &dn, &c32, rows, t, nb.min(8), false);
    let (tg, te, w32, bad) = run_f32(&st, &blas, &dn, &c32, rows, t, nb, false);
    println!("f32: Gram {tg:.1} ms, XsyevBatched {te:.1} ms ({:.2} ms per block, {bad} failed)", te / nb as f64);
    let (_, tl, _, badl) = run_f32(&st, &blas, &dn, &c32, rows, t, nb.min(32), true);
    println!("f32: Xsyevd loop {:.2} ms per block ({badl} failed)", tl / nb.min(32) as f64);
    let _ = run_f64(&st, &blas, &dn, &c64, rows, t, nb.min(8));
    let (tg6, te6, w64, bad6) = run_f64(&st, &blas, &dn, &c64, rows, t, nb);
    println!("f64: Gram {tg6:.1} ms, XsyevBatched {te6:.1} ms ({:.2} ms per block, {bad6} failed)", te6 / nb as f64);
    // reference: nalgebra f64 on the first 4 blocks; compare singular values (sqrt of eigenvalues)
    for b in 0..4.min(nb) {
        let a = DMatrix::<Complex64>::from_fn(rows, t, |i, j| host64[b * rows * t + i + rows * j]);
        let e = SymmetricEigen::new(a.adjoint() * &a);
        let mut sr: Vec<f64> = e.eigenvalues.iter().map(|&l| l.max(0.0).sqrt()).collect();
        sr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let s32: Vec<f64> = w32[b * t..(b + 1) * t].iter().map(|&l| (l.max(0.0) as f64).sqrt()).collect();
        let s64: Vec<f64> = w64[b * t..(b + 1) * t].iter().map(|&l| l.max(0.0).sqrt()).collect();
        let q = t / 4;
        let rel = |s: &[f64]| s.iter().zip(&sr).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / sr[q];
        println!("  block {b}: s_max {:.1}, s[T/4] {:.3}; max |Δs| / s[T/4]: f32 {:.2e}, f64 {:.2e}", sr[t - 1], sr[q], rel(&s32), rel(&s64));
    }
}
