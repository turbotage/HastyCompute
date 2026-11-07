//! CUDA execution: one problem per thread, the whole Levenberg–Marquardt loop in one kernel. The kernel source is
//! generated per model and option set: the model's straight-line programs (value; value + Jacobian (+ second
//! derivatives)) become device functions, the solver is a line-by-line translation of `lm.rs` / `chol.rs`, and the
//! source is compiled with NVRTC (cached on disk by `hasty-cuda`). Custom objectives (e.g. VarPro) plug in by
//! supplying their own `objective` / `derivs` device functions with the same signatures ([`GpuFit::with_source`]).

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};
use hasty_cuda::{CompileSettings, Device};

use crate::fit::Data;
use crate::lm::{HessianMode, LambdaRule, LmOptions, ModChol, Status};
use crate::model::Model;

pub trait GpuScalar: DeviceRepr + ValidAsZeroBits + Copy + Default + Unpin + 'static {
    const C_TYPE: &'static str;
    const EPS: f64;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}
impl GpuScalar for f32 {
    const C_TYPE: &'static str = "float";
    const EPS: f64 = f32::EPSILON as f64;
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl GpuScalar for f64 {
    const C_TYPE: &'static str = "double";
    const EPS: f64 = f64::EPSILON;
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] hasty_cuda::Error),
    #[error("CUDA driver: {0:?}")]
    Driver(#[from] cudarc::driver::DriverError),
}

/// the solver part of the kernel (generic over T, NP, NH, FULL, MODCHOL, NIELSEN; uses `objective` / `derivs`)
const SOLVER: &str = r#"
__device__ void mc_gmw81(T* a, int* perm) {
    T d[NP]; T l[NP * NP];
    for (int k = 0; k < NP; ++k) perm[k] = k;
    T gamma = 0, xi = 0;
    for (int i = 0; i < NP; ++i) {
        gamma = fmax_t(gamma, fabs_t(a[i * NP + i]));
        for (int j = 0; j < i; ++j) xi = fmax_t(xi, fabs_t(a[i * NP + j]));
    }
    T nu = sqrt_t((T)(NP * NP) - (T)1); if (nu < (T)1) nu = (T)1;
    T beta2 = fmax_t(fmax_t(gamma, xi / nu), (T)EPS);
    T delta = (T)EPS * fmax_t(gamma + xi, (T)1);
    for (int j = 0; j < NP; ++j) {
        int q = j;
        for (int k = j + 1; k < NP; ++k) if (fabs_t(a[k * NP + k]) > fabs_t(a[q * NP + q])) q = k;
        if (q != j) {
            for (int c = 0; c < NP; ++c) { T t = a[j * NP + c]; a[j * NP + c] = a[q * NP + c]; a[q * NP + c] = t; }
            for (int r = 0; r < NP; ++r) { T t = a[r * NP + j]; a[r * NP + j] = a[r * NP + q]; a[r * NP + q] = t; }
            for (int s = 0; s < j; ++s) { T t = l[j * NP + s]; l[j * NP + s] = l[q * NP + s]; l[q * NP + s] = t; }
            int t = perm[j]; perm[j] = perm[q]; perm[q] = t;
        }
        for (int s = 0; s < j; ++s) l[j * NP + s] = a[j * NP + s] / d[s];
        T theta = 0;
        for (int i = j + 1; i < NP; ++i) {
            T c = a[i * NP + j];
            for (int s = 0; s < j; ++s) c -= l[j * NP + s] * a[i * NP + s];
            a[i * NP + j] = c;
            theta = fmax_t(theta, fabs_t(c));
        }
        T cjj = a[j * NP + j];
        d[j] = fmax_t(fmax_t(delta, fabs_t(cjj)), theta * theta / beta2);
        for (int i = j + 1; i < NP; ++i) { T c = a[i * NP + j]; a[i * NP + i] -= c * c / d[j]; }
    }
    for (int i = 0; i < NP; ++i)
        for (int j = 0; j < NP; ++j)
            a[i * NP + j] = (j < i) ? l[i * NP + j] * sqrt_t(d[j]) : ((j == i) ? sqrt_t(d[i]) : (T)0);
}

__device__ void swap_sym(T* a, int i, int j) {
    if (i == j) return;
    for (int c = 0; c < NP; ++c) { T t = a[i * NP + c]; a[i * NP + c] = a[j * NP + c]; a[j * NP + c] = t; }
    for (int r = 0; r < NP; ++r) { T t = a[r * NP + i]; a[r * NP + i] = a[r * NP + j]; a[r * NP + j] = t; }
}
__device__ void chol_step(T* a, int j) {
    T ljj = sqrt_t(a[j * NP + j]);
    a[j * NP + j] = ljj;
    for (int i = j + 1; i < NP; ++i) a[i * NP + j] /= ljj;
    for (int i = j + 1; i < NP; ++i)
        for (int k = j + 1; k <= i; ++k) { T v = a[i * NP + k] - a[i * NP + j] * a[k * NP + j]; a[i * NP + k] = v; a[k * NP + i] = v; }
}
__device__ void mc_se99(T* a, int* perm) {
    for (int k = 0; k < NP; ++k) perm[k] = k;
    T tau = cbrt_t((T)EPS), taubar = tau * tau, mu = (T)0.1;
    T gamma = 0;
    for (int i = 0; i < NP; ++i) gamma = fmax_t(gamma, fabs_t(a[i * NP + i]));
    if (gamma == (T)0) gamma = (T)1;
    int j = 0; bool phase1 = true; T delta_prev = 0;
    while (j < NP) {
        int imax = j;
        for (int i = j + 1; i < NP; ++i) if (a[i * NP + i] > a[imax * NP + imax]) imax = i;
        if (a[imax * NP + imax] < taubar * gamma) { phase1 = false; break; }
        swap_sym(a, j, imax); { int t = perm[j]; perm[j] = perm[imax]; perm[imax] = t; }
        T ajj = a[j * NP + j]; bool ok = true;
        for (int i = j + 1; i < NP; ++i) { T aij = a[i * NP + j]; if (a[i * NP + i] - aij * aij / ajj < -mu * gamma) { ok = false; break; } }
        if (!ok) { phase1 = false; break; }
        chol_step(a, j);
        ++j;
    }
    if (!phase1) {
        if (j == NP - 1) {
            T ajj = a[j * NP + j];
            T dl = fmax_t(-ajj + fmax_t(tau * (-ajj) / ((T)1 - tau), taubar * gamma), (T)0);
            a[j * NP + j] = sqrt_t(ajj + dl);
        } else {
            int k0 = j; T g[NP];
            for (int i = k0; i < NP; ++i) { T s = 0; for (int c = k0; c < NP; ++c) if (c != i) s += fabs_t(a[i * NP + c]); g[i] = a[i * NP + i] - s; }
            for (int jj = k0; jj < NP - 2; ++jj) {
                int imax = jj;
                for (int i = jj + 1; i < NP; ++i) if (g[i] > g[imax]) imax = i;
                if (imax != jj) { swap_sym(a, jj, imax); int t = perm[jj]; perm[jj] = perm[imax]; perm[imax] = t; T tg = g[jj]; g[jj] = g[imax]; g[imax] = tg; }
                T normj = 0; for (int i = jj + 1; i < NP; ++i) normj += fabs_t(a[i * NP + jj]);
                T ajj = a[jj * NP + jj];
                T dl = fmax_t(fmax_t(-ajj + fmax_t(normj, taubar * gamma), delta_prev), (T)0);
                if (dl > (T)0) { a[jj * NP + jj] = ajj + dl; delta_prev = dl; }
                ajj = a[jj * NP + jj];
                if (ajj != normj) { T t = (T)1 - normj / ajj; for (int i = jj + 1; i < NP; ++i) g[i] += fabs_t(a[i * NP + jj]) * t; }
                chol_step(a, jj);
            }
            int p = NP - 2, q = NP - 1;
            T aa = a[p * NP + p], bb = a[q * NP + p], cc = a[q * NP + q];
            T tr = (T)0.5 * (aa + cc), disc = sqrt_t((T)0.25 * (aa - cc) * (aa - cc) + bb * bb);
            T lhi = tr + disc, llo = tr - disc;
            T dl = fmax_t(fmax_t(-llo + fmax_t(tau * (lhi - llo) / ((T)1 - tau), taubar * gamma), delta_prev), (T)0);
            if (dl > (T)0) { aa += dl; cc += dl; }
            T l11 = sqrt_t(aa), l21 = bb / l11;
            a[p * NP + p] = l11; a[q * NP + p] = l21; a[q * NP + q] = sqrt_t(fmax_t(cc - l21 * l21, (T)0));
        }
    }
    for (int i = 0; i < NP; ++i) for (int c = i + 1; c < NP; ++c) a[i * NP + c] = (T)0;
}

__device__ void chol_solve(const T* l, const int* perm, const T* b, T* x) {
    T y[NP];
    for (int k = 0; k < NP; ++k) { T s = b[perm[k]]; for (int c = 0; c < k; ++c) s -= l[k * NP + c] * y[c]; y[k] = s / l[k * NP + k]; }
    for (int k = NP - 1; k >= 0; --k) { T s = y[k]; for (int r = k + 1; r < NP; ++r) s -= l[r * NP + k] * y[r]; y[k] = s / l[k * NP + k]; }
    for (int k = 0; k < NP; ++k) x[perm[k]] = y[k];
}

// projected Newton decrement in the scaled variables (see lm.rs::newton_decrement); returns -1 if not applicable
__device__ T newton_decrement(const T* h, const T* g, const T* sc, const T* p, const T* lo, const T* hi, T* a) {
    int idx[NP]; int nf = 0;
    for (int j = 0; j < NP; ++j) { bool at_lo = p[j] <= lo[j] && g[j] > (T)0, at_hi = p[j] >= hi[j] && g[j] < (T)0; if (!(at_lo || at_hi)) idx[nf++] = j; }
    if (nf == 0) return (T)0;
    T dmax = 0;
    for (int r = 0; r < nf; ++r) { for (int c = 0; c < nf; ++c) a[r * nf + c] = sc[idx[r]] * sc[idx[c]] * h[idx[r] * NP + idx[c]]; dmax = fmax_t(dmax, a[r * nf + r]); }
    if (!(dmax > (T)0)) return (T)-1;
    T d2 = 0; T z[NP];
    for (int j = 0; j < nf; ++j) {
        T v = a[j * nf + j];
        for (int k = 0; k < j; ++k) v -= a[j * nf + k] * a[j * nf + k];
        if (!(v > (T)1e3 * (T)EPS * dmax)) return (T)-1;
        T ljj = sqrt_t(v); a[j * nf + j] = ljj;
        for (int i = j + 1; i < nf; ++i) { T u = a[i * nf + j]; for (int k = 0; k < j; ++k) u -= a[i * nf + k] * a[j * nf + k]; a[i * nf + j] = u / ljj; }
        T t = g[idx[j]] * sc[idx[j]];
        for (int k = 0; k < j; ++k) t -= a[j * nf + k] * z[k];
        z[j] = t / ljj; d2 += z[j] * z[j];
    }
    return d2;
}

extern "C" __global__ void qfit_lm(T* params, const T* lo, const T* hi, const T* sc, const T* vars, const T* fixed, const T* y, const T* w, int has_w,
    int nprob, int ndata, int max_iter, T tol_grad, T tol_step, T tol_fun, T tol_stat, T lambda0, T d_min, T mu, T eta, T down, T up,
    T* fout, int* iters, int* status)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nprob) return;
    const T* wp = has_w ? w : (const T*)0;
    T p[NP], pt[NP], g[NP], h[NP * NP], a[NP * NP], d[NP], s[NP], ng[NP], fx[NF1];
    int perm[NP];
    for (int k = 0; k < NF; ++k) fx[k] = fixed[k * nprob + tid];
    for (int j = 0; j < NP; ++j) { p[j] = params[j * nprob + tid]; p[j] = fmin_t(fmax_t(p[j], lo[j]), hi[j]); }
    T f = derivs(p, fx, vars, y, wp, ndata, nprob, tid, g, h, d);
    T lambda = lambda0, nu = (T)2;
    int st = 4, it = 1;
    for (; it <= max_iter; ++it) {
        if (!isfinite(f)) { st = 5; --it; break; }
        T pg = 0;
        for (int j = 0; j < NP; ++j) {
            bool at_lo = p[j] <= lo[j] && g[j] > (T)0, at_hi = p[j] >= hi[j] && g[j] < (T)0;
            if (!(at_lo || at_hi)) pg = fmax_t(pg, fabs_t(g[j] * sc[j]));
        }
        if (pg <= tol_grad) { st = 1; --it; break; }
        if (tol_stat > (T)0 && ndata * NOUT > NP) {
            T sig2 = (T)2 * f / (T)(ndata * NOUT - NP);
            T d2 = newton_decrement(h, g, sc, p, lo, hi, a);
            if (d2 >= (T)0 && d2 <= tol_stat * tol_stat * sig2) { st = 7; --it; break; }
        }
        T dmax = 0; for (int j = 0; j < NP; ++j) dmax = fmax_t(dmax, d[j] * sc[j] * sc[j]);
        T floor_ = fmax_t(d_min * dmax, (T)TINY);
        for (int i = 0; i < NP; ++i) for (int j = 0; j < NP; ++j) a[i * NP + j] = sc[i] * sc[j] * h[i * NP + j];
        for (int j = 0; j < NP; ++j) a[j * NP + j] += lambda * fmax_t(d[j] * sc[j] * sc[j], floor_);
        bool fr[NP];
        for (int j = 0; j < NP; ++j) { bool at_lo = p[j] <= lo[j] && g[j] > (T)0, at_hi = p[j] >= hi[j] && g[j] < (T)0; fr[j] = !(at_lo || at_hi); }
        for (int j = 0; j < NP; ++j) if (!fr[j]) { for (int c = 0; c < NP; ++c) { a[j * NP + c] = 0; a[c * NP + j] = 0; } a[j * NP + j] = 1; }
#if MODCHOL == 0
        mc_gmw81(a, perm);
#else
        mc_se99(a, perm);
#endif
        for (int j = 0; j < NP; ++j) ng[j] = fr[j] ? -g[j] * sc[j] : (T)0;
        chol_solve(a, perm, ng, s);
        T smax = 0, pmax = 0;
        for (int j = 0; j < NP; ++j) {
            pt[j] = fmin_t(fmax_t(p[j] + s[j] * sc[j], lo[j]), hi[j]);
            s[j] = pt[j] - p[j];
            smax = fmax_t(smax, fabs_t(s[j] / sc[j])); pmax = fmax_t(pmax, fabs_t(p[j] / sc[j]));
        }
        if (smax <= tol_step * (pmax + tol_step)) { st = 2; break; }
        T pred = 0;
        for (int i = 0; i < NP; ++i) { T hs = 0; for (int j = 0; j < NP; ++j) hs += h[i * NP + j] * s[j]; pred -= g[i] * s[i] + (T)0.5 * s[i] * hs; }
        T ft = objective(pt, fx, vars, y, wp, ndata, nprob, tid);
        T act = f - ft;
        T rho = pred > (T)0 ? act / pred : (T)-1;
#if NIELSEN
        bool accept = isfinite(ft) && pred > (T)0 && act > (T)0 && rho > (T)0;
#else
        bool accept = isfinite(ft) && pred > (T)0 && act > (T)0 && rho > mu;
#endif
        if (accept) {
            for (int j = 0; j < NP; ++j) p[j] = pt[j];
            T f_old = f;
            f = derivs(p, fx, vars, y, wp, ndata, nprob, tid, g, h, d);
#if NIELSEN
            { T t = (T)2 * rho - (T)1; lambda *= fmax_t((T)1 / (T)3, (T)1 - t * t * t); nu = (T)2; }
#else
            if (rho > eta) lambda *= down;
#endif
            lambda = fmax_t(lambda, (T)1e-15);
            if (f_old - f <= tol_fun * f_old) { st = 3; break; }
        } else {
#if NIELSEN
            lambda *= nu; nu *= (T)2;
#else
            lambda *= up;
#endif
            if (lambda > (T)1e16) { st = 6; break; }
        }
    }
    if (it > max_iter) it = max_iter;
    for (int j = 0; j < NP; ++j) params[j * nprob + tid] = p[j];
    fout[tid] = f; iters[tid] = it; status[tid] = st;
}
"#;

/// the symbolic-model objective as device functions (value; derivatives with full or Gauss–Newton Hessian)
fn model_source(m: &Model, ty: &str, full: bool) -> String {
    let nh = m.n_hes();
    let prog = if full { &m.prog_full } else { &m.prog_gn };
    let nder = prog.outputs.len();
    format!(
        r#"
__device__ __forceinline__ void m_val(const T* p, const T* v, const T* fx, T* out) {{
{val}}}
__device__ __forceinline__ void m_der(const T* p, const T* v, const T* fx, T* out) {{
{der}}}
__device__ T objective(const T* p, const T* fx, const T* vars, const T* y, const T* w, int ndata, int nprob, int tid) {{
    T v[NV1]; T m[NOUT]; T f = 0;
    for (int i = 0; i < ndata; ++i) {{
        for (int k = 0; k < NV; ++k) v[k] = vars[k * ndata + i];
        m_val(p, v, fx, m);
        T wi = w ? w[i * nprob + tid] : (T)1;
        for (int k = 0; k < NOUT; ++k) {{ T r = m[k] - y[(k * ndata + i) * nprob + tid]; f += (T)0.5 * wi * r * r; }}
    }}
    return f;
}}
__device__ T derivs(const T* p, const T* fx, const T* vars, const T* y, const T* w, int ndata, int nprob, int tid, T* g, T* h, T* d) {{
    T v[NV1]; T o[{nder}]; T f = 0;
    for (int j = 0; j < NP; ++j) {{ g[j] = 0; d[j] = 0; }}
    for (int j = 0; j < NP * NP; ++j) h[j] = 0;
    for (int i = 0; i < ndata; ++i) {{
        for (int k = 0; k < NV; ++k) v[k] = vars[k * ndata + i];
        m_der(p, v, fx, o);
        T wi = w ? w[i * nprob + tid] : (T)1;
        for (int k = 0; k < NOUT; ++k) {{
            T r = o[k] - y[(k * ndata + i) * nprob + tid];
            f += (T)0.5 * wi * r * r;
            const T* jk = o + NOUT + k * NP;
            for (int a = 0; a < NP; ++a) {{
                T ja = jk[a];
                g[a] += wi * r * ja; d[a] += wi * ja * ja;
                for (int b = 0; b <= a; ++b) {{
                    T v2 = ja * jk[b];
#if FULL
                    v2 += r * o[NOUT + NOUT * NP + k * {nh} + a * (a + 1) / 2 + b];
#endif
                    h[a * NP + b] += wi * v2;
                }}
            }}
        }}
    }}
    for (int a = 0; a < NP; ++a) for (int b = 0; b < a; ++b) h[b * NP + a] = h[a * NP + b];
    return f;
}}
"#,
        val = m.prog_val.cuda(ty, "out"),
        der = prog.cuda(ty, "out"),
        nder = nder.max(1),
        nh = nh,
    )
}

fn header(ty: &str, eps: f64, np: usize, nout: usize, nv: usize, nf: usize, o: &LmOptions) -> String {
    let f32 = ty == "float";
    let fun = |b: &str| if f32 { format!("{b}f") } else { b.to_string() };
    format!(
        "typedef {ty} T;\n#define NP {np}\n#define NOUT {nout}\n#define NV {nv}\n#define NV1 {nv1}\n#define NF {nf}\n#define NF1 {nf1}\n#define FULL {full}\n#define MODCHOL {mc}\n#define NIELSEN {ni}\n#define EPS {eps:e}\n#define TINY {tiny:e}\n\
         #define fmax_t {fmax}\n#define fmin_t {fmin}\n#define fabs_t {fabs}\n#define sqrt_t {sqrt}\n#define cbrt_t {cbrt}\n",
        nv1 = nv.max(1),
        nf1 = nf.max(1),
        full = (o.hessian == HessianMode::Full) as i32,
        mc = match o.modchol {
            ModChol::Gmw81 => 0,
            ModChol::Se99 => 1,
        },
        ni = matches!(o.rule, LambdaRule::Nielsen) as i32,
        tiny = if f32 { f32::MIN_POSITIVE as f64 } else { f64::MIN_POSITIVE },
        fmax = fun("fmax"),
        fmin = fun("fmin"),
        fabs = fun("fabs"),
        sqrt = fun("sqrt"),
        cbrt = fun("cbrt"),
    )
}

/// a compiled solver for one model, scalar type and option set
pub struct GpuFit {
    pub dev: Device,
    pub stream: Arc<CudaStream>,
    func: CudaFunction,
    np: usize,
    nout: usize,
    nv: usize,
    nf: usize,
    pub source: String,
}

/// a batch's data on the device (layout as [`Data`])
pub struct GpuData<T: GpuScalar> {
    pub n_prob: usize,
    pub n_data: usize,
    nv: usize,
    nf: usize,
    nout: usize,
    vars: cudarc::driver::CudaSlice<T>,
    fixed: cudarc::driver::CudaSlice<T>,
    y: cudarc::driver::CudaSlice<T>,
    w: cudarc::driver::CudaSlice<T>,
    has_w: i32,
}

impl<T: GpuScalar> GpuData<T> {
    /// the data y [n_out][n_data][n_prob] on the device
    pub fn y(&self) -> &cudarc::driver::CudaSlice<T> {
        &self.y
    }
    pub fn y_mut(&mut self) -> &mut cudarc::driver::CudaSlice<T> {
        &mut self.y
    }
    pub fn upload(st: &Arc<CudaStream>, d: &Data<T>) -> Result<GpuData<T>, Error> {
        let (n, nd) = (d.n_prob, d.n_data);
        assert!(nd > 0 && n > 0);
        let nv = d.vars.len() / nd;
        assert_eq!(d.vars.len(), nv * nd);
        let nf = d.fixed.len() / n;
        assert_eq!(d.fixed.len(), nf * n);
        let nout = d.y.len() / (nd * n);
        assert_eq!(d.y.len(), nout * nd * n);
        if let Some(w) = d.w {
            assert_eq!(w.len(), nd * n);
        }
        let dummy = [T::default()];
        Ok(GpuData {
            n_prob: n,
            n_data: nd,
            nv,
            nf,
            nout,
            vars: st.clone_htod(if d.vars.is_empty() { &dummy[..] } else { d.vars })?,
            fixed: st.clone_htod(if d.fixed.is_empty() { &dummy[..] } else { d.fixed })?,
            y: st.clone_htod(d.y)?,
            w: st.clone_htod(d.w.unwrap_or(&dummy[..]))?,
            has_w: d.w.is_some() as i32,
        })
    }
}

/// per-problem results on the device
pub struct GpuOutDev<T> {
    pub f: cudarc::driver::CudaSlice<T>,
    pub iters: cudarc::driver::CudaSlice<i32>,
    pub status: cudarc::driver::CudaSlice<i32>,
}

#[derive(Clone, Debug)]
pub struct GpuOut<T> {
    pub f: Vec<T>,
    pub iters: Vec<i32>,
    pub status: Vec<Status>,
}

impl GpuFit {
    /// compile the solver for a symbolic model
    pub fn new<T: GpuScalar>(dev: &Device, m: &Model, o: &LmOptions) -> Result<GpuFit, Error> {
        let src = model_source(m, T::C_TYPE, o.hessian == HessianMode::Full);
        Self::with_source::<T>(dev, m.n_param(), m.n_out(), m.n_var(), m.n_fixed(), &src, o)
    }
    /// compile the solver around user device code defining
    /// `T objective(const T* p, const T* fx, const T* vars, const T* y, const T* w, int ndata, int nprob, int tid)` and
    /// `T derivs(… same …, T* g, T* h, T* d)` (h full NP×NP, d = Marquardt scaling diagonal)
    pub fn with_source<T: GpuScalar>(dev: &Device, np: usize, nout: usize, nv: usize, nf: usize, objective_src: &str, o: &LmOptions) -> Result<GpuFit, Error> {
        let src = format!("{}{}{}", header(T::C_TYPE, T::EPS, np, nout, nv, nf, o), objective_src, SOLVER);
        let func = dev.load_function("qfit_lm", &src, "qfit_lm", &CompileSettings::default())?;
        Ok(GpuFit { dev: dev.clone(), stream: dev.default_stream(), func, np, nout, nv, nf, source: src })
    }

    /// fit all problems; `params` [n_param][n_prob] in/out (uploads the data; see [`GpuFit::fit_on`] to reuse it)
    pub fn fit<T: GpuScalar>(&self, d: &Data<T>, params: &mut [T], lo: &[T], hi: &[T], o: &LmOptions) -> Result<GpuOut<T>, Error> {
        let gd = GpuData::upload(&self.stream, d)?;
        self.fit_on(&gd, params, lo, hi, o)
    }

    /// fit on data already on the device (several starts / option sets on the same data)
    pub fn fit_on<T: GpuScalar>(&self, gd: &GpuData<T>, params: &mut [T], lo: &[T], hi: &[T], o: &LmOptions) -> Result<GpuOut<T>, Error> {
        let st = &self.stream;
        assert_eq!(params.len(), self.np * gd.n_prob);
        let mut p_d = st.clone_htod(params)?;
        let r = self.fit_dev(gd, &mut p_d, lo, hi, o)?;
        params.copy_from_slice(&st.clone_dtoh(&p_d)?);
        Ok(GpuOut { f: st.clone_dtoh(&r.f)?, iters: st.clone_dtoh(&r.iters)?, status: st.clone_dtoh(&r.status)?.into_iter().map(Status::from_i32).collect() })
    }

    /// fully on the device: `p_d` [n_param][n_prob] in/out; per-problem results stay on the device
    pub fn fit_dev<T: GpuScalar>(&self, gd: &GpuData<T>, p_d: &mut cudarc::driver::CudaSlice<T>, lo: &[T], hi: &[T], o: &LmOptions) -> Result<GpuOutDev<T>, Error> {
        let st = &self.stream;
        let n = gd.n_prob;
        assert_eq!(p_d.len(), self.np * n);
        assert_eq!(gd.nv, self.nv, "data has a different number of vars");
        assert_eq!(gd.nf, self.nf, "data has a different number of fixed inputs");
        assert_eq!(gd.nout, self.nout, "data has a different number of outputs");
        let lo_d = st.clone_htod(lo)?;
        let hi_d = st.clone_htod(hi)?;
        let lo64: Vec<f64> = lo.iter().map(|x| x.to_f64()).collect();
        let hi64: Vec<f64> = hi.iter().map(|x| x.to_f64()).collect();
        let sc: Vec<T> = crate::lm::scales(o, &lo64, &hi64).into_iter().map(T::from_f64).collect();
        let sc_d = st.clone_htod(&sc)?;
        let (vars_d, fixed_d, y_d, w_d, has_w) = (&gd.vars, &gd.fixed, &gd.y, &gd.w, gd.has_w);
        let mut f_d = st.alloc_zeros::<T>(n)?;
        let mut it_d = st.alloc_zeros::<i32>(n)?;
        let mut stat_d = st.alloc_zeros::<i32>(n)?;
        let (nprob, ndata, max_iter) = (n as i32, gd.n_data as i32, o.max_iter as i32);
        let (mu, eta, down, up) = match o.rule {
            LambdaRule::Classic { mu, eta, down, up } => (mu, eta, down, up),
            LambdaRule::Nielsen => (0.0, 0.0, 1.0, 2.0),
        };
        let sc = [o.tol_grad, o.tol_step, o.tol_fun, o.tol_stat, o.lambda0, o.d_min, mu, eta, down, up].map(T::from_f64);
        let block = 128u32;
        let cfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(block).max(1), 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            st.launch_builder(&self.func)
                .arg(&mut *p_d)
                .arg(&lo_d)
                .arg(&hi_d)
                .arg(&sc_d)
                .arg(vars_d)
                .arg(fixed_d)
                .arg(y_d)
                .arg(w_d)
                .arg(&has_w)
                .arg(&nprob)
                .arg(&ndata)
                .arg(&max_iter)
                .arg(&sc[0])
                .arg(&sc[1])
                .arg(&sc[2])
                .arg(&sc[3])
                .arg(&sc[4])
                .arg(&sc[5])
                .arg(&sc[6])
                .arg(&sc[7])
                .arg(&sc[8])
                .arg(&sc[9])
                .arg(&mut f_d)
                .arg(&mut it_d)
                .arg(&mut stat_d)
                .launch(cfg)?;
        }
        Ok(GpuOutDev { f: f_d, iters: it_d, status: stat_d })
    }
}
