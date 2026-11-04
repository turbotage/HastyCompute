//! CUDA C++ source generation for one FFT "axis pass": a shared-memory Stockham autosort kernel that can optionally
//!   * read only the first `n_in` entries of every line (the rest are implicit zeros: zero-padded input, never read),
//!   * write only the first `n_out` entries (cropped output),
//!   * multiply loaded values by diagonal arrays (`pre`) and stored values by diagonal arrays (`post`), optionally accumulating into the output,
//!   * fuse "forward transform -> pointwise multiply by a kernel array -> inverse transform" in a single kernel (`conv`).
//!
//! Line addressing is fully runtime: a line is `(a, b)`; element `n` of the line is at `a*la + b*lb + n*sn` (separately for input
//! and output). If the transformed axis is contiguous (`sn == 1`) one thread group handles one line; otherwise a thread block handles a tile
//! of `lpb` consecutive `a` values (which must be the stride-1 dimension) so global accesses stay coalesced.

use std::fmt::Write;

use crate::radix::emit_butterfly;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DiagOp {
    /// v *= d
    Mult,
    /// v *= conj(d)
    MultConj,
}

/// Storage format of the k-space multiplier `kern` used by fused convolution passes (smaller = less DRAM/PCIe traffic).
/// All formats are multiplied by the runtime scalar `kscale` after decoding (use it to bring fp16 data into range).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelFormat {
    /// interleaved complex in the transform precision
    Complex,
    /// real scalars in the transform precision (kernel known to be real, e.g. real symmetric weights)
    Real,
    /// complex as two IEEE binary16 packed in a u32 (re in the low half)
    ComplexHalf,
    /// complex as two bfloat16 packed in a u32 (re in the low half)
    ComplexBf16,
}

/// One term `coef * xi_x^px * xi_y^py * xi_z^pz` of a real polynomial diagonal; `xi_a = (idx_a - (N_a-1)/2) * 2/N_a` are the
/// FOV-normalised voxel coordinates. Evaluated in registers from the element position (needs `vox` in the pass config).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolyTerm {
    pub coef: f64,
    pub p: [u8; 3],
}

#[derive(Clone, Debug)]
pub struct AxisCfg {
    pub n: usize,
    pub radices: Vec<usize>,
    pub contiguous: bool,
    /// lines per block (contiguous) or tile width in lines (strided)
    pub lpb: usize,
    /// threads per line
    pub tpl: usize,
    pub ctype: &'static str,
    /// direction of the (first) transform; with `conv` the second one has the opposite direction
    pub forward: bool,
    pub scale: f64,
    pub n_in: usize,
    pub n_out: usize,
    /// forward -> multiply by `kern` -> inverse inside one kernel
    pub conv: bool,
    pub pre: Vec<DiagOp>,
    pub post: Vec<DiagOp>,
    pub accumulate: bool,
    pub kfmt: KernelFormat,
    /// real polynomial diagonals applied after `pre` arrays on load / after `post` arrays on store (X-axis passes of a rank-3 pipeline)
    pub poly_pre: bool,
    pub poly_post: bool,
    /// voxel grid `[Nx, Ny, Nz]` the polynomial coordinates refer to (x = position along the pass axis, y = `la`, z = `lb`)
    pub vox: [usize; 3],
}

impl AxisCfg {
    pub fn simple(
        n: usize,
        radices: Vec<usize>,
        contiguous: bool,
        lpb: usize,
        tpl: usize,
        ctype: &'static str,
        forward: bool,
        scale: f64,
    ) -> Self {
        Self {
            n,
            radices,
            contiguous,
            lpb,
            tpl,
            ctype,
            forward,
            scale,
            n_in: n,
            n_out: n,
            conv: false,
            pre: vec![],
            post: vec![],
            accumulate: false,
            kfmt: KernelFormat::Complex,
            poly_pre: false,
            poly_post: false,
            vox: [1, 1, 1],
        }
    }
    pub fn block_threads(&self) -> usize {
        self.lpb * self.tpl
    }
    pub fn passes(&self) -> usize {
        self.radices.len() * if self.conv { 2 } else { 1 }
    }
    /// All radices divide the largest one: every thread then owns exactly `rmax` values per pass, so passes can run in place
    /// in a single shared buffer (read all -> barrier -> write all -> barrier).
    pub fn inplace(&self) -> bool {
        let rmax = *self.radices.iter().max().unwrap();
        self.radices.iter().all(|r| rmax % r == 0) && self.tpl * rmax == self.n
    }
    /// bytes of dynamic shared memory (only needed when more than one pass)
    pub fn smem_bytes(&self, real_bytes: usize) -> usize {
        if self.passes() > 1 {
            (if self.inplace() { 1 } else { 2 }) * self.n * self.lpb * 2 * real_bytes
        } else {
            0
        }
    }
}

pub(crate) fn poly_expr(table: &str, nterms: &str, x: &str, y: &str, z: &str, vox: [usize; 3]) -> String {
    format!(
        "poly_eval_({table}, {nterms}, ((real)({x}) - (real){:.17e}) * (real){:.17e}, ((real)({y}) - (real){:.17e}) * (real){:.17e}, ((real)({z}) - (real){:.17e}) * (real){:.17e})",
        (vox[0] as f64 - 1.0) / 2.0,
        2.0 / vox[0] as f64,
        (vox[1] as f64 - 1.0) / 2.0,
        2.0 / vox[1] as f64,
        (vox[2] as f64 - 1.0) / 2.0,
        2.0 / vox[2] as f64
    )
}

struct Pass {
    r: usize,
    ns: usize,
    sign: i32,
    /// last pass of the forward half of a `conv` kernel: multiply by the kernel array before storing
    fuse_kernel: bool,
}

pub(crate) fn sfx(sign: i32) -> &'static str {
    if sign < 0 {
        "m"
    } else {
        "p"
    }
}

/// Type definitions and complex/polynomial/half-precision helpers shared by all generated axis kernels.
pub(crate) fn prelude(ctype: &str) -> String {
    let mut s = String::new();
    writeln!(s, "typedef {ctype} real;").unwrap();
    writeln!(s, "struct cx {{ real x, y; }};").unwrap();
    writeln!(s, "#define RL(v) ((real)(v))").unwrap();
    writeln!(
        s,
        r#"__device__ __forceinline__ cx make_cx(real x, real y) {{ cx r; r.x = x; r.y = y; return r; }}
__device__ __forceinline__ cx cadd(cx a, cx b) {{ return make_cx(a.x + b.x, a.y + b.y); }}
__device__ __forceinline__ cx csub(cx a, cx b) {{ return make_cx(a.x - b.x, a.y - b.y); }}
__device__ __forceinline__ cx cscale(cx a, real c) {{ return make_cx(a.x * c, a.y * c); }}
__device__ __forceinline__ cx cmul(cx a, cx b) {{ return make_cx(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }}
__device__ __forceinline__ cx cmulc(cx a, cx b) {{ return make_cx(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y); }}
__device__ __forceinline__ real ipow_(real x, int p) {{ real r = (real)1; for (int i = 0; i < p; ++i) r *= x; return r; }}
__device__ __forceinline__ real poly_eval_(const real* t, int n, real cx_, real cy_, real cz_) {{ real acc = (real)0; for (int i = 0; i < n; ++i) acc += t[4 * i] * ipow_(cx_, (int)t[4 * i + 1]) * ipow_(cy_, (int)t[4 * i + 2]) * ipow_(cz_, (int)t[4 * i + 3]); return acc; }}
__device__ __forceinline__ real h2r(unsigned int h) {{ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short)h)); return (real)f; }}
__device__ __forceinline__ real b2r(unsigned int h) {{ return (real)__uint_as_float(h << 16); }}"#
    )
    .unwrap();
    s
}

/// Parameter type of the k-space multiplier `kern` for a kernel format.
pub(crate) fn kern_type(kfmt: KernelFormat) -> &'static str {
    match kfmt {
        KernelFormat::Complex => "const cx*",
        KernelFormat::Real => "const real*",
        KernelFormat::ComplexHalf | KernelFormat::ComplexBf16 => "const unsigned int*",
    }
}

/// Expression for `kern` at element address `GIN(e)` scaled by `kscale`, multiplied into `v` (fused convolution).
pub(crate) fn kern_mul(kfmt: KernelFormat, v: &str) -> String {
    match kfmt {
        KernelFormat::Complex => format!("cmul({v}, cscale(kern[GIN(e)], kscale))"),
        KernelFormat::Real => format!("cscale({v}, kern[GIN(e)] * kscale)"),
        KernelFormat::ComplexHalf => format!("cmul({v}, make_cx(h2r(kern[GIN(e)] & 0xffffu) * kscale, h2r(kern[GIN(e)] >> 16) * kscale))"),
        KernelFormat::ComplexBf16 => format!("cmul({v}, make_cx(b2r(kern[GIN(e)] & 0xffffu) * kscale, b2r(kern[GIN(e)] >> 16) * kscale))"),
    }
}

pub fn generate(cfg: &AxisCfg) -> String {
    let n = cfg.n;
    let sign1 = if cfg.forward { -1 } else { 1 };
    let mut passes = Vec::new();
    for half in 0..(if cfg.conv { 2 } else { 1 }) {
        let sign = if half == 0 { sign1 } else { -sign1 };
        let mut ns = 1usize;
        for (i, &r) in cfg.radices.iter().enumerate() {
            passes.push(Pass {
                r,
                ns,
                sign,
                fuse_kernel: cfg.conv && half == 0 && i == cfg.radices.len() - 1,
            });
            ns *= r;
        }
    }
    let p = passes.len();
    let mut s = prelude(cfg.ctype);
    let kty = match cfg.kfmt {
        KernelFormat::Complex => "const cx*",
        KernelFormat::Real => "const real*",
        KernelFormat::ComplexHalf | KernelFormat::ComplexBf16 => "const unsigned int*",
    };
    let mut done: Vec<(usize, i32)> = vec![];
    for ps in &passes {
        if !done.contains(&(ps.r, ps.sign)) {
            s.push_str(&emit_butterfly(ps.r, ps.sign));
            done.push((ps.r, ps.sign));
        }
    }
    writeln!(
        s,
        "#define N {n}\n#define N_IN {}\n#define N_OUT {}\n#define LPB {}\n#define TPL {}",
        cfg.n_in, cfg.n_out, cfg.lpb, cfg.tpl
    )
    .unwrap();
    writeln!(
        s,
        "extern \"C\" __global__ void __launch_bounds__({}) fft_axis(const cx* __restrict__ in, cx* out, const cx* __restrict__ tw_m, const cx* __restrict__ tw_p, {kty} __restrict__ kern, real kscale,\n  const cx* __restrict__ p0, const cx* __restrict__ p1, const cx* __restrict__ p2, const cx* __restrict__ p3,\n  const cx* __restrict__ q0, const cx* __restrict__ q1, const cx* __restrict__ q2, const cx* __restrict__ q3,\n  const real* __restrict__ ppre, int nppre, const real* __restrict__ ppost, int nppost,\n  long la_cnt, long lb_cnt, long in_la, long in_lb, long in_sn, long out_la, long out_lb, long out_sn) {{",
        cfg.block_threads()
    )
    .unwrap();
    writeln!(s, "  extern __shared__ cx smem[];").unwrap();
    if cfg.contiguous {
        writeln!(
            s,
            r#"  const int tid = threadIdx.x;
  const int li = tid / TPL, tl = tid % TPL;
  const long line = (long)blockIdx.x * LPB + li;
  const bool act = line < la_cnt * lb_cnt;
  const long la = line % la_cnt, lb = line / la_cnt;
  const long ib = la * in_la + lb * in_lb, ob_ = la * out_la + lb * out_lb;
#define SIDX(n) (li * N + (n))"#
        )
        .unwrap();
    } else {
        writeln!(
            s,
            r#"  const int tid = threadIdx.x;
  const int li = tid % LPB, tl = tid / LPB;
  const long la = (long)blockIdx.x * LPB + li;
  const long lb = blockIdx.y;
  const bool act = la < la_cnt;
  const long ib = la * in_la + lb * in_lb, ob_ = la * out_la + lb * out_lb;
#define SIDX(n) ((n) * LPB + li)"#
        )
        .unwrap();
    }
    writeln!(
        s,
        "#define GIN(n) (ib + (long)(n) * in_sn)\n#define GOUT(n) (ob_ + (long)(n) * out_sn)"
    )
    .unwrap();
    let inplace = cfg.inplace() && p > 1;
    if p > 1 {
        if inplace {
            writeln!(s, "  cx* bufs[2] = {{ smem, smem }};").unwrap();
        } else {
            writeln!(s, "  cx* bufs[2] = {{ smem, smem + N * LPB }};").unwrap();
        }
    }
    let rmax = *cfg.radices.iter().max().unwrap();
    let pre_ptr = ["p0", "p1", "p2", "p3"];
    let post_ptr = ["q0", "q1", "q2", "q3"];
    for (pi, ps) in passes.iter().enumerate() {
        let (r, ns) = (ps.r, ps.ns);
        let m = n / r;
        let first = pi == 0;
        let last = pi == p - 1;
        let sf = sfx(ps.sign);
        let twp = if ps.sign < 0 { "tw_m" } else { "tw_p" };
        let mult = if inplace { rmax / r } else { 1 };
        writeln!(
            s,
            "  // pass {pi}: radix {r}, Ns = {ns}, sign {}{}{}",
            ps.sign,
            if inplace { ", in place" } else { "" },
            if ps.fuse_kernel {
                ", fused kernel multiply"
            } else {
                ""
            }
        )
        .unwrap();
        writeln!(s, "  {{").unwrap();
        if !first {
            writeln!(s, "    const cx* src = bufs[{}];", (pi - 1) % 2).unwrap();
        }
        if !last {
            writeln!(s, "    cx* dst = bufs[{}];", pi % 2).unwrap();
        }
        let load = |s: &mut String, jj: &str, vv: &str| {
            writeln!(
                s,
                "      #pragma unroll\n      for (int t = 0; t < {r}; ++t) {{"
            )
            .unwrap();
            if first {
                writeln!(s, "        const int e = {jj} + t * {m};").unwrap();
                if cfg.n_in < n {
                    writeln!(s, "        cx x = make_cx(RL(0), RL(0));\n        if (act && e < N_IN) {{ x = in[GIN(e)];").unwrap();
                } else {
                    writeln!(s, "        cx x = make_cx(RL(0), RL(0));\n        if (act) {{ x = in[GIN(e)];").unwrap();
                }
                for (k, op) in cfg.pre.iter().enumerate() {
                    let f = if *op == DiagOp::Mult { "cmul" } else { "cmulc" };
                    writeln!(s, "          x = {f}(x, {}[GIN(e)]);", pre_ptr[k]).unwrap();
                }
                if cfg.poly_pre {
                    writeln!(
                        s,
                        "          x = cscale(x, {});",
                        poly_expr("ppre", "nppre", "e", "la", "lb", cfg.vox)
                    )
                    .unwrap();
                }
                writeln!(s, "        }}\n        {vv}[t] = x;").unwrap();
            } else {
                writeln!(s, "        {vv}[t] = src[SIDX({jj} + t * {m})];").unwrap();
            }
            writeln!(s, "      }}").unwrap();
        };
        let compute_store = |s: &mut String, jj: &str, vv: &str| {
            if ns > 1 {
                let tws = n / (ns * r);
                writeln!(s, "      {{ const int k = {jj} % {ns};").unwrap();
                writeln!(s, "      #pragma unroll\n      for (int t = 1; t < {r}; ++t) {vv}[t] = cmul({vv}[t], {twp}[t * k * {tws}]);").unwrap();
                writeln!(s, "      const int ob = ({jj} / {ns}) * {} + k;", ns * r).unwrap();
            } else {
                writeln!(s, "      {{ const int ob = {jj} * {r};").unwrap();
            }
            writeln!(s, "      bfly{r}{sf}({vv});").unwrap();
            writeln!(
                s,
                "      #pragma unroll\n      for (int u = 0; u < {r}; ++u) {{"
            )
            .unwrap();
            if last {
                writeln!(s, "        const int e = ob + u * {ns};").unwrap();
                let guard = if cfg.n_out < n {
                    "act && e < N_OUT"
                } else {
                    "act"
                };
                writeln!(s, "        if ({guard}) {{ cx w = {vv}[u];").unwrap();
                if cfg.scale != 1.0 {
                    writeln!(s, "          w = cscale(w, RL({:.17e}));", cfg.scale).unwrap();
                }
                for (k, op) in cfg.post.iter().enumerate() {
                    let f = if *op == DiagOp::Mult { "cmul" } else { "cmulc" };
                    writeln!(s, "          w = {f}(w, {}[GOUT(e)]);", post_ptr[k]).unwrap();
                }
                if cfg.poly_post {
                    writeln!(
                        s,
                        "          w = cscale(w, {});",
                        poly_expr("ppost", "nppost", "e", "la", "lb", cfg.vox)
                    )
                    .unwrap();
                }
                if cfg.accumulate {
                    writeln!(s, "          out[GOUT(e)] = cadd(out[GOUT(e)], w); }}").unwrap();
                } else {
                    writeln!(s, "          out[GOUT(e)] = w; }}").unwrap();
                }
            } else if ps.fuse_kernel {
                writeln!(s, "        const int e = ob + u * {ns};").unwrap();
                let kexpr = match cfg.kfmt {
                    KernelFormat::Complex => "cscale(kern[GIN(e)], kscale)".to_string(),
                    KernelFormat::Real => "make_cx(kern[GIN(e)] * kscale, RL(0))".to_string(),
                    KernelFormat::ComplexHalf => "make_cx(h2r(kern[GIN(e)] & 0xffffu) * kscale, h2r(kern[GIN(e)] >> 16) * kscale)".to_string(),
                    KernelFormat::ComplexBf16 => "make_cx(b2r(kern[GIN(e)] & 0xffffu) * kscale, b2r(kern[GIN(e)] >> 16) * kscale)".to_string(),
                };
                let kmul = if cfg.kfmt == KernelFormat::Real {
                    format!("cscale({vv}[u], kern[GIN(e)] * kscale)")
                } else {
                    format!("cmul({vv}[u], {kexpr})")
                };
                writeln!(s, "        dst[SIDX(e)] = act ? {kmul} : {vv}[u];").unwrap();
            } else {
                writeln!(s, "        dst[SIDX(ob + u * {ns})] = {vv}[u];").unwrap();
            }
            writeln!(s, "      }} }}").unwrap();
        };
        if inplace {
            writeln!(s, "    cx vv[{mult}][{r}];").unwrap();
            writeln!(s, "    #pragma unroll\n    for (int mm = 0; mm < {mult}; ++mm) {{ const int jj = tl + mm * TPL;").unwrap();
            load(&mut s, "jj", "vv[mm]");
            writeln!(s, "    }}").unwrap();
            if !first && !last {
                writeln!(s, "    __syncthreads();").unwrap();
            }
            writeln!(s, "    #pragma unroll\n    for (int mm = 0; mm < {mult}; ++mm) {{ const int jj = tl + mm * TPL;").unwrap();
            compute_store(&mut s, "jj", "vv[mm]");
            writeln!(s, "    }}").unwrap();
        } else {
            writeln!(
                s,
                "    for (int jj = tl; jj < {m}; jj += TPL) {{\n      cx v[{r}];"
            )
            .unwrap();
            load(&mut s, "jj", "v");
            compute_store(&mut s, "jj", "v");
            writeln!(s, "    }}").unwrap();
        }
        writeln!(s, "  }}").unwrap();
        if !last {
            writeln!(s, "  __syncthreads();").unwrap();
        }
    }
    writeln!(s, "#undef GIN\n#undef GOUT\n#undef SIDX\n}}").unwrap();
    s
}
