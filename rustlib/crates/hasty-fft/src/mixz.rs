//! Fused multi-field slowest-axis pass: for every (x, y) line, forward-transform `nf` fields along Z, mix them pointwise in k-space with
//! a list of kernels (`out_o = sum coef * K_k * in_i`, optional conj(K)), and inverse-transform the results back, all in shared memory.
//! Replaces `2 * nf` global Z passes plus a separate mixing pass of the unfused multi-field engine by one read and one write of the
//! pruned (`Zn` of `Z2` planes) field data plus one read of every kernel line.

use std::fmt::Write;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, Real};

use crate::factor::factorize;
use crate::plan::FftError;
use crate::radix::emit_butterfly;

#[derive(Clone, Copy, Debug)]
pub struct MixEntry {
    pub out: usize,
    pub inp: usize,
    pub kernel: usize,
    pub coef: (f64, f64),
    pub conj: bool,
}

pub struct MixZPlan<R: Real> {
    func: CudaFunction,
    tw_m: CudaSlice<R::Complex>,
    tw_p: CudaSlice<R::Complex>,
    lpb: usize,
    block: u32,
    smem: u32,
    pub nf: usize,
    pub z2: usize,
    pub x2: usize,
    pub y2: usize,
}

struct Seq<'a> {
    radices: &'a [usize],
    n: usize,
}

/// emits the passes of one transform of `seq` (sign, direction) from `src` to `dst` using buffers `a` (region) and `t` (temp)
#[allow(clippy::too_many_arguments)]
fn emit_transform(s: &mut String, seq: &Seq, sign: i32, from_global: bool, to_global: bool, region: &str, temp: &str, n_in: usize, n_out: usize, scale: Option<f64>) {
    let n = seq.n;
    let p = seq.radices.len();
    let sf = if sign < 0 { "m" } else { "p" };
    let twp = if sign < 0 { "tw_m" } else { "tw_p" };
    let mut ns = 1usize;
    for (pi, &r) in seq.radices.iter().enumerate() {
        let m = n / r;
        let first = pi == 0;
        let last = pi == p - 1;
        // buffer written by pass pi
        let dst_buf = if last && to_global {
            None
        } else if from_global {
            // forward: last pass writes region, alternate backwards
            Some(if (p - 1 - pi) % 2 == 0 { region } else { temp })
        } else {
            // inverse (from region): pass 0 reads region
            Some(if pi % 2 == 0 { temp } else { region })
        };
        let src_buf = if first {
            if from_global {
                None
            } else {
                Some(region)
            }
        } else if from_global {
            Some(if (p - pi) % 2 == 0 { region } else { temp })
        } else {
            Some(if (pi - 1) % 2 == 0 { temp } else { region })
        };
        writeln!(s, "    // pass {pi}: radix {r}, Ns {ns}").unwrap();
        writeln!(s, "    for (int jj = tl; jj < {m}; jj += TPL) {{\n      cx v[{r}];\n      #pragma unroll\n      for (int t = 0; t < {r}; ++t) {{").unwrap();
        match src_buf {
            None => {
                writeln!(s, "        const int e = jj + t * {m};\n        v[t] = (act && e < {n_in}) ? in_f[base + (long)e * sn] : make_cx(RL(0), RL(0));").unwrap();
            }
            Some(b) => {
                writeln!(s, "        v[t] = {b}[(jj + t * {m}) * LPB + li];").unwrap();
            }
        }
        writeln!(s, "      }}").unwrap();
        if ns > 1 {
            let tws = n / (ns * r);
            writeln!(s, "      const int k = jj % {ns};\n      #pragma unroll\n      for (int t = 1; t < {r}; ++t) v[t] = cmul(v[t], {twp}[t * k * {tws}]);\n      const int ob = (jj / {ns}) * {} + k;", ns * r).unwrap();
        } else {
            writeln!(s, "      const int ob = jj * {r};").unwrap();
        }
        writeln!(s, "      bfly{r}{sf}(v);\n      #pragma unroll\n      for (int u = 0; u < {r}; ++u) {{").unwrap();
        match dst_buf {
            None => {
                let sc = match scale {
                    Some(c) => format!("cscale(v[u], RL({c:.17e}))"),
                    None => "v[u]".to_string(),
                };
                writeln!(s, "        const int e = ob + u * {ns};\n        if (act && e < {n_out}) out_f[base + (long)e * sn] = {sc};").unwrap();
            }
            Some(b) => {
                writeln!(s, "        {b}[(ob + u * {ns}) * LPB + li] = v[u];").unwrap();
            }
        }
        writeln!(s, "      }}\n    }}\n    __syncthreads();").unwrap();
        ns *= r;
    }
}

pub fn mixz_source(ctype: &str, z2: usize, zn: usize, nf: usize, lpb: usize, tpl: usize, entries: &[MixEntry]) -> Result<String, FftError> {
    let radices = factorize(z2).ok_or(FftError::UnsupportedLength(z2))?;
    let mut s = String::new();
    writeln!(s, "typedef {ctype} real;\nstruct cx {{ real x, y; }};\n#define RL(v) ((real)(v))").unwrap();
    writeln!(
        s,
        r#"__device__ __forceinline__ cx make_cx(real x, real y) {{ cx r; r.x = x; r.y = y; return r; }}
__device__ __forceinline__ cx cadd(cx a, cx b) {{ return make_cx(a.x + b.x, a.y + b.y); }}
__device__ __forceinline__ cx csub(cx a, cx b) {{ return make_cx(a.x - b.x, a.y - b.y); }}
__device__ __forceinline__ cx cscale(cx a, real c) {{ return make_cx(a.x * c, a.y * c); }}
__device__ __forceinline__ cx cmul(cx a, cx b) {{ return make_cx(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }}"#
    )
    .unwrap();
    let mut done = vec![];
    for sg in [-1, 1] {
        for &r in &radices {
            if !done.contains(&(r, sg)) {
                s.push_str(&emit_butterfly(r, sg));
                done.push((r, sg));
            }
        }
    }
    writeln!(s, "#define Z2 {z2}\n#define NF {nf}\n#define LPB {lpb}\n#define TPL {tpl}\n#define BLK {}", lpb * tpl).unwrap();
    writeln!(
        s,
        "extern \"C\" __global__ void __launch_bounds__(BLK) mixz(cx* const* fields, const cx* const* kernels, const cx* __restrict__ tw_m, const cx* __restrict__ tw_p, long x2, long lbs, long sn) {{"
    )
    .unwrap();
    writeln!(s, "  extern __shared__ cx smem[];\n  const int tid = threadIdx.x, li = tid % LPB, tl = tid / LPB;\n  const long la = (long)blockIdx.x * LPB + li, lb = blockIdx.y;\n  const bool act = la < x2;\n  const long base = la + lb * lbs;\n  cx* temp = smem + (size_t)NF * Z2 * LPB;").unwrap();
    let seq = Seq { radices: &radices, n: z2 };
    // forward transforms
    writeln!(s, "  for (int f = 0; f < NF; ++f) {{\n    cx* region = smem + (size_t)f * Z2 * LPB;\n    const cx* in_f = fields[f]; cx* out_f = nullptr; (void)out_f;").unwrap();
    emit_transform(&mut s, &seq, -1, true, false, "region", "temp", zn, z2, None);
    writeln!(s, "  }}").unwrap();
    // mixing
    writeln!(s, "  for (int g = tid; g < Z2 * LPB; g += BLK) {{\n    const int kz = g / LPB, l = g % LPB;\n    const long la2 = (long)blockIdx.x * LPB + l;\n    if (la2 >= x2) continue;\n    const long ga = la2 + lb * lbs + (long)kz * sn;\n    cx v[NF];\n    #pragma unroll\n    for (int i = 0; i < NF; ++i) v[i] = smem[(size_t)i * Z2 * LPB + g];").unwrap();
    for o in 0..nf {
        writeln!(s, "    {{ cx a = make_cx(RL(0), RL(0));").unwrap();
        for e in entries.iter().filter(|e| e.out == o) {
            let cj = if e.conj { "-" } else { "" };
            writeln!(
                s,
                "      {{ cx kv = kernels[{}][ga]; kv.y = {cj}kv.y; cx t = cmul(kv, v[{}]); a.x += RL({:.17e}) * t.x - RL({:.17e}) * t.y; a.y += RL({:.17e}) * t.y + RL({:.17e}) * t.x; }}",
                e.kernel, e.inp, e.coef.0, e.coef.1, e.coef.0, e.coef.1
            )
            .unwrap();
        }
        writeln!(s, "      smem[(size_t){o} * Z2 * LPB + g] = a; }}").unwrap();
    }
    writeln!(s, "  }}\n  __syncthreads();").unwrap();
    // inverse transforms (unnormalised; the caller's final pass carries 1/prod(pad))
    writeln!(s, "  for (int f = 0; f < NF; ++f) {{\n    cx* region = smem + (size_t)f * Z2 * LPB;\n    const cx* in_f = nullptr; (void)in_f; cx* out_f = fields[f];").unwrap();
    emit_transform(&mut s, &seq, 1, false, true, "region", "temp", z2, zn, None);
    writeln!(s, "  }}\n}}").unwrap();
    Ok(s)
}

impl<R: Real> MixZPlan<R> {
    /// returns `Ok(None)` if the working set does not fit in shared memory (use the unfused path)
    pub fn new(dev: &Device, z2: usize, zn: usize, x2: usize, y2: usize, nf: usize, entries: &[MixEntry]) -> Result<Option<Self>, FftError> {
        let radices = factorize(z2).ok_or(FftError::UnsupportedLength(z2))?;
        let rmax = *radices.iter().max().unwrap();
        let tpl = (z2 / rmax).max(1);
        let max_smem = dev.ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)? as usize;
        let per_lpb = (nf + 1) * z2 * 2 * R::BYTES;
        let mut lpb = 16usize;
        while lpb > 1 && lpb * per_lpb > max_smem {
            lpb /= 2;
        }
        if lpb * per_lpb > max_smem {
            return Ok(None);
        }
        while lpb * tpl > 1024 && lpb > 1 {
            lpb /= 2;
        }
        let src = mixz_source(R::CTYPE, z2, zn, nf, lpb, tpl, entries)?;
        let name = format!("mixz_z{z2}_n{zn}_f{nf}_e{}_{}", entries.len(), R::CTYPE);
        let func = dev.load_function(&name, &src, "mixz", &CompileSettings::default())?;
        let smem = lpb * per_lpb;
        if smem > 48 * 1024 {
            func.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)?;
        }
        let table = |sg: f64| -> Vec<R::Complex> {
            (0..z2)
                .map(|m| {
                    let a = sg * 2.0 * std::f64::consts::PI * m as f64 / z2 as f64;
                    R::complex(a.cos(), a.sin())
                })
                .collect()
        };
        let stream = dev.default_stream();
        Ok(Some(Self { func, tw_m: stream.clone_htod(&table(-1.0))?, tw_p: stream.clone_htod(&table(1.0))?, lpb, block: (lpb * tpl) as u32, smem: smem as u32, nf, z2, x2, y2 }))
    }

    pub fn lpb(&self) -> usize {
        self.lpb
    }

    /// `fields` / `kernels`: device arrays of device pointers; every field and kernel is a `(Z2, Y2, X2)` array (x fastest).
    pub fn launch(&self, stream: &Arc<CudaStream>, fields: &CudaSlice<u64>, kernels: &CudaSlice<u64>) -> Result<(), FftError> {
        let (twm, _g1) = self.tw_m.device_ptr(stream);
        let (twp, _g2) = self.tw_p.device_ptr(stream);
        let x2 = self.x2 as i64;
        let lbs = self.x2 as i64;
        let sn = (self.x2 * self.y2) as i64;
        let cfg = LaunchConfig { grid_dim: (self.x2.div_ceil(self.lpb) as u32, self.y2 as u32, 1), block_dim: (self.block, 1, 1), shared_mem_bytes: self.smem };
        unsafe {
            stream.launch_builder(&self.func).arg(fields).arg(kernels).arg(&twm).arg(&twp).arg(&x2).arg(&lbs).arg(&sn).launch(cfg)?;
        }
        Ok(())
    }
}
