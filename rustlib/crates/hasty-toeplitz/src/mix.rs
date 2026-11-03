//! Execution of a [`MixSpec`]: pruned forward transforms of every field, a pointwise k-space mixing step, pruned inverse transforms.

use std::fmt::Write;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, Real};
use hasty_fft::{DiagOp, MixEntry, MixZPlan, PadFft3, PadFftOps, PolyTable};

use crate::spec::{FieldDiag, MixSpec};
use crate::Error;

fn mix_source<R: Real>(spec: &MixSpec) -> String {
    let n = spec.nfield();
    let mut s = String::new();
    writeln!(s, "typedef {} real;\nstruct cx {{ real x, y; }};", R::CTYPE).unwrap();
    writeln!(s, "__device__ __forceinline__ cx cmul(cx a, cx b) {{ cx r; r.x = a.x * b.x - a.y * b.y; r.y = a.x * b.y + a.y * b.x; return r; }}").unwrap();
    writeln!(s, "extern \"C\" __global__ void __launch_bounds__(128) mix(cx* const* fields, const cx* const* kernels, size_t n) {{").unwrap();
    writeln!(
        s,
        "  const size_t g = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (g >= n) return;"
    )
    .unwrap();
    writeln!(s, "  cx v[{n}];").unwrap();
    for i in 0..n {
        writeln!(s, "  v[{i}] = fields[{i}][g];").unwrap();
    }
    for o in 0..n {
        writeln!(s, "  {{ cx a; a.x = (real)0; a.y = (real)0;").unwrap();
        for e in spec.entries.iter().filter(|e| e.out == o) {
            let k = e.kernel;
            let conj = if e.conj { "-" } else { "" };
            writeln!(
                s,
                "    {{ cx kv = kernels[{k}][g]; kv.y = {conj}kv.y; cx t = cmul(kv, v[{}]); const real cr = (real){:.17e}, ci = (real){:.17e}; a.x += cr * t.x - ci * t.y; a.y += cr * t.y + ci * t.x; }}",
                e.inp, e.coef.re, e.coef.im
            )
            .unwrap();
        }
        writeln!(s, "    fields[{o}][g] = a; }}").unwrap();
    }
    writeln!(s, "}}").unwrap();
    s
}

pub struct MixOp<R: Real> {
    pub spec: MixSpec,
    pub dims: [usize; 3],
    fft: PadFft3<R>,
    kernels: Vec<CudaSlice<R::Complex>>,
    kptr: CudaSlice<u64>,
    scratch: Vec<CudaSlice<R::Complex>>,
    fptr: CudaSlice<u64>,
    pre_poly: Vec<Option<PolyTable<R>>>,
    post_poly: Vec<Option<PolyTable<R>>>,
    arrays: Vec<CudaSlice<R::Complex>>,
    mix: CudaFunction,
    /// connected components of the mixing graph, each handled by one fused multi-field Z pass (`None` if the group does not fit in shared memory)
    groups: Option<Vec<(Vec<usize>, MixZPlan<R>, CudaSlice<u64>)>>,
    use_fused: bool,
}

impl<R: Real> MixOp<R> {
    /// `kernels[k]` is the `(2N)^3` k-space multiplier of `spec.kernels[k]`; `arrays` the user arrays referenced by `FieldDiag::Array(i)`
    /// (dense `dims`-shaped, complex).
    pub fn new(
        dev: &Device,
        spec: MixSpec,
        dims: [usize; 3],
        kernels: Vec<CudaSlice<R::Complex>>,
        arrays: Vec<CudaSlice<R::Complex>>,
    ) -> Result<Self, Error> {
        let stream = dev.default_stream();
        let pad = [2 * dims[0], 2 * dims[1], 2 * dims[2]];
        assert_eq!(kernels.len(), spec.kernels.len());
        let side = |v: &[FieldDiag]| -> (bool, bool) {
            (
                v.iter().any(|d| matches!(d, FieldDiag::Poly(_))),
                v.iter().any(|d| matches!(d, FieldDiag::Array(_))),
            )
        };
        let (pre_poly_any, pre_arr_any) = side(&spec.pre);
        let (post_poly_any, post_arr_any) = side(&spec.post);
        assert!(
            !(pre_poly_any && pre_arr_any) && !(post_poly_any && post_arr_any),
            "a side may use polynomial or array diagonals, not both"
        );
        let ops = PadFftOps {
            pre: if pre_arr_any {
                vec![DiagOp::Mult]
            } else {
                vec![]
            },
            poly_pre: pre_poly_any,
            post: if post_arr_any {
                vec![DiagOp::MultConj]
            } else {
                vec![]
            },
            poly_post: post_poly_any,
            accumulate: true,
        };
        let fft = PadFft3::<R>::new(dev, dims, pad, &ops)?;
        let npad: usize = pad.iter().product();
        let n = spec.nfield();
        let scratch: Vec<_> = (0..n)
            .map(|_| stream.alloc_zeros::<R::Complex>(npad))
            .collect::<Result<_, _>>()?;
        let ptrs = |v: &[CudaSlice<R::Complex>]| -> Result<CudaSlice<u64>, Error> {
            let host: Vec<u64> = v
                .iter()
                .map(|b| {
                    let (p, _g) = b.device_ptr(&stream);
                    p
                })
                .collect();
            Ok(stream.clone_htod(&host)?)
        };
        let kptr = ptrs(&kernels)?;
        let fptr = ptrs(&scratch)?;
        let one = hasty_fft::PolyTerm {
            coef: 1.0,
            p: [0, 0, 0],
        };
        let mk = |d: &FieldDiag, poly_mode: bool| -> Result<Option<PolyTable<R>>, Error> {
            if !poly_mode {
                return Ok(None);
            }
            Ok(Some(match d {
                FieldDiag::Poly(t) => PolyTable::new(&stream, t)?,
                _ => PolyTable::new(&stream, &[one])?,
            }))
        };
        let pre_poly = spec
            .pre
            .iter()
            .map(|d| mk(d, pre_poly_any))
            .collect::<Result<Vec<_>, _>>()?;
        let post_poly = spec
            .post
            .iter()
            .map(|d| mk(d, post_poly_any))
            .collect::<Result<Vec<_>, _>>()?;
        let module = dev.load_module(
            &format!("mix_{}_{}", spec.label, R::CTYPE),
            &mix_source::<R>(&spec),
            &CompileSettings::default(),
        )?;
        let mix = module.load_function("mix")?;
        // connected components of the field-coupling graph (diagonal specs such as CP decouple into independent single-field passes)
        let mut comp: Vec<usize> = (0..n).collect();
        fn find(c: &mut Vec<usize>, i: usize) -> usize {
            let mut r = i;
            while c[r] != r {
                r = c[r];
            }
            let mut j = i;
            while c[j] != r {
                let nx = c[j];
                c[j] = r;
                j = nx;
            }
            r
        }
        for e in &spec.entries {
            let (a, b) = (find(&mut comp, e.out), find(&mut comp, e.inp));
            if a != b {
                comp[a] = b;
            }
        }
        let mut members: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
        for i in 0..n {
            let r = find(&mut comp, i);
            members.entry(r).or_default().push(i);
        }
        let mut groups_v = Vec::new();
        let mut all_fit = true;
        for (_, fields) in members {
            let local = |g: usize| fields.iter().position(|&f| f == g).unwrap();
            let entries: Vec<MixEntry> = spec
                .entries
                .iter()
                .filter(|e| fields.contains(&e.out))
                .map(|e| MixEntry { out: local(e.out), inp: local(e.inp), kernel: e.kernel, coef: (e.coef.re, e.coef.im), conj: e.conj })
                .collect();
            match MixZPlan::<R>::new(dev, pad[0], dims[0], pad[2], pad[1], fields.len(), &entries)? {
                Some(plan) => {
                    let host: Vec<u64> = fields
                        .iter()
                        .map(|&f| {
                            let (p, _g) = scratch[f].device_ptr(&stream);
                            p
                        })
                        .collect();
                    groups_v.push((fields, plan, stream.clone_htod(&host)?));
                }
                None => {
                    all_fit = false;
                    break;
                }
            }
        }
        // narrow tiles (< 8 lines per block) lose more to uncoalesced global traffic than fusion saves: fall back to the unfused path
        let narrow = groups_v.iter().any(|(f, p, _)| f.len() > 1 && p.lpb() < 8);
        let groups = if all_fit && !narrow { Some(groups_v) } else { None };
        let use_fused = groups.is_some();
        Ok(Self {
            spec,
            dims,
            fft,
            kernels,
            kptr,
            scratch,
            fptr,
            pre_poly,
            post_poly,
            arrays,
            mix,
            groups,
            use_fused,
        })
    }

    /// force the unfused path (forward, pointwise mix, inverse) for comparison
    pub fn set_fused(&mut self, on: bool) {
        self.use_fused = on && self.groups.is_some();
    }
    pub fn is_fused(&self) -> bool {
        self.use_fused
    }

    pub fn n_fft_pairs(&self) -> usize {
        self.spec.nfield()
    }
    pub fn kernel_bytes(&self) -> usize {
        self.kernels.len() * self.kernels.first().map(|k| k.len()).unwrap_or(0) * 2 * R::BYTES
    }

    /// `out = N x` (or `out += N x`)
    pub fn apply(&mut self, stream: &Arc<CudaStream>, x: &CudaSlice<R::Complex>, out: &mut CudaSlice<R::Complex>, accumulate: bool) -> Result<(), Error> {
        let n = self.spec.nfield();
        let fused = self.use_fused;
        for i in 0..n {
            let pre: Vec<&CudaSlice<R::Complex>> = match &self.spec.pre[i] {
                FieldDiag::Array(a) => vec![&self.arrays[*a]],
                _ => vec![],
            };
            if fused {
                self.fft.forward_xy(stream, x, &mut self.scratch[i], &pre, self.pre_poly[i].as_ref())?;
            } else {
                self.fft.forward(stream, x, &mut self.scratch[i], &pre, self.pre_poly[i].as_ref())?;
            }
        }
        if fused {
            for (_, plan, fp) in self.groups.as_ref().unwrap() {
                plan.launch(stream, fp, &self.kptr)?;
            }
        } else {
            let npad = self.fft.len_pad();
            let np_i = npad;
            unsafe {
                stream
                    .launch_builder(&self.mix)
                    .arg(&self.fptr)
                    .arg(&self.kptr)
                    .arg(&np_i)
                    .launch(LaunchConfig { grid_dim: ((npad as u32).div_ceil(128), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 })?;
            }
        }
        if !accumulate {
            stream.memset_zeros(out)?;
        }
        for o in 0..n {
            let post: Vec<&CudaSlice<R::Complex>> = match &self.spec.post[o] {
                FieldDiag::Array(a) => vec![&self.arrays[*a]],
                _ => vec![],
            };
            if fused {
                self.fft.inverse_yx(stream, &mut self.scratch[o], out, &post, self.post_poly[o].as_ref())?;
            } else {
                self.fft.inverse(stream, &mut self.scratch[o], out, &post, self.post_poly[o].as_ref())?;
            }
        }
        Ok(())
    }
}
