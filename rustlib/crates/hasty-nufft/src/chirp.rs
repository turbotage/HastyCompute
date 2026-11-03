//! Chirp-aware ("modulated-row") NUFFT: a type-2 / adjoint pair for the operator
//!     y_j = sum_i f_i exp(-i k_j . (i - N/2)) exp(-i Phi_j(xi_i)),     Phi_j(xi) = sum_q w_jq m_q(xi),
//! evaluated with ONE shared fine-grid FFT per coil and a per-sample compact k-space stencil that absorbs the phase.
//!
//! Derivation (see `research/concomitant_stencils_and_kernels.md`): with the NUFFT identity  sum_u phi(kappa-u) e^{-iux} = psi(x) e^{-i kappa x}
//! the stencil of the modulated row is  Gamma_j = phi_kappa (*) ghat_j  with  ghat_j(s) = delta_{s0} + hhat(s; w_j),
//! hhat(.; w) = DFT over the fine-grid period of  W(xi) (e^{-i Phi_w(xi)} - 1)  (W = smooth window, 1 on the FOV, 0 at the period edge).
//! The map w -> hhat is represented by a POD basis of the family + DEIM interpolation points, so a sample costs L sin/cos, an LxL solve and an
//! L x (2S+1)^3 contraction. Everything here is built once per (grid, sigma, tolerance, w-box).

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device};
use nalgebra::DMatrix;
use num_complex::Complex64;

use crate::plan::{Dims, NufftError, NufftOpts, NufftPlan, SpreadMethod};

#[derive(Clone, Debug)]
pub struct ChirpFamily {
    /// box of the four concomitant coefficients: `|w_q| <= w_box[q]` (radians; fields are `z^2, x^2+y^2, xz, yz` in FOV-normalised coordinates)
    pub w_box: [f64; 4],
    /// stencil half-width in fine-grid cells
    pub s_half: usize,
    /// POD basis: keep modes until the relative tail energy is below `pod_tol` (or `l_max`)
    pub pod_tol: f64,
    pub l_max: usize,
    /// FOV lattice points per axis used for the fit
    pub nfov: usize,
    /// relative Tikhonov ridge of the stencil least-squares fit (keeps the stencil coefficients bounded)
    pub ridge: f64,
    pub n_train: usize,
    /// train on these coefficient vectors (e.g. a subsample of the actual sequence's `w_j`) instead of the box: the sequence-specific family
    /// is a low-dimensional subset of the box, so the basis is much smaller. Precomputable once per sequence/scanner.
    pub train: Option<Vec<[f64; 4]>>,
}

impl Default for ChirpFamily {
    fn default() -> Self {
        Self {
            w_box: [1.0; 4],
            s_half: 4,
            pod_tol: 1e-4,
            l_max: 40,
            nfov: 13,
            ridge: 1e-7,
            n_train: 300,
            train: None,
        }
    }
}

/// Host-side tables describing the chirp spectrum family for one fine grid.
pub struct ChirpTables {
    pub l: usize,
    pub s_half: usize,
    /// DEIM points: FOV-normalised physical coordinates (x fastest)
    pub pts: Vec<[f64; 3]>,
    /// window values at the DEIM points (all 1: the fit is restricted to the FOV)
    pub win: Vec<f64>,
    /// `binv[l * L + i]`: coefficient l from sample value i (complex)
    pub binv: Vec<Complex64>,
    /// `h[l * NS^3 + s]`, `s = (sz*NS + sy)*NS + sx` with offsets `idx - S`; complex least-squares stencil spectrum of basis function l
    pub h: Vec<Complex64>,
    /// representation error measured on held-out w: (max, mean) of relative L2 error of the row over the FOV
    pub heldout_err: (f64, f64),
    pub singular_values: Vec<f64>,
    pub max_abs_h: f64,
}

fn m_q(q: usize, x: f64, y: f64, z: f64) -> f64 {
    match q {
        0 => z * z,
        1 => x * x + y * y,
        2 => x * z,
        _ => y * z,
    }
}

impl ChirpTables {
    /// `modes`: image size `[Nx, Ny, Nz]` (x fastest); `nf`: fine grid `[nf0, nf1, nf2]` (from the NUFFT plan).
    /// Positions use `xi_a(i) = (i - (N_a-1)/2) * 2/N_a` for image index `i` (mode `k = i - N_a/2`, lattice coordinate `xi' = 2k/N = xi - 1/N`).
    ///
    /// The stencil spectrum solves, for every `w`, the least-squares problem
    ///     min_g  sum_{p in FOV} | sum_s g(s) e^{-i pi s . xi'_p / sg} - (e^{-i Phi_w(xi_p)} - 1) |^2 + ridge ||g||^2
    /// which is linear in the sampled chirp; the family `{e^{-i Phi_w} - 1}` is compressed by POD and interpolated by DEIM.
    pub fn build(modes: [usize; 3], nf: [usize; 3], fam: &ChirpFamily) -> Self {
        let sg = [
            nf[0] as f64 / modes[0] as f64,
            nf[1] as f64 / modes[1] as f64,
            nf[2] as f64 / modes[2] as f64,
        ];
        let shift = [
            1.0 / modes[0] as f64,
            1.0 / modes[1] as f64,
            1.0 / modes[2] as f64,
        ];
        let nfv = fam.nfov;
        let sh = fam.s_half;
        let ns = 2 * sh + 1;
        let nstencil = ns * ns * ns;
        let nfov3 = nfv * nfv * nfv;
        // FOV lattice (physical coordinates), uniform incl. the edges of the cube
        let lat = |t: usize| -> f64 { -1.0 + 2.0 * t as f64 / (nfv - 1) as f64 };
        let pidx = |tx: usize, ty: usize, tz: usize| (tz * nfv + ty) * nfv + tx;
        let mut pts_all = vec![[0f64; 3]; nfov3];
        for tz in 0..nfv {
            for ty in 0..nfv {
                for tx in 0..nfv {
                    pts_all[pidx(tx, ty, tz)] = [lat(tx), lat(ty), lat(tz)];
                }
            }
        }
        // training coefficients
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut rnd = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let mut train: Vec<[f64; 4]> = Vec::new();
        if let Some(t) = &fam.train {
            train = t.clone();
        } else {
            let nt = fam.n_train;
            for t in 0..nt {
                if t < 16 {
                    train.push([0, 1, 2, 3].map(|q| if (t >> q) & 1 == 1 { fam.w_box[q] } else { -fam.w_box[q] }));
                } else {
                    let r = ((t as f64) / nt as f64).sqrt();
                    train.push([0, 1, 2, 3].map(|q| fam.w_box[q] * r * rnd()));
                }
            }
        }
        let nt = train.len();
        let chirp_minus1 = |w: &[f64; 4], p: &[f64; 3]| -> Complex64 {
            let ph: f64 = (0..4).map(|q| w[q] * m_q(q, p[0], p[1], p[2])).sum();
            Complex64::from_polar(1.0, -ph) - 1.0
        };
        let mut y = DMatrix::<Complex64>::zeros(nfov3, nt);
        for (t, w) in train.iter().enumerate() {
            for p in 0..nfov3 {
                y[(p, t)] = chirp_minus1(w, &pts_all[p]);
            }
        }
        let gram = y.adjoint() * &y;
        let svd = gram.svd(true, false);
        let u = svd.u.unwrap();
        let sig: Vec<f64> = svd.singular_values.iter().cloned().collect();
        let total: f64 = sig.iter().sum();
        let (mut acc, mut l) = (0.0, 0);
        for (i, s) in sig.iter().enumerate() {
            acc += s;
            l = i + 1;
            if (total - acc) / total < fam.pod_tol * fam.pod_tol || l >= fam.l_max {
                break;
            }
        }
        let mut b = DMatrix::<Complex64>::zeros(nfov3, l);
        for a in 0..l {
            let v = &y * u.column(a) / Complex64::new(sig[a].sqrt(), 0.0);
            b.set_column(a, &v);
        }
        // DEIM
        let mut sel: Vec<usize> = Vec::new();
        sel.push(
            (0..nfov3)
                .max_by(|&i, &j| b[(i, 0)].norm().partial_cmp(&b[(j, 0)].norm()).unwrap())
                .unwrap(),
        );
        for a in 1..l {
            let k = sel.len();
            let bm = DMatrix::<Complex64>::from_fn(k, k, |i, j| b[(sel[i], j)]);
            let rhs = nalgebra::DVector::<Complex64>::from_fn(k, |i, _| b[(sel[i], a)]);
            let c = bm.lu().solve(&rhs).expect("DEIM solve");
            let mut r = b.column(a).clone_owned();
            for j in 0..k {
                r -= b.column(j) * c[j];
            }
            sel.push(
                (0..nfov3)
                    .max_by(|&i, &j| r[i].norm().partial_cmp(&r[j].norm()).unwrap())
                    .unwrap(),
            );
        }
        let bp = DMatrix::<Complex64>::from_fn(l, l, |i, j| b[(sel[i], j)]);
        let bp_inv = bp.try_inverse().expect("DEIM matrix singular");
        // least-squares stencil operator  P = (A^H A + mu I)^-1 A^H,  A[p, s] = exp(-i pi s.xi'_p / sg)
        let mut amat = DMatrix::<Complex64>::zeros(nfov3, nstencil);
        for p in 0..nfov3 {
            let xp = [
                pts_all[p][0] - shift[0],
                pts_all[p][1] - shift[1],
                pts_all[p][2] - shift[2],
            ];
            for sz in 0..ns {
                for sy in 0..ns {
                    for sx in 0..ns {
                        let ph = std::f64::consts::PI
                            * ((sx as f64 - sh as f64) * xp[0] / sg[0]
                                + (sy as f64 - sh as f64) * xp[1] / sg[1]
                                + (sz as f64 - sh as f64) * xp[2] / sg[2]);
                        amat[(p, (sz * ns + sy) * ns + sx)] = Complex64::from_polar(1.0, -ph);
                    }
                }
            }
        }
        let aha = amat.adjoint() * &amat;
        let smax = aha.diagonal().iter().map(|c| c.re).fold(0.0, f64::max);
        let mut reg = aha.clone();
        for i in 0..nstencil {
            reg[(i, i)] += Complex64::new(fam.ridge * smax, 0.0);
        }
        let hmat = reg
            .lu()
            .solve(&(amat.adjoint() * &b))
            .expect("stencil LS solve"); // nstencil x l
        let mut h = vec![Complex64::new(0.0, 0.0); l * nstencil];
        let mut max_abs_h: f64 = 0.0;
        for a in 0..l {
            for s in 0..nstencil {
                h[a * nstencil + s] = hmat[(s, a)];
                max_abs_h = max_abs_h.max(hmat[(s, a)].norm());
            }
        }
        let pts: Vec<[f64; 3]> = sel.iter().map(|&p| pts_all[p]).collect();
        let win = vec![1.0; l];
        let mut binv = vec![Complex64::new(0.0, 0.0); l * l];
        for i in 0..l {
            for j in 0..l {
                binv[i * l + j] = bp_inv[(i, j)];
            }
        }
        let mut tables = ChirpTables {
            l,
            s_half: sh,
            pts,
            win,
            binv,
            h,
            heldout_err: (0.0, 0.0),
            singular_values: sig,
            max_abs_h,
        };
        tables.heldout_err = tables.validate(modes, sg, fam, &mut rnd);
        tables
    }

    /// Persist the tables (they depend only on grid, sigma, kernel width, S and the coefficient family: build once per scanner/sequence).
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(b"HCHIRP01")?;
        for v in [self.l as u64, self.s_half as u64, self.pts.len() as u64, self.h.len() as u64, self.singular_values.len() as u64] {
            f.write_all(&v.to_le_bytes())?;
        }
        let wf = |f: &mut std::io::BufWriter<std::fs::File>, v: f64| f.write_all(&v.to_le_bytes());
        for p in &self.pts {
            for &c in p {
                wf(&mut f, c)?;
            }
        }
        for &v in &self.win {
            wf(&mut f, v)?;
        }
        for c in self.binv.iter().chain(self.h.iter()) {
            wf(&mut f, c.re)?;
            wf(&mut f, c.im)?;
        }
        for &v in &self.singular_values {
            wf(&mut f, v)?;
        }
        wf(&mut f, self.heldout_err.0)?;
        wf(&mut f, self.heldout_err.1)?;
        wf(&mut f, self.max_abs_h)?;
        f.flush()
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != b"HCHIRP01" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a chirp table file"));
        }
        let mut u = [0u8; 8];
        let mut rd_u = |f: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<usize> {
            f.read_exact(&mut u)?;
            Ok(u64::from_le_bytes(u) as usize)
        };
        let (l, s_half, npts, nh, nsv) = (rd_u(&mut f)?, rd_u(&mut f)?, rd_u(&mut f)?, rd_u(&mut f)?, rd_u(&mut f)?);
        let mut d = [0u8; 8];
        let mut rd = |f: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<f64> {
            f.read_exact(&mut d)?;
            Ok(f64::from_le_bytes(d))
        };
        let mut pts = Vec::with_capacity(npts);
        for _ in 0..npts {
            pts.push([rd(&mut f)?, rd(&mut f)?, rd(&mut f)?]);
        }
        let win = (0..npts).map(|_| rd(&mut f)).collect::<Result<Vec<_>, _>>()?;
        let mut binv = Vec::with_capacity(l * l);
        for _ in 0..l * l {
            binv.push(Complex64::new(rd(&mut f)?, rd(&mut f)?));
        }
        let mut h = Vec::with_capacity(nh);
        for _ in 0..nh {
            h.push(Complex64::new(rd(&mut f)?, rd(&mut f)?));
        }
        let singular_values = (0..nsv).map(|_| rd(&mut f)).collect::<Result<Vec<_>, _>>()?;
        let heldout_err = (rd(&mut f)?, rd(&mut f)?);
        let max_abs_h = rd(&mut f)?;
        Ok(Self { l, s_half, pts, win, binv, h, heldout_err, singular_values, max_abs_h })
    }

    /// stencil spectrum `ghat(s; w) = delta + sum_l c_l(w) h_l(s)` on the (2S+1)^3 window (host, f64)
    pub fn ghat(&self, w: &[f64; 4]) -> Vec<Complex64> {
        let l = self.l;
        let f: Vec<Complex64> = (0..l)
            .map(|i| {
                let p = self.pts[i];
                let ph: f64 = (0..4).map(|q| w[q] * m_q(q, p[0], p[1], p[2])).sum();
                (Complex64::from_polar(1.0, -ph) - 1.0) * self.win[i]
            })
            .collect();
        let c: Vec<Complex64> = (0..l)
            .map(|a| (0..l).map(|i| self.binv[a * l + i] * f[i]).sum())
            .collect();
        let ns = 2 * self.s_half + 1;
        let mut g = vec![Complex64::new(0.0, 0.0); ns * ns * ns];
        for a in 0..l {
            for s in 0..ns * ns * ns {
                g[s] += c[a] * self.h[a * ns * ns * ns + s];
            }
        }
        g[(self.s_half * ns + self.s_half) * ns + self.s_half] += 1.0;
        g
    }

    /// Row representation error on random FOV points for held-out w:  || sum_s ghat(s;w) e^{-i pi s.xi'/sg} - e^{-i Phi_w} || / || e^{-i Phi_w} ||.
    fn validate(
        &self,
        modes: [usize; 3],
        sg: [f64; 3],
        fam: &ChirpFamily,
        rnd: &mut dyn FnMut() -> f64,
    ) -> (f64, f64) {
        let mut worst: f64 = 0.0;
        let mut mean = 0.0;
        let ntest = 20;
        let ns = 2 * self.s_half + 1;
        let npts = 300;
        let pts: Vec<[f64; 3]> = (0..npts).map(|_| [rnd(), rnd(), rnd()]).collect();
        for _ in 0..ntest {
            let w = match &fam.train {
                Some(t) => {
                    let k = (((rnd() * 0.5 + 0.5) * t.len() as f64) as usize).min(t.len() - 1);
                    t[k]
                }
                None => [0, 1, 2, 3].map(|q| fam.w_box[q] * rnd()),
            };
            let g = self.ghat(&w);
            let (mut num, mut den) = (0.0, 0.0);
            for p in &pts {
                let ph: f64 = (0..4).map(|q| w[q] * m_q(q, p[0], p[1], p[2])).sum();
                let exact = Complex64::from_polar(1.0, -ph);
                let mut acc = Complex64::new(0.0, 0.0);
                for sz in 0..ns {
                    for sy in 0..ns {
                        for sx in 0..ns {
                            let pe = std::f64::consts::PI
                                * ((sx as f64 - self.s_half as f64)
                                    * (p[0] - 1.0 / modes[0] as f64)
                                    / sg[0]
                                    + (sy as f64 - self.s_half as f64)
                                        * (p[1] - 1.0 / modes[1] as f64)
                                        / sg[1]
                                    + (sz as f64 - self.s_half as f64)
                                        * (p[2] - 1.0 / modes[2] as f64)
                                        / sg[2]);
                            acc += g[(sz * ns + sy) * ns + sx] * Complex64::from_polar(1.0, -pe);
                        }
                    }
                }
                num += (acc - exact).norm_sqr();
                den += exact.norm_sqr();
            }
            let e = (num / den).sqrt();
            worst = worst.max(e);
            mean += e / ntest as f64;
        }
        (worst, mean)
    }
}

// ------------------------------------------------------------------------------------------------ GPU plan

struct ChirpGpu {
    interp: CudaFunction,
    spread: CudaFunction,
    l: usize,
    smem: u32,
    pts: CudaSlice<f32>,
    ptsw: CudaSlice<f32>,
    binv: CudaSlice<hasty_cuda::C32>,
    h: CudaSlice<hasty_cuda::C32>,
    w: Option<[CudaSlice<f32>; 4]>,
    gen: CudaFunction,
    gather_sm: CudaFunction,
    scatter_sm: CudaFunction,
    lmax: usize,
    tile_bytes: u32,
    stencils: Option<CudaSlice<u32>>,
}

/// Type-2 (`forward`: y = A f) and adjoint (`adjoint`: f = A^H c) of the concomitant-modulated operator, f32, 3-D, centred modes.
pub struct ChirpNufft {
    pub plan: NufftPlan<f32>,
    pub tables: ChirpTables,
    gpu: ChirpGpu,
}

impl ChirpNufft {
    pub fn new(
        dev: &Device,
        modes: [usize; 3],
        opts: NufftOpts,
        fam: ChirpFamily,
    ) -> Result<Self, NufftError> {
        Self::with_tables(dev, modes, opts, fam, None)
    }

    /// like `new` but reusing tables built (or loaded from disk) earlier; the family is then only used for `s_half`
    pub fn with_tables(
        dev: &Device,
        modes: [usize; 3],
        opts: NufftOpts,
        fam: ChirpFamily,
        tables: Option<ChirpTables>,
    ) -> Result<Self, NufftError> {
        let mut opts = opts;
        opts.spread = SpreadMethod::Gm; // the chirp spreader is its own kernel
        opts.footprint_pad = 2 * fam.s_half;
        let plan = NufftPlan::<f32>::new(dev, &modes, opts)?;
        let tables = match tables {
            Some(t) => t,
            None => ChirpTables::build(modes, plan.nf, &fam),
        };
        let (w_es, s) = (plan.ker.w, fam.s_half);
        let src = format!(
            "#define W {}\n#define S {}\n#define BETA {:.17e}\n#define BSZ {}\n{}",
            w_es,
            s,
            plan.ker.beta,
            plan.bs(),
            include_str!("chirp.cu")
        );
        let module = dev.load_module(
            &format!("chirp_w{w_es}_s{s}_b{}", plan.bs()),
            &src,
            &CompileSettings::default(),
        )?;
        let interp = module.load_function("chirp_interp")?;
        let spread = module.load_function("chirp_spread")?;
        let gen = module.load_function("chirp_gen")?;
        let gather_sm = module.load_function("chirp_gather_sm")?;
        let scatter_sm = module.load_function("chirp_scatter_sm")?;
        let ns = 2 * s + 1;
        let jw = w_es + 2 * s;
        let t1n = (ns * jw * jw).max(ns * ns * ns);
        let l = tables.l;
        let smem = ((2 * l + t1n + jw * jw * jw) * 8) as u32;
        let attr = cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        if smem > 48 * 1024 {
            interp.set_attribute(attr, smem as i32)?;
            spread.set_attribute(attr, smem as i32)?;
            gen.set_attribute(attr, smem as i32)?;
        }
        let tile_bytes = ((plan.bs() + jw).pow(3) * 8) as u32;
        if tile_bytes > 48 * 1024 {
            gather_sm.set_attribute(attr, tile_bytes as i32)?;
            scatter_sm.set_attribute(attr, tile_bytes as i32)?;
        }
        let stream = dev.default_stream();
        let pts: Vec<f32> = tables
            .pts
            .iter()
            .flat_map(|p| p.iter().map(|&v| v as f32))
            .collect();
        let ptsw: Vec<f32> = tables.win.iter().map(|&v| v as f32).collect();
        let c32 = |v: &[Complex64]| -> Vec<hasty_cuda::C32> {
            v.iter()
                .map(|c| hasty_cuda::C32::new(c.re as f32, c.im as f32))
                .collect()
        };
        let gpu = ChirpGpu {
            interp,
            spread,
            l,
            smem,
            pts: stream.clone_htod(&pts)?,
            ptsw: stream.clone_htod(&ptsw)?,
            binv: stream.clone_htod(&c32(&tables.binv))?,
            h: stream.clone_htod(&c32(&tables.h))?,
            w: None,
            gen,
            gather_sm,
            scatter_sm,
            lmax: l,
            tile_bytes,
            stencils: None,
        };
        Ok(Self { plan, tables, gpu })
    }

    /// `coords[a]`: k-space coordinate a in radians per voxel; `w`: the four concomitant coefficients per point.
    pub fn set_points(
        &mut self,
        stream: &Arc<CudaStream>,
        coords: &[&CudaSlice<f32>],
        w: [&CudaSlice<f32>; 4],
        m: usize,
    ) -> Result<(), NufftError> {
        self.plan.set_points(stream, coords, m)?;
        self.gpu.w = Some([
            stream.clone_dtod(w[0])?,
            stream.clone_dtod(w[1])?,
            stream.clone_dtod(w[2])?,
            stream.clone_dtod(w[3])?,
        ]);
        Ok(())
    }

    /// Generate and store every sample's stencil (packed fp16, bin-sorted order). One-off per trajectory + coefficient set; the stored
    /// stencils make forward/adjoint pure tiled gather/scatter for every coil and iteration. Needs `M * (W+2S)^3 * 4` bytes.
    pub fn precompute_stencils(&mut self, stream: &Arc<CudaStream>) -> Result<(), NufftError> {
        let p = self.plan.points().ok_or_else(|| NufftError::Invalid("set_points not called".into()))?;
        let w = self.gpu.w.as_ref().ok_or_else(|| NufftError::Invalid("w not set".into()))?;
        let jw = self.plan.ker.w + 2 * self.tables.s_half;
        let mut buf = stream.alloc_zeros::<u32>(p.m * jw * jw * jw)?;
        let (m_i, l_i, lmax_i) = (p.m as i32, self.gpu.l as i32, self.gpu.lmax as i32);
        unsafe {
            let mut b = stream.launch_builder(&self.gpu.gen);
            b.arg(&p.sc).arg(&p.perm).arg(&w[0]).arg(&w[1]).arg(&w[2]).arg(&w[3]).arg(&mut buf).arg(&m_i).arg(&l_i);
            b.arg(&self.gpu.pts).arg(&self.gpu.ptsw).arg(&self.gpu.binv).arg(&self.gpu.h).arg(&lmax_i);
            b.launch(LaunchConfig { grid_dim: (p.m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: self.gpu.smem })?;
        }
        self.gpu.stencils = Some(buf);
        Ok(())
    }

    pub fn has_stored_stencils(&self) -> bool {
        self.gpu.stencils.is_some()
    }

    pub fn drop_stored_stencils(&mut self) {
        self.gpu.stencils = None;
    }

    fn launch_stored(g: &ChirpGpu, plan: &NufftPlan<f32>, stream: &Arc<CudaStream>, c_ptr: u64, ntr: usize, adjoint: bool) -> Result<(), NufftError> {
        let p = plan.points().ok_or_else(|| NufftError::Invalid("set_points not called".into()))?;
        let st = g.stencils.as_ref().ok_or_else(|| NufftError::Invalid("stencils not precomputed".into()))?;
        let d: Dims = plan.dims();
        let (m_i, ntr_i) = (p.m as i32, ntr as i32);
        let gs = plan.grid_elems() as u64;
        let (gp, _gg) = plan.grid.device_ptr(stream);
        let cfg = LaunchConfig { grid_dim: (p.nsub as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: g.tile_bytes };
        unsafe {
            let f = if adjoint { &g.scatter_sm } else { &g.gather_sm };
            let mut b = stream.launch_builder(f);
            b.arg(&p.sc).arg(&p.perm).arg(st);
            b.arg(&gp).arg(&c_ptr);
            b.arg(&p.sub_bin).arg(&p.sub_start).arg(&p.sub_cnt).arg(&m_i).arg(&ntr_i).arg(&gs).arg(&d);
            b.launch(cfg)?;
        }
        Ok(())
    }

    fn launch(
        g: &ChirpGpu,
        plan: &NufftPlan<f32>,
        stream: &Arc<CudaStream>,
        c_ptr: u64,
        ntr: usize,
        adjoint: bool,
    ) -> Result<(), NufftError> {
        let p = plan
            .points()
            .ok_or_else(|| NufftError::Invalid("set_points not called".into()))?;
        let w =
            g.w.as_ref()
                .ok_or_else(|| NufftError::Invalid("w not set".into()))?;
        let d: Dims = plan.dims();
        let (m_i, ntr_i, l_i) = (p.m as i32, ntr as i32, g.l as i32);
        let gs = plan.grid_elems() as u64;
        let (gp, _gg) = plan.grid.device_ptr(stream);
        let cfg = LaunchConfig {
            grid_dim: (p.m as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: g.smem,
        };
        unsafe {
            let f = if adjoint { &g.spread } else { &g.interp };
            let mut b = stream.launch_builder(f);
            b.arg(&p.sc)
                .arg(&p.perm)
                .arg(&w[0])
                .arg(&w[1])
                .arg(&w[2])
                .arg(&w[3]);
            b.arg(&gp)
                .arg(&c_ptr)
                .arg(&m_i)
                .arg(&ntr_i)
                .arg(&gs)
                .arg(&d)
                .arg(&l_i);
            b.arg(&g.pts).arg(&g.ptsw).arg(&g.binv).arg(&g.h).arg(&l_i);
            b.launch(cfg)?;
        }
        Ok(())
    }

    /// `c_j = sum_i f_i exp(-i k_j.(i - N/2)) exp(-i Phi_j(xi_i))`
    pub fn forward(
        &mut self,
        stream: &Arc<CudaStream>,
        f: &CudaSlice<hasty_cuda::C32>,
        c: &mut CudaSlice<hasty_cuda::C32>,
    ) -> Result<(), NufftError> {
        let Self { plan, gpu, .. } = self;
        let gpu: &ChirpGpu = gpu;
        if gpu.stencils.is_some() {
            plan.type2_with(stream, f, c, -1, &|pl, st, cptr, ntr| Self::launch_stored(gpu, pl, st, cptr, ntr, false))
        } else {
            plan.type2_with(stream, f, c, -1, &|pl, st, cptr, ntr| Self::launch(gpu, pl, st, cptr, ntr, false))
        }
    }

    /// `f_i = sum_j conj(A_ji) c_j`
    pub fn adjoint(
        &mut self,
        stream: &Arc<CudaStream>,
        c: &CudaSlice<hasty_cuda::C32>,
        f: &mut CudaSlice<hasty_cuda::C32>,
    ) -> Result<(), NufftError> {
        let Self { plan, gpu, .. } = self;
        let gpu: &ChirpGpu = gpu;
        if gpu.stencils.is_some() {
            plan.type1_with(stream, c, f, 1, &|pl, st, cptr, ntr| Self::launch_stored(gpu, pl, st, cptr, ntr, true))
        } else {
            plan.type1_with(stream, c, f, 1, &|pl, st, cptr, ntr| Self::launch(gpu, pl, st, cptr, ntr, true))
        }
    }
}
