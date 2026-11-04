//! Voxelise the atlas tissue groups (`tools/phantom/pack_meshes.py`) on a grid by the generalised winding number
//! w(p) = Σ_triangles Ω(p) / 4π (robust to open / holed atlas surfaces; inside where w > ½).
//!   phantomvox <phantom_meshes.hv> <out.hv> --center x,y,z --size sx,sy,sz --vox v      (metres)
//!
//! Per group (flags from the pack):
//! - `per_member`: each member is evaluated only inside its own bounding box (+ pad) — muscles and bones, where the
//!   group spans the body but each member is small.
//! - `envelope`: the group is a shell (skin, organ walls, myocardium). Its triangles are classified by ray casting
//!   into *outer* (≥ half of the normal-hemisphere rays escape the group) and *inner*; `env` = winding of the outer
//!   triangles only = the solid enclosed by the shell (cavity / lumen = env ∧ ¬w).
//! - `member_envelope`: additionally `member` = 1 + argmax_m of the winding of member m's inner triangles, reversed
//!   (which chamber's endocardium encloses a cavity voxel); 0 where no member exceeds ¼.
//! Output (hasty value): `groups`: list of {name, box_lo [x,y,z], box_dims [x,y,z], w, env?, member?} with u8 tensors
//! [Z, Y, X] over the group's box (w, env: clamp(w, 0, 2) × 127.5); grid origin_m / voxel_m / dims_xyz.

use std::time::Instant;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device};
use hasty_value::{DType, Tensor, Value};

const SRC: &str = r#"
// winding number of a triangle set at the voxel centres of an evaluation box, accumulated (× sign) into a box buffer
extern "C" __global__ void winding(const float* __restrict__ tri, int ntri, float* __restrict__ w, float sign,
                                   int x0, int y0, int z0, int bx, int by, int bz,        // evaluation box (grid)
                                   int dx0, int dy0, int dz0, int dnx, int dny,           // destination box (grid)
                                   float ox, float oy, float oz, float vox)               // grid origin (local)
{
    __shared__ float t[256 * 9];
    long long id = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long nb = (long long)bx * by * bz;
    bool live = id < nb;
    int i = 0, j = 0, k = 0;
    float px = 0.f, py = 0.f, pz = 0.f;
    if (live) {
        i = x0 + (int)(id % bx); j = y0 + (int)((id / bx) % by); k = z0 + (int)(id / ((long long)bx * by));
        px = ox + (i + 0.5f) * vox; py = oy + (j + 0.5f) * vox; pz = oz + (k + 0.5f) * vox;
    }
    float acc = 0.f;
    for (int base = 0; base < ntri; base += 256) {
        int n = min(256, ntri - base);
        __syncthreads();
        for (int q = threadIdx.x; q < n * 9; q += blockDim.x) t[q] = tri[(long long)base * 9 + q];
        __syncthreads();
        if (!live) continue;
        for (int m = 0; m < n; m++) {
            const float* v = t + 9 * m;
            float ax = v[0] - px, ay = v[1] - py, az = v[2] - pz;
            float bx_ = v[3] - px, by_ = v[4] - py, bz_ = v[5] - pz;
            float cx = v[6] - px, cy = v[7] - py, cz = v[8] - pz;
            float la = sqrtf(ax * ax + ay * ay + az * az), lb = sqrtf(bx_ * bx_ + by_ * by_ + bz_ * bz_), lc = sqrtf(cx * cx + cy * cy + cz * cz);
            float num = ax * (by_ * cz - bz_ * cy) + ay * (bz_ * cx - bx_ * cz) + az * (bx_ * cy - by_ * cx);
            float den = la * lb * lc + (ax * bx_ + ay * by_ + az * bz_) * lc + (bx_ * cx + by_ * cy + bz_ * cz) * la + (cx * ax + cy * ay + cz * az) * lb;
            acc += 2.f * atan2f(num, den);
        }
    }
    if (live) w[((long long)(k - dz0) * dny + (j - dy0)) * dnx + (i - dx0)] += sign * acc * 0.07957747154594767f;  // 1 / 4π
}

// does the ray (from triangle ray_tri[r]'s centroid along ray_dir[r]) escape the triangle set? → esc[tri] += 1
extern "C" __global__ void escape(const float* __restrict__ tri, int ntri, const int* __restrict__ ray_tri,
                                  const float* __restrict__ ray_dir, int nray, int* __restrict__ esc)
{
    __shared__ float t[256 * 9];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    bool live = id < nray;
    int self = -1;
    float ox = 0.f, oy = 0.f, oz = 0.f, dx = 0.f, dy = 0.f, dz = 0.f;
    if (live) {
        self = ray_tri[id];
        const float* v = tri + 9LL * self;
        dx = ray_dir[3 * id]; dy = ray_dir[3 * id + 1]; dz = ray_dir[3 * id + 2];
        ox = (v[0] + v[3] + v[6]) * (1.f / 3.f); oy = (v[1] + v[4] + v[7]) * (1.f / 3.f); oz = (v[2] + v[5] + v[8]) * (1.f / 3.f);
    }
    bool hit = false;
    for (int base = 0; base < ntri; base += 256) {
        int n = min(256, ntri - base);
        __syncthreads();
        for (int q = threadIdx.x; q < n * 9; q += blockDim.x) t[q] = tri[(long long)base * 9 + q];
        __syncthreads();
        if (!live || hit) continue;
        for (int m = 0; m < n; m++) {
            if (base + m == self) continue;
            const float* v = t + 9 * m;
            float e1x = v[3] - v[0], e1y = v[4] - v[1], e1z = v[5] - v[2];
            float e2x = v[6] - v[0], e2y = v[7] - v[1], e2z = v[8] - v[2];
            float px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
            float det = e1x * px + e1y * py + e1z * pz;
            if (fabsf(det) < 1e-16f) continue;
            float inv = 1.f / det;
            float sx = ox - v[0], sy = oy - v[1], sz = oz - v[2];
            float u = (sx * px + sy * py + sz * pz) * inv;
            if (u < 0.f || u > 1.f) continue;
            float qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
            float vv = (dx * qx + dy * qy + dz * qz) * inv;
            if (vv < 0.f || u + vv > 1.f) continue;
            float tt = (e2x * qx + e2y * qy + e2z * qz) * inv;
            if (tt > 2e-6f) { hit = true; break; }
        }
    }
    if (live && !hit) atomicAdd(&esc[self], 1);
}

extern "C" __global__ void argmax_update(const float* __restrict__ c, float* __restrict__ best, unsigned char* __restrict__ id,
                                         long long n, int m)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n && c[i] > best[i]) { best[i] = c[i]; id[i] = (unsigned char)m; }
}
"#;

fn arg3(args: &[String], name: &str) -> [f64; 3] {
    let i = args.iter().position(|a| a == name).unwrap_or_else(|| panic!("{name} x,y,z"));
    let v: Vec<f64> = args[i + 1].split(',').map(|x| x.parse().unwrap()).collect();
    [v[0], v[1], v[2]]
}

fn flag(g: &Value, k: &str) -> bool {
    matches!(g.get(k), Some(Value::Bool(true)))
}

/// 64 near-uniform directions (Fibonacci sphere)
fn sphere_dirs(n: usize) -> Vec<[f64; 3]> {
    let ga = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).sqrt();
            let a = ga * i as f64;
            [r * a.cos(), r * a.sin(), z]
        })
        .collect()
}

struct Grid {
    dims: [usize; 3],
    origin: [f64; 3],
    vox: f64,
}

type Box3 = ([usize; 3], [usize; 3]);

fn bbox(tri: &[f32], a: usize, b: usize, pad: f64, g: &Grid, shift: [f64; 3]) -> Option<Box3> {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for t in a..b {
        for p in 0..3 {
            for ax in 0..3 {
                let x = tri[9 * t + 3 * p + ax] as f64 + shift[ax];
                lo[ax] = lo[ax].min(x);
                hi[ax] = hi[ax].max(x);
            }
        }
    }
    let l: [i64; 3] = [0, 1, 2].map(|ax| ((lo[ax] - pad - g.origin[ax]) / g.vox).floor() as i64 - 1);
    let h: [i64; 3] = [0, 1, 2].map(|ax| ((hi[ax] + pad - g.origin[ax]) / g.vox).ceil() as i64 + 1);
    let l2 = [0, 1, 2].map(|ax| l[ax].clamp(0, g.dims[ax] as i64) as usize);
    let h2 = [0, 1, 2].map(|ax| h[ax].clamp(0, g.dims[ax] as i64) as usize);
    if (0..3).any(|ax| h2[ax] <= l2[ax]) {
        None
    } else {
        Some((l2, h2))
    }
}

struct K {
    st: std::sync::Arc<CudaStream>,
    wind: cudarc::driver::CudaFunction,
    esc: cudarc::driver::CudaFunction,
    amax: cudarc::driver::CudaFunction,
}

impl K {
    /// accumulate sign × winding of `tri` (local coordinates) over evaluation box `e` into the buffer of box `d`
    fn winding(&self, tri: &[f32], w: &mut CudaSlice<f32>, sign: f32, e: Box3, d: Box3, g: &Grid, local_origin: [f64; 3]) -> f64 {
        if tri.is_empty() {
            return 0.0;
        }
        let dt = self.st.clone_htod(tri).unwrap();
        let ntri = (tri.len() / 9) as i32;
        let bd = [0, 1, 2].map(|a| e.1[a] - e.0[a]);
        let nb: usize = bd.iter().product();
        let dd = [0, 1, 2].map(|a| d.1[a] - d.0[a]);
        let (x0, y0, z0, bx, by, bz) = (e.0[0] as i32, e.0[1] as i32, e.0[2] as i32, bd[0] as i32, bd[1] as i32, bd[2] as i32);
        let (dx0, dy0, dz0, dnx, dny) = (d.0[0] as i32, d.0[1] as i32, d.0[2] as i32, dd[0] as i32, dd[1] as i32);
        let (ox, oy, oz, vf) = (local_origin[0] as f32, local_origin[1] as f32, local_origin[2] as f32, g.vox as f32);
        let cfg = LaunchConfig { grid_dim: ((nb as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.st
                .launch_builder(&self.wind)
                .arg(&dt).arg(&ntri).arg(w).arg(&sign)
                .arg(&x0).arg(&y0).arg(&z0).arg(&bx).arg(&by).arg(&bz)
                .arg(&dx0).arg(&dy0).arg(&dz0).arg(&dnx).arg(&dny)
                .arg(&ox).arg(&oy).arg(&oz).arg(&vf)
                .launch(cfg)
                .unwrap();
        }
        nb as f64 * ntri as f64
    }

    /// fraction of normal-hemisphere rays from each triangle that escape the triangle set
    fn escape_fraction(&self, tri: &[f32], dirs: &[[f64; 3]]) -> Vec<f32> {
        let nt = tri.len() / 9;
        let (mut rt, mut rd, mut active) = (vec![], vec![], vec![0i32; nt]);
        for t in 0..nt {
            let v = |p: usize, a: usize| tri[9 * t + 3 * p + a] as f64;
            let e1 = [v(1, 0) - v(0, 0), v(1, 1) - v(0, 1), v(1, 2) - v(0, 2)];
            let e2 = [v(2, 0) - v(0, 0), v(2, 1) - v(0, 1), v(2, 2) - v(0, 2)];
            let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let ln = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if ln < 1e-14 {
                continue;
            }
            for d in dirs {
                if (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) / ln > 0.1 {
                    rt.push(t as i32);
                    rd.extend(d.iter().map(|&x| x as f32));
                    active[t] += 1;
                }
            }
        }
        let dt = self.st.clone_htod(tri).unwrap();
        let drt = self.st.clone_htod(&rt).unwrap();
        let drd = self.st.clone_htod(&rd).unwrap();
        let mut esc = self.st.alloc_zeros::<i32>(nt).unwrap();
        let (ntri, nray) = (nt as i32, rt.len() as i32);
        let cfg = LaunchConfig { grid_dim: ((rt.len() as u32).div_ceil(256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.st.launch_builder(&self.esc).arg(&dt).arg(&ntri).arg(&drt).arg(&drd).arg(&nray).arg(&mut esc).launch(cfg).unwrap();
        }
        let e = self.st.clone_dtoh(&esc).unwrap();
        (0..nt).map(|t| if active[t] == 0 { 1.0 } else { e[t] as f32 / active[t] as f32 }).collect()
    }
}

fn select(tri: &[f32], keep: impl Fn(usize) -> bool, flip: bool) -> Vec<f32> {
    let mut out = Vec::new();
    for t in 0..tri.len() / 9 {
        if keep(t) {
            let s = &tri[9 * t..9 * t + 9];
            if flip {
                out.extend_from_slice(&s[0..3]);
                out.extend_from_slice(&s[6..9]);
                out.extend_from_slice(&s[3..6]);
            } else {
                out.extend_from_slice(s);
            }
        }
    }
    out
}

fn q8(w: &[f32]) -> Vec<u8> {
    w.iter().map(|&x| (x.clamp(0.0, 2.0) * 127.5).round() as u8).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (inp, out) = (&args[0], &args[1]);
    let center = arg3(&args, "--center");
    let size = arg3(&args, "--size");
    let vox: f64 = args[args.iter().position(|a| a == "--vox").unwrap() + 1].parse().unwrap();
    let dims: [usize; 3] = [0, 1, 2].map(|a| (size[a] / vox).round() as usize);
    let g = Grid { dims, origin: [0, 1, 2].map(|a| center[a] - 0.5 * dims[a] as f64 * vox), vox };
    let v = hasty_value::read(&mut std::io::BufReader::new(std::fs::File::open(inp).unwrap())).unwrap();
    let Value::List(groups) = v.get("groups").unwrap() else { panic!("groups") };
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let m = dev.load_module("phantom_winding_v2", SRC, &CompileSettings::default()).unwrap();
    let k = K { st: st.clone(), wind: m.load_function("winding").unwrap(), esc: m.load_function("escape").unwrap(), amax: m.load_function("argmax_update").unwrap() };
    let dirs = sphere_dirs(64);
    let t0 = Instant::now();
    let mut res_groups = vec![];
    for gv in groups.iter() {
        let name = gv.get("name").and_then(|x| x.as_str()).unwrap().to_string();
        let raw = gv.get("tri").and_then(|x| x.as_tensor()).unwrap().to_vec::<f32>().unwrap();
        let off = gv.get("offsets").and_then(|x| x.as_tensor()).unwrap();
        let offs: Vec<usize> = (0..off.numel()).map(|i| off.get_f64(i).0 as usize).collect();
        let (per_member, envelope, member_env) = (flag(gv, "per_member"), flag(gv, "envelope"), flag(gv, "member_envelope"));
        let tg = Instant::now();
        // local coordinates (float precision): shift by the group's mean vertex
        let nt = raw.len() / 9;
        let mut shift = [0f64; 3];
        for t in 0..nt * 3 {
            for a in 0..3 {
                shift[a] += raw[3 * t + a] as f64 / (nt * 3) as f64;
            }
        }
        let tri: Vec<f32> = raw.iter().enumerate().map(|(i, &x)| (x as f64 - shift[i % 3]) as f32).collect();
        let lo = [0, 1, 2].map(|a| g.origin[a] - shift[a]);
        // evaluation jobs and the group's output box
        let mut jobs: Vec<(usize, usize, Box3)> = vec![];
        if per_member {
            for w in offs.windows(2) {
                if let Some(b) = bbox(&tri, w[0], w[1], 5e-3, &g, shift) {
                    jobs.push((w[0], w[1], b));
                }
            }
        } else if name == "body" {
            jobs.push((0, nt, ([0; 3], dims)));
        } else if let Some(b) = bbox(&tri, 0, nt, 5e-3, &g, shift) {
            jobs.push((0, nt, b));
        }
        if jobs.is_empty() {
            println!("{name:12} outside the field of view");
            continue;
        }
        let mut obox: Box3 = jobs[0].2;
        for (_, _, b) in &jobs {
            for a in 0..3 {
                obox.0[a] = obox.0[a].min(b.0[a]);
                obox.1[a] = obox.1[a].max(b.1[a]);
            }
        }
        let od = [0, 1, 2].map(|a| obox.1[a] - obox.0[a]);
        let nb: usize = od.iter().product();
        let mut wbuf = st.alloc_zeros::<f32>(nb).unwrap();
        let mut pairs = 0.0;
        for &(a, b, e) in &jobs {
            pairs += k.winding(&tri[9 * a..9 * b], &mut wbuf, 1.0, e, obox, &g, lo);
        }
        let w = st.clone_dtoh(&wbuf).unwrap();
        let mut entry = vec![
            ("name".to_string(), Value::Str(name.clone())),
            ("box_lo".into(), Value::List(obox.0.iter().map(|&x| Value::Int(x as i64)).collect())),
            ("box_dims".into(), Value::List(od.iter().map(|&x| Value::Int(x as i64)).collect())),
            ("w".into(), Value::Tensor(Tensor::from_bytes(DType::U8, vec![od[2], od[1], od[0]], &q8(&w)).unwrap())),
        ];
        let inside = w.iter().filter(|&&x| x > 0.5).count();
        let mut msg = format!("{name:12} w>½ {:8.1} cm³", inside as f64 * (vox * 100.0).powi(3));
        if envelope {
            let frac = k.escape_fraction(&tri, &dirs);
            let outer = |t: usize| frac[t] >= 0.5;
            let n_out = (0..nt).filter(|&t| outer(t)).count();
            let mut ebuf = st.alloc_zeros::<f32>(nb).unwrap();
            pairs += k.winding(&select(&tri, outer, false), &mut ebuf, 1.0, obox, obox, &g, lo);
            let env = st.clone_dtoh(&ebuf).unwrap();
            let cav = env.iter().zip(&w).filter(|&(&e, &x)| e > 0.5 && x <= 0.5).count();
            msg += &format!(", outer {n_out}/{nt} triangles, enclosed {:8.1} cm³ (cavity {:7.1})", env.iter().filter(|&&x| x > 0.5).count() as f64 * (vox * 100.0).powi(3), cav as f64 * (vox * 100.0).powi(3));
            entry.push(("env".into(), Value::Tensor(Tensor::from_bytes(DType::U8, vec![od[2], od[1], od[0]], &q8(&env)).unwrap())));
            if member_env {
                let mut best = st.clone_htod(&vec![0.25f32; nb]).unwrap();
                let mut id = st.alloc_zeros::<u8>(nb).unwrap();
                let mut cbuf = st.alloc_zeros::<f32>(nb).unwrap();
                for (mi, wm) in offs.windows(2).enumerate() {
                    st.memset_zeros(&mut cbuf).unwrap();
                    let inner = select(&tri, |t| t >= wm[0] && t < wm[1] && !outer(t), true);
                    pairs += k.winding(&inner, &mut cbuf, 1.0, obox, obox, &g, lo);
                    let (n, mm) = (nb as i64, mi as i32 + 1);
                    let cfg = LaunchConfig { grid_dim: ((nb as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    unsafe {
                        st.launch_builder(&k.amax).arg(&cbuf).arg(&mut best).arg(&mut id).arg(&n).arg(&mm).launch(cfg).unwrap();
                    }
                }
                let ids = st.clone_dtoh(&id).unwrap();
                entry.push(("member".into(), Value::Tensor(Tensor::from_bytes(DType::U8, vec![od[2], od[1], od[0]], &ids).unwrap())));
            }
        }
        if let Some(mem) = gv.get("members") {
            entry.push(("members".into(), mem.clone()));
        }
        println!("{msg}; {:.2e} pairs, {:.1}s", pairs, tg.elapsed().as_secs_f64());
        res_groups.push(Value::Dict(entry));
    }
    let res = Value::Dict(vec![
        ("groups".into(), Value::List(res_groups)),
        ("origin_m".into(), Value::List(g.origin.iter().map(|&x| Value::Float(x)).collect())),
        ("voxel_m".into(), Value::Float(vox)),
        ("dims_xyz".into(), Value::List(dims.iter().map(|&x| Value::Int(x as i64)).collect())),
        ("note".into(), Value::Str("w, env: u8 = clamp(winding, 0, 2) × 127.5 over the group box [Z, Y, X]; env = winding of the outer (escaping) triangles; member = 1 + chamber whose reversed inner surface encloses the voxel most; atlas Z-Anatomy CC BY-SA 4.0".into())),
    ]);
    hasty_value::write(&res, &hasty_value::Options::zstd(9), &mut std::io::BufWriter::new(std::fs::File::create(out).unwrap())).unwrap();
    println!("wrote {out} ({:?} voxels, {:.0}s)", dims, t0.elapsed().as_secs_f64());
}
