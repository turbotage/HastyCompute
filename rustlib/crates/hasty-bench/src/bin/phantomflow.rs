//! Ground-truth blood flow of the atlas phantom on any grid (GPU): vessel lumens as truncated-cone capsules along the
//! graph centrelines, each carrying its network flow harmonics with Womersley profiles.
//!   phantomflow <vessels.hv> <out.hv> --center x,y,z --size sx,sy,sz --vox v [--sub S] [--harmonics H] [--cell c]
//! (`vessels.hv` from `tools/phantom/pack_vessels.py`; metres.)
//!
//! Every voxel is sampled at S³ sub-voxel points. A point is blood if it lies inside a capsule (distance to the axis
//! d < r(s)); the capsule whose surface is farthest away (smallest d − r) owns it: a branch still running inside its
//! parent owns nothing (its flow is handed to the parent in `pack_vessels.py`), pressed neighbours split their overlap
//! at its mid-surface. Its velocity
//! is the capsule tangent × Q_n(s) / (πr²) · g(d / r; α_n) per heart-rate harmonic n (Womersley, α_n = r √(ω_n ρ / μ)).
//! A point outside every lumen but within a wall thickness of one is vessel wall.
//! Contacts: where another vessel owns part of a capsule's cross-section — a neighbour pressed against it (D-shaped
//! lumens) or a branch near / along its origin, whose flow still passes through that cross-section — Each capsule's harmonic h is scaled by
//! ∫g_h dA / ∫_owned g_h dA over its mid cross-section (96 equal-area points), so every vessel keeps its network flux.
//! Output (sparse, voxels with blood or wall): `idx` u32 (linear z·ny·nx + y·nx + x), `blood` / `wall` f32 fractions,
//! `cls` u8 (class of the deepest blood point), `sample` u32 (its capsule = centreline sample index, u32::MAX if
//! none), `vh` f32 [N, H+1, 3, 2]: harmonics of the voxel-mean velocity (Σ over blood points / S³; tissue counts as 0),
//! v(t) = Re[vh_0 + 2 Σ_n vh_n e^{iω_n t}]; the blood-mean velocity is vh / blood. Grid geometry, omega, period.

use std::time::Instant;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device};
use hasty_value::{DType, Tensor, Value};

const SRC: &str = r#"
__device__ __forceinline__ void cone(const float* P, const float* R, const float* W, unsigned i, float px, float py, float pz,
                                     float& s, float& d, float& r, float& w)
{
    float ax = P[3 * i], ay = P[3 * i + 1], az = P[3 * i + 2];
    float bx = P[3 * i + 3] - ax, by = P[3 * i + 4] - ay, bz = P[3 * i + 5] - az;
    float qx = px - ax, qy = py - ay, qz = pz - az;
    float l2 = bx * bx + by * by + bz * bz;
    s = l2 > 0.f ? fminf(fmaxf((qx * bx + qy * by + qz * bz) / l2, 0.f), 1.f) : 0.f;
    float ex = qx - s * bx, ey = qy - s * by, ez = qz - s * bz;
    d = sqrtf(ex * ex + ey * ey + ez * ez);
    r = R[i] + s * (R[i + 1] - R[i]);
    w = W[i] + s * (W[i + 1] - W[i]);
}

__device__ __forceinline__ int cell_of(float px, float py, float pz, float cx0, float cy0, float cz0, float cell, int cnx, int cny, int cnz)
{
    int a = (int)floorf((px - cx0) / cell), b = (int)floorf((py - cy0) / cell), c = (int)floorf((pz - cz0) / cell);
    if (a < 0 || b < 0 || c < 0 || a >= cnx || b >= cny || c >= cnz) return -1;
    return (c * cny + b) * cnx + a;
}

// candidate voxels: the centre is within reach (r + wall + margin) of some capsule
extern "C" __global__ void candidates(const float* P, const float* R, const float* W, const unsigned* cstart, const unsigned* clist,
                                      int cnx, int cny, int cnz, float cx0, float cy0, float cz0, float cell,
                                      int nx, int ny, int nz, float ox, float oy, float oz, float vox, float margin,
                                      unsigned char* flag)
{
    long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (v >= (long long)nx * ny * nz) return;
    int i = (int)(v % nx), j = (int)((v / nx) % ny), k = (int)(v / ((long long)nx * ny));
    float px = ox + (i + 0.5f) * vox, py = oy + (j + 0.5f) * vox, pz = oz + (k + 0.5f) * vox;
    int c = cell_of(px, py, pz, cx0, cy0, cz0, cell, cnx, cny, cnz);
    unsigned char f = 0;
    if (c >= 0) {
        for (unsigned e = cstart[c]; e < cstart[c + 1]; e++) {
            float s, d, r, w;
            cone(P, R, W, clist[e], px, py, pz, s, d, r, w);
            if (d < r + w + margin) { f = 1; break; }
        }
    }
    flag[v] = f;
}

__device__ __forceinline__ void prof_at(const float* prof, int na, int nxi, float amax, float al, float xi, float& gr, float& gi)
{
    float fx = fminf(fmaxf(xi, 0.f), 1.f) * (nxi - 1);
    int x0 = min((int)fx, nxi - 2);
    float wx = fx - x0;
    float fa = fminf(al / amax, 1.f) * (na - 1);
    int a0 = min((int)fa, na - 2);
    float wa = fa - a0;
    const float* g00 = prof + ((long long)a0 * nxi + x0) * 2;
    const float* g10 = g00 + nxi * 2;
    gr = (1 - wa) * ((1 - wx) * g00[0] + wx * g00[2]) + wa * ((1 - wx) * g10[0] + wx * g10[2]);
    gi = (1 - wa) * ((1 - wx) * g00[1] + wx * g00[3]) + wa * ((1 - wx) * g10[1] + wx * g10[3]);
}

// owner of a point: the capsule with the smallest d / r among those containing it (−1: none)
__device__ int owner(const float* P, const float* R, const float* W, const unsigned* cstart, const unsigned* clist,
                     int cnx, int cny, int cnz, float cx0, float cy0, float cz0, float cell, float px, float py, float pz)
{
    int c = cell_of(px, py, pz, cx0, cy0, cz0, cell, cnx, cny, cnz);
    if (c < 0) return -1;
    float bq = 1e30f;
    int bi = -1;
    for (unsigned e = cstart[c]; e < cstart[c + 1]; e++) {
        float s, d, r, w;
        cone(P, R, W, clist[e], px, py, pz, s, d, r, w);
        if (d < r && d - r < bq) { bq = d - r; bi = (int)clist[e]; }
    }
    return bi;
}

extern "C" __global__ void own_frac(const unsigned* caps, int ncap, const float* P, const float* R, const float* W,
                                    const unsigned* sedge, const int* enodes, const float* eleave,
                                    const unsigned* cstart, const unsigned* clist, int cnx, int cny, int cnz, float cx0, float cy0, float cz0, float cell,
                                    const float* om, float rho_mu, const float* prof, int na, int nxi, float amax, int h1,
                                    float* fac, float* owned_area)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ncap) return;
    unsigned i = caps[id];
    float tx = P[3 * i + 3] - P[3 * i], ty = P[3 * i + 4] - P[3 * i + 1], tz = P[3 * i + 5] - P[3 * i + 2];
    float tl = sqrtf(tx * tx + ty * ty + tz * tz);
    if (tl <= 0.f) return;
    tx /= tl; ty /= tl; tz /= tl;
    float mx = P[3 * i] + 0.5f * tx * tl, my = P[3 * i + 1] + 0.5f * ty * tl, mz = P[3 * i + 2] + 0.5f * tz * tl;
    float r = 0.5f * (R[i] + R[i + 1]);
    // basis u, v ⟂ t
    float ax = fabsf(tx) < 0.9f ? 1.f : 0.f, ay = 1.f - ax;
    float ux = ty * 0.f - tz * ay, uy = tz * ax - tx * 0.f, uz = tx * ay - ty * ax;
    float ul = sqrtf(ux * ux + uy * uy + uz * uz);
    ux /= ul; uy /= ul; uz /= ul;
    float vx = ty * uz - tz * uy, vy = tz * ux - tx * uz, vz = tx * uy - ty * ux;
    unsigned e0 = sedge[i];
    const int K = 6, A = 16;
    int n_own = 0;
    float* f = fac + (long long)i * h1 * 2;
    // accumulate owned and total profile integrals per harmonic in fac (re, im of owned) and a local total
    for (int h = 0; h < h1; h++) { f[2 * h] = 0.f; f[2 * h + 1] = 0.f; }
    float tot_r[64], tot_i[64];
    for (int h = 0; h < h1; h++) { tot_r[h] = 0.f; tot_i[h] = 0.f; }
    for (int k = 0; k < K; k++) {
        float xi = sqrtf((k + 0.5f) / K);
        for (int a = 0; a < A; a++) {
            float ang = 6.283185307f * (a + 0.5f * (k & 1)) / A;
            float cu = cosf(ang) * xi * r, cv = sinf(ang) * xi * r;
            float px = mx + cu * ux + cv * vx, py = my + cu * uy + cv * vy, pz = mz + cu * uz + cv * vz;
            int o = owner(P, R, W, cstart, clist, cnx, cny, cnz, cx0, cy0, cz0, cell, px, py, pz);
            bool own = true;
            if (o >= 0) {
                unsigned e1 = sedge[o];
                // same stream: the same edge, or an edge continuing it through a shared node (leaving directions
                // opposite); branches and parallel duplicate splines leave alike and are compensated
                own = e1 == e0;
                for (int x = 0; x < 2 && !own; x++)
                    for (int y = 0; y < 2 && !own; y++)
                        if (enodes[2 * e0 + x] == enodes[2 * e1 + y]) {
                            const float* l0 = eleave + (2 * e0 + x) * 3;
                            const float* l1 = eleave + (2 * e1 + y) * 3;
                            own = l0[0] * l1[0] + l0[1] * l1[1] + l0[2] * l1[2] < -0.5f;
                        }
            }
            n_own += own;
            for (int h = 0; h < h1; h++) {
                float gr, gi;
                prof_at(prof, na, nxi, amax, r * sqrtf(om[h] * rho_mu), xi, gr, gi);
                tot_r[h] += gr; tot_i[h] += gi;
                if (own) { f[2 * h] += gr; f[2 * h + 1] += gi; }
            }
        }
    }
    owned_area[i] = n_own / (float)(K * A);
    for (int h = 0; h < h1; h++) {
        // factor = total / owned (complex); owned part floored at ½ of the total (factor ≤ 2: emerging branches)
        float orr = f[2 * h], oi = f[2 * h + 1];
        float tr = tot_r[h], ti = tot_i[h];
        float on = sqrtf(orr * orr + oi * oi), tn = sqrtf(tr * tr + ti * ti);
        if (on < 0.5f * tn) { if (on <= 0.f) { orr = 0.5f * tr; oi = 0.5f * ti; } else { float sc = 0.5f * tn / on; orr *= sc; oi *= sc; } }
        float den = orr * orr + oi * oi;
        f[2 * h] = (tr * orr + ti * oi) / den;
        f[2 * h + 1] = (ti * orr - tr * oi) / den;
    }
}

extern "C" __global__ void truth(const unsigned* cand, long long ncand, const float* P, const float* R, const float* W,
                                 const unsigned char* C, const float* Q, int hq, int h1,
                                 const unsigned* cstart, const unsigned* clist, int cnx, int cny, int cnz, float cx0, float cy0, float cz0, float cell,
                                 int nx, int ny, float ox, float oy, float oz, float vox, int S,
                                 const float* om, float rho_mu, const float* prof, int na, int nxi, float amax, const float* fac,
                                 const float* WV, float* blood, float* wallf, unsigned char* cls, unsigned* samp, float* vh, float* wv)
{
    long long id = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (id >= ncand) return;
    unsigned v = cand[id];
    int i = (int)(v % nx), j = (int)((v / nx) % ny), k = (int)(v / ((unsigned)nx * ny));
    float inv = 1.f / (float)(S * S * S);
    float* acc = vh + id * (long long)h1 * 6;
    int nb = 0, nw = 0, bestc = 0;
    float best = 1e30f;
    unsigned bests = 0xffffffffu;
    for (int c3 = 0; c3 < S; c3++)
    for (int b3 = 0; b3 < S; b3++)
    for (int a3 = 0; a3 < S; a3++) {
        float px = ox + (i + (a3 + 0.5f) / S) * vox, py = oy + (j + (b3 + 0.5f) / S) * vox, pz = oz + (k + (c3 + 0.5f) / S) * vox;
        int c = cell_of(px, py, pz, cx0, cy0, cz0, cell, cnx, cny, cnz);
        if (c < 0) continue;
        float bq = 1e30f, bs = 0.f, bx = 0.f;  // bq: signed depth d − r of the owner, bx: its d / r
        int bi = -1;
        bool inwall = false;
        for (unsigned e = cstart[c]; e < cstart[c + 1]; e++) {
            unsigned q = clist[e];
            float s, d, r, w;
            cone(P, R, W, q, px, py, pz, s, d, r, w);
            if (d < r) {
                if (d - r < bq) { bq = d - r; bx = d / r; bi = (int)q; bs = s; }
            } else if (d < r + w) inwall = true;
        }
        if (bi < 0) { nw += inwall; continue; }
        nb++;
        if (bx < best) { best = bx; bestc = C[bi]; bests = (unsigned)bi; }
        unsigned q = (unsigned)bi;
        float tx = P[3 * q + 3] - P[3 * q], ty = P[3 * q + 4] - P[3 * q + 1], tz = P[3 * q + 5] - P[3 * q + 2];
        float tl = sqrtf(tx * tx + ty * ty + tz * tz);
        if (tl <= 0.f) continue;
        tx /= tl; ty /= tl; tz /= tl;
        float r = R[q] + bs * (R[q + 1] - R[q]);
        float sc = inv / (3.14159265358979f * r * r);
        for (int c = 0; c < 3; c++) wv[id * 3 + c] += inv * (WV[3 * q + c] + bs * (WV[3 * q + 3 + c] - WV[3 * q + c]));
        const float* fq = fac + (long long)q * h1 * 2;
        for (int h = 0; h < h1; h++) {
            float gr0, gi0;
            prof_at(prof, na, nxi, amax, r * sqrtf(om[h] * rho_mu), bx, gr0, gi0);
            float gr = gr0 * fq[2 * h] - gi0 * fq[2 * h + 1], gi = gr0 * fq[2 * h + 1] + gi0 * fq[2 * h];
            const float* qa = Q + ((long long)q * hq + h) * 2;
            const float* qb = Q + ((long long)(q + 1) * hq + h) * 2;
            float qr = qa[0] + bs * (qb[0] - qa[0]), qi = qa[1] + bs * (qb[1] - qa[1]);
            float ur = (gr * qr - gi * qi) * sc, ui = (gr * qi + gi * qr) * sc;
            float* o = acc + h * 6;
            o[0] += tx * ur; o[1] += tx * ui;
            o[2] += ty * ur; o[3] += ty * ui;
            o[4] += tz * ur; o[5] += tz * ui;
        }
    }
    blood[id] = nb * inv;
    wallf[id] = nw * inv;
    cls[id] = (unsigned char)bestc;
    samp[id] = bests;
}
"#;

fn arg3(args: &[String], name: &str) -> [f64; 3] {
    let i = args.iter().position(|a| a == name).unwrap_or_else(|| panic!("{name} x,y,z"));
    let v: Vec<f64> = args[i + 1].split(',').map(|x| x.parse().unwrap()).collect();
    [v[0], v[1], v[2]]
}

fn argf(args: &[String], name: &str, d: f64) -> f64 {
    args.iter().position(|a| a == name).map(|i| args[i + 1].parse().unwrap()).unwrap_or(d)
}

fn tensor_f32(v: &Value, k: &str) -> (Vec<f32>, Vec<usize>) {
    let t = v.get(k).and_then(|x| x.as_tensor()).unwrap_or_else(|| panic!("{k}"));
    (t.to_vec::<f32>().unwrap(), t.shape().to_vec())
}

fn float(v: &Value, k: &str) -> f64 {
    match v.get(k) {
        Some(Value::Float(x)) => *x,
        Some(Value::Int(x)) => *x as f64,
        _ => panic!("{k}"),
    }
}

struct Common {
    eoff: Vec<usize>,
    wall: Vec<f32>,
    cls: Vec<u8>,
    q: Vec<f32>,
    prof: Vec<f32>,
    omega: Vec<f64>,
    enodes: Vec<i32>,
    sedge: Vec<u32>,
    hq: usize,
    h1: usize,
    na: usize,
    nxi: usize,
    rho_mu: f32,
    amax: f32,
}

struct Grid {
    dims: [usize; 3],
    origin: [f64; 3],
    vox: f64,
    sub: i32,
    cell: f64,
}

#[derive(Default)]
struct Sparse {
    idx: Vec<u32>,
    blood: Vec<f32>,
    wall: Vec<f32>,
    cls: Vec<u8>,
    samp: Vec<u32>,
    vh: Vec<f32>,
    wv: Vec<f32>,
}

struct Kern {
    st: std::sync::Arc<cudarc::driver::CudaStream>,
    cand: cudarc::driver::CudaFunction,
    truth: cudarc::driver::CudaFunction,
    own: cudarc::driver::CudaFunction,
}

/// voxelise one geometry (centreline points, radii, per-sample wall velocity) on the grid
fn evaluate(k: &Kern, c: &Common, g: &Grid, pts: &[f32], rad: &[f32], wvel: &[f32], verbose: bool) -> Sparse {
    let t0 = Instant::now();
    let st = &k.st;
    let (dims, origin, vox) = (g.dims, g.origin, g.vox);
    let nv: usize = dims.iter().product();
    let ns = rad.len();
    let (h1, eoff, wall) = (c.h1, &c.eoff, &c.wall);
    // unit directions leaving each edge's end nodes into the edge: [E, 2 (node a, node b), 3]
    let ne = eoff.len() - 1;
    let mut eleave = vec![0f32; ne * 6];
    for e in 0..ne {
        let (a, b) = (eoff[e], eoff[e + 1]);
        if b - a < 2 {
            continue;
        }
        let j = (b - a - 1).min(4);
        for (end, (from, to)) in [(a, a + j), (b - 1, b - 1 - j)].into_iter().enumerate() {
            let d: Vec<f32> = (0..3).map(|x| pts[3 * to + x] - pts[3 * from + x]).collect();
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-12);
            for x in 0..3 {
                eleave[e * 6 + end * 3 + x] = d[x] / l;
            }
        }
    }
    // capsules (sample i → i+1 inside one edge) and a uniform cell grid over the field of view
    let margin = 0.5 * 3f64.sqrt() * vox;
    let cellsz = g.cell;
    let cdims: [usize; 3] = [0, 1, 2].map(|a| ((dims[a] as f64 * vox) / cellsz).ceil() as usize);
    let ncell: usize = cdims.iter().product();
    let mut caps: Vec<(u32, [usize; 3], [usize; 3])> = vec![];
    for w in eoff.windows(2) {
        for i in w[0]..w[1].saturating_sub(1) {
            let reach = (rad[i] + wall[i]).max(rad[i + 1] + wall[i + 1]) as f64 + margin;
            let mut lo = [0usize; 3];
            let mut hi = [0usize; 3];
            let mut inside = true;
            for a in 0..3 {
                let (x0, x1) = (pts[3 * i + a] as f64, pts[3 * i + 3 + a] as f64);
                let l = (x0.min(x1) - reach - origin[a]) / cellsz;
                let h = (x0.max(x1) + reach - origin[a]) / cellsz;
                if h < 0.0 || l >= cdims[a] as f64 {
                    inside = false;
                    break;
                }
                lo[a] = l.max(0.0).floor() as usize;
                hi[a] = (h.floor() as usize).min(cdims[a] - 1);
            }
            if inside {
                caps.push((i as u32, lo, hi));
            }
        }
    }
    let mut cnt = vec![0u32; ncell + 1];
    for (_, lo, hi) in &caps {
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    cnt[(z * cdims[1] + y) * cdims[0] + x + 1] += 1;
                }
            }
        }
    }
    for i in 0..ncell {
        cnt[i + 1] += cnt[i];
    }
    let mut fill = cnt.clone();
    let mut list = vec![0u32; cnt[ncell] as usize];
    for (i, lo, hi) in &caps {
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let kk = (z * cdims[1] + y) * cdims[0] + x;
                    list[fill[kk] as usize] = *i;
                    fill[kk] += 1;
                }
            }
        }
    }
    if verbose {
        println!("{} capsules in the field of view of {ns} samples; {} cell entries over {:?} cells of {:.1} mm", caps.len(), list.len(), cdims, cellsz * 1e3);
    }
    let d_p = st.clone_htod(pts).unwrap();
    let d_r = st.clone_htod(rad).unwrap();
    let d_w = st.clone_htod(wall).unwrap();
    let d_c = st.clone_htod(&c.cls).unwrap();
    let d_q = st.clone_htod(&c.q).unwrap();
    let d_prof = st.clone_htod(&c.prof).unwrap();
    let d_wv = st.clone_htod(wvel).unwrap();
    let om32: Vec<f32> = c.omega.iter().map(|&x| x as f32).collect();
    let d_om = st.clone_htod(&om32).unwrap();
    let d_cs = st.clone_htod(&cnt).unwrap();
    let d_cl = st.clone_htod(&list).unwrap();
    let (cnx, cny, cnz) = (cdims[0] as i32, cdims[1] as i32, cdims[2] as i32);
    let (cx0, cy0, cz0, cell) = (origin[0] as f32, origin[1] as f32, origin[2] as f32, cellsz as f32);
    let (nx, ny, nz) = (dims[0] as i32, dims[1] as i32, dims[2] as i32);
    let (ox, oy, oz, vf, mg) = (origin[0] as f32, origin[1] as f32, origin[2] as f32, vox as f32, margin as f32);
    let (na32, nxi32, h132, hq32, s32) = (c.na as i32, c.nxi as i32, h1 as i32, c.hq as i32, g.sub);
    // 0. contact factors per capsule and harmonic
    let capids: Vec<u32> = caps.iter().map(|x| x.0).collect();
    let mut facinit = vec![0f32; ns * h1 * 2];
    for x in facinit.chunks_mut(2) {
        x[0] = 1.0;
    }
    let mut d_fac = st.clone_htod(&facinit).unwrap();
    let mut d_oa = st.clone_htod(&vec![1f32; ns]).unwrap();
    if !capids.is_empty() {
        let (d_caps, d_se, d_en, d_el) = (st.clone_htod(&capids).unwrap(), st.clone_htod(&c.sedge).unwrap(), st.clone_htod(&c.enodes).unwrap(), st.clone_htod(&eleave).unwrap());
        let nc = capids.len() as i32;
        let cfg = LaunchConfig { grid_dim: ((capids.len() as u32).div_ceil(64).max(1), 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            st.launch_builder(&k.own)
                .arg(&d_caps).arg(&nc).arg(&d_p).arg(&d_r).arg(&d_w).arg(&d_se).arg(&d_en).arg(&d_el)
                .arg(&d_cs).arg(&d_cl).arg(&cnx).arg(&cny).arg(&cnz).arg(&cx0).arg(&cy0).arg(&cz0).arg(&cell)
                .arg(&d_om).arg(&c.rho_mu).arg(&d_prof).arg(&na32).arg(&nxi32).arg(&c.amax).arg(&h132)
                .arg(&mut d_fac).arg(&mut d_oa)
                .launch(cfg)
                .unwrap();
        }
    }
    if verbose {
        let oa = st.clone_dtoh(&d_oa).unwrap();
        let pressed: Vec<f32> = capids.iter().map(|&i| oa[i as usize]).filter(|&a| a < 0.999).collect();
        println!(
            "contacts: {} of {} capsules share their cross-section with a foreign vessel (owned area: median {:.2}, min {:.2}) ({:.1}s)",
            pressed.len(),
            capids.len(),
            { let mut p = pressed.clone(); p.sort_by(|a, b| a.partial_cmp(b).unwrap()); p.get(p.len() / 2).copied().unwrap_or(1.0) },
            pressed.iter().cloned().fold(1.0f32, f32::min),
            t0.elapsed().as_secs_f64()
        );
    }
    // 1. candidates
    let mut d_flag = st.alloc_zeros::<u8>(nv).unwrap();
    let cfg = LaunchConfig { grid_dim: ((nv as u64).div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        st.launch_builder(&k.cand)
            .arg(&d_p).arg(&d_r).arg(&d_w).arg(&d_cs).arg(&d_cl)
            .arg(&cnx).arg(&cny).arg(&cnz).arg(&cx0).arg(&cy0).arg(&cz0).arg(&cell)
            .arg(&nx).arg(&ny).arg(&nz).arg(&ox).arg(&oy).arg(&oz).arg(&vf).arg(&mg)
            .arg(&mut d_flag)
            .launch(cfg)
            .unwrap();
    }
    let flag = st.clone_dtoh(&d_flag).unwrap();
    drop(d_flag);
    let cand: Vec<u32> = flag.iter().enumerate().filter(|(_, &f)| f != 0).map(|(i, _)| i as u32).collect();
    if verbose {
        println!("{} candidate voxels of {nv} ({:.1}s)", cand.len(), t0.elapsed().as_secs_f64());
    }
    // 2. truth, in chunks
    let chunk = ((6usize << 30) / (h1 * 6 * 4 + 32)).min(cand.len().max(1));
    let mut o = Sparse::default();
    for part in cand.chunks(chunk) {
        let n = part.len();
        let d_cand = st.clone_htod(part).unwrap();
        let mut d_blood = st.alloc_zeros::<f32>(n).unwrap();
        let mut d_wall = st.alloc_zeros::<f32>(n).unwrap();
        let mut d_cls = st.alloc_zeros::<u8>(n).unwrap();
        let mut d_samp = st.alloc_zeros::<u32>(n).unwrap();
        let mut d_vh = st.alloc_zeros::<f32>(n * h1 * 6).unwrap();
        let mut d_wvo = st.alloc_zeros::<f32>(n * 3).unwrap();
        let nl = n as i64;
        let cfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(128), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            st.launch_builder(&k.truth)
                .arg(&d_cand).arg(&nl).arg(&d_p).arg(&d_r).arg(&d_w)
                .arg(&d_c).arg(&d_q).arg(&hq32).arg(&h132)
                .arg(&d_cs).arg(&d_cl).arg(&cnx).arg(&cny).arg(&cnz).arg(&cx0).arg(&cy0).arg(&cz0).arg(&cell)
                .arg(&nx).arg(&ny).arg(&ox).arg(&oy).arg(&oz).arg(&vf).arg(&s32)
                .arg(&d_om).arg(&c.rho_mu).arg(&d_prof).arg(&na32).arg(&nxi32).arg(&c.amax).arg(&d_fac)
                .arg(&d_wv).arg(&mut d_blood).arg(&mut d_wall).arg(&mut d_cls).arg(&mut d_samp).arg(&mut d_vh).arg(&mut d_wvo)
                .launch(cfg)
                .unwrap();
        }
        let (b, w, cl, s, vh, wvo) = (
            st.clone_dtoh(&d_blood).unwrap(),
            st.clone_dtoh(&d_wall).unwrap(),
            st.clone_dtoh(&d_cls).unwrap(),
            st.clone_dtoh(&d_samp).unwrap(),
            st.clone_dtoh(&d_vh).unwrap(),
            st.clone_dtoh(&d_wvo).unwrap(),
        );
        for t in 0..n {
            if b[t] > 0.0 || w[t] > 0.0 {
                o.idx.push(part[t]);
                o.blood.push(b[t]);
                o.wall.push(w[t]);
                o.cls.push(cl[t]);
                o.samp.push(s[t]);
                o.vh.extend_from_slice(&vh[t * h1 * 6..(t + 1) * h1 * 6]);
                o.wv.extend_from_slice(&wvo[t * 3..t * 3 + 3]);
            }
        }
    }
    if verbose {
        let vol = |f: &[f32]| f.iter().map(|&x| x as f64).sum::<f64>() * vox.powi(3) * 1e6;
        println!("{} voxels with blood or wall; blood {:.1} mL, wall {:.1} mL ({:.1}s)", o.idx.len(), vol(&o.blood), vol(&o.wall), t0.elapsed().as_secs_f64());
    }
    o
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (inp, out) = (&args[0], &args[1]);
    let center = arg3(&args, "--center");
    let size = arg3(&args, "--size");
    let vox = argf(&args, "--vox", 1e-3);
    let dims: [usize; 3] = [0, 1, 2].map(|a| (size[a] / vox).round() as usize);
    let g = Grid {
        dims,
        origin: [0, 1, 2].map(|a| center[a] - 0.5 * dims[a] as f64 * vox),
        vox,
        sub: argf(&args, "--sub", 4.0) as i32,
        cell: argf(&args, "--cell", 4e-3),
    };
    assert!(dims.iter().product::<usize>() < u32::MAX as usize, "grid too large for u32 voxel indices");
    let t0 = Instant::now();
    let v = hasty_value::read(&mut std::io::BufReader::new(std::fs::File::open(inp).unwrap())).unwrap();
    let (pts, _) = tensor_f32(&v, "points");
    let (rad, _) = tensor_f32(&v, "radius");
    let (wall, _) = tensor_f32(&v, "wall");
    let (q, qs) = tensor_f32(&v, "q");
    let (prof, ps) = tensor_f32(&v, "prof");
    let cls: Vec<u8> = v.get("cls").and_then(|x| x.as_tensor()).unwrap().to_vec::<u8>().unwrap();
    let eo = v.get("edge_offsets").and_then(|x| x.as_tensor()).unwrap();
    let eoff: Vec<usize> = (0..eo.numel()).map(|i| eo.get_f64(i).0 as usize).collect();
    let om_t = v.get("omega").and_then(|x| x.as_tensor()).unwrap();
    let omega: Vec<f64> = (0..om_t.numel()).map(|i| om_t.get_f64(i).0).collect();
    let (period, rho, mu, amax) = (float(&v, "period"), float(&v, "rho"), float(&v, "mu"), float(&v, "alpha_max"));
    let hq = qs[1];
    let h1 = (argf(&args, "--harmonics", (hq - 1) as f64) as usize + 1).min(hq);
    assert!(h1 <= 64, "at most 63 harmonics");
    let ns = rad.len();
    let ints = |k: &str| -> Vec<i32> {
        let t = v.get(k).and_then(|x| x.as_tensor()).unwrap_or_else(|| panic!("{k}"));
        (0..t.numel()).map(|i| t.get_f64(i).0 as i32).collect()
    };
    let mut sedge = vec![0u32; ns];
    for (e, w) in eoff.windows(2).enumerate() {
        for x in &mut sedge[w[0]..w[1]] {
            *x = e as u32;
        }
    }
    let c = Common {
        eoff, wall, cls, q, prof, omega: omega.clone(), enodes: ints("edge_nodes"), sedge, hq, h1, na: ps[0], nxi: ps[1],
        rho_mu: (rho / mu) as f32, amax: amax as f32,
    };
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let m = dev.load_module("phantom_flow_v7", SRC, &CompileSettings::default()).unwrap();
    let k = Kern { st: st.clone(), cand: m.load_function("candidates").unwrap(), truth: m.load_function("truth").unwrap(), own: m.load_function("own_frac").unwrap() };
    let fl = |x: &[f64]| Value::List(x.iter().map(|&a| Value::Float(a)).collect());
    let grid_meta = |d: &mut Vec<(String, Value)>| {
        d.push(("origin_m".into(), fl(&g.origin)));
        d.push(("voxel_m".into(), Value::Float(vox)));
        d.push(("dims_xyz".into(), Value::List(dims.iter().map(|&x| Value::Int(x as i64)).collect())));
        d.push(("sub".into(), Value::Int(g.sub as i64)));
        d.push(("omega".into(), fl(&omega[..h1])));
        d.push(("period".into(), Value::Float(period)));
    };
    let base = |o: &Sparse| -> Vec<(String, Value)> {
        let n = o.idx.len();
        vec![
            ("idx".into(), Value::Tensor(Tensor::from_slice(&[n], &o.idx).unwrap())),
            ("blood".into(), Value::Tensor(Tensor::from_slice(&[n], &o.blood).unwrap())),
            ("wall".into(), Value::Tensor(Tensor::from_slice(&[n], &o.wall).unwrap())),
            ("cls".into(), Value::Tensor(Tensor::from_bytes(DType::U8, vec![n], &o.cls).unwrap())),
            ("sample".into(), Value::Tensor(Tensor::from_slice(&[n], &o.samp).unwrap())),
        ]
    };
    let res = if let Some(fi) = args.iter().position(|a| a == "--frames") {
        // moving geometry: per frame (cardiac time tc) the displaced centrelines, radii and wall velocities
        let fv = hasty_value::read(&mut std::io::BufReader::new(std::fs::File::open(&args[fi + 1]).unwrap())).unwrap();
        let (fp, fps) = tensor_f32(&fv, "points");
        let (fr, _) = tensor_f32(&fv, "radius");
        let (fw, _) = tensor_f32(&fv, "wvel");
        let tct = fv.get("tc").and_then(|x| x.as_tensor()).unwrap();
        let tcs: Vec<f64> = (0..tct.numel()).map(|i| tct.get_f64(i).0).collect();
        let nf = fps[0];
        assert_eq!(fps[1], ns, "frames: points [F, N, 3] with N = the vessel samples");
        let mut frames = vec![];
        for f in 0..nf {
            let o = evaluate(&k, &c, &g, &fp[f * ns * 3..(f + 1) * ns * 3], &fr[f * ns..(f + 1) * ns], &fw[f * ns * 3..(f + 1) * ns * 3], f == 0);
            // velocity at the frame's cardiac time: Re[vh0 + 2 Σ vh_n e^{iω_n tc}] + wall motion
            let n = o.idx.len();
            let mut vel = vec![0f32; n * 3];
            let tc = tcs[f];
            let e: Vec<(f64, f64)> = (0..h1).map(|hh| ((omega[hh] * tc).cos(), (omega[hh] * tc).sin())).collect();
            for t in 0..n {
                for x in 0..3 {
                    let mut acc = o.vh[(t * h1) * 6 + x * 2] as f64;
                    for (hh, &(cs, sn)) in e.iter().enumerate().skip(1) {
                        let (re, im) = (o.vh[(t * h1 + hh) * 6 + x * 2] as f64, o.vh[(t * h1 + hh) * 6 + x * 2 + 1] as f64);
                        acc += 2.0 * (re * cs - im * sn);
                    }
                    vel[t * 3 + x] = acc as f32 + o.wv[t * 3 + x];
                }
            }
            let mut d = base(&o);
            d.push(("vel".into(), Value::Tensor(Tensor::from_slice(&[n, 3], &vel).unwrap())));
            d.push(("tc".into(), Value::Float(tc)));
            frames.push(Value::Dict(d));
            println!("frame {f}: tc {:.3} s, {n} voxels ({:.1}s)", tc, t0.elapsed().as_secs_f64());
        }
        let mut d = vec![("frames".to_string(), Value::List(frames))];
        grid_meta(&mut d);
        d.push(("note".into(), Value::Str("per frame sparse voxels (idx = z·ny·nx + y·nx + x): vel = voxel-mean blood velocity at the frame's time incl. vessel motion (Σ over blood points / S³); blood-mean = vel / blood; Z-Anatomy CC BY-SA 4.0 centrelines".into())));
        Value::Dict(d)
    } else {
        let wv0 = vec![0f32; ns * 3];
        let o = evaluate(&k, &c, &g, &pts, &rad, &wv0, true);
        let n = o.idx.len();
        let mut d = base(&o);
        d.push(("vh".into(), Value::Tensor(Tensor::from_slice(&[n, h1, 3, 2], &o.vh).unwrap())));
        grid_meta(&mut d);
        d.push(("note".into(), Value::Str("sparse voxels (idx = z·ny·nx + y·nx + x); vh: harmonics of the voxel-mean velocity [N, H+1, xyz, re/im], v(t) = Re[vh0 + 2 Σ vh_n e^{iω_n t}]; blood-mean = vh / blood; Z-Anatomy CC BY-SA 4.0 centrelines".into())));
        Value::Dict(d)
    };
    hasty_value::write(&res, &hasty_value::Options::zstd(3), &mut std::io::BufWriter::new(std::fs::File::create(out).unwrap())).unwrap();
    println!("wrote {out} ({:?} grid, {:.0}s)", dims, t0.elapsed().as_secs_f64());
}
