//! Prototype benchmark of sample-side (gather / scatter) kernel designs of the segmented NUFFT
//! (research/sample_kernels.md). Standalone: the engine is only called for the baseline (`--mode base`).
//!
//!   proto_gather --mode base|fwd|adj [--m 4000000] [--ncb 10] [--nseg 10] [--order 0|2] [--traj real|cube]
//!                [--designs halo,gm] [--reps 5] [--check 1500] [--tune quick|full] [--json out.jsonl]
//!
//! Problem: 1 mm box 160^3, fine grid 200^3 (σ 1.25), W 10, ES window; one pass of NCH = ncb · nseg channels,
//! channel-interleaved grid (as segnufft.cu). Samples (`--traj real`): the yarn-ball readouts of encode 0 of
//! data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/samples.hv (a contiguous block of rotations), k scaled
//! so that max |k_a| maps to 0.98 π (the 2 mm trajectory fills the 1 mm grid; the centre-dense yarn shape is kept).
//!
//! base: today's tile kernels (`SegNufft::autotune`, all ncb coils in one batch, static), kernel times only.
//! fwd: design 2 (`halo`: home tile + (TT+W−1)^3 halo in shared memory) and design 3 (`gm`: same exact-support
//!      loop reading the grid through L1/L2), each autotuned over its NVRTC defines; best verified against an f64
//!      CPU reference on `--check` samples.
//! adj: design 4 (`smspread`: home-tile shared-memory spreading, halo added with global atomics), verified at random
//!      grid points against an f64 CPU sum over all samples.
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{sys, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg};
use hasty_cuda::{compile_cubin, CompileSettings, Device, C32};

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct Item {
    tile: i32,
    first: i32,
    count: i32,
    ufirst: i32,
}
unsafe impl DeviceRepr for Item {}
#[repr(C)]
#[derive(Clone, Copy)]
struct Dims {
    nf0: i32,
    nf1: i32,
    nf2: i32,
}
unsafe impl DeviceRepr for Dims {}

const W: usize = 10;
const SRC: &str = include_str!("../../../hasty-nufft/src/proto_gather.cu");

struct Args {
    mode: String,
    m: usize,
    ncb: usize,
    nseg: usize,
    order: u32,
    traj: String,
    designs: Vec<String>,
    reps: usize,
    check: usize,
    tune: String,
    json: Option<String>,
    fixed: Option<String>,
    /// fwd: time, check and log every candidate (not only each design's best)
    each: bool,
}

fn args() -> Args {
    let v: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| v.iter().position(|a| a == k).map(|i| v[i + 1].clone());
    Args {
        mode: get("--mode").unwrap_or("fwd".into()),
        m: get("--m").map_or(4_000_000, |s| s.parse().unwrap()),
        ncb: get("--ncb").map_or(10, |s| s.parse().unwrap()),
        nseg: get("--nseg").map_or(10, |s| s.parse().unwrap()),
        order: get("--order").map_or(0, |s| s.parse().unwrap()),
        traj: get("--traj").unwrap_or("real".into()),
        designs: get("--designs").unwrap_or("halo,gm".into()).split(',').map(|s| s.to_string()).collect(),
        reps: get("--reps").map_or(5, |s| s.parse().unwrap()),
        check: get("--check").map_or(1500, |s| s.parse().unwrap()),
        tune: get("--tune").unwrap_or("quick".into()),
        json: get("--json"),
        fixed: get("--fixed"),
        each: v.iter().any(|x| x == "--each"),
    }
}

/// k·Δ (radians) of m samples
fn trajectory(kind: &str, m: usize) -> Vec<[f32; 3]> {
    if kind == "cube" {
        let mut s = 99u64;
        let mut u = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5 };
        return (0..m).map(|_| [(u() * 6.2) as f32, (u() * 6.2) as f32, (u() * 6.2) as f32]).collect();
    }
    use hasty_value::lazy::LazyFile;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/samples.hv");
    let lf = LazyFile::open(&path).expect("samples.hv");
    let tr = lf.root.get("array").and_then(|l| l.tensor()).expect("array").clone();
    let (nro, nadc, ncol) = (tr.shape[0], tr.shape[1], tr.shape[2]);
    let need = m.div_ceil(nadc);
    // contiguous block of rotations in the middle of the scan; encode 0 = readouts i ≡ 0 mod 7
    let lo = (nro / 2 / 7) * 7;
    let hi = (lo + 7 * need).min(nro);
    let t = lf.read_box(&tr, &[lo, 0, 0], &[hi, nadc, ncol]).expect("read");
    let b = t.bytes();
    let val = |i: usize| -> f64 {
        match tr.dtype {
            hasty_value::DType::F32 => f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()) as f64,
            hasty_value::DType::F64 => f64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap()),
            d => panic!("samples.hv dtype {d:?}"),
        }
    };
    let mut k: Vec<[f64; 3]> = Vec::with_capacity(m);
    'o: for r in (0..hi - lo).step_by(7) {
        for a in 0..nadc {
            let o = (r * nadc + a) * ncol;
            k.push([val(o), val(o + 1), val(o + 2)]);
            if k.len() == m {
                break 'o;
            }
        }
    }
    let kmax = k.iter().flat_map(|p| p.iter().map(|v| v.abs())).fold(0.0, f64::max);
    let s = 0.98 * std::f64::consts::PI / kmax;
    println!("trajectory: {} samples of {} readouts (encode 0, rotations from {}), |k|max {kmax:.1} rad/m -> 0.98 π", k.len(), k.len().div_ceil(nadc), lo / 7);
    k.iter().map(|p| [(p[0] * s) as f32, (p[1] * s) as f32, (p[2] * s) as f32]).collect()
}

fn hmix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}
/// grid value of flat index t (pg_fill)
fn gval(t: u64) -> (f64, f64) {
    let a = hmix((t as u32).wrapping_mul(2).wrapping_add(1));
    let b = hmix((t as u32).wrapping_mul(2).wrapping_add(2).wrapping_add((t >> 32) as u32));
    (((a >> 8) as f32 * (1.0 / 8388608.0) - 1.0) as f64, ((b >> 8) as f32 * (1.0 / 8388608.0) - 1.0) as f64)
}

/// ES kernel and derivatives (f64 port of es5; z as the kernels compute it, in f32)
fn es5(z: f32, beta: f64) -> [f64; 5] {
    let z = z as f64;
    let a = 1.0 - z * z;
    if a <= 0.0 {
        return [0.0; 5];
    }
    let f = (beta * (a.sqrt() - 1.0)).exp();
    if z.abs() > 0.98 {
        return [f, 0.0, 0.0, 0.0, 0.0];
    }
    let (b, ia, is) = (beta, 1.0 / a, 1.0 / a.sqrt());
    let (is3, is5, is7) = (is * ia, is * ia * ia, is * ia * ia * ia);
    let (u1, u2, u3, u4) = (-b * z * is, -b * is3, -3.0 * b * z * is5, -3.0 * b * (1.0 + 4.0 * z * z) * is7);
    [f, u1 * f, (u2 + u1 * u1) * f, (u3 + 3.0 * u1 * u2 + u1 * u1 * u1) * f, (u4 + 4.0 * u1 * u3 + 3.0 * u2 * u2 + 6.0 * u1 * u1 * u2 + u1 * u1 * u1 * u1) * f]
}

/// per-sample f64 weights over the W^3 support (order 0 or 2, the kernels' formulas) and the support corner
/// order 3: plus the degree-3 (real, + hc wy_b wz_{3-a-b}) and degree-4 (imaginary, − hc wy_b wz_{4-a-b}) terms of
/// segnufft_knot.cu stage_weights_hk, hc = the sample's 25 coefficients (degree 3 then 4, a outer, b inner)
fn weights(sc: [f32; 3], d: &[f64; 6], order: u32, beta: f64, nf: usize, hc: Option<&[f64; 25]>) -> ([usize; 3], Vec<(f64, f64)>) {
    let mut k = [[[0f64; W]; 5]; 3];
    let mut c0 = [0usize; 3];
    for a in 0..3 {
        let i0 = (sc[a] - (W as f32) * 0.5).ceil() as i64;
        c0[a] = i0.rem_euclid(nf as i64) as usize;
        for t in 0..W {
            let z = ((i0 + t as i64) as f32 - sc[a]) * (2.0 / W as f32);
            let o = es5(z, beta);
            for q in 0..5 {
                k[a][q][t] = o[q];
            }
        }
    }
    let mut e = [0f64; 15];
    let (mx, my) = ([2, 0, 0, 1, 1, 0], [0, 2, 0, 1, 0, 1]);
    for p in 0..6 {
        for q in 0..6 {
            let (ea, eb) = (mx[p] + mx[q], my[p] + my[q]);
            e[ea * (11 - ea) / 2 + eb] += -0.5 * d[p] * d[q];
        }
    }
    let mut w = vec![(0.0, 0.0); W * W * W];
    for tz in 0..W {
        for ty in 0..W {
            let (wy, wz) = ([0, 1, 2, 3, 4].map(|q| k[1][q][ty]), [0, 1, 2, 3, 4].map(|q| k[2][q][tz]));
            let (mut r, mut qq) = ([0f64; 5], [0f64; 5]);
            r[0] = wy[0] * wz[0];
            // order 1 (§15): the ΔC terms Q only (stage_weights CORR 1); order 2 adds the e terms
            if order >= 1 {
                qq[0] = d[1] * wy[2] * wz[0] + d[2] * wy[0] * wz[2] + d[5] * wy[1] * wz[1];
                qq[1] = d[3] * wy[1] * wz[0] + d[4] * wy[0] * wz[1];
                qq[2] = d[0] * wy[0] * wz[0];
            }
            if order >= 2 {
                let mut idx = 0;
                for a in 0..=4 {
                    for b in 0..=4 - a {
                        r[a] += e[idx] * wy[b] * wz[4 - a - b];
                        idx += 1;
                    }
                }
            }
            if let (true, Some(hc)) = (order >= 3, hc) {
                let mut j = 0;
                for a in 0..=3 {
                    for b in 0..=3 - a {
                        r[a] += hc[j] * wy[b] * wz[3 - a - b];
                        j += 1;
                    }
                }
                for a in 0..=4 {
                    for b in 0..=4 - a {
                        qq[a] -= hc[j] * wy[b] * wz[4 - a - b];
                        j += 1;
                    }
                }
            }
            for tx in 0..W {
                let wr: f64 = if order >= 1 { (0..5).map(|a| k[0][a][tx] * r[a]).sum() } else { k[0][0][tx] * r[0] };
                let wi: f64 = if order >= 1 { (0..5).map(|a| k[0][a][tx] * qq[a]).sum() } else { 0.0 };
                w[(tz * W + ty) * W + tx] = (wr, wi);
            }
        }
    }
    (c0, w)
}

struct Problem {
    nf: usize,
    nch: usize,
    sc: Vec<[f32; 3]>, // grid coordinates, original order
    d: Vec<[f32; 6]>,  // ΔC (kernel-argument units), original order
    a: Vec<C32>,       // [j][l], original order
    /// order 3: per-sample u (float4, original order) and H' (4 x 25, q-major), hc_ja = Σ_q u_jq H'_qa (as the engine's
    /// uw / htab)
    u: Vec<[f32; 4]>,
    htab: Vec<f32>,
}

/// per-sample device arrays in sorted order; depend only on (home tile dims, rx), so all (smax, sb) item lists of one
/// tile shape share them (one copy per tile shape: at 4 M x 12 segments ≈ 0.56 GB each)
struct SBufs {
    td: [usize; 3],
    rx: usize,
    keys: Vec<(u64, u64)>,
    perm_h: Vec<u32>,
    pos: CudaSlice<f32>,
    dco: CudaSlice<f32>,
    acf: CudaSlice<C32>,
    perm: CudaSlice<u32>,
    /// order 3 (vs): u in the sorted order (float4 per sample; 4 floats when unused)
    uw: CudaSlice<f32>,
}

struct Sorted {
    td: [usize; 3],
    smax: usize,
    sb: usize,
    rx: usize,
    items: CudaSlice<Item>,
    items_h: Vec<Item>,
    nitems: usize,
    units: CudaSlice<u32>,
    nunits: usize,
    b: Arc<SBufs>,
    /// built by sort_vs (items: ufirst = zlo | zhi << 16)
    vs: bool,
}

/// samples sorted by (home tile, k0, j0, i0 / rx, i0); work units = runs of equal (tile, k0, j0, i0 / rx) cut into
/// ≤ sb samples; items = a tile's units cut into ≤ smax samples (units never split); a sentinel item ends the list
fn sort(st: &Arc<CudaStream>, p: &Problem, td: [usize; 3], smax: usize, sb: usize, rx: usize, cache: &mut Vec<Arc<SBufs>>) -> Sorted {
    let nf = p.nf;
    let m = p.sc.len();
    let bufs = match cache.iter().find(|b| b.td == td && b.rx == rx) {
        Some(b) => b.clone(),
        None => {
            let nt = [0, 1, 2].map(|a| nf.div_ceil(td[a]));
            let corner = |s: &[f32; 3]| -> [usize; 3] { [0, 1, 2].map(|a| ((s[a] - W as f32 * 0.5).ceil() as i64).rem_euclid(nf as i64) as usize) };
            let key = |j: usize| -> (u64, u64) {
                let c = corner(&p.sc[j]);
                let t = ((c[2] / td[2] * nt[1] + c[1] / td[1]) * nt[0] + c[0] / td[0]) as u64;
                (t, ((c[2] * nf + c[1]) * nf + c[0] / rx) as u64 * nf as u64 + c[0] as u64)
            };
            let keys: Vec<(u64, u64)> = (0..m).map(key).collect();
            let mut perm: Vec<u32> = (0..m as u32).collect();
            perm.sort_by_key(|&j| keys[j as usize]);
            let pos: Vec<f32> = perm.iter().flat_map(|&j| { let s = p.sc[j as usize]; [s[0], s[1], s[2], 0.0] }).collect();
            let mut dco = vec![0f32; 6 * m];
            for (i, &j) in perm.iter().enumerate() {
                for q in 0..6 {
                    dco[q * m + i] = p.d[j as usize][q];
                }
            }
            let ns = p.a.len() / m;
            let acf: Vec<C32> = perm.iter().flat_map(|&j| (0..ns).map(move |l| (j as usize, l))).map(|(j, l)| p.a[j * ns + l]).collect();
            let b = Arc::new(SBufs { td, rx, pos: st.clone_htod(&pos).unwrap(), dco: st.clone_htod(&dco).unwrap(), acf: st.clone_htod(&acf).unwrap(), perm: st.clone_htod(&perm).unwrap(), uw: st.alloc_zeros::<f32>(4).unwrap(), keys, perm_h: perm });
            cache.push(b.clone());
            b
        }
    };
    let (keys, perm) = (&bufs.keys, &bufs.perm_h);
    let grp = |j: u32| { let k = keys[j as usize]; (k.0, k.1 / nf as u64) };
    let (mut items, mut units) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < m {
        // one tile
        let t = keys[perm[i] as usize].0;
        let mut e = i;
        while e < m && keys[perm[e] as usize].0 == t {
            e += 1;
        }
        // units of the tile
        let mut tu: Vec<(usize, usize)> = Vec::new();
        let mut a = i;
        while a < e {
            let g = grp(perm[a]);
            let mut b = a;
            while b < e && b - a < sb && grp(perm[b]) == g {
                b += 1;
            }
            tu.push((a, b - a));
            a = b;
        }
        // items: ≤ smax samples, whole units
        let mut k = 0;
        while k < tu.len() {
            let first = tu[k].0;
            let ufirst = units.len();
            let mut cnt = 0;
            while k < tu.len() && cnt + tu[k].1 <= smax {
                units.push((((tu[k].0 - first) as u32) << 8) | tu[k].1 as u32);
                cnt += tu[k].1;
                k += 1;
            }
            items.push(Item { tile: t as i32, first: first as i32, count: cnt as i32, ufirst: ufirst as i32 });
        }
        i = e;
    }
    let nitems = items.len();
    items.push(Item { tile: 0, first: m as i32, count: 0, ufirst: units.len() as i32 });
    let nunits = units.len();
    Sorted { td, smax, sb, rx, items: st.clone_htod(&items).unwrap(), items_h: items, nitems, units: st.clone_htod(&units).unwrap(), nunits, b: bufs, vs: false }
}

/// vs (§12): samples sorted by (home tile, oz, oy, ox); items = runs of <= smax samples of one tile; inside an item
/// re-sorted by (class key oy * tx + ox, oz). Item.ufirst = zlo | zhi << 16 (item's oz range, tile-relative).
fn sort_vs(st: &Arc<CudaStream>, p: &Problem, td: [usize; 3], smax: usize, cache: &mut Vec<Arc<SBufs>>) -> Sorted {
    let nf = p.nf;
    let m = p.sc.len();
    let marker = 1000 + smax;
    let nt = [0, 1, 2].map(|a| nf.div_ceil(td[a]));
    let bufs = match cache.iter().find(|b| b.td == td && b.rx == marker) {
        Some(b) => b.clone(),
        None => {
            let corner = |s: &[f32; 3]| -> [usize; 3] { [0, 1, 2].map(|a| ((s[a] - W as f32 * 0.5).ceil() as i64).rem_euclid(nf as i64) as usize) };
            // key: (tile, oz << 20 | oy << 10 | ox)
            let keys: Vec<(u64, u64)> = (0..m)
                .map(|j| {
                    let c = corner(&p.sc[j]);
                    let t = ((c[2] / td[2] * nt[1] + c[1] / td[1]) * nt[0] + c[0] / td[0]) as u64;
                    (t, (((c[2] % td[2]) << 20) | ((c[1] % td[1]) << 10) | (c[0] % td[0])) as u64)
                })
                .collect();
            let mut perm: Vec<u32> = (0..m as u32).collect();
            perm.sort_by_key(|&j| (keys[j as usize], j));
            // items, then the in-item class order
            let mut i = 0;
            while i < m {
                let t = keys[perm[i] as usize].0;
                let mut e = i;
                while e < m && e - i < smax && keys[perm[e] as usize].0 == t {
                    e += 1;
                }
                let cls = |j: u32| { let k = keys[j as usize].1; let (oz, oy, ox) = (k >> 20, (k >> 10) & 1023, k & 1023); (oy * td[0] as u64 + ox, oz, j) };
                perm[i..e].sort_by_key(|&j| cls(j));
                i = e;
            }
            let pos: Vec<f32> = perm.iter().flat_map(|&j| { let s = p.sc[j as usize]; [s[0], s[1], s[2], 0.0] }).collect();
            let mut dco = vec![0f32; 6 * m];
            for (i, &j) in perm.iter().enumerate() {
                for q in 0..6 {
                    dco[q * m + i] = p.d[j as usize][q];
                }
            }
            let ns = p.a.len() / m;
            let acf: Vec<C32> = perm.iter().flat_map(|&j| (0..ns).map(move |l| (j as usize, l))).map(|(j, l)| p.a[j * ns + l]).collect();
            let uw: Vec<f32> = if p.u.is_empty() { vec![0f32; 4] } else { perm.iter().flat_map(|&j| p.u[j as usize]).collect() };
            let b = Arc::new(SBufs { td, rx: marker, pos: st.clone_htod(&pos).unwrap(), dco: st.clone_htod(&dco).unwrap(), acf: st.clone_htod(&acf).unwrap(), perm: st.clone_htod(&perm).unwrap(), uw: st.clone_htod(&uw).unwrap(), keys, perm_h: perm });
            cache.push(b.clone());
            b
        }
    };
    let (keys, perm) = (&bufs.keys, &bufs.perm_h);
    let mut items = Vec::new();
    let mut i = 0;
    while i < m {
        let t = keys[perm[i] as usize].0;
        let mut e = i;
        let (mut zlo, mut zhi) = (u64::MAX, 0u64);
        while e < m && e - i < smax && keys[perm[e] as usize].0 == t {
            let oz = keys[perm[e] as usize].1 >> 20;
            zlo = zlo.min(oz);
            zhi = zhi.max(oz);
            e += 1;
        }
        items.push(Item { tile: t as i32, first: i as i32, count: (e - i) as i32, ufirst: (zlo | (zhi << 16)) as i32 });
        i = e;
    }
    let nitems = items.len();
    items.push(Item { tile: 0, first: m as i32, count: 0, ufirst: 0 });
    Sorted { td, smax, sb: 0, rx: 0, items: st.clone_htod(&items).unwrap(), items_h: items, nitems, units: st.clone_htod(&[0u32]).unwrap(), nunits: 0, b: bufs, vs: true }
}

/// vs work accounting (host, from the items): [item fill (samples / (items · smax)), sample-weighted samples active per
/// (item, plane), sample-weighted bodies per (plane, window step), mean planes per item, channel-lane efficiency]
fn vs_work(p: &Problem, sr: &Sorted, c: &Cfg) -> [f64; 5] {
    let m = p.sc.len() as f64;
    let (mut act, mut stepb, mut planes) = (0f64, 0f64, 0f64);
    for it in &sr.items_h[..sr.nitems] {
        let n = it.count as f64;
        let npl = ((it.ufirst >> 16) - (it.ufirst & 0xffff) + W as i32) as f64;
        act += n * (n * W as f64 / npl);
        stepb += n * (n * W as f64 / npl / c.ty as f64);
        planes += npl;
    }
    let g = p.nch.div_ceil(16);
    [m / (sr.nitems * c.smax) as f64, act / m, stepb / m, planes / sr.nitems as f64, p.nch as f64 / (16 * g) as f64]
}

/// ps work accounting from the sample layout (host side, exact for the kernel's loop bounds): per warp (32 / NCG slots
/// of FSB samples, consecutive samples of an item) the kernel runs HXP x-points x (union of y rows) x (union of z
/// planes) for every slot, live or not. Returns (useful MACs / issued x-loop MAC slots, mean rows per active warp, mean
/// planes per active warp, live samples per active-warp slot, active (warp, plane) pairs / all (warp, plane) pairs of
/// the launched blocks, live samples / block sample capacity).
fn ps_work(p: &Problem, sr: &Sorted, c: &Cfg) -> [f64; 6] {
    let nf = p.nf as i64;
    let (ncg, fsb, nt) = (c.tps, c.sb, c.nt);
    let ws = 32 / ncg * fsb;
    let (hxp, hzp) = ((c.tt + W - 1) as f64, (c.tz + W - 1) as f64);
    let (ntx, nty) = (p.nf.div_ceil(c.tt) as i64, p.nf.div_ceil(c.ty) as i64);
    let w = W as i64;
    let (mut useful, mut issued, mut rows, mut planes, mut nact, mut live, mut wp_act, mut wp_all, mut cap) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for it in &sr.items_h[..sr.nitems] {
        let t = it.tile as i64;
        let og = [(t % ntx) * c.tt as i64, ((t / ntx) % nty) * c.ty as i64, (t / (ntx * nty)) * c.tz as i64];
        let n = it.count as usize;
        cap += (nt / ncg * fsb) as f64;
        wp_all += (nt / 32) as f64 * hzp;
        for w0 in (0..n).step_by(ws) {
            let (mut ylo, mut yhi, mut zlo, mut zhi) = (i64::MAX, i64::MIN, i64::MAX, i64::MIN);
            let k = (n - w0).min(ws);
            for s in w0..w0 + k {
                let j = sr.b.perm_h[it.first as usize + s] as usize;
                let sc = p.sc[j];
                let off = |a: usize| ((sc[a] - W as f32 * 0.5).ceil() as i64).rem_euclid(nf) - og[a];
                let (oy, oz) = (off(1), off(2));
                ylo = ylo.min(oy); yhi = yhi.max(oy); zlo = zlo.min(oz); zhi = zhi.max(oz);
            }
            let (r, pl) = ((yhi + w - ylo) as f64, (zhi + w - zlo) as f64);
            useful += k as f64 * (W * W * W) as f64;
            issued += ws as f64 * hxp * r * pl;
            rows += r; planes += pl; nact += 1.0; live += k as f64 / ws as f64; wp_act += pl;
        }
    }
    [useful / issued, rows / nact, planes / nact, live / nact, wp_act / wp_all, p.sc.len() as f64 / cap]
}

#[derive(Clone, Copy, Debug)]
struct Cfg {
    halo: usize, // 1 halo, 0 gm, 2 adjoint smspread
    tt: usize,
    ch: usize,
    tps: usize,
    nt: usize,
    smax: usize,
    padc: usize,
    sb: usize,
    rx: usize,
    /// ps: home tile y / z extents (tt = x extent); 0 for the other designs
    ty: usize,
    tz: usize,
    /// ps: 1 = segment coefficients read from global at the output (PS_ACFG), not staged in the sample record
    acfg: usize,
    /// ps: 1 = next plane copied one float4 at a time inside the row loop (PS_PFS) instead of a whole-plane register prefetch
    pfs: usize,
    /// ps: minimum resident blocks per SM for ptxas (PS_MINB, register cap); 0 = none
    minb: usize,
    /// vs (halo 4, §12): channel groups (16 channels each) per block, window-position splits per block, zero-padded x
    /// weights (VXPAD); vs uses tt / ty / tz = TX / TY / TZ, smax = SMAX, nt = 32 nwc nws
    nwc: usize,
    nws: usize,
    vxpad: usize,
    /// vs tunables of research/sample_kernels.md §13: VT (per-(sample, plane) weight table, order ≥ 2), VSHR (half 1
    /// hands its evicted row to half 0 by SHFL), VPF (next window row one step ahead), VS2 (two samples per iteration,
    /// order 0), VPT (parallel range-table build); `minb` doubles as VMINB for vs; carve = preferred shared-memory
    /// carve-out in % (driver attribute; −1 = default)
    vt: usize,
    shr: usize,
    pf: usize,
    s2: usize,
    pt: usize,
    carve: i32,
    /// vt: software-pipelined table build (VTP, §13)
    vtp: usize,
}

fn sw(order: u32, nseg: usize) -> usize {
    sw_a(order, nseg, 0)
}
fn sw_a(order: u32, nseg: usize, acfg: usize) -> usize {
    let nk = if order >= 2 { 5 } else { 1 };
    (3 * nk * W + 3 + if order >= 2 { 21 } else { 0 } + if acfg == 1 { 0 } else { 2 * nseg }) | 1
}
/// vs record stride (floats), as VSW in proto_gather.cu
fn vs_sw(c: &Cfg, order: u32) -> usize {
    let kxn = if c.vxpad == 1 { c.tt + W - 1 } else { W };
    let kxs = (kxn + 2).div_ceil(4) * 4;
    if c.vt == 1 {
        // §13.1 record: kx 60, wy/wz 80 + 20, e/d 24 (+ hc 28 at order 3), ox/oy/oz 4
        return if order >= 3 { 216 } else { 188 };
    }
    let sw0 = if order >= 2 { 5 * kxs + 56 + 52 + 15 + 6 } else { 2 * kxs + 8 + 10 };
    (sw0 + 2).div_ceil(4) * 4
}
fn smem(c: &Cfg, order: u32, nseg: usize) -> usize {
    if c.halo == 4 {
        // vs: records + range table (NPLM x NCLS words) + accumulators (nwc x 2 planes of SMAX * 16 + 16 floats)
        let tab = ((c.tz + W - 1) * c.tt * c.ty * 4 + 15) & !15;
        // vt: + per warp a double-buffered 10 x 10 complex weight table (1600 B)
        return c.smax * vs_sw(c, order) * 4 + tab + c.nwc * 2 * (c.smax * 16 + 16) * 4 + if c.vt == 1 { c.nwc * c.nws * 1600 } else { 0 };
    }
    let hw = c.tt + W - 1;
    if c.halo == 3 {
        // ps: sample records + two planes of (tx + W - 1)(ty + W - 1) points x ch channels
        return ((c.smax * sw_a(order, nseg, c.acfg) * 4 + 15) & !15) + 2 * hw * (c.ty + W - 1) * c.ch * 8;
    }
    ((c.smax * sw(order, nseg) * 4 + 15) & !15) + if c.halo != 0 { hw * hw * hw * (c.ch + c.padc) * 8 } else { 0 }
}
/// home tile extents of a configuration
fn tdims(c: &Cfg) -> [usize; 3] {
    if c.halo == 3 || c.halo == 4 { [c.tt, c.ty, c.tz] } else { [c.tt; 3] }
}

fn source(c: &Cfg, a: &Args, beta: f64) -> (String, String) {
    let src = format!(
        "#define W {W}\n#define BETA {beta:.17e}\n#define CORR {}\n#define NCB {}\n#define NSEG {}\n#define TT {}\n#define CH {}\n#define TPS {}\n#define NT {}\n#define SMAX {}\n#define HALO {}\n#define PADC {}\n#define SB {}\n#define RX {}\n{SRC}",
        a.order,
        a.ncb,
        a.nseg,
        c.tt,
        if c.halo == 4 { 1 } else { c.ch },
        if c.halo == 4 { 1 } else { c.tps },
        c.nt,
        c.smax,
        (c.halo == 1 || c.halo == 2) as usize,
        c.padc,
        if c.halo >= 3 { 1 } else { c.sb },
        if c.halo >= 3 { 1 } else { c.rx }
    );
    let src = if c.halo == 4 { format!("#define VS 1\n#define TX {}\n#define TY {}\n#define TZ {}\n#define NWC {}\n#define NWS {}\n#define VXPAD {}\n#define VT {}\n#define VSHR {}\n#define VPF {}\n#define VS2 {}\n#define VPT {}\n#define VMINB {}\n#define VTP {}\n{src}", c.tt, c.ty, c.tz, c.nwc, c.nws, c.vxpad, c.vt, c.shr, c.pf, c.s2, c.pt, c.minb, c.vtp) } else { src };
    let src = if c.halo == 3 { format!("#define PS 1\n#define TX {}\n#define TY {}\n#define TZ {}\n#define NCG {}\n#define FC {}\n#define FSB {}\n#define PS_ACFG {}\n#define PS_PFS {}\n#define PS_MINB {}\n{src}", c.tt, c.ty, c.tz, c.tps, c.ch / c.tps, c.sb, c.acfg, c.pfs, c.minb) } else { src };
    // PG_DEFS=A,B: extra defines (timing ablations of pg_fwd_ps, e.g. PS_NOLOAD,PS_NOSYNC; research/sample_kernels.md §11)
    let pd = pg_defs();
    let src = format!("{}{src}", pd.iter().map(|d| format!("#define {d} 1\n")).collect::<String>());
    let name = format!("proto_gather_o{}_c{}_l{}_h{}_t{}_ch{}_p{}_n{}_s{}_pad{}_sb{}_rx{}_y{}_z{}{}{}", a.order, a.ncb, a.nseg, c.halo, c.tt, c.ch, c.tps, c.nt, c.smax, c.padc, c.sb, c.rx, c.ty, c.tz, format!("{}{}{}", if c.acfg == 1 { "_acfg" } else { "" }, if c.pfs == 1 { "_pfs" } else { "" }, if c.minb > 0 && c.halo == 3 { format!("_mb{}", c.minb) } else { String::new() }) + &if c.halo == 4 { format!("_vs{}_{}_{}{}", c.nwc, c.nws, c.vxpad, if c.vt + c.shr + c.pf + c.s2 + c.pt + c.minb + c.vtp > 0 { format!("_t{}{}{}{}{}m{}p{}", c.vt, c.shr, c.pf, c.s2, c.pt, c.minb, c.vtp) } else { String::new() }) } else { String::new() }, pd.iter().map(|d| format!("_{}", d.to_lowercase())).collect::<String>());
    (name, src)
}

/// grid y extent: vs launches one block row per nwc channel groups
fn gridy(c: &Cfg, nch: usize) -> u32 {
    if c.halo == 4 { (nch.div_ceil(16) / c.nwc) as u32 } else { 1 }
}
/// every (sample, coil) output written by exactly one thread with a plain store (else y is zeroed and atomics add)
fn owns(c: &Cfg, a: &Args) -> bool {
    if c.halo == 4 {
        return gridy(c, a.ncb * a.nseg) == 1 || (c.nwc * 16) % a.nseg == 0;
    }
    (c.ch / c.tps) % a.nseg == 0 && c.ch % a.nseg == 0
}

fn pg_defs() -> Vec<String> {
    std::env::var("PG_DEFS").unwrap_or_default().split(',').filter(|d| !d.is_empty()).map(|d| d.trim().to_string()).collect()
}

/// second autotune stage of vs (§13): the kept tunables on the best shape of the first stage. Order 0: VSHR, VPT,
/// VS2, VPF and combinations (each within ±5 % alone, case-dependent); order >= 2 (vt): VTP, VSHR, VPT.
fn vs_refine(b: &Cfg, order: u32) -> Vec<Cfg> {
    let combos: &[(usize, usize, usize, usize, usize)] = if order >= 2 && b.vt == 1 {
        // (shr, pt, s2, pf, vtp)
        &[(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 0, 0, 1), (1, 0, 0, 0, 1), (1, 1, 0, 0, 1), (1, 1, 0, 0, 0)]
    } else if order >= 2 {
        &[(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (1, 1, 0, 0, 0)]
    } else {
        &[(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 1, 1, 0, 0), (1, 1, 0, 0, 0), (1, 1, 1, 0, 0), (1, 0, 0, 1, 0), (1, 1, 1, 1, 0)]
    };
    combos.iter().map(|&(shr, pt, s2, pf, vtp)| Cfg { shr, pt, s2, pf, vtp, ..*b }).collect()
}

/// one `--fixed` vs configuration: tx,ty,tz,smax,nwc,nws[,vxpad][,key=value ...]
fn parse_vs(fx: &str) -> Cfg {
    let toks: Vec<&str> = fx.split(',').map(|t| t.trim()).collect();
    let v: Vec<usize> = toks.iter().filter(|t| !t.contains('=')).map(|t| t.parse().unwrap()).collect();
    let (nwc, nws) = (v[4], v[5]);
    let mut c = Cfg { halo: 4, tt: v[0], ty: v[1], tz: v[2], smax: v[3], nwc, nws, vxpad: v.get(6).copied().unwrap_or(0), ch: 16 * nwc, tps: 1, nt: 32 * nwc * nws, padc: 0, sb: 0, rx: 0, acfg: 0, pfs: 0, minb: 0, vt: 0, shr: 0, pf: 0, s2: 0, pt: 0, carve: -1, vtp: 0 };
    for t in toks.iter().filter(|t| t.contains('=')) {
        let (k, val) = t.split_once('=').unwrap();
        match k {
            "vt" => c.vt = val.parse().unwrap(),
            "shr" => c.shr = val.parse().unwrap(),
            "pf" => c.pf = val.parse().unwrap(),
            "s2" => c.s2 = val.parse().unwrap(),
            "pt" => c.pt = val.parse().unwrap(),
            "minb" => c.minb = val.parse().unwrap(),
            "carve" => c.carve = val.parse().unwrap(),
            "vtp" => c.vtp = val.parse().unwrap(),
            _ => panic!("--fixed vs: unknown key {k}"),
        }
    }
    c
}

/// design 7 `ra` (§15): output tile tx × ty × tz, nxg x-groups per warp (pt = tx / nxg points per lane), cb channels per
/// lane (32 / nxg channel blocks: cpw = 32 / nxg · cb channels per block row), chk samples staged per chunk, tile lists
/// cut into items of ≤ lmax samples (longer lists: split items with atomic output)
#[derive(Clone, Copy, Debug)]
struct RaCfg {
    tx: usize,
    ty: usize,
    tz: usize,
    nxg: usize,
    cb: usize,
    chk: usize,
    lmax: usize,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct RItem {
    tile: i32,
    first: i32,
    count: i32,
    split: i32,
}
unsafe impl DeviceRepr for RItem {}
fn ra_cpw(c: &RaCfg) -> usize {
    32 / c.nxg * c.cb
}
/// dynamic shared memory of pg_adj_ra (as RSW / RZ_OFF / RW_OFF in proto_gather.cu), and the output parking area
fn ra_smem(c: &RaCfg, a: &Args) -> usize {
    let r4 = |n: usize| n.div_ceil(4) * 4;
    let nka = match a.order { 0 => 1, 1 => 3, _ => 5 };
    let rr_d = 4 + nka * (r4(c.tx) + r4(c.ty) + r4(c.tz));
    let rsw = r4(rr_d + if a.order >= 1 { 6 } else { 0 } + if a.order >= 2 { 15 } else { 0 } + if a.order >= 3 { 25 } else { 0 });
    let cbp = c.cb.div_ceil(2) * 2;
    let ncbrm = (ra_cpw(c).div_ceil(a.nseg) + 1).min(a.ncb);
    let raw = c.chk * 16 + if a.order >= 3 { c.chk * 16 } else { 0 } + c.chk * a.nseg * 8 + c.chk * ncbrm * 8 + if a.order >= 1 { c.chk * 24 } else { 0 } + 2 * c.chk * 4 + if a.order >= 1 { c.chk * c.tz * 32 * 4 } else { 0 };
    let stage = c.chk * rsw * 4 + c.chk * (32 / c.nxg) * cbp * 8 + if a.order >= 1 { c.chk * c.ty * c.tz * c.tx * 8 } else { 0 } + raw;
    let park = c.tz * c.tx * (ra_cpw(c) | 1) * 8;
    stage.max(park)
}
fn ra_candidates(a: &Args, nf: usize, smax_opt: usize) -> Vec<RaCfg> {
    let nch = a.ncb * a.nseg;
    let full = a.tune == "full";
    let tiles: &[[usize; 3]] = if full { &[[4, 4, 8], [4, 4, 4], [2, 4, 8], [2, 8, 8], [4, 8, 4], [4, 8, 8], [2, 4, 4], [8, 4, 4]] } else { &[[4, 4, 8], [4, 4, 4], [2, 4, 8], [2, 8, 8], [4, 8, 4]] };
    let mut v = Vec::new();
    for &[tx, ty, tz] in tiles {
        if nf % tx != 0 || nf % ty != 0 || nf % tz != 0 {
            continue;
        }
        for nxg in [1usize, 2, 4] {
            if tx % nxg != 0 {
                continue;
            }
            let pt = tx / nxg;
            let ncl = 32 / nxg;
            for cb in 1..=12usize {
                let rows = nch.div_ceil(ncl * cb);
                let eff = nch as f64 / (rows * ncl * cb) as f64;
                // accumulators ≤ 128 floats (o0) / 112 (o ≥ 1: weight registers); lane efficiency ≥ 0.75 and no
                // smaller cb with the same block rows
                let accf = ty * pt * cb * 2;
                if accf > if a.order >= 1 { 112 } else { 128 } || eff < 0.75 || (cb > 1 && nch.div_ceil(ncl * (cb - 1)) == rows) {
                    continue;
                }
                for chk in [16usize, 32] {
                    for lmax in if full { vec![1024usize, 4096] } else { vec![2048] } {
                        let c = RaCfg { tx, ty, tz, nxg, cb, chk, lmax };
                        if ra_smem(&c, a) <= smax_opt {
                            v.push(c);
                        }
                    }
                }
            }
        }
    }
    v
}
/// per output tile the samples whose W^3 support meets it (original sample indices, increasing), items of ≤ lmax
fn ra_lists(p: &Problem, td: [usize; 3], lmax: usize) -> (Vec<u32>, Vec<RItem>) {
    let nf = p.nf;
    let nt = [0, 1, 2].map(|q| nf.div_ceil(td[q]));
    let ntiles = nt[0] * nt[1] * nt[2];
    let axis_tiles = |s: &[f32; 3], q: usize, out: &mut [usize; 8]| -> usize {
        let i0 = (s[q] - W as f32 * 0.5).ceil() as i64;
        let mut n = 0;
        for t in 0..W as i64 {
            let tile = ((i0 + t).rem_euclid(nf as i64) as usize) / td[q];
            if !out[..n].contains(&tile) {
                out[n] = tile;
                n += 1;
            }
        }
        n
    };
    let mut cnt = vec![0u32; ntiles + 1];
    let each = |j: usize, f: &mut dyn FnMut(usize)| {
        let (mut ax, mut ay, mut az) = ([0usize; 8], [0usize; 8], [0usize; 8]);
        let (nx, ny, nz) = (axis_tiles(&p.sc[j], 0, &mut ax), axis_tiles(&p.sc[j], 1, &mut ay), axis_tiles(&p.sc[j], 2, &mut az));
        for &z in &az[..nz] {
            for &y in &ay[..ny] {
                for &x in &ax[..nx] {
                    f((z * nt[1] + y) * nt[0] + x);
                }
            }
        }
    };
    for j in 0..p.sc.len() {
        each(j, &mut |t| cnt[t + 1] += 1);
    }
    for t in 0..ntiles {
        cnt[t + 1] += cnt[t];
    }
    let mut tl = vec![0u32; cnt[ntiles] as usize];
    let mut fill = cnt.clone();
    for j in 0..p.sc.len() {
        each(j, &mut |t| { tl[fill[t] as usize] = j as u32; fill[t] += 1; });
    }
    let mut items = Vec::new();
    for t in 0..ntiles {
        let (b, e) = (cnt[t] as usize, cnt[t + 1] as usize);
        let split = (e - b > lmax) as i32;
        let mut f = b;
        while f < e {
            let n = lmax.min(e - f);
            items.push(RItem { tile: t as i32, first: f as i32, count: n as i32, split });
            f += n;
        }
    }
    (tl, items)
}

fn divisors(n: usize) -> Vec<usize> {
    (1..=n).filter(|d| n % d == 0).collect()
}

fn candidates(a: &Args, design: &str, smax_opt: usize) -> Vec<Cfg> {
    let nch = a.ncb * a.nseg;
    let full = a.tune == "full";
    let mut v = Vec::new();
    let nts: &[usize] = if full { &[128, 256, 512] } else { &[128, 256] };
    match design {
        "halo" | "smspread" => {
            let halo = if design == "halo" { 1 } else { 2 };
            for tt in [2usize, 3, 4, 6, 8] {
                for ch in divisors(nch) {
                    for tps in divisors(ch) {
                        let fc = ch / tps;
                        if fc > 16 || tps > 32 {
                            continue;
                        }
                        for &nt in nts {
                            for smax in [64usize, 128, 256] {
                                if smax < nt / tps && smax != 256 {
                                    continue;
                                }
                                for padc in if full { vec![0usize, 2] } else { vec![0] } {
                                    for (sb, rx) in if halo == 1 { vec![(1usize, 1usize), (2, 2), (4, 2)] } else { vec![(1, 1)] } {
                                    let c = Cfg { halo, tt, ch, tps, nt, smax, padc, sb, rx, ty: 0, tz: 0, acfg: 0, pfs: 0, minb: 0, nwc: 0, nws: 0, vxpad: 0, vt: 0, shr: 0, pf: 0, s2: 0, pt: 0, carve: -1, vtp: 0 };
                                    if smem(&c, a.order, a.nseg) <= smax_opt && (ch % 2 == 0 || padc == 0) && tt % rx == 0 {
                                        v.push(c);
                                    }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // halo designs: keep the largest channel chunk per (tt, tps, nt, smax) and the one below (staging
            // amortisation grows with ch; smaller chunks only cost more passes over the samples)
            if !full {
                let mut keep = Vec::new();
                for c in &v {
                    let bigger = v.iter().filter(|o| o.tt == c.tt && o.nt == c.nt && o.smax == c.smax && o.ch > c.ch && o.ch / o.tps == c.ch / c.tps).count();
                    if bigger <= 1 {
                        keep.push(*c);
                    }
                }
                v = keep;
            }
        }
        "ps" => {
            // plane-streamed home tile: (tx, ty, tz), ncg channel groups x fc channels, fsb samples per slot
            let full = a.tune == "full";
            // flat tiles (tz 1-2): every warp of a block is active in (almost) every streamed plane, but the items get
            // emptier and the planes bigger; slower in every probe of §11, so full tuning only
            let tiles: &[[usize; 3]] = if full { &[[4, 4, 4], [4, 4, 8], [4, 8, 8], [2, 4, 4], [2, 8, 8], [4, 8, 4], [2, 4, 8], [4, 4, 16], [8, 8, 8], [4, 8, 2], [4, 16, 2], [4, 16, 1], [2, 16, 2], [4, 8, 1], [4, 32, 1]] } else { &[[4, 4, 4], [4, 4, 8], [4, 8, 8], [2, 8, 8]] };
            // PS_PFS (plane copy inside the row loop): fewer registers but ≈ 25 more instructions per row; slower in the
            // §11 probes, full tuning only. PS_MINB: --fixed only (spills).
            let pfss: &[usize] = if full { &[0, 1] } else { &[0] };
            // PS_ACFG: segment coefficients from global at the output; with many segments it frees the sample
            // record's 2 NSEG words (quick: on for nseg >= 4, full: both)
            let acfgs: &[usize] = if full && a.nseg > 1 { &[0, 1] } else if a.nseg >= 4 { &[1] } else { &[0] };
            for &[tx, ty, tz] in tiles {
                for ncg in [1usize, 2, 4, 8] {
                    for fc in [2usize, 4, 6, 8, 10, 12, 16] {
                        let ch = ncg * fc;
                        if nch % ch != 0 || (a.order >= 2 && fc > 10) {
                            continue;
                        }
                        for &nt in nts {
                            for fsb in [1usize, 2, 3] {
                                if a.order >= 2 && fsb > 1 && tx > 2 {
                                    continue;
                                }
                                if fsb == 3 && fc > 4 {
                                    continue;
                                }
                                for &acfg in acfgs {
                                for &pfs in pfss {
                                let c = Cfg { halo: 3, tt: tx, ch, tps: ncg, nt, smax: nt / ncg * fsb, padc: 0, sb: fsb, rx: 1, ty, tz, acfg, pfs, minb: 0, nwc: 0, nws: 0, vxpad: 0, vt: 0, shr: 0, pf: 0, s2: 0, pt: 0, carve: -1, vtp: 0 };
                                // planes within 64 KB; per-thread plane prefetch ≤ 16 float4
                                let plane = (tx + W - 1) * (ty + W - 1) * ch * 8;
                                if smem(&c, a.order, a.nseg) <= smax_opt && plane <= 32 << 10 && plane / 16 <= 16 * nt {
                                    v.push(c);
                                }
                                }
                                }
                            }
                        }
                    }
                }
            }
        }
        "vs" => {
            // value-stationary rows (§12): footprint tx x ty (the L2 staging budget wants ≈ 16 cells), tz, items of
            // smax, nwc channel groups x nws window splits per block
            let g = nch.div_ceil(16);
            let tiles: &[[usize; 3]] = if full { &[[2, 8, 8], [2, 8, 16], [4, 4, 8], [4, 4, 16], [2, 4, 8], [2, 4, 16], [1, 8, 16], [4, 8, 8], [1, 16, 8]] } else if a.order >= 2 { &[[2, 4, 8], [2, 8, 8], [4, 4, 8], [4, 4, 16], [2, 4, 16]] } else { &[[2, 8, 8], [2, 8, 16], [4, 4, 8], [4, 4, 16], [2, 4, 16]] };
            let smaxs: &[usize] = if a.order >= 2 { &[16, 32, 48] } else if full { &[32, 48, 64, 96, 128] } else { &[32, 64, 96, 128] };
            // order >= 2: the per-lane x-moment body (vt 0, §12) and the shared weight table (vt 1, §13); order 3: vt only
            let vts: &[usize] = if a.order >= 3 { &[1] } else if a.order == 2 { &[0, 1] } else { &[0] };
            for &vt in vts {
            let smaxs: &[usize] = if vt == 1 { &[16, 32, 48, 64] } else { smaxs };
            for &[tx, ty, tz] in tiles {
                for &smax in smaxs {
                    // channel groups per block: small (more blocks) or all (plain stores)
                    for nwc in divisors(g).into_iter().filter(|&n| n <= 4 || n == g) {
                        for nws in [1usize, 2, 4, 8] {
                            // block size: __launch_bounds__(NT) caps registers at 64 K / NT; TX 2 o0 needs ≈ 160,
                            // TX 4 ≈ 226, o2 ≈ 254 (§12.5)
                            let ntmax = if a.order < 2 && tx <= 2 { 384 } else { 256 };
                            if ty % nws != 0 || 32 * nwc * nws > ntmax {
                                continue;
                            }
                            for vxpad in if vt == 1 { vec![0usize] } else if full || a.order >= 2 { vec![0usize, 1] } else { vec![0] } {
                                if vxpad == 1 && tx == 1 {
                                    continue;
                                }
                                let c = Cfg { halo: 4, tt: tx, ch: 16 * nwc, tps: 1, nt: 32 * nwc * nws, smax, padc: 0, sb: 0, rx: 0, ty, tz, acfg: 0, pfs: 0, minb: 0, nwc, nws, vxpad, vt, shr: 0, pf: 0, s2: 0, pt: 0, carve: -1, vtp: 0 };
                                if smem(&c, a.order, a.nseg) <= smax_opt {
                                    v.push(c);
                                }
                            }
                        }
                    }
                }
            }
            }
        }
        _ => {
            for tt in [4usize, 8] {
                for fc in [1usize, 2, 4, 8] {
                    for tps in [1usize, 2, 4, 8, 16, 32] {
                        let ch = fc * tps;
                        if ch > nch || nch % ch != 0 {
                            continue;
                        }
                        for &nt in nts {
                            for smax in if full { vec![32usize, 64, 128] } else { vec![64] } {
                                for (sb, rx) in [(1usize, 1usize), (2, 2), (4, 2)] {
                                    v.push(Cfg { halo: 0, tt, ch, tps, nt, smax, padc: 0, sb, rx, ty: 0, tz: 0, acfg: 0, pfs: 0, minb: 0, nwc: 0, nws: 0, vxpad: 0, vt: 0, shr: 0, pf: 0, s2: 0, pt: 0, carve: -1, vtp: 0 });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    v
}

fn main() {
    let a = args();
    if a.mode == "stats" {
        // CPU only: how many samples share a support corner (cell), i.e. how much a value load could be shared
        let nf = hasty_nufft::grid_default_size(160, 1.25, W);
        let kd = trajectory(&a.traj, a.m);
        let m = kd.len();
        let corner = |k: &[f32; 3], a: usize| -> i64 {
            let g = ((k[a] as f64 / (2.0 * std::f64::consts::PI)) * nf as f64).rem_euclid(nf as f64) as f32;
            ((g - W as f32 * 0.5).ceil() as i64).rem_euclid(nf as i64)
        };
        let keys: Vec<[i64; 3]> = kd.iter().map(|k| [corner(k, 0), corner(k, 1), corner(k, 2)]).collect();
        for r in [1i64, 2, 3, 4] {
            // groups: same (j0, k0), i0 in the same r-wide bin
            let mut h = std::collections::HashMap::<(i64, i64, i64), usize>::new();
            for k in &keys {
                *h.entry((k[0] / r, k[1], k[2])).or_default() += 1;
            }
            let mut sizes: Vec<usize> = h.values().cloned().collect();
            sizes.sort_unstable();
            let frac = |s: usize| sizes.iter().filter(|&&v| v >= s).map(|v| *v).sum::<usize>() as f64 / m as f64;
            println!("  x-bin {r}: {} groups, mean {:.2} samples/group, samples in groups >= 2: {:.2}, >= 4: {:.2}, >= 8: {:.2}; union efficiency {:.2}", h.len(), m as f64 / h.len() as f64, frac(2), frac(4), frac(8), W as f64 / (W as f64 + r as f64 - 1.0));
        }
        let mut occ = std::collections::HashMap::<i64, usize>::new();
        for k in &keys {
            *occ.entry((k[2] / 4 * 64 + k[1] / 4) * 64 + k[0] / 4).or_default() += 1;
        }
        let mut v: Vec<usize> = occ.values().cloned().collect();
        v.sort_unstable();
        println!("  TT 4 tiles: {} occupied of {}, samples/tile median {}, p90 {}, max {}", v.len(), (nf / 4).pow(3), v[v.len() / 2], v[v.len() * 9 / 10], v[v.len() - 1]);
        // home tiles (tx, ty, tz) for the plane-streamed design: sample-weighted samples per tile, and the halo staging
        // bytes per useful FMA (order 0, 2 FMA per MAC, items of <= 128 samples restage the halo)
        for td in [[2usize, 4, 4], [4, 4, 4], [4, 4, 8], [4, 8, 8], [2, 8, 8], [8, 8, 8], [4, 8, 16]] {
            let nt = [0, 1, 2].map(|q| nf.div_ceil(td[q]));
            let mut h = std::collections::HashMap::<usize, usize>::new();
            for k in &keys {
                *h.entry(((k[2] as usize / td[2]) * nt[1] + k[1] as usize / td[1]) * nt[0] + k[0] as usize / td[0]).or_default() += 1;
            }
            let sw: f64 = h.values().map(|&n| (n * n) as f64).sum::<f64>() / m as f64;
            let items: usize = h.values().map(|&n| n.div_ceil(128)).sum();
            let halo = ((td[0] + W - 1) * (td[1] + W - 1) * (td[2] + W - 1)) as f64;
            println!("  home tile {td:?}: {} tiles, sample-weighted samples/tile {sw:.0}, items(128) {items}, staging {:.3} B per useful FMA", h.len(), items as f64 * halo * 8.0 / (m as f64 * 2000.0));
        }
        // value-stationary design (vs, §12): items = runs of <= smax z-sorted samples of one home tile; the item loads
        // only the planes / rows its samples touch: (tx + W - 1) x (dy + W) x (dz + W) points (dy, dz = corner ranges).
        // Also: mean samples active per (item, plane) (the reuse of a loaded value row), sample-weighted.
        for td in [[1usize, 1, 200], [1, 2, 200], [2, 1, 200], [2, 2, 200], [1, 4, 200], [4, 1, 200], [2, 4, 200], [4, 4, 200], [4, 4, 8], [4, 8, 8], [8, 8, 8]] {
            let nt = [0, 1, 2].map(|q| nf.div_ceil(td[q]));
            let mut h = std::collections::HashMap::<usize, Vec<[i64; 3]>>::new();
            for k in &keys {
                h.entry(((k[2] as usize / td[2]) * nt[1] + k[1] as usize / td[1]) * nt[0] + k[0] as usize / td[0]).or_default().push(*k);
            }
            for smax in [32usize, 64, 128, 256] {
                let (mut pts, mut nitems, mut act, mut actw) = (0f64, 0usize, 0f64, 0f64);
                for v in h.values() {
                    let mut v = v.clone();
                    v.sort_by_key(|k| (k[2], k[1], k[0]));
                    for it in v.chunks(smax) {
                        let (ylo, yhi) = (it.iter().map(|k| k[1]).min().unwrap(), it.iter().map(|k| k[1]).max().unwrap());
                        let (zlo, zhi) = (it.iter().map(|k| k[2]).min().unwrap(), it.iter().map(|k| k[2]).max().unwrap());
                        let np = (zhi - zlo + W as i64) as f64;
                        pts += ((td[0] + W - 1) as i64 * (yhi - ylo + W as i64)) as f64 * np;
                        nitems += 1;
                        // each sample is active in W planes: mean active per plane = n W / np; sample-weighted
                        let a = it.len() as f64 * W as f64 / np;
                        act += a * it.len() as f64;
                        actw += it.len() as f64;
                    }
                }
                println!("  vs tile {td:?} smax {smax:>3}: {nitems} items, staging {:.3} B per useful FMA (o0), samples active per plane {:.1} (sample-weighted)", pts * 8.0 / (m as f64 * 2000.0), act / actw);
            }
        }
        return;
    }
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let attr = |x| dev.ctx.attribute(x).unwrap() as usize;
    let nsm = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    let clk = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE) as f64 * 1e3;
    let smax_opt = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
    let peak_fma = nsm as f64 * 128.0 * clk;
    let (nf, nch) = (hasty_nufft::grid_default_size(160, 1.25, W), a.ncb * a.nseg);
    let beta = hasty_nufft::EsKernel::with_width(W, 1.25).beta;
    println!("device {} SMs, clock {:.2} GHz, FP32 FMA peak {:.1} T/s, smem opt-in {} B; grid {nf}^3, NCH {nch} ({} coils x {} seg), order {}, M {}", nsm, clk * 1e-9, peak_fma * 1e-12, smax_opt, a.ncb, a.nseg, a.order, a.m);
    let kd = trajectory(&a.traj, a.m);
    let m = kd.len();
    let mut s = 4242u64;
    let mut u = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5 };
    let json = |rec: serde_json::Value| {
        if let Some(p) = &a.json {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap();
            writeln!(f, "{rec}").unwrap();
        }
    };
    let ev = (dev.ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap(), dev.ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap()); // timing events (cudarc default flags disable timing)
    // median of reps after a warm-up; aborts early (returns the first time) when the first call exceeds `cut`
    let time = |f: &mut dyn FnMut(), reps: usize, cut: f32| -> f32 {
        let mut t = Vec::new();
        for i in 0..=reps {
            ev.0.record(&st).unwrap();
            f();
            ev.1.record(&st).unwrap();
            ev.1.synchronize().unwrap();
            let ms = ev.0.elapsed_ms(&ev.1).unwrap();
            if i == 0 && ms > cut {
                return ms;
            }
            if i >= 1 {
                t.push(ms);
            }
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        t[t.len() / 2]
    };
    let macs = m as f64 * (W * W * W) as f64 * nch as f64;
    let fma_per_mac = if a.order >= 2 { 4.0 } else { 2.0 };
    let tag = std::env::args().skip_while(|x| x != "--tag").nth(1).unwrap_or_default();
    let defs = std::env::var("HASTY_SEG_DEFS").unwrap_or_default();
    let kvenv = std::env::var("HASTY_SEG_KV").unwrap_or_default();
    let base_rec = serde_json::json!({"m": m, "ncb": a.ncb, "nseg": a.nseg, "nch": nch, "order": a.order, "traj": a.traj, "tag": tag, "defs": defs, "kv_env": kvenv});

    if a.mode == "base" {
        use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};
        hasty_nufft::set_thread_cb_cap(Some(a.ncb));
        // --knot: the production knot-major kernels (segnufft_knot.cu, variants 0/1 timed by the autotune) on one knot:
        // 2 bins (B-spline degree 0), every sample in bin 0, knot 0 timed
        let knot = std::env::args().any(|x| x == "--knot");
        let so = SegNufftOpts { sigma: 1.25, width: W, nseg: a.nseg, order: a.order, frames: if knot { 2 } else { 1 }, bspline: if knot { 0 } else { 1 }, max_grid_bytes: 22 << 30 };
        let mut op = SegNufft::new_temporal(&dev, [160; 3], [1e-3; 3], a.ncb, so, knot).unwrap();
        assert_eq!(op.cb, a.ncb, "one batch");
        let nv = 160 * 160 * 160;
        let fit: Vec<usize> = (0..nv).step_by(97).collect();
        let psi = |v: usize, t: f64| 2000.0 * ((v % 113) as f64 / 113.0 - 0.5) * t;
        let (sp, at, t0, dt) = build_segments(nv, a.nseg, 2e-3, 8e-3, 128, &fit, &psi);
        let seg = op.segments(&st, &sp, &at, t0, dt).unwrap();
        drop(sp);
        let c: [Vec<f32>; 3] = [0, 1, 2].map(|q| kd.iter().map(|p| p[q]).collect());
        let tau: Vec<f32> = (0..m).map(|_| (5e-3 + u() * 6e-3) as f32).collect();
        let dcv: Vec<f32> = (0..6 * m).map(|_| (u() * 10.0) as f32).collect();
        let ph0 = vec![0f32; m];
        let tp = Instant::now();
        let tk: Vec<f32> = vec![0.25; m];
        // order 3: the degree-3/4 tables (points_hi: u SoA 4 x m, H' 4 x NHI; values only need to be finite)
        let pts = if a.order >= 3 {
            let uu: Vec<f32> = (0..4 * m).map(|_| u() as f32).collect();
            let hh: Vec<f64> = (0..4 * hasty_nufft::NHI).map(|_| u() * 1e5).collect();
            op.points_hi(&st, [&c[0], &c[1], &c[2]], &tau, &ph0, &dcv, &uu, &hh, if knot { Some(&tk) } else { None }).unwrap()
        } else {
            op.points(&st, [&c[0], &c[1], &c[2]], &tau, &ph0, &dcv, if knot { Some(&tk) } else { None }).unwrap()
        };
        st.synchronize().unwrap();
        println!("  engine points {:.1} s, grid {:?}", tp.elapsed().as_secs_f64(), op.plan.nf);
        let def = (op.tiles_fwd, op.tiles_adj);
        let tt = Instant::now();
        let (kf, ka) = op.autotune(&st, &seg, &pts).unwrap();
        println!("  base (today's tile kernels, autotuned {:.0} s): forward {kf:.2} ms {:?}, adjoint {ka:.2} ms {:?} (default {:?} / {:?})", tt.elapsed().as_secs_f64(), op.tiles_fwd, op.tiles_adj, def.0, def.1);
        for (dir, ms) in [("fwd", kf), ("adj", ka)] {
            let r = serde_json::json!({"design": if knot { "base_knot" } else { "base" }, "dir": dir, "ms": ms, "gmac_s": macs / (ms as f64 * 1e-3) * 1e-9, "peak_frac": macs * fma_per_mac / (ms as f64 * 1e-3) / peak_fma,
                "cfg": format!("{:?}", if dir == "fwd" { op.tiles_fwd } else { op.tiles_adj })});
            let mut b = base_rec.clone();
            b.as_object_mut().unwrap().extend(r.as_object().unwrap().clone());
            json(b);
        }
        return;
    }

    if a.mode == "redbench" {
        // global float REDs in the flush pattern of an accumulator-stationary adjoint (pg_red_bench, §15): every home
        // tile flushes its footprint box for all channels; tiles cover the whole nf^3 grid. Rates in G float-ops/s.
        let nchp = 16 * nch.div_ceil(16);
        let mut g = st.alloc_zeros::<C32>(nf * nf * nf * nchp).unwrap();
        for td in [[4usize, 4, 16], [2, 4, 16], [4, 4, 8]] {
            for mode in [0usize, 1, 2, 3] {
                let f = dev.load_function(&format!("proto_gather_redb_{}_{}_{}_{mode}", td[0], td[1], td[2]), &format!("#define REDB 1\n#define RED_MODE {mode}\n#define TXB {}\n#define TYB {}\n#define TZB {}\n#define W 10\n#define BETA 1\n#define CORR 0\n#define NCB 1\n#define NSEG 1\n#define TT 4\n#define CH 1\n#define TPS 1\n#define NT 128\n#define SMAX 1\n{SRC}", td[0], td[1], td[2]), "pg_red_bench", &CompileSettings::default()).unwrap();
                let nt = [0, 1, 2].map(|q| nf.div_ceil(td[q]));
                let (ntx, nty, nfi, nchi) = (nt[0] as i32, nt[1] as i32, nf as i32, nchp as i32);
                let cfg = LaunchConfig { grid_dim: ((nt[0] * nt[1] * nt[2]) as u32, 1, 1), block_dim: (nchp as u32 * 2, 1, 1), shared_mem_bytes: 0 };
                let mut run = || unsafe { st.launch_builder(&f).arg(&mut g).arg(&nfi).arg(&nchi).arg(&ntx).arg(&nty).launch(cfg).unwrap(); };
                let ms = time(&mut run, 3, f32::INFINITY) as f64;
                let pts = (nt[0] * nt[1] * nt[2]) as f64 * ((td[0] + W - 1) * (td[1] + W - 1) * (td[2] + W - 1)) as f64;
                let flops = pts * nchp as f64 * 2.0;
                println!("  REDB tile {td:?} mode {mode} ({}): {ms:8.2} ms, {:.2} G footprint points x {nchp} ch, {:.0} G float-ops/s, {:.2} TB/s of 4-B values", ["RED stride-8", "RED packed", "plain RMW", "plain store"][mode], pts * 1e-9, flops / (ms * 1e-3) * 1e-9, flops * 4.0 / (ms * 1e-3) * 1e-12);
                json(serde_json::json!({"design": "redbench", "tile": td, "mode": mode, "ms": ms, "points": pts, "nch": nchp, "gflop_s": flops / (ms * 1e-3) * 1e-9, "tag": tag}));
            }
        }
        return;
    }

    if a.mode == "ldsbench" {
        // shared-memory float4 broadcast loads (pg_lds_bench, §11): lanes read ncg distinct float4, KF FFMA per load,
        // k blocks of 8 warps per SM (one wave); rates per SM
        let mut out = st.alloc_zeros::<f32>(nsm * 6 * 256).unwrap();
        let iters: i32 = 20000;
        // LDSB_W=1|2: LDS.32 / LDS.64 variants (§12; ncg 33 = all lanes distinct)
        let ldsw: usize = std::env::var("LDSB_W").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        if ldsw != 4 {
            for kf in [0usize, 4, 8] {
                let f = dev.load_function(&format!("proto_gather_ldsb_w{ldsw}_kf{kf}"), &format!("#define LDSB 1\n#define LDSW {ldsw}\n#define KF {kf}\n#define W 10\n#define BETA 1\n#define CORR 0\n#define NCB 1\n#define NSEG 1\n#define TT 4\n#define CH 1\n#define TPS 1\n#define NT 128\n#define SMAX 1\n{SRC}"), "pg_lds_bench", &CompileSettings::default()).unwrap();
                for k in [2usize, 3] {
                    for ncg in [1i32, 2, 4, 16, 33] {
                        let cfg = LaunchConfig { grid_dim: ((nsm * k) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                        let mut run = || unsafe { st.launch_builder(&f).arg(&mut out).arg(&iters).arg(&ncg).launch(cfg).unwrap(); };
                        let ms = time(&mut run, 3, f32::INFINITY) as f64;
                        let lds = (nsm * k * 8) as f64 * iters as f64 * 16.0;
                        let per_sm_ns = lds / nsm as f64 / (ms * 1e6);
                        let ffma = lds * kf as f64 * 32.0 / (ms * 1e-3) / peak_fma;
                        println!("  LDSB w{ldsw} kf {kf:>2} warps/SM {:>2} ncg {ncg:>2}: {ms:8.3} ms, {:.2} clk per warp-LDS at 2.79 GHz, FFMA {:.1} % of peak", k * 8, 2.79 / per_sm_ns, 100.0 * ffma);
                        json(serde_json::json!({"design": "ldsbench", "ldsw": ldsw, "kf": kf, "warps_sm": k * 8, "ncg": ncg, "ms": ms, "lds_per_sm_ns": per_sm_ns, "ffma_frac": ffma, "tag": tag}));
                    }
                }
            }
            return;
        }
        // LDSB_BLK=1: blocked address pattern (lane / (32 / ncg)) instead of interleaved (lane % ncg), §13
        let blk: usize = std::env::var("LDSB_BLK").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        for kf in [0usize, 4, 8, 16] {
            let f = dev.load_function(&format!("proto_gather_ldsb_kf{kf}_b{blk}"), &format!("#define LDSB 1\n#define LDSBLK {blk}\n#define KF {kf}\n#define W 10\n#define BETA 1\n#define CORR 0\n#define NCB 1\n#define NSEG 1\n#define TT 4\n#define CH 1\n#define TPS 1\n#define NT 128\n#define SMAX 1\n{SRC}"), "pg_lds_bench", &CompileSettings::default()).unwrap();
            for k in [1usize, 2, 3, 6] {
                for ncg in [1i32, 2, 4, 8, 32] {
                    let cfg = LaunchConfig { grid_dim: ((nsm * k) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    let mut run = || unsafe { st.launch_builder(&f).arg(&mut out).arg(&iters).arg(&ncg).launch(cfg).unwrap(); };
                    let ms = time(&mut run, 3, f32::INFINITY) as f64;
                    let lds = (nsm * k * 8) as f64 * iters as f64 * 16.0; // warp-level LDS.128
                    let per_sm_ns = lds / nsm as f64 / (ms * 1e6);
                    let ffma = lds * kf as f64 * 32.0 / (ms * 1e-3) / peak_fma;
                    println!("  LDSB kf {kf:>2} warps/SM {:>2} ncg {ncg:>2}: {ms:8.3} ms, {per_sm_ns:.3} warp-LDS.128 per SM per ns ({:.2} clk each at 2.79 GHz), FFMA {:.1} % of peak (device clock)", k * 8, 2.79 / per_sm_ns, 100.0 * ffma);
                    json(serde_json::json!({"design": "ldsbench", "kf": kf, "blk": blk, "warps_sm": k * 8, "ncg": ncg, "ms": ms, "lds_per_sm_ns": per_sm_ns, "ffma_frac": ffma, "tag": tag}));
                }
            }
        }
        return;
    }

    // problem: grid coordinates in [0, nf), ΔC (kernel units, |d| ≤ 0.05), segment coefficients
    let p = Problem {
        nf,
        nch,
        sc: kd.iter().map(|k| k.map(|v| { let mut g = ((v as f64 / (2.0 * std::f64::consts::PI)) * nf as f64).rem_euclid(nf as f64) as f32; if g >= nf as f32 { g -= nf as f32; } g })).collect(),
        d: (0..m).map(|_| [0; 6].map(|_| (u() * 0.1) as f32)).collect(),
        a: (0..m * a.nseg).map(|_| C32::new((u() * 2.0) as f32, (u() * 2.0) as f32)).collect(),
        // order 3: |u| ≤ 0.5, |H'| ≤ 0.05 → |hc| ≲ 0.05 (the size of the ΔC terms)
        u: if a.order >= 3 { (0..m).map(|_| [0; 4].map(|_| u() as f32)).collect() } else { Vec::new() },
        htab: if a.order >= 3 { (0..100).map(|_| (u() * 0.1) as f32).collect() } else { Vec::new() },
    };
    assert!(a.order <= 2 || (a.mode != "fwd" || a.designs.iter().all(|d| d == "vs")), "order 3: vs (vt) only");
    let htab_d = st.clone_htod(&if p.htab.is_empty() { vec![0f32; 100] } else { p.htab.clone() }).unwrap();
    let hi = a.order >= 3;
    // hc of sample j (original order), f64 from the f32 inputs
    let hc_of = |j: usize| -> [f64; 25] { std::array::from_fn(|k| (0..4).map(|q| p.u[j][q] as f64 * p.htab[q * 25 + k] as f64).sum()) };
    let ng = nf * nf * nf * nch;
    let dims = Dims { nf0: nf as i32, nf1: nf as i32, nf2: nf as i32 };
    let mi = m as i32;
    let fill = |g: &mut CudaSlice<C32>| {
        let f = dev.load_function("proto_gather_fill", &format!("#define W 10\n#define BETA 1\n#define CORR 0\n#define NCB 1\n#define NSEG 1\n#define TT 4\n#define CH 1\n#define TPS 1\n#define NT 128\n#define SMAX 1\n{SRC}"), "pg_fill", &CompileSettings::default()).unwrap();
        let n = ng as u64;
        unsafe { st.launch_builder(&f).arg(g).arg(&n).launch(LaunchConfig { grid_dim: (4096, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }).unwrap() };
    };
    let mut grid = st.alloc_zeros::<C32>(ng).unwrap();
    let mut y = st.alloc_zeros::<C32>(a.ncb * m).unwrap();
    println!("  grid {:.2} GiB, y {:.2} GiB, {} useful complex MACs per pass", (ng * 8) as f64 / (1u64 << 30) as f64, (a.ncb * m * 8) as f64 / (1u64 << 30) as f64, macs);
    let compile_all = |cs: &[Cfg]| -> Vec<Option<CudaFunction>> {
        // parallel NVRTC (4 threads), then load
        let srcs: Vec<(String, String)> = cs.iter().map(|c| source(c, &a, beta)).collect();
        let cubins: Vec<_> = std::thread::scope(|sc| {
            let chunks: Vec<Vec<usize>> = (0..4).map(|t| (t..srcs.len()).step_by(4).collect()).collect();
            let hs: Vec<_> = chunks.into_iter().map(|ix| { let srcs = &srcs; let arch = dev.arch.clone(); sc.spawn(move || ix.into_iter().map(|i| (i, compile_cubin(&srcs[i].0, &srcs[i].1, &arch, &CompileSettings::default()))).collect::<Vec<_>>()) }).collect();
            let mut all: Vec<_> = hs.into_iter().flat_map(|h| h.join().unwrap()).collect();
            all.sort_by_key(|x| x.0);
            all.into_iter().map(|x| x.1).collect::<Vec<_>>()
        });
        cs.iter()
            .zip(cubins)
            .map(|(c, cb)| {
                let cb = match cb { Ok(b) => b, Err(e) => { println!("    compile {c:?}: {}", e.to_string().lines().take(3).collect::<Vec<_>>().join(" ")); return None; } };
                let md = match dev.ctx.load_module(cudarc::nvrtc::Ptx::from_binary(cb)) { Ok(m) => m, Err(e) => { println!("    load {c:?}: {e:?}"); return None; } };
                let name = match c.halo { 1 => "pg_fwd_halo", 0 => "pg_fwd_gm", 3 => "pg_fwd_ps", 4 => "pg_fwd_vs", _ => "pg_adj_sm" };
                let f = match md.load_function(name) { Ok(f) => f, Err(e) => { println!("    function {name} {c:?}: {e:?}"); return None; } };
                if let Err(e) = f.set_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem(c, a.order, a.nseg) as i32) { println!("    smem {} B {c:?}: {e:?}", smem(c, a.order, a.nseg)); return None; }
                if c.halo == 4 && c.carve >= 0 {
                    // vs (§13): preferred shared-memory carve-out (percent of the unified L1 / shared memory)
                    if let Err(e) = f.set_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, c.carve) { println!("    carve {} {c:?}: {e:?}", c.carve); return None; }
                }
                if c.halo == 0 {
                    // prefer L1 over shared memory
                    let _ = f.set_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 0);
                }
                Some(f)
            })
            .collect()
    };

    if a.mode == "fwd" {
        fill(&mut grid);
        let mut sorted: Vec<Sorted> = Vec::new();
        let mut sbufs: Vec<Arc<SBufs>> = Vec::new();
        let mut best_all: Vec<(String, f32, Cfg)> = Vec::new();
        for design in &a.designs {
            let mut cs = candidates(&a, design, smax_opt);
            if let Some(fx) = a.fixed.as_ref().filter(|_| design == "vs") {
                // tx,ty,tz,smax,nwc,nws[,vxpad][,key=value ...] with keys vt, shr, pf, s2, pt, minb, carve (§13);
                // several configurations separated by ';' (with --each every one is timed, checked and logged)
                cs = fx.split(';').filter(|x| !x.trim().is_empty()).map(parse_vs).collect();
            } else if let Some(fx) = a.fixed.as_ref().filter(|_| design == "ps") {
                // tx,ty,tz,ncg,fc,fsb,nt[,acfg]
                let v: Vec<usize> = fx.split(',').map(|s| s.parse().unwrap()).collect();
                cs = vec![Cfg { halo: 3, tt: v[0], ty: v[1], tz: v[2], tps: v[3], ch: v[3] * v[4], sb: v[5], nt: v[6], smax: v[6] / v[3] * v[5], padc: 0, rx: 1, acfg: v.get(7).copied().unwrap_or(0), pfs: v.get(8).copied().unwrap_or(0), minb: v.get(9).copied().unwrap_or(0), nwc: 0, nws: 0, vxpad: 0, vt: 0, shr: 0, pf: 0, s2: 0, pt: 0, carve: -1, vtp: 0 }];
            } else if let Some(fx) = &a.fixed {
                let v: Vec<usize> = fx.split(',').map(|s| s.parse().unwrap()).collect();
                cs = vec![Cfg { halo: if design == "halo" { 1 } else { 0 }, tt: v[0], ch: v[1], tps: v[2], nt: v[3], smax: v[4], padc: v[5], sb: v[6], rx: v[7], ty: 0, tz: 0, acfg: 0, pfs: 0, minb: 0, nwc: 0, nws: 0, vxpad: 0, vt: 0, shr: 0, pf: 0, s2: 0, pt: 0, carve: -1, vtp: 0 }];
            }
            let tc = Instant::now();
            let mut round = cs.clone();
            let mut fs = compile_all(&round);
            println!("  {design}: {} candidates, compiled in {:.1} s", cs.len(), tc.elapsed().as_secs_f64());
            let mut best = (f32::INFINITY, cs[0]);
            // vs autotune in two stages (§13): the shape search with default tunables, then the tunable combinations
            // (vs_refine) on the best shape
            let refinable = design == "vs" && a.fixed.is_none() && !a.each;
            for pass in 0..2 {
            for (c, f) in round.iter().zip(fs) {
                let Some(f) = f else { continue };
                let isvs = c.halo == 4;
                let so = match sorted.iter().position(|s| s.vs == isvs && s.td == tdims(c) && s.smax == c.smax && s.sb == c.sb && s.rx == c.rx) {
                    Some(i) => i,
                    None => { sorted.push(if isvs { sort_vs(&st, &p, tdims(c), c.smax, &mut sbufs) } else { sort(&st, &p, tdims(c), c.smax, c.sb, c.rx, &mut sbufs) }); sorted.len() - 1 }
                };
                let sr = &sorted[so];
                let cfg = LaunchConfig { grid_dim: (sr.nitems as u32, gridy(c, nch), 1), block_dim: (c.nt as u32, 1, 1), shared_mem_bytes: smem(c, a.order, a.nseg) as u32 };
                let own = owns(c, &a);
                let mut run = || {
                    if !own {
                        st.memset_zeros(&mut y).unwrap();
                    }
                    {
                    let mut lb = st.launch_builder(&f);
                    lb.arg(&sr.b.pos).arg(&sr.b.dco).arg(&sr.b.acf).arg(&sr.b.perm).arg(&sr.items).arg(&sr.units).arg(&grid).arg(&mut y).arg(&mi).arg(&dims);
                    if hi {
                        lb.arg(&sr.b.uw).arg(&htab_d);
                    }
                    unsafe { lb.launch(cfg).unwrap() };
                }
                };
                let ms = time(&mut run, if best.0.is_finite() { 2 } else { 1 }, 3.0 * best.0);
                let regs = f.num_regs().unwrap_or(-1);
                println!("    {design} t {:?} ch {:>3} tps {:>2} (fc {:>2}) nt {:>3} smax {:>3} sb {} rx {}{} smem {:>6} regs {regs:>3}: {ms:8.2} ms", tdims(c), c.ch, c.tps, c.ch / c.tps, c.nt, c.smax, c.sb, c.rx, format!("{}{}{}", if c.acfg == 1 { " acfg" } else { "" }, if c.pfs == 1 { " pfs" } else { "" }, if c.minb > 0 && c.halo == 3 { format!(" minb {}", c.minb) } else { String::new() }) + &if c.halo == 4 { format!(" nwc {} nws {} vxpad {} vt {} vtp {} shr {} pf {} s2 {} pt {} minb {} carve {}", c.nwc, c.nws, c.vxpad, c.vt, c.vtp, c.shr, c.pf, c.s2, c.pt, c.minb, c.carve) } else { String::new() }, smem(c, a.order, a.nseg));
                if ms < best.0 {
                    best = (ms, *c);
                }
            }
            if pass == 0 && refinable {
                round = vs_refine(&best.1, a.order);
                fs = compile_all(&round);
                println!("  {design}: refinement of the best shape, {} variants", round.len());
            } else {
                break;
            }
            }
            if a.each {
                best_all.extend(cs.iter().map(|c| (design.clone(), f32::NAN, *c)));
            } else {
                best_all.push((design.clone(), best.0, best.1));
            }
        }
        // verify and time each design's best with full reps
        for (design, _, c) in &best_all {
            let fs = compile_all(&[*c]);
            let Some(f) = fs[0].clone() else { println!("  SKIP (compile / load failed) {c:?}"); continue; };
            let sr = sorted.iter().find(|s| s.vs == (c.halo == 4) && s.td == tdims(c) && s.smax == c.smax && s.sb == c.sb && s.rx == c.rx).unwrap();
            let cfg = LaunchConfig { grid_dim: (sr.nitems as u32, gridy(c, nch), 1), block_dim: (c.nt as u32, 1, 1), shared_mem_bytes: smem(c, a.order, a.nseg) as u32 };
            let own = owns(c, &a);
            let mut run = || {
                if !own {
                    st.memset_zeros(&mut y).unwrap();
                }
                {
                    let mut lb = st.launch_builder(&f);
                    lb.arg(&sr.b.pos).arg(&sr.b.dco).arg(&sr.b.acf).arg(&sr.b.perm).arg(&sr.items).arg(&sr.units).arg(&grid).arg(&mut y).arg(&mi).arg(&dims);
                    if hi {
                        lb.arg(&sr.b.uw).arg(&htab_d);
                    }
                    unsafe { lb.launch(cfg).unwrap() };
                }
            };
            let ms = time(&mut run, a.reps, f32::INFINITY);
            // resources without the profiler: registers, local memory (spills), resident blocks per SM
            let regs = f.num_regs().unwrap_or(-1);
            let lmem = f.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES).unwrap_or(-1);
            let bps = f.occupancy_max_active_blocks_per_multiprocessor(c.nt as u32, smem(c, a.order, a.nseg), None).map(|v| v as i64).unwrap_or(-1);
            let warps_sm = bps * (c.nt as i64 / 32);
            let wk = if c.halo == 3 { Some(ps_work(&p, sr, c)) } else { None };
            if c.halo == 4 {
                let w = vs_work(&p, sr, c);
                println!("  WORK vs: item fill {:.3}, samples active per (item, plane) {:.1}, bodies per (plane, window step) {:.1}, planes per item {:.1}, channel-lane efficiency {:.3}; FFMA-pipe use of issued work {:.1} %", w[0], w[1], w[2], w[3], w[4],
                    100.0 * macs * fma_per_mac / (ms as f64 * 1e-3) / peak_fma / w[4] * if c.vt == 1 { if a.order >= 3 { 329.0 / 200.0 } else { 287.0 / 200.0 } } else if a.order >= 2 { 706.0 / 200.0 } else { 112.0 / 100.0 });
            }
            println!("  RES {design}: regs {regs}, local {lmem} B, smem {} B, blocks/SM {bps}, warps/SM {warps_sm} of 48", smem(c, a.order, a.nseg));
            if let Some(w) = &wk {
                println!("  WORK {design}: useful/issued x-loop MAC slots {:.3} (x {:.3} = W/HXP; rows/warp {:.2}, planes/warp {:.2}, live slots {:.3}); active (warp, plane) {:.3}; block fill {:.3}; FFMA-pipe use of issued work {:.1} %",
                    w[0], W as f64 / (c.tt + W - 1) as f64, w[1], w[2], w[3], w[4], w[5], 100.0 * macs * fma_per_mac / (ms as f64 * 1e-3) / peak_fma / w[0]);
            }
            if let Ok(dir) = std::env::var("PG_CUBIN_DIR") {
                let (name, src) = source(c, &a, beta);
                if let Ok(cb) = compile_cubin(&name, &src, &dev.arch, &CompileSettings::default()) {
                    let _ = std::fs::create_dir_all(&dir);
                    let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{name}.cubin")), cb);
                }
            }
            run();
            let yh = st.clone_dtoh(&y).unwrap();
            // CPU f64 reference on a strided subset of the original samples
            let step = (m / a.check.max(1)).max(1);
            let (mut e2, mut r2, mut emax) = (0.0f64, 0.0f64, 0.0f64);
            let chk: Vec<usize> = (0..m).step_by(step).collect();
            let res: Vec<Vec<(f64, f64)>> = std::thread::scope(|scp| {
                let hs: Vec<_> = (0..4).map(|t| { let (chk, p, a2) = (&chk, &p, &a); scp.spawn(move || chk.iter().skip(t).step_by(4).map(|&j| {
                    let dd = p.d[j].map(|v| v as f64);
                    let hcj = if a2.order >= 3 { Some(hc_of(j)) } else { None };
                    let (c0, w) = weights(p.sc[j], &dd, a2.order, beta, nf, hcj.as_ref());
                    let mut out = vec![(0.0, 0.0); a2.ncb];
                    for tz in 0..W { for ty in 0..W { for tx in 0..W {
                        let (gx, gy, gz) = ((c0[0] + tx) % nf, (c0[1] + ty) % nf, (c0[2] + tz) % nf);
                        let wv = w[(tz * W + ty) * W + tx];
                        let base = ((gz * nf + gy) * nf + gx) * nch;
                        for ch in 0..nch {
                            let g = gval((base + ch) as u64);
                            let (c, l) = (ch / a2.nseg, ch % a2.nseg);
                            let av = p.a[j * a2.nseg + l];
                            let (vr, vi) = (wv.0 * g.0 - wv.1 * g.1, wv.0 * g.1 + wv.1 * g.0);
                            let (ar, ai) = (av.re as f64, av.im as f64);
                            out[c].0 += ar * vr - ai * vi;
                            out[c].1 += ar * vi + ai * vr;
                        }
                    }}}
                    out
                }).collect::<Vec<_>>()) }).collect();
                let parts: Vec<Vec<Vec<(f64, f64)>>> = hs.into_iter().map(|h| h.join().unwrap()).collect();
                let mut all = vec![Vec::new(); chk.len()];
                for (t, part) in parts.into_iter().enumerate() {
                    for (k, v) in part.into_iter().enumerate() {
                        all[t + 4 * k] = v;
                    }
                }
                all
            });
            for (k, &j) in chk.iter().enumerate() {
                for c in 0..a.ncb {
                    let (rr, ri) = res[k][c];
                    let g = yh[c * m + j];
                    let e = ((g.re as f64 - rr).powi(2) + (g.im as f64 - ri).powi(2)).sqrt();
                    e2 += e * e;
                    r2 += rr * rr + ri * ri;
                    emax = emax.max(e);
                }
            }
            let rel = (e2 / r2).sqrt();
            let rms = (r2 / (chk.len() * a.ncb) as f64).sqrt();
            let hw = c.tt + W - 1;
            let stage_bytes = match c.halo { 1 => sr.nitems as f64 * (hw * hw * hw) as f64 * nch as f64 * 8.0, 3 => sr.nitems as f64 * (hw * (c.ty + W - 1) * (c.tz + W - 1)) as f64 * nch as f64 * 8.0, 4 => sr.items_h[..sr.nitems].iter().map(|it| ((it.ufirst >> 16) - (it.ufirst & 0xffff) + W as i32) as f64 * (2 * (c.ty + 4) * hw) as f64).sum::<f64>() * (16 * nch.div_ceil(16)) as f64 * 8.0, _ => 0.0 };
            let l1_bytes = macs * 8.0;
            println!("  BEST {design}: {ms:.2} ms {:?} items {} | {:.0} G cMAC/s, {:.1} % of FP32 FMA peak ({} FMA/MAC) | value loads {:.1} TB/s (L1/smem), halo staging {:.0} GB/s | rel err {rel:.2e} (max {:.2e} of rms {:.2e}) on {} samples",
                c, sr.nitems, macs / (ms as f64 * 1e-3) * 1e-9, 100.0 * macs * fma_per_mac / (ms as f64 * 1e-3) / peak_fma, fma_per_mac, l1_bytes / (ms as f64 * 1e-3) * 1e-12, stage_bytes / (ms as f64 * 1e-3) * 1e-9, emax, rms, chk.len());
            let r = serde_json::json!({"design": design, "dir": "fwd", "ms": ms, "cfg": format!("{c:?}"), "items": sr.nitems, "gmac_s": macs / (ms as f64 * 1e-3) * 1e-9,
                "peak_frac": macs * fma_per_mac / (ms as f64 * 1e-3) / peak_fma, "l1_tb_s": l1_bytes / (ms as f64 * 1e-3) * 1e-12, "stage_gb_s": stage_bytes / (ms as f64 * 1e-3) * 1e-9, "rel_err": rel,
                "pg_defs": pg_defs().join(","), "regs": regs, "local_b": lmem, "blocks_sm": bps, "work": wk.map(|w| w.to_vec())});
            let mut b = base_rec.clone();
            b.as_object_mut().unwrap().extend(r.as_object().unwrap().clone());
            json(b);
        }
        return;
    }

    if a.mode == "adj" && a.designs.iter().any(|d| d == "ra") {
        // design 7 `ra` (§15): row-blocked output adjoint, autotuned; f64 point check + adjointness against the f64
        // forward on a sample subset
        let yh: Vec<C32> = (0..a.ncb * m).map(|_| C32::new((u() * 2.0) as f32, (u() * 2.0) as f32)).collect();
        st.memcpy_htod(&yh, &mut y).unwrap();
        let pos_d = st.clone_htod(&p.sc.iter().flat_map(|s| [s[0], s[1], s[2], 0.0]).collect::<Vec<f32>>()).unwrap();
        let mut dco_h = vec![0f32; 6 * m];
        for j in 0..m {
            for q in 0..6 {
                dco_h[q * m + j] = p.d[j][q];
            }
        }
        let dco_d = st.clone_htod(&dco_h).unwrap();
        drop(dco_h);
        let acf_d = st.clone_htod(&p.a).unwrap();
        let uw_d = st.clone_htod(&if p.u.is_empty() { vec![0f32; 4] } else { p.u.iter().flat_map(|v| *v).collect::<Vec<f32>>() }).unwrap();
        let mut cs: Vec<RaCfg> = ra_candidates(&a, nf, smax_opt);
        if let Some(fx) = &a.fixed {
            cs = fx.split(';').filter(|x| !x.trim().is_empty()).map(|x| { let v: Vec<usize> = x.split(',').map(|t| t.trim().parse().unwrap()).collect(); RaCfg { tx: v[0], ty: v[1], tz: v[2], nxg: v[3], cb: v[4], chk: v[5], lmax: v[6] } }).collect();
        }
        println!("  ra: {} candidates", cs.len());
        let pdefs = pg_defs();
        let ra_src = |c: &RaCfg| -> (String, String) {
            let s = format!("{}#define RA 1\n#define TX {}\n#define TY {}\n#define TZ {}\n#define NXG {}\n#define CB {}\n#define CHK {}\n#define W {W}\n#define BETA {beta:.17e}\n#define CORR {}\n#define NCB {}\n#define NSEG {}\n#define TT 4\n#define CH 1\n#define TPS 1\n#define NT 128\n#define SMAX 1\n#define HALO 0\n{SRC}",
                pdefs.iter().map(|d| format!("#define {d} 1\n")).collect::<String>(), c.tx, c.ty, c.tz, c.nxg, c.cb, c.chk, a.order, a.ncb, a.nseg);
            (format!("proto_gather_ra_o{}_c{}_l{}_{}x{}x{}_g{}_b{}_k{}{}", a.order, a.ncb, a.nseg, c.tx, c.ty, c.tz, c.nxg, c.cb, c.chk, pdefs.iter().map(|d| format!("_{}", d.to_lowercase())).collect::<String>()), s)
        };
        let compile = |cs: &[RaCfg]| -> Vec<Option<CudaFunction>> {
            let srcs: Vec<(String, String)> = cs.iter().map(&ra_src).collect();
            let cubins: Vec<_> = std::thread::scope(|sc| {
                let hs: Vec<_> = (0..4).map(|t| { let srcs = &srcs; let arch = dev.arch.clone(); sc.spawn(move || (t..srcs.len()).step_by(4).map(|i| (i, compile_cubin(&srcs[i].0, &srcs[i].1, &arch, &CompileSettings::default()))).collect::<Vec<_>>()) }).collect();
                let mut all: Vec<_> = hs.into_iter().flat_map(|h| h.join().unwrap()).collect();
                all.sort_by_key(|x| x.0);
                all.into_iter().map(|x| x.1).collect::<Vec<_>>()
            });
            cs.iter().zip(cubins).map(|(c, cb)| {
                let cb = match cb { Ok(b) => b, Err(e) => { println!("    compile {c:?}: {}", e.to_string().lines().take(3).collect::<Vec<_>>().join(" ")); return None; } };
                let md = dev.ctx.load_module(cudarc::nvrtc::Ptx::from_binary(cb)).ok()?;
                let f = md.load_function("pg_adj_ra").ok()?;
                if let Err(e) = f.set_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ra_smem(c, &a) as i32) { println!("    smem {c:?}: {e:?}"); return None; }
                Some(f)
            }).collect()
        };
        // tile lists per (tile shape, lmax), built on demand; only one kept on the device at a time
        let mut lists: Option<([usize; 4], CudaSlice<u32>, CudaSlice<RItem>, Vec<RItem>, usize)> = None;
        let get_lists = |lists: &mut Option<([usize; 4], CudaSlice<u32>, CudaSlice<RItem>, Vec<RItem>, usize)>, c: &RaCfg| {
            let key = [c.tx, c.ty, c.tz, c.lmax];
            if lists.as_ref().map_or(true, |l| l.0 != key) {
                *lists = None;
                let tb = Instant::now();
                let (tl, items) = ra_lists(&p, [c.tx, c.ty, c.tz], c.lmax);
                let nsplit = items.iter().filter(|i| i.split != 0).count();
                println!("    tile lists {:?} lmax {}: {} entries ({:.1} per sample), {} items ({nsplit} split), {:.1} s", [c.tx, c.ty, c.tz], c.lmax, tl.len(), tl.len() as f64 / m as f64, items.len(), tb.elapsed().as_secs_f64());
                let n = items.len();
                *lists = Some((key, st.clone_htod(&tl).unwrap(), st.clone_htod(&items).unwrap(), items, n));
            }
        };
        let launch = |f: &CudaFunction, c: &RaCfg, tl: &CudaSlice<u32>, it: &CudaSlice<RItem>, n: usize, yv: &CudaSlice<C32>, g: &mut CudaSlice<C32>| {
            let cfg = LaunchConfig { grid_dim: (n as u32, nch.div_ceil(ra_cpw(c)) as u32, 1), block_dim: (32 * c.tz as u32, 1, 1), shared_mem_bytes: ra_smem(c, &a) as u32 };
            let mut lb = st.launch_builder(f);
            lb.arg(&pos_d).arg(&dco_d).arg(&acf_d).arg(tl).arg(it).arg(yv).arg(g).arg(&mi).arg(&dims);
            if hi {
                lb.arg(&uw_d).arg(&htab_d);
            }
            unsafe { lb.launch(cfg).unwrap() };
        };
        let tm = time(&mut || st.memset_zeros(&mut grid).unwrap(), 3, f32::INFINITY);
        println!("  grid memset {tm:.2} ms (not included in the kernel times below)");
        let fs = compile(&cs);
        let mut best = (f32::INFINITY, cs[0]);
        let mut order_ix: Vec<usize> = (0..cs.len()).collect();
        order_ix.sort_by_key(|&i| (cs[i].tx, cs[i].ty, cs[i].tz, cs[i].lmax));
        for &i in &order_ix {
            let (c, Some(f)) = (&cs[i], &fs[i]) else { continue };
            get_lists(&mut lists, c);
            let l = lists.as_ref().unwrap();
            let mut run = || { st.memset_zeros(&mut grid).unwrap(); launch(f, c, &l.1, &l.2, l.4, &y, &mut grid); };
            let ms = time(&mut run, if a.each { a.reps } else { 2 }, if best.0.is_finite() && !a.each { 3.0 * (best.0 + tm) } else { f32::INFINITY }) - tm;
            let regs = f.num_regs().unwrap_or(-1);
            let lmem = f.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES).unwrap_or(-1);
            let bps = f.occupancy_max_active_blocks_per_multiprocessor(32 * c.tz as u32, ra_smem(c, &a), None).map(|v| v as i64).unwrap_or(-1);
            println!("    ra {:?} smem {:>6} regs {regs:>3} local {lmem} blocks/SM {bps}: {ms:8.2} ms ({:.1} %)", c, ra_smem(c, &a), 100.0 * macs * fma_per_mac / (ms as f64 * 1e-3) / peak_fma);
            if ms < best.0 {
                best = (ms, *c);
            }
        }
        // checks (and the final timing) of the best candidate, or of every candidate with --each
        let targets: Vec<usize> = if a.each { (0..cs.len()).filter(|&i| fs[i].is_some()).collect() } else { vec![cs.iter().position(|x| format!("{x:?}") == format!("{:?}", best.1)).unwrap()] };
        for &i in &targets {
        let c = cs[i];
        let f = fs[i].clone().unwrap();
        get_lists(&mut lists, &c);
        let l = lists.as_ref().unwrap();
        let ms = time(&mut || { st.memset_zeros(&mut grid).unwrap(); launch(&f, &c, &l.1, &l.2, l.4, &y, &mut grid); }, a.reps, f32::INFINITY) - tm;
        let regs = f.num_regs().unwrap_or(-1);
        let lmem = f.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES).unwrap_or(-1);
        let bps = f.occupancy_max_active_blocks_per_multiprocessor(32 * c.tz as u32, ra_smem(&c, &a), None).map(|v| v as i64).unwrap_or(-1);
        // (a) f64 adjoint at grid points near random samples, every channel
        let npt = 48;
        let pts: Vec<[usize; 3]> = (0..npt).map(|k| { let j = (k * 7919 * 131 + 17) % m; let s = p.sc[j]; [(s[0] as usize + k % 3) % nf, (s[1] as usize + k % 5) % nf, s[2] as usize % nf] }).collect();
        let mut refv = vec![(0.0f64, 0.0f64); npt * nch];
        for j in 0..m {
            let c0: [usize; 3] = [0, 1, 2].map(|q| (((p.sc[j][q] - W as f32 * 0.5).ceil() as i64).rem_euclid(nf as i64)) as usize);
            let mut wj: Option<Vec<(f64, f64)>> = None;
            for (k, q) in pts.iter().enumerate() {
                let t: [usize; 3] = [0, 1, 2].map(|x| (q[x] + nf - c0[x]) % nf);
                if t.iter().all(|&v| v < W) {
                    let w = wj.get_or_insert_with(|| { let dd = p.d[j].map(|v| v as f64); let hcj = if hi { Some(hc_of(j)) } else { None }; weights(p.sc[j], &dd, a.order, beta, nf, hcj.as_ref()).1 });
                    let wv = w[(t[2] * W + t[1]) * W + t[0]];
                    for ch in 0..nch {
                        let (cc, ll) = (ch / a.nseg, ch % a.nseg);
                        let av = p.a[j * a.nseg + ll];
                        let yv = yh[cc * m + j];
                        let (ur, ui) = (av.re as f64 * yv.re as f64 + av.im as f64 * yv.im as f64, av.re as f64 * yv.im as f64 - av.im as f64 * yv.re as f64);
                        refv[k * nch + ch].0 += wv.0 * ur + wv.1 * ui;
                        refv[k * nch + ch].1 += wv.0 * ui - wv.1 * ur;
                    }
                }
            }
        }
        let (mut e2, mut r2) = (0.0, 0.0);
        for (k, q) in pts.iter().enumerate() {
            let o = ((q[2] * nf + q[1]) * nf + q[0]) * nch;
            let mut h = vec![C32::new(0.0, 0.0); nch];
            st.memcpy_dtoh(&grid.slice(o..o + nch), &mut h).unwrap();
            for ch in 0..nch {
                let (rr, ri) = refv[k * nch + ch];
                e2 += (h[ch].re as f64 - rr).powi(2) + (h[ch].im as f64 - ri).powi(2);
                r2 += rr * rr + ri * ri;
            }
        }
        let rel = (e2 / r2).sqrt();
        // (b) adjointness: y supported on a strided subset S; <A x, y> (f64 forward of the hash grid x on S) against
        // <x, A^H y> (this kernel, summed over the whole grid)
        let step = (m / a.check.max(1)).max(1);
        let sub: Vec<usize> = (0..m).step_by(step).collect();
        let mut ys = vec![C32::new(0.0, 0.0); a.ncb * m];
        for &j in &sub {
            for cc in 0..a.ncb {
                ys[cc * m + j] = yh[cc * m + j];
            }
        }
        let ys_d = st.clone_htod(&ys).unwrap();
        st.memset_zeros(&mut grid).unwrap();
        launch(&f, &c, &l.1, &l.2, l.4, &ys_d, &mut grid);
        st.synchronize().unwrap();
        let s1: (f64, f64) = std::thread::scope(|scp| {
            let hs: Vec<_> = (0..4).map(|t| { let (sub, p, a2, ys) = (&sub, &p, &a, &ys); scp.spawn(move || {
                let mut s = (0.0f64, 0.0f64);
                for &j in sub.iter().skip(t).step_by(4) {
                    let dd = p.d[j].map(|v| v as f64);
                    let hcj = if a2.order >= 3 { Some(hc_of(j)) } else { None };
                    let (c0, w) = weights(p.sc[j], &dd, a2.order, beta, nf, hcj.as_ref());
                    let mut out = vec![(0.0, 0.0); a2.ncb];
                    for tz in 0..W { for ty in 0..W { for tx in 0..W {
                        let (gx, gy, gz) = ((c0[0] + tx) % nf, (c0[1] + ty) % nf, (c0[2] + tz) % nf);
                        let wv = w[(tz * W + ty) * W + tx];
                        let base = ((gz * nf + gy) * nf + gx) * nch;
                        for ch in 0..nch {
                            let gv = gval((base + ch) as u64);
                            let (cc, l) = (ch / a2.nseg, ch % a2.nseg);
                            let av = p.a[j * a2.nseg + l];
                            let (vr, vi) = (wv.0 * gv.0 - wv.1 * gv.1, wv.0 * gv.1 + wv.1 * gv.0);
                            let (ar, ai) = (av.re as f64, av.im as f64);
                            out[cc].0 += ar * vr - ai * vi;
                            out[cc].1 += ar * vi + ai * vr;
                        }
                    }}}
                    for cc in 0..a2.ncb {
                        let yv = ys[cc * m + j];
                        // conj(Ax) y
                        s.0 += out[cc].0 * yv.re as f64 + out[cc].1 * yv.im as f64;
                        s.1 += out[cc].0 * yv.im as f64 - out[cc].1 * yv.re as f64;
                    }
                }
                s
            }) }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
        });
        let mut s2 = (0.0f64, 0.0f64);
        let slab = 1usize << 25;
        let mut hbuf = vec![C32::new(0.0, 0.0); slab];
        let mut o = 0usize;
        while o < ng {
            let n = slab.min(ng - o);
            st.memcpy_dtoh(&grid.slice(o..o + n), &mut hbuf[..n]).unwrap();
            for (k, v) in hbuf[..n].iter().enumerate() {
                let x = gval((o + k) as u64);
                s2.0 += x.0 * v.re as f64 + x.1 * v.im as f64;
                s2.1 += x.0 * v.im as f64 - x.1 * v.re as f64;
            }
            o += n;
        }
        let adj = ((s1.0 - s2.0).powi(2) + (s1.1 - s2.1).powi(2)).sqrt() / (s1.0 * s1.0 + s1.1 * s1.1).sqrt();
        println!("  RES ra: regs {regs}, local {lmem} B, smem {} B, blocks/SM {bps}, warps/SM {} of 48", ra_smem(&c, &a), bps * c.tz as i64);
        println!("  {} ra: {ms:.2} ms (kernel; + memset {tm:.2} ms) {c:?} items {} | {:.0} G cMAC/s, {:.1} % of FP32 FMA peak ({fma_per_mac} FMA/MAC) | rel err {rel:.2e} at {npt} points x {nch} ch | adjointness |<Ax,y> - <x,A^H y>| / |<Ax,y>| = {adj:.2e} on {} samples",
            if a.each { "CHECK" } else { "BEST" }, l.4, macs / (ms as f64 * 1e-3) * 1e-9, 100.0 * macs * fma_per_mac / (ms as f64 * 1e-3) / peak_fma, sub.len());
        let r = serde_json::json!({"design": "ra", "dir": "adj", "ms": ms, "memset_ms": tm, "cfg": format!("{c:?}"), "items": l.4, "checked_each": a.each, "gmac_s": macs / (ms as f64 * 1e-3) * 1e-9,
            "peak_frac": macs * fma_per_mac / (ms as f64 * 1e-3) / peak_fma, "rel_err": rel, "adjointness": adj, "regs": regs, "local_b": lmem, "blocks_sm": bps, "pg_defs": pg_defs().join(",")});
        let mut b = base_rec.clone();
        b.as_object_mut().unwrap().extend(r.as_object().unwrap().clone());
        json(b);
        }
        return;
    }

    if a.mode == "adj" {
        let yh: Vec<C32> = (0..a.ncb * m).map(|_| C32::new((u() * 2.0) as f32, (u() * 2.0) as f32)).collect();
        st.memcpy_htod(&yh, &mut y).unwrap();
        let mut cs = candidates(&a, "smspread", smax_opt);
        if let Some(fx) = &a.fixed {
            let v: Vec<usize> = fx.split(',').map(|s| s.parse().unwrap()).collect();
            cs = vec![Cfg { halo: 2, tt: v[0], ch: v[1], tps: v[2], nt: v[3], smax: v[4], padc: v[5], sb: v[6], rx: v[7], ty: 0, tz: 0, acfg: 0, pfs: 0, minb: 0, nwc: 0, nws: 0, vxpad: 0, vt: 0, shr: 0, pf: 0, s2: 0, pt: 0, carve: -1, vtp: 0 }];
        }
        let fs = compile_all(&cs);
        let mut sorted: Vec<Sorted> = Vec::new();
        let mut sbufs: Vec<Arc<SBufs>> = Vec::new();
        let mut best = (f32::INFINITY, cs[0]);
        for (c, f) in cs.iter().zip(fs.iter()) {
            let Some(f) = f else { continue };
            let so = match sorted.iter().position(|s| s.td == tdims(c) && s.smax == c.smax && s.sb == c.sb && s.rx == c.rx) { Some(i) => i, None => { sorted.push(sort(&st, &p, tdims(c), c.smax, c.sb, c.rx, &mut sbufs)); sorted.len() - 1 } };
            let sr = &sorted[so];
            let cfg = LaunchConfig { grid_dim: (sr.nitems as u32, 1, 1), block_dim: (c.nt as u32, 1, 1), shared_mem_bytes: smem(c, a.order, a.nseg) as u32 };
            let mut run = || {
                st.memset_zeros(&mut grid).unwrap();
                unsafe { st.launch_builder(f).arg(&sr.b.pos).arg(&sr.b.dco).arg(&sr.b.acf).arg(&sr.b.perm).arg(&sr.items).arg(&y).arg(&mut grid).arg(&mi).arg(&dims).launch(cfg).unwrap() };
            };
            let ms = time(&mut run, 2, if best.0.is_finite() { 2.0 * best.0 } else { f32::INFINITY });
            println!("    smspread tt {:>2} ch {:>3} tps {:>2} nt {:>3} smax {:>3} smem {:>6}: {ms:8.2} ms (incl. grid memset)", c.tt, c.ch, c.tps, c.nt, c.smax, smem(c, a.order, a.nseg));
            if ms < best.0 {
                best = (ms, *c);
            }
        }
        let c = best.1;
        let i = cs.iter().position(|x| x.tt == c.tt && x.ch == c.ch && x.tps == c.tps && x.nt == c.nt && x.smax == c.smax && x.padc == c.padc).unwrap();
        let f = fs[i].clone().unwrap();
        let sr = sorted.iter().find(|s| s.td == tdims(&c) && s.smax == c.smax && s.sb == c.sb && s.rx == c.rx).unwrap();
        let cfg = LaunchConfig { grid_dim: (sr.nitems as u32, 1, 1), block_dim: (c.nt as u32, 1, 1), shared_mem_bytes: smem(&c, a.order, a.nseg) as u32 };
        let tm = time(&mut || st.memset_zeros(&mut grid).unwrap(), a.reps, f32::INFINITY);
        let mut run = || {
            st.memset_zeros(&mut grid).unwrap();
            unsafe { st.launch_builder(&f).arg(&sr.b.pos).arg(&sr.b.dco).arg(&sr.b.acf).arg(&sr.b.perm).arg(&sr.items).arg(&y).arg(&mut grid).arg(&mi).arg(&dims).launch(cfg).unwrap() };
        };
        let ms = time(&mut run, a.reps, f32::INFINITY);
        // check at random grid points: G[p][ch] = Σ_j conj(w_j(p)) conj(a_jl) y[c][j]
        let npt = 12;
        let pts: Vec<[usize; 3]> = (0..npt).map(|k| { let j = (k * 7919 * 131) % m; let s = p.sc[j]; [s[0] as usize % nf, s[1] as usize % nf, s[2] as usize % nf] }).collect();
        let mut refv = vec![(0.0f64, 0.0f64); npt * nch];
        for j in 0..m {
            let c0: Vec<usize> = (0..3).map(|q| (((p.sc[j][q] - W as f32 * 0.5).ceil() as i64).rem_euclid(nf as i64)) as usize).collect();
            for (k, q) in pts.iter().enumerate() {
                let t: Vec<usize> = (0..3).map(|x| (q[x] + nf - c0[x]) % nf).collect();
                if t.iter().all(|&v| v < W) {
                    let dd = p.d[j].map(|v| v as f64);
                    let (_, w) = weights(p.sc[j], &dd, a.order, beta, nf, None);
                    let wv = w[(t[2] * W + t[1]) * W + t[0]];
                    for ch in 0..nch {
                        let (cc, l) = (ch / a.nseg, ch % a.nseg);
                        let av = p.a[j * a.nseg + l];
                        let yv = yh[cc * m + j];
                        // conj(w) conj(a) y
                        let (ur, ui) = (av.re as f64 * yv.re as f64 + av.im as f64 * yv.im as f64, av.re as f64 * yv.im as f64 - av.im as f64 * yv.re as f64);
                        refv[k * nch + ch].0 += wv.0 * ur + wv.1 * ui;
                        refv[k * nch + ch].1 += wv.0 * ui - wv.1 * ur;
                    }
                }
            }
        }
        let (mut e2, mut r2) = (0.0, 0.0);
        for (k, q) in pts.iter().enumerate() {
            let o = ((q[2] * nf + q[1]) * nf + q[0]) * nch;
            let mut h = vec![C32::new(0.0, 0.0); nch];
            st.memcpy_dtoh(&grid.slice(o..o + nch), &mut h).unwrap();
            for ch in 0..nch {
                let (rr, ri) = refv[k * nch + ch];
                e2 += (h[ch].re as f64 - rr).powi(2) + (h[ch].im as f64 - ri).powi(2);
                r2 += rr * rr + ri * ri;
            }
        }
        let rel = (e2 / r2).sqrt();
        let hw = c.tt + W - 1;
        let atom = sr.nitems as f64 * (hw * hw * hw) as f64 * nch as f64;
        println!("  BEST smspread: {ms:.2} ms incl. grid memset {tm:.2} ms, {c:?} items {} | {:.0} G cMAC/s, {:.1} % of FP32 FMA peak | {:.2} G global atomic (cx) adds ({:.1} G/s) | rel err {rel:.2e} at {npt} points x {nch} ch",
            sr.nitems, macs / (ms as f64 * 1e-3) * 1e-9, 100.0 * macs * fma_per_mac / (ms as f64 * 1e-3) / peak_fma, atom * 1e-9, atom * 1e-9 / ((ms - tm) as f64 * 1e-3));
        let r = serde_json::json!({"design": "smspread", "dir": "adj", "ms": ms, "memset_ms": tm, "cfg": format!("{c:?}"), "items": sr.nitems, "gmac_s": macs / (ms as f64 * 1e-3) * 1e-9,
            "peak_frac": macs * fma_per_mac / (ms as f64 * 1e-3) / peak_fma, "atomics_g": atom * 1e-9, "rel_err": rel});
        let mut b = base_rec.clone();
        b.as_object_mut().unwrap().extend(r.as_object().unwrap().clone());
        json(b);
    }
}
