//! GPU: block-floating-point storage of the y ↔ z intermediates of the pruned channel FFT (`PrunedChanFft3::set_store`,
//! research/bfp_storage.md §7 step 1) against the f32 in-place passes.
//!
//! Per case (grid, channels, image block, k support, slab size, mantissa bits; forward and inverse):
//! * writer exact: the BFP buffer equals the host codec (`hasty_bfp`) applied to the f32 intermediate of the
//!   block-pruned plan, bit for bit, block by block (lines outside the pass's line range stored as zeros, blocks the
//!   support skips left with exponent byte 0, partial tiles at the end of every row);
//! * reader exact: the BFP plan's output equals the f32 passes applied to the decoded buffer, bit for bit, on the
//!   cells the transform defines (the support forward, the image block inverse);
//! * every block a reader uses was written or cleared: the buffer is filled with NaN blocks before every call;
//! * slabs (1 and 3 planes) give the same buffer and output bit for bit as whole-grid passes;
//! * error: the buffer's rounding error and the transform's error against the f32 plan, as c · 2^−b.
//!
//!   cargo test --release -p hasty-fft --test bfp_store -- --nocapture --test-threads 1

use hasty_bfp::Bfp;
use hasty_cuda::{Device, C32};
use hasty_fft::{KSupport, PrunedChanFft3, Store};

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32
    }
}

fn in_wrapped(i: usize, n: usize, (h, t): (usize, usize)) -> bool {
    i < h || i >= n - t
}

/// k-support of a ball of radius `r` (index units) around k = 0 on a wrapped grid `n = [n0, n1, n2]` (x fastest), or
/// of the cube |k_a| ≤ r_a when `ball` is false (as tests/chanfft.rs)
fn ksup(n: [usize; 3], r: [f64; 3], ball: bool) -> KSupport {
    let c = |i: usize, m: usize| {
        let i = i as f64;
        if i > m as f64 / 2.0 { i - m as f64 } else { i }
    };
    if ball {
        let q = |x: usize, y: usize| 1.0 - (c(x, n[0]) / r[0]).powi(2) - (c(y, n[1]) / r[1]).powi(2);
        KSupport::from_columns(n, |x, y| q(x, y) >= 0.0, |x, y| KSupport::hull(n[2], |z| c(z, n[2]).abs() <= r[2] * q(x, y).max(0.0).sqrt()))
    } else {
        KSupport::from_masks(n, |x, y| c(x, n[0]).abs() <= r[0] && c(y, n[1]).abs() <= r[1], |z| c(z, n[2]).abs() <= r[2])
    }
}

/// a non-convex support: a ball with a ring of columns removed and some columns of a second, offset blob added (the
/// hulls of `rows` / `ycols` then cover columns outside the support)
fn ksup_holes(n: [usize; 3], r: f64) -> KSupport {
    let c = |i: usize, m: usize| {
        let i = i as f64;
        if i > m as f64 / 2.0 { i - m as f64 } else { i }
    };
    let xy = |x: usize, y: usize| {
        let (a, b) = (c(x, n[0]), c(y, n[1]));
        let rr = (a * a + b * b).sqrt();
        (rr <= r && !(rr > 0.45 * r && rr < 0.6 * r)) || ((a - 0.8 * r).abs() <= 1.0 && (b + 0.7 * r).abs() <= 1.5)
    };
    KSupport::from_columns(n, xy, |x, y| KSupport::hull(n[2], |z| c(z, n[2]).abs() <= 2.0 + ((x + 3 * y) % 4) as f64))
}

fn same(x: &[C32], y: &[C32], sel: &dyn Fn(usize) -> bool) -> (usize, usize) {
    let mut d = 0;
    let mut k = 0;
    for q in (0..x.len()).filter(|&q| sel(q)) {
        k += 1;
        if x[q].re.to_bits() != y[q].re.to_bits() || x[q].im.to_bits() != y[q].im.to_bits() {
            d += 1;
        }
    }
    (d, k)
}

fn rel(x: &[C32], y: &[C32], sel: &dyn Fn(usize) -> bool) -> f64 {
    let (mut d, mut s) = (0.0f64, 0.0f64);
    for q in (0..x.len()).filter(|&q| sel(q)) {
        d += ((x[q].re - y[q].re) as f64).powi(2) + ((x[q].im - y[q].im) as f64).powi(2);
        s += (y[q].re as f64).powi(2) + (y[q].im as f64).powi(2);
    }
    (d / s.max(1e-300)).sqrt()
}

/// Geometry of the BFP intermediate: planes zc of the image block's z range × rows y × lines la of a plane in
/// 16-line tiles; and which lines / blocks a pass writes.
struct Geo {
    n: [usize; 3],
    keep: [(usize, usize); 3],
    lp: usize,
    nt: usize,
    nz: usize,
}

impl Geo {
    fn new(n: [usize; 3], nch: usize, keep: [(usize, usize); 3]) -> Self {
        let lp = n[2] * nch;
        Self { n, keep, lp, nt: lp.div_ceil(16), nz: keep[0].0 + keep[0].1 }
    }
    /// grid z of compact plane zc
    fn z(&self, zc: usize) -> usize {
        if zc < self.keep[0].0 { zc } else { zc + self.n[0] - self.nz }
    }
    /// flat grid index of (z, y, la)
    fn g(&self, z: usize, y: usize, la: usize) -> usize {
        (z * self.n[1] + y) * self.lp + la
    }
    fn blk(&self, zc: usize, y: usize, tile: usize) -> usize {
        (zc * self.n[1] + y) * self.nt + tile
    }
    /// tiles of a line range (h, t) of a plane rounded out to whole tiles
    fn tile_in(&self, tile: usize, (h, t): (usize, usize)) -> bool {
        let th = h.div_ceil(16);
        let ts = if t > 0 { (self.lp - t) / 16 } else { self.nt };
        tile < th || tile >= ts.max(th)
    }
    fn lane_in(&self, la: usize, (h, t): (usize, usize)) -> bool {
        la < self.lp && in_wrapped(la, self.lp, (h, t))
    }
}

/// expected BFP buffer: block (zc, y, tile) is written iff `written(zc, y, tile)`, its lanes hold `val(zc, y, la)`
/// where `active(y, la)`, zeros elsewhere; unwritten blocks have exponent byte 0 (k-support plans) and are not compared
fn expect(geo: &Geo, bits: u32, val: &dyn Fn(usize, usize, usize) -> C32, active: &dyn Fn(usize, usize) -> bool, written: &dyn Fn(usize, usize) -> bool) -> (Vec<u32>, Vec<u8>, Vec<bool>) {
    let f = Bfp::new(bits);
    let b = bits as usize;
    let nb = geo.nz * geo.n[1] * geo.nt;
    let mut mant = vec![0u32; nb * b];
    let mut ex = vec![0u8; nb];
    let mut wr = vec![false; nb];
    for zc in 0..geo.nz {
        for y in 0..geo.n[1] {
            for tile in 0..geo.nt {
                if !written(y, tile) {
                    continue;
                }
                let mut x = [0f32; 32];
                for k in 0..16 {
                    let la = tile * 16 + k;
                    if la < geo.lp && active(y, la) {
                        let v = val(zc, y, la);
                        x[2 * k] = v.re;
                        x[2 * k + 1] = v.im;
                    }
                }
                let k = geo.blk(zc, y, tile);
                ex[k] = f.encode_block(&x, &mut mant[k * b..(k + 1) * b]);
                wr[k] = true;
            }
        }
    }
    (mant, ex, wr)
}

/// compare the device buffer with the expected one; returns (differing written blocks, written blocks, unwritten
/// blocks without exponent 0)
fn cmp_buf(bits: u32, dev: &(Vec<u32>, Vec<u8>, usize), exp: &(Vec<u32>, Vec<u8>, Vec<bool>)) -> (usize, usize, usize) {
    let b = bits as usize;
    let (mut bad, mut n, mut stale) = (0, 0, 0);
    for k in 0..exp.2.len() {
        if exp.2[k] {
            n += 1;
            if dev.1[k] != exp.1[k] || dev.0[k * b..(k + 1) * b] != exp.0[k * b..(k + 1) * b] {
                bad += 1;
            }
        } else if dev.1[k] != 0 {
            stale += 1;
        }
    }
    (bad, n, stale)
}

/// decoded device buffer as a function of (zc, y, la): zeros for blocks with exponent byte 0
fn decoded(geo: &Geo, bits: u32, dev: &(Vec<u32>, Vec<u8>, usize)) -> Vec<C32> {
    let f = Bfp::new(bits);
    let b = bits as usize;
    let nb = dev.1.len();
    let mut out = vec![C32::new(0.0, 0.0); nb * 16];
    let mut x = [0f32; 32];
    for k in 0..nb {
        if dev.1[k] == 0 {
            continue;
        }
        f.decode_block(&dev.0[k * b..(k + 1) * b], dev.1[k], &mut x);
        for j in 0..16 {
            out[k * 16 + j] = C32::new(x[2 * j], x[2 * j + 1]);
        }
    }
    let _ = geo;
    out
}

struct Out {
    /// buffer rounding error / 2^−b (forward, inverse) and transform error against the f32 plan / 2^−b
    c_buf: [f64; 2],
    c_out: [f64; 2],
}

/// one case at `bits`; `slabs`: plane counts to check against whole-grid passes
fn case(dev: &Device, name: &str, n: [usize; 3], nch: usize, keep: [(usize, usize); 3], ks: Option<&KSupport>, bits: u32, slabs: &[usize]) -> Out {
    let st = dev.default_stream();
    let geo = Geo::new(n, nch, keep);
    let len = n.iter().product::<usize>() * nch;
    let q2 = (2.0f64).powi(-(bits as i32));
    // f32 plans: the same pruning (error reference) and block-only (values of every line of the intermediates)
    let a = PrunedChanFft3::<f32>::new_ksupport(dev, n, nch, keep, 1.0, ks).unwrap();
    let r = PrunedChanFft3::<f32>::new(dev, n, nch, keep, 1.0).unwrap();
    let mut b = PrunedChanFft3::<f32>::new_ksupport(dev, n, nch, keep, 1.0, ks).unwrap();
    // the dense layout (separate buffer: compared with the host codec block by block); the in-place layout (mantissas in
    // the grid buffer) must give the same results bit for bit
    b.set_store_opts(dev, Store::Bfp(bits), true).unwrap();
    assert_eq!((b.store(), b.store_inplace()), (Store::Bfp(bits), false));
    let mut c = PrunedChanFft3::<f32>::new_ksupport(dev, n, nch, keep, 1.0, ks).unwrap();
    c.set_store(dev, Store::Bfp(bits)).unwrap();
    assert_eq!(c.store_inplace(), (n[2] * nch) % 16 == 0, "in place exactly when a plane's lines are whole tiles");
    let idx = |q: usize| -> (usize, usize, usize) {
        let p = q / nch;
        (p / (n[1] * n[2]), (p / n[2]) % n[1], p % n[2])
    };
    let blk = |q: usize| {
        let (z, y, x) = idx(q);
        in_wrapped(z, n[0], keep[0]) && in_wrapped(y, n[1], keep[1]) && in_wrapped(x, n[2], keep[2])
    };
    let sup = |q: usize| match ks {
        None => true,
        Some(k) => {
            let (z, y, x) = idx(q);
            let zr = k.zcols[y * n[2] + x];
            zr != (0, 0) && in_wrapped(z, n[0], zr)
        }
    };
    let mut rng = Rng(0x9e3779b97f4a7c15 ^ (bits as u64 * 7919) ^ (nch as u64 * 31) ^ (n[1] as u64));
    let img: Vec<C32> = (0..len).map(|q| if blk(q) { C32::new(rng.u(), rng.u()) } else { C32::new(1e3, -7e2) }).collect();
    let spec: Vec<C32> = (0..len).map(|q| if sup(q) { C32::new(rng.u(), rng.u()) } else { C32::new(0.0, 0.0) }).collect();
    // line ranges and the written-block rule of the passes (k support: the y pass's lines are the kx range
    // proj[0] · nch; the forward writes ky entries of proj[1] that some line of the tile writes (ycols); the inverse z
    // pass writes the tiles of the row's kx range rows[ky] · nch, all compact z planes)
    let chr = |(h, t): (usize, usize)| (h * nch, t * nch);
    let ylines = ks.map_or((geo.lp, 0), |k| chr(k.proj[0]));
    let fwd_written = |y: usize, tile: usize| -> bool {
        match ks {
            None => true,
            Some(k) => {
                geo.tile_in(tile, ylines)
                    && in_wrapped(y, n[1], k.proj[1])
                    && (0..16).map(|j| tile * 16 + j).any(|la| geo.lane_in(la, ylines) && in_wrapped(y, n[1], k.ycols[la / nch]))
            }
        }
    };
    let fwd_active = |_y: usize, la: usize| geo.lane_in(la, ylines);
    let inv_rows = |y: usize| ks.map_or((geo.lp, 0), |k| chr(k.rows[y]));
    let inv_written = |y: usize, tile: usize| -> bool { ks.is_none() || (inv_rows(y) != (0, 0) && geo.tile_in(tile, inv_rows(y))) };
    let inv_active = |y: usize, la: usize| geo.lane_in(la, inv_rows(y));

    // ---- forward ----
    // f32 intermediate of every line (block-only plan, x and y passes), the f32 reference output
    let mut ri = st.clone_htod(&img).unwrap();
    r.debug_f32_passes(&st, &mut ri, true, &[0, 1]).unwrap();
    let ri = st.clone_dtoh(&ri).unwrap();
    let mut ra = st.clone_htod(&img).unwrap();
    a.forward(&st, &mut ra).unwrap();
    let ra = st.clone_dtoh(&ra).unwrap();
    let run_b = |b: &mut PrunedChanFft3<f32>, x: &[C32], fwd: bool| {
        b.debug_poison_store(&st).unwrap();
        let mut d = st.clone_htod(x).unwrap();
        if fwd { b.forward(&st, &mut d).unwrap() } else { b.inverse(&st, &mut d).unwrap() }
        (st.clone_dtoh(&d).unwrap(), b.debug_store_buffer(&st).unwrap().unwrap())
    };
    let (rb, bufb) = run_b(&mut b, &img, true);
    let val_f = |zc: usize, y: usize, la: usize| ri[geo.g(geo.z(zc), y, la)];
    let exp = expect(&geo, bits, &val_f, &fwd_active, &fwd_written);
    let (bad, nw, stale) = cmp_buf(bits, &bufb, &exp);
    assert_eq!(bufb.2, geo.nt);
    assert!(bad == 0 && stale == 0, "{name} bfp{bits} forward: BFP buffer differs from the host codec in {bad} of {nw} blocks ({stale} unwritten blocks not cleared)");
    // buffer rounding error over the active lines of written blocks
    let dec = decoded(&geo, bits, &bufb);
    let (mut d2, mut s2) = (0.0f64, 0.0f64);
    for zc in 0..geo.nz {
        for y in 0..n[1] {
            for la in 0..geo.lp {
                if fwd_written(y, la / 16) && fwd_active(y, la) {
                    let (u, v) = (dec[geo.blk(zc, y, la / 16) * 16 + la % 16], val_f(zc, y, la));
                    d2 += ((u.re - v.re) as f64).powi(2) + ((u.im - v.im) as f64).powi(2);
                    s2 += (v.re as f64).powi(2) + (v.im as f64).powi(2);
                }
            }
        }
    }
    let cbf = (d2 / s2).sqrt() / q2;
    // reader exact: the block-only z pass on the decoded buffer equals the BFP output on the support
    let mut zin = ri.clone();
    for zc in 0..geo.nz {
        for y in 0..n[1] {
            for la in 0..geo.lp {
                zin[geo.g(geo.z(zc), y, la)] = dec[geo.blk(zc, y, la / 16) * 16 + la % 16];
            }
        }
    }
    let mut dz = st.clone_htod(&zin).unwrap();
    r.debug_f32_passes(&st, &mut dz, true, &[2]).unwrap();
    let rz = st.clone_dtoh(&dz).unwrap();
    let (dfo, nfo) = same(&rb, &rz, &sup);
    assert!(dfo == 0, "{name} bfp{bits} forward: BFP z pass differs from the f32 z pass on the decoded buffer in {dfo} of {nfo} values");
    let cof = rel(&rb, &ra, &sup) / q2;

    // ---- inverse ----
    let mut ri2 = st.clone_htod(&spec).unwrap();
    r.debug_f32_passes(&st, &mut ri2, false, &[0]).unwrap();
    let ri2 = st.clone_dtoh(&ri2).unwrap();
    let mut ra2 = st.clone_htod(&spec).unwrap();
    a.inverse(&st, &mut ra2).unwrap();
    let ra2 = st.clone_dtoh(&ra2).unwrap();
    let (rb2, bufb2) = run_b(&mut b, &spec, false);
    let val_i = |zc: usize, y: usize, la: usize| ri2[geo.g(geo.z(zc), y, la)];
    let exp2 = expect(&geo, bits, &val_i, &inv_active, &inv_written);
    let (bad, nw2, stale) = cmp_buf(bits, &bufb2, &exp2);
    assert!(bad == 0 && stale == 0, "{name} bfp{bits} inverse: BFP buffer differs from the host codec in {bad} of {nw2} blocks ({stale} unwritten blocks not cleared)");
    let dec2 = decoded(&geo, bits, &bufb2);
    let (mut d2, mut s2) = (0.0f64, 0.0f64);
    let mut yin = ri2.clone();
    for zc in 0..geo.nz {
        for y in 0..n[1] {
            for la in 0..geo.lp {
                let u = dec2[geo.blk(zc, y, la / 16) * 16 + la % 16];
                yin[geo.g(geo.z(zc), y, la)] = u;
                if inv_written(y, la / 16) && inv_active(y, la) {
                    let v = val_i(zc, y, la);
                    d2 += ((u.re - v.re) as f64).powi(2) + ((u.im - v.im) as f64).powi(2);
                    s2 += (v.re as f64).powi(2) + (v.im as f64).powi(2);
                }
            }
        }
    }
    let cbi = (d2 / s2).sqrt() / q2;
    let mut dy = st.clone_htod(&yin).unwrap();
    r.debug_f32_passes(&st, &mut dy, false, &[1, 2]).unwrap();
    let ry = st.clone_dtoh(&dy).unwrap();
    let (dio, nio) = same(&rb2, &ry, &blk);
    assert!(dio == 0, "{name} bfp{bits} inverse: BFP y pass differs from the f32 y + x passes on the decoded buffer in {dio} of {nio} values");
    let coi = rel(&rb2, &ra2, &blk) / q2;

    // ---- in place: the same outputs bit for bit (whole grid and slabs) ----
    let mut inpl = String::new();
    if c.store_inplace() {
        for &dzp in [0usize].iter().chain(slabs) {
            c.set_slab(dzp);
            let (cf, _) = run_b(&mut c, &img, true);
            let (ci, _) = run_b(&mut c, &spec, false);
            let (e1, _) = same(&cf, &rb, &sup);
            let (e2, _) = same(&ci, &rb2, &blk);
            assert!(e1 + e2 == 0, "{name} bfp{bits}: in-place layout (slab {dzp}) differs from the dense one (forward {e1}, inverse {e2} values)");
        }
        c.set_slab(0);
        inpl = format!(", in place = dense ({:.0} % of the dense buffer's bytes)", 100.0 * c.store_bytes() as f64 / b.store_bytes() as f64);
    }

    // ---- slabs: identical buffers and outputs ----
    for &dzp in slabs {
        b.set_slab(dzp);
        let (sf, sbf) = run_b(&mut b, &img, true);
        let (si, sbi) = run_b(&mut b, &spec, false);
        let (e1, _) = same(&sf, &rb, &sup);
        let (e2, _) = same(&si, &rb2, &blk);
        let e3 = (0..exp.2.len()).filter(|&k| exp.2[k] && (sbf.1[k] != bufb.1[k] || sbf.0[k * bits as usize..(k + 1) * bits as usize] != bufb.0[k * bits as usize..(k + 1) * bits as usize])).count();
        let e4 = (0..exp2.2.len()).filter(|&k| exp2.2[k] && (sbi.1[k] != bufb2.1[k] || sbi.0[k * bits as usize..(k + 1) * bits as usize] != bufb2.0[k * bits as usize..(k + 1) * bits as usize])).count();
        assert!(e1 + e2 + e3 + e4 == 0, "{name} bfp{bits}: slab {dzp} differs from whole-grid passes (forward {e1}, inverse {e2} values; buffer blocks {e3} / {e4})");
        b.set_slab(0);
    }
    eprintln!(
        "{name:<34} bfp{bits}: blocks written {nw} / {nw2} of {}; buffer error {:.3} / {:.3} · 2^-b; forward (support) {:.3}, inverse (block) {:.3} · 2^-b vs f32; exact writer + reader{}{inpl}",
        exp.2.len(),
        cbf,
        cbi,
        cof,
        coi,
        if slabs.is_empty() { String::new() } else { format!(", slabs {slabs:?} identical") }
    );
    Out { c_buf: [cbf, cbi], c_out: [cof, coi] }
}

/// pruned / unpruned (k support), slabs on / off, odd channel counts, rows of the plane ending in a partial tile, an
/// in-place (single shared buffer) y and z pass, a z block without tail, a non-convex support; bits 16 / 20 / 22 /
/// 24 and the single-run odd widths 17 / 23
#[test]
fn bfp_store_matches_codec_and_f32() {
    let dev = Device::new(0).expect("CUDA device");
    // n = [n2, n1, n0]; supports in [x, y, z] order
    let cases: Vec<(&str, [usize; 3], usize, [(usize, usize); 3], Option<KSupport>)> = vec![
        ("block, 3 ch (partial tiles)", [20, 24, 30], 3, [(8, 8), (10, 9), (12, 12)], None),
        ("ball, 3 ch", [20, 24, 30], 3, [(8, 8), (10, 9), (12, 12)], Some(ksup([30, 24, 20], [11.0, 9.5, 8.0], true))),
        ("cube, 3 ch", [20, 24, 30], 3, [(8, 8), (10, 9), (12, 12)], Some(ksup([30, 24, 20], [12.0, 10.0, 9.0], false))),
        ("non-convex, 5 ch", [18, 22, 26], 5, [(7, 6), (9, 8), (10, 9)], Some(ksup_holes([26, 22, 18], 9.0))),
        ("full support, 2 ch", [20, 24, 30], 2, [(8, 8), (10, 9), (12, 12)], Some(KSupport::full([30, 24, 20]))),
        ("block, 8 ch, z tail 0 (whole tiles)", [16, 20, 24], 8, [(16, 0), (10, 10), (1, 0)], None),
        ("ball, 8 ch, z tail 0", [16, 20, 24], 8, [(16, 0), (10, 10), (1, 0)], Some(ksup([24, 20, 16], [4.0, 3.0, 2.5], true))),
        ("block, in-place y / z, 3 ch", [32, 16, 20], 3, [(12, 11), (6, 5), (8, 7)], None),
        ("ball, in-place y / z, 7 ch", [32, 16, 20], 7, [(12, 11), (6, 5), (8, 7)], Some(ksup([20, 16, 32], [7.0, 6.0, 11.0], true))),
        // y length 384: the padded shared-memory stride does not fit 99 KB per block, the unpadded one does
        ("ball, y 384 (unpadded smem), 3 ch", [20, 384, 6], 3, [(8, 8), (154, 153), (3, 2)], Some(ksup([6, 384, 20], [3.0, 180.0, 8.0], true))),
        // planes of whole 16-line tiles (n0 · nch % 16 = 0): the in-place layout
        ("block, 8 ch (whole tiles)", [20, 24, 30], 8, [(8, 8), (10, 9), (12, 12)], None),
        ("non-convex, 8 ch (whole tiles)", [18, 22, 26], 8, [(7, 6), (9, 8), (10, 9)], Some(ksup_holes([26, 22, 18], 9.0))),
        ("ball, in-place y / z, 4 ch (whole tiles)", [32, 16, 20], 4, [(12, 11), (6, 5), (8, 7)], Some(ksup([20, 16, 32], [7.0, 6.0, 11.0], true))),
    ];
    let mut worst_buf = [f64::INFINITY, 0.0f64];
    let mut worst_out = 0.0f64;
    // HASTY_BFP_TEST_CASES (e.g. "0,3" for the compute-sanitizer runs): the case indices to run
    let sel: Option<Vec<usize>> = std::env::var("HASTY_BFP_TEST_CASES").ok().map(|v| v.split(',').map(|c| c.parse().unwrap()).collect());
    for (ci, (name, n, nch, keep, ks)) in cases.iter().enumerate() {
        if sel.as_ref().is_some_and(|s| !s.contains(&ci)) {
            continue;
        }
        // HASTY_BFP_TEST_BITS (e.g. "22,23" for the compute-sanitizer runs): the widths to test
        let bl: Vec<u32> = std::env::var("HASTY_BFP_TEST_BITS").map_or(vec![16, 20, 22, 24, 17, 23], |v| v.split(',').map(|b| b.parse().unwrap()).collect());
        for &bits in &bl {
            let slabs: &[usize] = if bits == 22 { &[1, 3] } else { &[] };
            let o = case(&dev, name, *n, *nch, *keep, ks.as_ref(), bits, slabs);
            for c in o.c_buf {
                worst_buf = [worst_buf[0].min(c), worst_buf[1].max(c)];
            }
            worst_out = worst_out.max(o.c_out[0]).max(o.c_out[1]);
            // the doc's prediction (§3.4, channel blocks of FFT intermediates): 1.95–2.0 · 2^−b per buffer
            for c in o.c_buf {
                assert!((1.5..=2.5).contains(&c), "{name} bfp{bits}: buffer error {c:.3} · 2^-b outside 1.5 … 2.5");
            }
            // one stored buffer per direction: the transform's error is the buffer's, plus f32 rounding noise
            // (≈ 1e-7 of the z / y passes: comparable to 2^-24)
            let f32_noise = 3e-7 / (2.0f64).powi(-(bits as i32));
            for c in o.c_out {
                assert!(c <= 2.6 + f32_noise, "{name} bfp{bits}: transform error {c:.3} · 2^-b");
            }
        }
    }
    eprintln!("buffer error range {:.3} … {:.3} · 2^-b; worst transform error {worst_out:.3} · 2^-b", worst_buf[0], worst_buf[1]);
}

/// the store switch and its errors: f32 back to in place, bit-identical to a plan that never had BFP; invalid widths
#[test]
fn bfp_store_switch() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let (n, nch, keep) = ([20usize, 24, 30], 3usize, [(8usize, 8usize), (10, 9), (12, 12)]);
    let ks = ksup([30, 24, 20], [11.0, 9.5, 8.0], true);
    for k in [None, Some(&ks)] {
        let a = PrunedChanFft3::<f32>::new_ksupport(&dev, n, nch, keep, 1.0, k).unwrap();
        let mut b = PrunedChanFft3::<f32>::new_ksupport(&dev, n, nch, keep, 1.0, k).unwrap();
        assert!(b.set_store(&dev, Store::Bfp(25)).is_err() && b.set_store(&dev, Store::Bfp(12)).is_err());
        assert_eq!(b.store(), Store::F32);
        b.set_store(&dev, Store::Bfp(22)).unwrap();
        assert!(b.store_bytes() > 0);
        b.set_store(&dev, Store::F32).unwrap();
        assert_eq!((b.store(), b.store_bytes()), (Store::F32, 0));
        let mut rng = Rng(5);
        let v: Vec<C32> = (0..a.len()).map(|_| C32::new(rng.u(), rng.u())).collect();
        for fwd in [true, false] {
            let run = |p: &PrunedChanFft3<f32>| {
                let mut d = st.clone_htod(&v).unwrap();
                if fwd { p.forward(&st, &mut d).unwrap() } else { p.inverse(&st, &mut d).unwrap() }
                st.clone_dtoh(&d).unwrap()
            };
            let (d, m) = same(&run(&a), &run(&b), &|_| true);
            assert_eq!(d, 0, "f32 after a BFP store differs ({d} of {m} values)");
        }
    }
    assert_eq!(Store::parse("bfp22"), Some(Store::Bfp(22)));
    assert_eq!(Store::parse("f32"), Some(Store::F32));
    assert_eq!(Store::parse("bfp15"), None);
    assert_eq!(Store::parse("bfp25"), None);
    assert_eq!(Store::Bfp(20).to_string(), "bfp20");
}

/// a short coil batch: channels at or past `set_active_channels` hold stale data (huge values); the BFP writers store
/// them as zeros, so the active channels' results equal those of a buffer whose unused channels are zero, bit for bit,
/// forward and inverse, block-pruned and k-support pruned
#[test]
fn bfp_store_short_batch() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let (n, nch, keep) = ([20usize, 24, 30], 8usize, [(8usize, 8usize), (10, 9), (12, 12)]);
    let ks = ksup([30, 24, 20], [11.0, 9.5, 8.0], true);
    for k in [None, Some(&ks)] {
        for (act, dense) in [(3usize, false), (5, false), (8, false), (3, true), (5, true)] {
            let mut b = PrunedChanFft3::<f32>::new_ksupport(&dev, n, nch, keep, 1.0, k).unwrap();
            b.set_store_opts(&dev, Store::Bfp(22), dense).unwrap();
            assert_eq!(b.store_inplace(), !dense);
            let mut rng = Rng(77 + act as u64);
            let base: Vec<C32> = (0..b.len()).map(|_| C32::new(rng.u(), rng.u())).collect();
            let zero: Vec<C32> = base.iter().enumerate().map(|(q, v)| if q % nch < act { *v } else { C32::new(0.0, 0.0) }).collect();
            let stale: Vec<C32> = base.iter().enumerate().map(|(q, v)| if q % nch < act { *v } else { C32::new(3e6, -2e6) }).collect();
            for fwd in [true, false] {
                let run = |x: &[C32]| {
                    let mut d = st.clone_htod(x).unwrap();
                    // the count applies to the next call only
                    b.set_active_channels(act);
                    if fwd { b.forward(&st, &mut d).unwrap() } else { b.inverse(&st, &mut d).unwrap() }
                    st.clone_dtoh(&d).unwrap()
                };
                let (rz, rs) = (run(&zero), run(&stale));
                // forward: every active cell (the support's for pruned plans is a subset); inverse: active channels
                let (d, m) = same(&rs, &rz, &|q| q % nch < act);
                let ok = if fwd && k.is_some() {
                    // the forward is defined on the support only
                    let idx = |q: usize| { let p = q / nch; (p / (n[1] * n[2]), (p / n[2]) % n[1], p % n[2]) };
                    let sup = |q: usize| { let (z, y, x) = idx(q); let zr = ks.zcols[y * n[2] + x]; zr != (0, 0) && in_wrapped(z, n[0], zr) };
                    same(&rs, &rz, &|q| q % nch < act && sup(q)).0 == 0
                } else if fwd {
                    d == 0
                } else {
                    // the inverse is defined on the image block
                    let idx = |q: usize| { let p = q / nch; (p / (n[1] * n[2]), (p / n[2]) % n[1], p % n[2]) };
                    let blk = |q: usize| { let (z, y, x) = idx(q); in_wrapped(z, n[0], keep[0]) && in_wrapped(y, n[1], keep[1]) && in_wrapped(x, n[2], keep[2]) };
                    same(&rs, &rz, &|q| q % nch < act && blk(q)).0 == 0
                };
                eprintln!("short batch {act} of {nch} channels, {}, k-support {}, {}: stale channels {} the active results ({d} of {m} values differ before restricting to the defined cells)", if dense { "dense" } else { "in place" }, k.is_some(), if fwd { "forward" } else { "inverse" }, if ok { "do not change" } else { "CHANGE" });
                assert!(ok, "short batch: stale channels changed the active channels' results");
            }
        }
    }
}
