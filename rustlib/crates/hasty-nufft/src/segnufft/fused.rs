//! Pad fused into the first forward grid-FFT pass and crop fused into the last inverse pass (default; HASTY_SEG_FUSE_PADCROP=0
//! restores the separate kernels; research/temporal_operator.md §23), and the two-pass L2 slab policy of the grid FFT. The x pass of the pruned transform (`hasty_fft::PrunedChanFft3`, lines along x
//! of every (channel, y, z) in the image block, channels innermost) generates its input from the image, the coil maps,
//! the segment diagonals and the deapodisation (`hasty_fft::IoHook::load`, the same arithmetic as `seg_pad`: grid values
//! equal up to FMA contraction, 1.3e-7), and the inverse x pass reduces its block lines over the batch's channels with
//! conj(S_c P_l) / φ̂ into the image (`IoHook::epilogue`: group-shuffle reduction, one f32 atomic add per voxel and
//! channel tile — the crop's x += Σ, f32 summation order differs). The image block of the grid is then neither written by a
//! pad nor read back by the first pass, nor written by the last pass and read back by a crop.

use super::*;
use hasty_fft::{IoHook, KArg, Store};

/// image index of grid index g on an axis of N image and nf grid points (inverse of the pad's (a − N/2) mod nf)
const PRELUDE: &str = r#"
__device__ __forceinline__ int hk_img(long g, int n, int nf) { const int h = n - n / 2; return g < h ? (int)g + n / 2 : (int)g - nf + n / 2; }
"#;

// hsm / hse / hso, hpm / hpe / hpo: dense BFP copies of the coil maps and the diagonals (mantissa words, exponent
// bytes, element offset of the batch's first coil / the frame's diagonals; `bfpmaps.rs`), null: read hs / hp (f32)
const PARAMS_PAD: &str = ", const cx* __restrict__ hx, const cx* __restrict__ hs, const cx* __restrict__ hp, const float* __restrict__ hi0, const float* __restrict__ hi1, const float* __restrict__ hi2, int hnch, int hN0, int hN1, int hN2, int hnf0, int hnf1, int hnf2, const unsigned* __restrict__ hsm, const unsigned char* __restrict__ hse, long hso, const unsigned* __restrict__ hpm, const unsigned char* __restrict__ hpe, long hpo, long hz0";
const PARAMS_CROP: &str = ", cx* hxo, const cx* __restrict__ hs, const cx* __restrict__ hp, const float* __restrict__ hi0, const float* __restrict__ hi1, const float* __restrict__ hi2, int hnch, int hN0, int hN1, int hN2, int hnf0, int hnf1, int hnf2, const unsigned* __restrict__ hsm, const unsigned char* __restrict__ hse, long hso, const unsigned* __restrict__ hpm, const unsigned char* __restrict__ hpe, long hpo, long hz0";

/// common prelude: NSEG, the staging rule and (HASTY_SEG_FUSE_PADCROP=gather) the per-element gather variant.
/// Staged (default where it fits): the block first loads its line's image row (/ φ̂), the coil rows its channel tile
/// touches and the NSEG segment rows into the spare shared buffer with coalesced loads, then pass 0 / the crop read
/// them from shared memory. Gather: every element loads its own S_c, P_l, x from global memory (latency-bound: the x
/// passes run one block per SM, measured 55 % of DRAM bandwidth).
fn hook_prelude(nseg: usize, (mb, db): (u32, u32)) -> String {
    let gather = std::env::var("HASTY_SEG_FUSE_PADCROP").is_ok_and(|v| v == "gather");
    // BFP maps (`bfpmaps.rs`): HK_MB / HK_DB mantissa bits of the coil-map / diagonal copies the hooks can decode (0:
    // f32 only); at run time a null mantissa pointer selects the f32 array
    let codec = if mb + db > 0 { hasty_bfp::BFP_CUH } else { "" };
    format!(
        r#"{PRELUDE}{codec}
#define HNSEG {nseg}
#define HK_ROWS_C ((LPB + HNSEG - 1) / HNSEG + 1)
#define HK_GATHER {}
#define HK_MB {mb}
#define HK_DB {db}
#if HK_MB || HK_DB
__device__ __forceinline__ cx hk_bfp(int b, const unsigned* __restrict__ m, const unsigned char* __restrict__ e, long i) {{
  float2 v;
  switch (b) {{
#if HK_MB
    case HK_MB: v = bfp_load_c<HK_MB>(m, e, i); break;
#endif
#if HK_DB && HK_DB != HK_MB
    case HK_DB: v = bfp_load_c<HK_DB>(m, e, i); break;
#endif
    default: v = make_float2(0.f, 0.f);
  }}
  return make_cx(v.x, v.y);
}}
#endif
#if HK_MB
#define HK_S(i) (hsm ? hk_bfp(HK_MB, hsm, hse, hso + (i)) : hs[i])
#else
#define HK_S(i) hs[i]
#endif
#if HK_DB
#define HK_P(i) (hpm ? hk_bfp(HK_DB, hpm, hpe, hpo + (i)) : hp[i])
#else
#define HK_P(i) hp[i]
#endif
"#,
        gather as u8
    )
}

/// pad hook: grid value of channel la = c · NSEG + l at grid (e, lb1, lb2) = S_c P_l x / φ̂ (as `pad_core`)
fn pad_hook(nseg: usize, bits: (u32, u32)) -> IoHook {
    let prelude = format!(
        r#"{}
// staged rows: 0 = x / φ̂, 1 .. HK_ROWS_C = coils from LA0 / NSEG, then the NSEG segment rows (row stride = block width)
#define HK_W (N_IN + N_IN_TAIL)
#define HK_STAGED (!HK_GATHER && !FFT_INPLACE && (1 + HK_ROWS_C + HNSEG) * HK_W <= N * LPB)
__device__ __forceinline__ cx hk_pad(long la, long lb1, long lb2, int e, const cx* __restrict__ hx, const cx* __restrict__ hs, const cx* __restrict__ hp,
    const float* __restrict__ hi0, const float* __restrict__ hi1, const float* __restrict__ hi2, int hnch, int hN0, int hN1, int hN2, int hnf0, int hnf1, int hnf2,
    const unsigned* __restrict__ hsm, const unsigned char* __restrict__ hse, long hso, const unsigned* __restrict__ hpm, const unsigned char* __restrict__ hpe, long hpo) {{
  if (la >= hnch) return make_cx(RL(0), RL(0));
  const int a0 = hk_img(e, hN0, hnf0), a1 = hk_img(lb1, hN1, hnf1), a2 = hk_img(lb2, hN2, hnf2);
  const long nv = (long)hN0 * hN1 * hN2, v = ((long)a2 * hN1 + a1) * hN0 + a0;
  const float fac = hi0[a0] * hi1[a1] * hi2[a2];
  const cx xv = hx[v];
  const cx xs = make_cx(xv.x * fac, xv.y * fac);
  const int c = (int)la / HNSEG, l = (int)la % HNSEG;
  return cmul(HK_S((long)c * nv + v), cmul(HK_P((long)l * nv + v), xs));
}}
__device__ __forceinline__ cx hk_pad_s(long la, long la0, int e, const cx* stage, int hnch, int hN0, int hnf0) {{
  if (la >= hnch) return make_cx(RL(0), RL(0));
  const int a0 = hk_img(e, hN0, hnf0);
  const int c = (int)(la / HNSEG - la0 / HNSEG), l = (int)la % HNSEG;
  return cmul(stage[(1 + c) * HK_W + a0], cmul(stage[(1 + HK_ROWS_C + l) * HK_W + a0], stage[a0]));
}}
#if HK_STAGED
#define HK_LOAD hk_pad_s(la, LA0, e, stage, hnch, hN0, hnf0)
#else
#define HK_LOAD hk_pad(la, lb1, lb2 + hz0, e, hx, hs, hp, hi0, hi1, hi2, hnch, hN0, hN1, hN2, hnf0, hnf1, hnf2, hsm, hse, hso, hpm, hpe, hpo)
#endif
"#,
        hook_prelude(nseg, bits)
    );
    let prologue = r#"
#if HK_STAGED
    const long nv = (long)hN0 * hN1 * hN2;
    const int a1 = hk_img(lb1, hN1, hnf1), a2 = hk_img(lb2 + hz0, hN2, hnf2);
    const long vrow = ((long)a2 * hN1 + a1) * hN0;
    const int c_lo = (int)(LA0 / HNSEG);
    for (int q = tid; q < (1 + HK_ROWS_C + HNSEG) * HK_W; q += LPB * TPL) {
      const int r = q / HK_W, a0 = q % HK_W;
      const long v = vrow + a0;
      cx w;
      if (r == 0) {
        const float fac = hi0[a0] * hi1[a1] * hi2[a2];
        const cx xv = hx[v];
        w = make_cx(xv.x * fac, xv.y * fac);
      } else if (r <= HK_ROWS_C) {
        const int c = c_lo + r - 1;
        w = c * HNSEG < hnch ? HK_S((long)c * nv + v) : make_cx(RL(0), RL(0));
      } else {
        w = HK_P((long)(r - 1 - HK_ROWS_C) * nv + v);
      }
      stage[q] = w;
    }
#endif"#;
    IoHook { params: PARAMS_PAD.into(), prelude, load: Some("HK_LOAD".into()), epilogue: None, prologue: Some(prologue.into()) }
}

/// crop epilogue: HK_HH consecutive threads per entry e (as many as the block has per entry, a power of two dividing
/// LPB and 32) sum conj(S_c P_l) g over their share of the tile's channels, reduce with group shuffles, and one adds the
/// sum / φ̂ to the image (an f32 atomic per voxel and channel tile). Staged (as the pad): the coil and segment rows of
/// the line first go to the spare shared buffer.
fn crop_hook(nseg: usize, bits: (u32, u32)) -> IoHook {
    let prelude = format!(
        r#"{}
#define HK_W (N_OUT + N_OUT_TAIL)
#define HK_STAGED (!HK_GATHER && !FFT_INPLACE && (HK_ROWS_C + HNSEG) * HK_W <= N * LPB)
#define HK_R (LPB * TPL / N)
#define HK_H (HK_R >= 32 ? 32 : HK_R >= 16 ? 16 : HK_R >= 8 ? 8 : HK_R >= 4 ? 4 : HK_R >= 2 ? 2 : 1)
#define HK_HH (HK_H > LPB ? LPB : HK_H)
#define HK_J (LPB / HK_HH)
"#,
        hook_prelude(nseg, bits)
    );
    let epi = r#"
    const long nv = (long)hN0 * hN1 * hN2;
    const int a1 = hk_img(lb1, hN1, hnf1), a2 = hk_img(lb2 + hz0, hN2, hnf2);
    const float f12 = hi1[a1] * hi2[a2];
    const long vrow = ((long)a2 * hN1 + a1) * hN0;
    const int c_lo = (int)(LA0 / HNSEG);
#if HK_STAGED
    // rows 0 .. HK_ROWS_C − 1: coils from c_lo, then the NSEG segment rows (row stride = block width)
    for (int q = tid; q < (HK_ROWS_C + HNSEG) * HK_W; q += LPB * TPL) {
      const int r = q / HK_W, a0 = q % HK_W;
      const long v = vrow + a0;
      const int c = c_lo + r;
      spare[q] = r < HK_ROWS_C ? (c * HNSEG < hnch ? HK_S((long)c * nv + v) : make_cx(RL(0), RL(0))) : HK_P((long)(r - HK_ROWS_C) * nv + v);
    }
    __syncthreads();
#define HK_SP(c, l, a0, v) cmul(spare[((c) - c_lo) * HK_W + (a0)], spare[(HK_ROWS_C + (l)) * HK_W + (a0)])
#else
#define HK_SP(c, l, a0, v) cmul(HK_S((long)(c) * nv + (v)), HK_P((long)(l) * nv + (v)))
#endif
    // HK_HH threads per entry (consecutive lanes, a power of two dividing LPB and 32), each summing HK_J of the
    // tile's lines in a rotated order (bank spread of fin[e * LPB + k]), reduced with group shuffles; one f32 atomic
    // add per entry and channel tile
    const int h = tid % HK_HH;
    const unsigned gmask = HK_HH == 32 ? 0xffffffffu : (((1u << HK_HH) - 1u) << ((threadIdx.x & 31) & ~(HK_HH - 1)));
    for (int e = tid / HK_HH; e < N; e += LPB * TPL / HK_HH) {
      const bool in_ = (e < N_OUT || e >= N - N_OUT_TAIL) && LA0 < hnch;
      cx w = make_cx(RL(0), RL(0));
      const int a0 = in_ ? hk_img(e, hN0, hnf0) : 0;
      const long v = vrow + a0;
      if (in_) {
        for (int j = 0; j < HK_J; ++j) {
          const int k = h * HK_J + (j + e) % HK_J;
          const long ch = LA0 + k;
          if (ch < hnch && ch < la_cnt) {
            const int c = (int)ch / HNSEG, l = (int)ch % HNSEG;
            w = cadd(w, cmulc(fin[e * LPB + k], HK_SP(c, l, a0, v)));   // g · conj(S_c P_l) (codegen cmulc(a, b) = a · conj(b))
          }
        }
      }
      #pragma unroll
      for (int o = HK_HH / 2; o > 0; o >>= 1) { w.x += __shfl_xor_sync(gmask, w.x, o); w.y += __shfl_xor_sync(gmask, w.y, o); }
      if (in_ && h == 0) {
        const float fac = hi0[a0] * f12;
        atomicAdd(&hxo[v].x, w.x * fac); atomicAdd(&hxo[v].y, w.y * fac);
      }
    }
#undef HK_SP"#;
    IoHook { params: PARAMS_CROP.into(), prelude, load: None, epilogue: Some(epi.into()), prologue: None }
}

/// fused pad / crop default: env HASTY_SEG_FUSE_PADCROP (1 / unset = on, staged inputs where they fit; `gather` = on
/// with per-element global loads; 0 = off, the separate seg_pad / seg_crop kernels)
pub fn fuse_padcrop_default() -> bool {
    std::env::var("HASTY_SEG_FUSE_PADCROP").map_or(true, |v| v != "0")
}

/// z planes per slab of the two-pass L2 grid FFT (`PrunedChanFft3::set_slab`): env HASTY_SEG_FFT_SLAB = 0 (off:
/// whole-grid passes), `auto` (default: as many planes as fit HASTY_SEG_FFT_L2FRAC, default 0.5, of the device's L2
/// (CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE); one plane while it fits HASTY_SEG_FFT_L2MAX, default 0.85, of the L2; else
/// off), or a plane count. RTX 4090 (72 MB L2), 2 mm thorax (33 MB planes): 1 plane measured best (§23).
pub(super) fn slab_default(dev: &Device, p: &PrunedChanFft3<f32>) -> usize {
    let envf = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    match std::env::var("HASTY_SEG_FFT_SLAB").as_deref() {
        Ok("0") => 0,
        Err(_) | Ok("") | Ok("auto") => {
            let dz = p.l2_slab_planes(dev, envf("HASTY_SEG_FFT_L2FRAC", 0.5)).unwrap_or(0);
            if dz == 0 && p.l2_slab_planes(dev, envf("HASTY_SEG_FFT_L2MAX", 0.85)).unwrap_or(0) >= 1 { 1 } else { dz }
        }
        Ok(v) => v.parse().unwrap_or_else(|_| panic!("HASTY_SEG_FFT_SLAB={v}: 0, auto or a plane count")),
    }
}

thread_local! {
    static FFT_STORE: std::cell::Cell<Option<Store>> = const { std::cell::Cell::new(None) };
    static GRID_STORE: std::cell::Cell<Option<Store>> = const { std::cell::Cell::new(None) };
}

/// Storage of the forward grid itself (the z pass's output, read by the gathers) for the engines built on this thread
/// from now on; None: back to the env rule of [`grid_store_default`]. For tests and comparisons in one process.
pub fn set_thread_grid_store(s: Option<Store>) {
    GRID_STORE.with(|c| c.set(s));
}

/// The forward grid (research/bfp_storage.md §10): [`set_thread_grid_store`], else env HASTY_SEG_GRID_STORE = f32
/// (default) | bfp16 … bfp24: the last forward z pass stores the grid slot as BFP in place (exponent bytes after the
/// slot's cells, 1/128 of the slot) and the plain-path gathers (tile, knot tile, vs / vt) decode it while staging grid
/// values. Only where a plane's lines n0 · nch are whole 16-line tiles (else f32); the multi-image, fat-group and other
/// paths that do not go through the engine's load keep an f32 grid. The adjoint grid stays f32.
pub fn grid_store_default() -> Store {
    if let Some(s) = GRID_STORE.with(|c| c.get()) {
        return s;
    }
    match std::env::var("HASTY_SEG_GRID_STORE").as_deref().map(str::trim) {
        Err(_) | Ok("") | Ok("f32") => Store::F32,
        Ok(v) => Store::parse(v).unwrap_or_else(|| panic!("HASTY_SEG_GRID_STORE={v}: f32 or bfp16 … bfp24")),
    }
}

/// apply the requested grid storage to a new pruned grid transform (f32 with a note where unavailable)
pub(super) fn apply_grid_store(dev: &Device, p: &mut PrunedChanFft3<f32>, s: Store) {
    if s == Store::F32 {
        return;
    }
    if let Err(e) = p.set_grid_store(dev, s) {
        STORE_NOTE.call_once(|| eprintln!("    [grid_store] {s} grid unavailable ({e}): f32 grid"));
    }
}

/// Storage of the grid FFT's y ↔ z intermediates for the engines built on this thread from now on (an engine keeps
/// its store over re-plans and multi-image passes); None: back to the env rule of [`fft_store_default`]. For tests and
/// comparisons in one process.
pub fn set_thread_fft_store(s: Option<Store>) {
    FFT_STORE.with(|c| c.set(s));
}

/// y → z (forward) and z → y (inverse) intermediates of the grid FFT (`PrunedChanFft3::set_store`,
/// research/bfp_storage.md §7 step 1, §8.1, §10; research/sample_kernels.md §20): [`set_thread_fft_store`], else env
/// HASTY_SEG_FFT_STORE = f32 (default since §10: f32 in place; with the L2-slab discard the BFP store is neutral in
/// time at 256³ and costs accuracy) | bfp16 … bfp24 (block floating point in place in the grid slot when a plane's lines
/// n0 · nch are whole 16-line tiles; f32 with one note where the passes do not cover BFP, e.g. y / z lengths > 396 on
/// 99 KB-per-block parts) | auto (bfp24 for whole-grid passes, f32 under L2 slabs, `fft_store_auto`). The x ↔ y
/// intermediates stay f32 (L2 slabs).
/// whether the y ↔ z store is the automatic rule (no thread override, HASTY_SEG_FFT_STORE=auto): then a transform
/// that runs two-pass L2 slabs keeps f32 intermediates and whole-grid passes use bfp24 (research/bfp_storage.md §10:
/// with the slab discard, the L2-resident BFP y pass costs more issue time than the BFP z passes save in DRAM). An
/// explicit width applies everywhere.
pub fn fft_store_auto() -> bool {
    FFT_STORE.with(|c| c.get()).is_none() && std::env::var("HASTY_SEG_FFT_STORE").as_deref().map(str::trim) == Ok("auto")
}

pub fn fft_store_default() -> Store {
    if let Some(s) = FFT_STORE.with(|c| c.get()) {
        return s;
    }
    match std::env::var("HASTY_SEG_FFT_STORE").as_deref().map(str::trim) {
        Err(_) => Store::F32,
        Ok("auto") => Store::Bfp(24),
        Ok(v) => Store::parse(v).unwrap_or_else(|| panic!("HASTY_SEG_FFT_STORE={v}: f32, auto or bfp16 … bfp24")),
    }
}

/// layout of the BFP intermediates: env HASTY_SEG_FFT_BFP_LAYOUT = inplace (default: mantissas in the grid slot
/// itself when a plane's lines n0 · nch are whole 16-line tiles, only the exponent bytes extra; otherwise the transform
/// stays f32 — the separate buffer would cost 0.56 of a slot and shrink the coil batch, the bigger lever) | dense (a
/// separate buffer whenever the store is BFP)
pub fn fft_store_dense() -> bool {
    match std::env::var("HASTY_SEG_FFT_BFP_LAYOUT").as_deref() {
        Err(_) | Ok("") | Ok("inplace") => false,
        Ok("dense") => true,
        Ok(v) => panic!("HASTY_SEG_FFT_BFP_LAYOUT={v}: inplace or dense"),
    }
}

/// whether a transform of `nch` channels on grid `nf` (x fastest) keeps BFP intermediates: in place possible, or the
/// dense layout asked for by env (else it runs f32, [`apply_store`])
pub(super) fn store_active(store: Store, nf: [usize; 3], nch: usize) -> bool {
    store != Store::F32 && (fft_store_dense() || (nf[0] * nch) % 16 == 0)
}

/// extra device bytes of the BFP intermediates of a grid transform (`nf` x fastest, image `dims`, `nch` channels)
pub(super) fn store_extra(store: Store, nf: [usize; 3], dims: [usize; 3], nch: usize) -> usize {
    if !store_active(store, nf, nch) {
        return 0;
    }
    PrunedChanFft3::<f32>::store_bytes_for(store, fft_store_dense(), [nf[2], nf[1], nf[0]], nch, dims[2])
}

/// bytes of the engine's own BFP intermediates counted in its coil-batch budget (`max_grid_bytes`): a dense buffer in
/// full; in place none — its exponent bytes (≤ 1/128 of the slot) live outside the budget like the FFT plans' own
/// buffers, so that a budget of exactly n coils keeps n coils (the knot ledger measures them with the operator, and
/// `seg_slot_bytes` adds them for planners)
pub(super) fn store_budget_bytes(store: Store, nf: [usize; 3], dims: [usize; 3], nch: usize) -> usize {
    if store != Store::F32 && !fft_store_dense() && (nf[0] * nch) % 16 == 0 {
        0
    } else {
        store_extra(store, nf, dims, nch)
    }
}

/// the coil batch with the BFP intermediates counted in the grid budget: the largest batch ≤ `cb` whose slots
/// (`per_coil` bytes per coil) plus the BFP buffer ([`store_budget_bytes`]) fit `budget` (f32: `cb` unchanged), equal
/// batches again
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_store_budget(cb: usize, nc: usize, budget: usize, per_coil: usize, store: Store, nf: [usize; 3], dims: [usize; 3], nseg: usize) -> usize {
    if store == Store::F32 {
        return cb;
    }
    let mut c = cb;
    while c > 1 && c * per_coil + store_budget_bytes(store, nf, dims, c * nseg) > budget {
        c -= 1;
    }
    if c == cb { cb } else { balanced_batch(nc, c) }
}

/// Prefer a coil batch whose grid planes are whole 16-line tiles (n0 · cb · nseg a multiple of 16: the BFP
/// intermediates in place, only the exponent bytes extra) over the dense fallback (a separate buffer, 0.56 of a slot at
/// bfp22), when it needs no more passes over the samples: among the batches ≤ `cb_f32` that fit `budget` with their
/// store bytes and are equal-sized (`balanced_batch`), the fewest passes, then the fewest transformed channels, then
/// the largest. f32, the dense layout by env, an in-place `cb` already, or no such batch: `cb` unchanged. Env
/// HASTY_SEG_FFT_INPLACE_PREF=0: off (the batch of `fit_store_budget` as before).
#[allow(clippy::too_many_arguments)]
pub(super) fn prefer_inplace(cb_f32: usize, cb: usize, nc: usize, budget: usize, per_coil: usize, store: Store, nf: [usize; 3], dims: [usize; 3], nseg: usize) -> usize {
    let inplace = |c: usize| (nf[0] * c * nseg) % 16 == 0;
    if store == Store::F32 || fft_store_dense() || inplace(cb) || nc == 0 || std::env::var("HASTY_SEG_FFT_INPLACE_PREF").is_ok_and(|v| v == "0") {
        return cb;
    }
    let passes = |c: usize| nc.div_ceil(c);
    let best = (1..=cb_f32.max(1))
        .filter(|&c| inplace(c) && balanced_batch(nc, c) == c && (c == 1 || c * per_coil + store_budget_bytes(store, nf, dims, c * nseg) <= budget))
        .min_by_key(|&c| (passes(c), passes(c) * c, std::cmp::Reverse(c)));
    match best {
        Some(c) if passes(c) <= passes(cb) => c,
        _ => cb,
    }
}

static STORE_NOTE: std::sync::Once = std::sync::Once::new();

/// mantissa bits of a BFP store (0: f32)
pub(super) fn store_bits(s: Store) -> u32 {
    match s {
        Store::F32 => 0,
        Store::Bfp(b) => b,
    }
}

/// apply the engine's store to a new pruned grid transform (f32 if the plan's passes do not cover BFP, with one note
/// per process; env HASTY_SEG_FFT_STORE_LOG=1: a note per transform)
pub(super) fn apply_store(dev: &Device, p: &mut PrunedChanFft3<f32>, s: Store, explicit: bool) {
    if s != Store::F32 && !explicit && p.slab() > 0 {
        // automatic default under two-pass L2 slabs: f32 intermediates (fft_store_auto)
        return;
    }
    if s != Store::F32 {
        if !fft_store_dense() && !p.bfp_inplace_possible() {
            // no dense fallback by default: the transform stays f32 (its lines are not whole 16-line tiles)
            if std::env::var("HASTY_SEG_FFT_STORE_LOG").is_ok_and(|v| v != "0") {
                eprintln!("    [fft_store] {s}: plane lines not whole 16-line tiles, f32 for this transform");
            }
            return;
        }
        if let Err(e) = p.set_store_opts(dev, s, fft_store_dense()) {
            if std::env::var("HASTY_SEG_FFT_STORE_LOG").is_ok_and(|v| v != "0") {
                eprintln!("    [fft_store] {s} intermediates unavailable ({e}): f32");
            } else {
                STORE_NOTE.call_once(|| eprintln!("    [fft_store] {s} intermediates unavailable ({e}): f32 for this transform (printed once; HASTY_SEG_FFT_STORE_LOG=1 for every one)"));
            }
        }
    }
}

impl GridFft {
    /// free the BFP intermediate buffer before a re-plan builds the next transform (no two buffers at once); returns
    /// the requested store for the new one
    pub(super) fn release_store(&mut self, dev: &Device) -> Store {
        if let Some(p) = self.pruned.as_mut() {
            let _ = p.set_store(dev, Store::F32);
        }
        self.store
    }

    /// build the hooked x passes once (Some(true): available; Some(false): the plan declines, e.g. no pruned plan, a
    /// Bluestein x length or a block shape the crop's lane reduction does not cover)
    pub(super) fn ensure_fused(&mut self, dev: &Device, nseg: usize, bits: (u32, u32)) -> bool {
        if let Some(ok) = self.fused {
            return ok;
        }
        let ok = match self.pruned.as_mut() {
            Some(p) => match p.set_hooks(dev, Some(pad_hook(nseg, bits)), Some(crop_hook(nseg, bits))) {
                Ok(()) => true,
                Err(e) => {
                    eprintln!("    [fuse_padcrop] hooked grid-FFT passes unavailable ({e}): separate pad / crop");
                    false
                }
            },
            None => false,
        };
        self.fused = Some(ok);
        ok
    }
}

impl SegNufft {
    /// Two-pass L2 grid FFT: `Some(dz)` slabs of dz z planes (0: off), None: the env default (`HASTY_SEG_FFT_SLAB`).
    /// Kept over re-plans. Identical results (same kernels on the same lines).
    pub fn set_fft_slab(&mut self, dz: Option<usize>) {
        self.fft_slab = dz;
        let (s, explicit) = (self.gfft.store, self.gfft.store_explicit);
        if let Some(p) = self.gfft.pruned.as_mut() {
            p.set_slab(dz.unwrap_or_else(|| slab_default(&self.dev, p)));
            if !explicit {
                // the automatic store depends on the slab mode (fft_store_auto)
                let _ = p.set_store(&self.dev, Store::F32);
                apply_store(&self.dev, p, s, false);
            }
        }
    }

    /// whether the fused pad / crop passes are built and used by this engine's plain single-image calls
    pub fn fused_padcrop_active(&self) -> bool {
        self.fuse_padcrop && self.gfft.fused == Some(true)
    }

    /// z planes per slab of the current grid FFT (0: whole-grid passes or no pruned plan)
    pub fn fft_slab(&self) -> usize {
        self.gfft.pruned.as_ref().map_or(0, |p| p.slab())
    }

    /// Storage of the grid FFT's y ↔ z intermediates of this engine from now on (all its transforms, kept over
    /// re-plans): f32 in place or `Store::Bfp(b)` (`PrunedChanFft3::set_store`; a transform whose passes do not cover
    /// BFP stays f32, with a note). Re-budgets the coil batch as at construction (the store's bytes counted in
    /// `max_grid_bytes`, in-place batches preferred): if the batch changes, the slots, kernels and launch shapes are
    /// rebuilt at their defaults (autotune again) and a note is printed. The multi-image passes are re-planned at their
    /// next use when the store needs more bytes than before. Errors only from the allocation / compilation.
    pub fn set_fft_store(&mut self, s: Store) -> Result<(), NufftError> {
        let (nf, dims) = ([self.plan.nf[0], self.plan.nf[1], self.plan.nf[2]], [self.plan.modes[0], self.plan.modes[1], self.plan.modes[2]]);
        let (pc, nc) = (self.per_coil_bytes(), self.nc_all());
        let before = store_extra(self.gfft.store, nf, dims, self.cb * self.nseg);
        // (the multi-image passes are re-planned below when the store needs more bytes than before)
        let cb = fit_store_budget(self.cb_f32, nc, self.max_grid_bytes, pc, s, nf, dims, self.nseg);
        let cb = prefer_inplace(self.cb_f32, cb, nc, self.max_grid_bytes, pc, s, nf, dims, self.nseg);
        self.gfft.store = s;
        if cb != self.cb {
            return self.rebatch(cb);
        }
        if store_extra(s, nf, dims, self.cb * self.nseg) > before {
            self.multi.clear();
        }
        let dev = self.dev.clone();
        for g in std::iter::once(&mut self.gfft).chain(self.multi.iter_mut().filter_map(|m| m.fft.as_mut())) {
            g.store = s;
            g.store_explicit = true;
            if let Some(p) = g.pruned.as_mut() {
                p.set_store(&dev, Store::F32)?;
                apply_store(&dev, p, s, true);
            }
        }
        Ok(())
    }

    /// storage of the forward grid the gathers read (HASTY_SEG_GRID_STORE; f32 unless a BFP grid is active)
    pub fn grid_store(&self) -> Store {
        self.gfft.pruned.as_ref().map_or(Store::F32, |p| p.grid_store())
    }

    /// storage of the grid FFT's y ↔ z intermediates (f32: in place)
    pub fn fft_store(&self) -> Store {
        self.gfft.pruned.as_ref().map_or(Store::F32, |p| p.store())
    }

    /// device bytes of the grid FFT's BFP intermediate buffer (0 with f32)
    pub fn fft_store_bytes(&self) -> usize {
        self.gfft.pruned.as_ref().map_or(0, |p| p.store_bytes())
    }

    /// the FFT-side switches for reports: slab planes, fused pad / crop, y ↔ z storage
    pub(super) fn fft_desc_tail(&self) -> String {
        let s = self.fft_slab();
        let st = match self.fft_store() {
            Store::F32 => String::new(),
            b => format!(
                ", y ↔ z intermediates {b} ({}, {:.3} GB extra)",
                if self.gfft.pruned.as_ref().is_some_and(|p| p.store_inplace()) { "in the grid slot" } else { "separate buffer" },
                self.fft_store_bytes() as f64 / 1e9
            ),
        };
        let st = st
            + &if self.grid_store() != Store::F32 { format!(", grid {} (in the slot)", self.grid_store()) } else { String::new() }
            + &if (self.maps_store, self.diag_store) != (Store::F32, Store::F32) { format!(", coil maps {} / diagonals {} (fused pad / crop)", self.maps_store, self.diag_store) } else { String::new() };
        format!(
            ", x pass {:?} lines × threads, {}, pad / crop {}{st}",
            self.gfft.pruned.as_ref().map(|p| p.x_shape()),
            if s > 0 { format!("two-pass L2 slabs of {s} z planes ({:.1} MB)", (s * self.gfft.pruned.as_ref().unwrap().plane_bytes()) as f64 / 1e6) } else { "whole-grid passes".into() },
            if self.fuse_padcrop { "fused into the x passes (not in this FFT time)" } else { "separate kernels" }
        )
    }

    /// whether this call's pad / crop run fused into the grid FFT (switch on, plain single-image path, hooks built)
    pub(super) fn fused_padcrop(&mut self) -> bool {
        let bits = (store_bits(self.maps_store), store_bits(self.diag_store));
        self.fuse_padcrop && !self.wf && self.warp.is_none() && self.gfft.ensure_fused(&self.dev, self.nseg, bits)
    }

    /// the hooks' arguments: x frame `xp`, coil maps `sp` (from coil c0), segment diagonals `pp` of the frame (the
    /// caller holds their guards over the launch), then the BFP copies `bm` = (mantissa, exponent, element offset) of the
    /// maps and the diagonals (null pointers: the f32 arrays)
    fn hook_args(&self, st: &Arc<CudaStream>, xp: u64, sp: u64, pp: u64, ncb: usize, bm: [(u64, u64, i64); 2]) -> Vec<KArg> {
        let ip = [0, 1, 2].map(|a| self.plan.invphi[a].device_ptr(st).0);
        let (m, nf) = (self.plan.modes, self.plan.nf);
        vec![
            KArg::Ptr(xp),
            KArg::Ptr(sp),
            KArg::Ptr(pp),
            KArg::Ptr(ip[0]),
            KArg::Ptr(ip[1]),
            KArg::Ptr(ip[2]),
            KArg::I32((ncb * self.nseg) as i32),
            KArg::I32(m[0] as i32),
            KArg::I32(m[1] as i32),
            KArg::I32(m[2] as i32),
            KArg::I32(nf[0] as i32),
            KArg::I32(nf[1] as i32),
            KArg::I32(nf[2] as i32),
            KArg::Ptr(bm[0].0),
            KArg::Ptr(bm[0].1),
            KArg::I64(bm[0].2),
            KArg::Ptr(bm[1].0),
            KArg::Ptr(bm[1].1),
            KArg::I64(bm[1].2),
        ]
    }

    /// (mantissa, exponent, element offset) of the BFP copies the hooks read for coil c0 and frame f's diagonals, null
    /// for f32 (or a copy whose width the hooks were not built for); builds the maps' copy when it is missing
    fn hook_bfp(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, f: usize, c0: usize) -> Result<[(u64, u64, i64); 2], NufftError> {
        let nv = self.nv();
        if self.maps_store != Store::F32 && self.smaps_bfp.is_none() {
            self.smaps_bfp = bfpmaps::pack_dense(&self.dev, st, self.smaps.as_ref().unwrap(), self.maps_store)?;
        }
        let ptrs = |b: Option<&bfpmaps::BfpDev>, want: Store, off: usize| -> (u64, u64, i64) {
            match b {
                Some(b) if Store::Bfp(b.bits) == want => (b.mant.device_ptr(st).0, b.ex.device_ptr(st).0, off as i64),
                _ => (0, 0, 0),
            }
        };
        Ok([ptrs(self.smaps_bfp.as_ref(), self.maps_store, c0 * nv), ptrs(seg.segph_bfp.as_ref(), self.diag_store, seg.diag_off(f, self.nseg * nv))])
    }

    /// slot ← FFT(pad(S_c P_l x_f)) with the pad generated inside the first FFT pass
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_frame_fused(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, slot: usize, x: &CudaSlice<C32>, f: usize, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let bm = self.hook_bfp(st, seg, f, c0)?;
        let nv = self.nv();
        let xv = x.slice(f * nv..(f + 1) * nv);
        let sv = self.smaps.as_ref().unwrap().slice(c0 * nv..);
        let pv = seg.diag(f, self.nseg * nv);
        let ((xp, _gx), (sp, _gs), (pp, _gp)) = (xv.device_ptr(st), sv.device_ptr(st), pv.device_ptr(st));
        let args = self.hook_args(st, xp, sp, pp, ncb, bm);
        let gex = self.grid_ex_ptr(st, slot);
        let p = self.gfft.pruned.as_ref().unwrap();
        p.set_active_channels(ncb * self.nseg);
        p.set_grid_exponents(gex);
        p.forward_hooked(st, &mut self.slots[slot], &args)?;
        Ok(())
    }

    /// x_f += crop(IFFT(slot)) with the crop reduced inside the last inverse pass
    #[allow(clippy::too_many_arguments)]
    pub(super) fn flush_frame_fused(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, slot: usize, x: &mut CudaSlice<C32>, f: usize, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let bm = self.hook_bfp(st, seg, f, c0)?;
        let nv = self.nv();
        let mut xv = x.slice_mut(f * nv..(f + 1) * nv);
        let sv = self.smaps.as_ref().unwrap().slice(c0 * nv..);
        let pv = seg.diag(f, self.nseg * nv);
        let ((xp, _gx), (sp, _gs), (pp, _gp)) = (xv.device_ptr_mut(st), sv.device_ptr(st), pv.device_ptr(st));
        let args = self.hook_args(st, xp, sp, pp, ncb, bm);
        let p = self.gfft.pruned.as_ref().unwrap();
        p.set_active_channels(ncb * self.nseg);
        p.inverse_hooked(st, &mut self.slots[slot], &args)?;
        Ok(())
    }
}

impl SegNufft {
    /// storage of the coil maps / segment diagonals the fused pad / crop read (HASTY_SEG_MAPS_STORE /
    /// HASTY_SEG_DIAG_STORE, `set_thread_map_stores`)
    pub fn map_stores(&self) -> (Store, Store) {
        (self.maps_store, self.diag_store)
    }

    /// Debug / test: the BFP copy of the coil maps (mantissa words, exponent bytes), built if missing; None for f32
    #[doc(hidden)]
    pub fn debug_maps_bfp(&mut self, st: &Arc<CudaStream>) -> Result<Option<(Vec<u32>, Vec<u8>)>, NufftError> {
        if self.maps_store != Store::F32 && self.smaps_bfp.is_none() {
            self.smaps_bfp = bfpmaps::pack_dense(&self.dev, st, self.smaps.as_ref().unwrap(), self.maps_store)?;
        }
        match &self.smaps_bfp {
            Some(b) => Ok(Some((st.clone_dtoh(&b.mant)?, st.clone_dtoh(&b.ex)?))),
            None => Ok(None),
        }
    }
}

impl SegSegments {
    /// Debug / test: the BFP copy of the diagonals (mantissa words, exponent bytes); None for f32
    #[doc(hidden)]
    pub fn debug_diag_bfp(&self, st: &Arc<CudaStream>) -> Result<Option<(Vec<u32>, Vec<u8>)>, NufftError> {
        match &self.segph_bfp {
            Some(b) => Ok(Some((st.clone_dtoh(&b.mant)?, st.clone_dtoh(&b.ex)?))),
            None => Ok(None),
        }
    }
}

impl SegNufft {
    /// Debug / test: slot 0 after loading frame `f` of `x` for coils c0 .. c0 + ncb with the fused (`fused`) or the
    /// separate pad (the forward grid FFT included), copied to the host. With a BFP grid (HASTY_SEG_GRID_STORE) the
    /// slot holds BFP blocks and their exponent tail (decode on the host, as tests/seg_bfp_dram.rs does).
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn debug_loaded_slot(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, x: &CudaSlice<C32>, f: usize, c0: usize, ncb: usize, fused: bool, zero: bool) -> Result<Vec<C32>, NufftError> {
        let keep = self.fuse_padcrop;
        self.fuse_padcrop = fused;
        if zero {
            st.memset_zeros(&mut self.slots[0])?;
        }
        let r = self.load_frame(st, seg, 0, x, f, c0, ncb);
        self.fuse_padcrop = keep;
        r?;
        Ok(st.clone_dtoh(&self.slots[0])?)
    }
}
