# Implementation round: what was built, what was measured, what it changes

All numbers: RTX 4090, CUDA 13.0, f32, 2026-11-02. Code in `rustlib/`, raw tables in `rustlib/bench/baselines.md`, reproduction commands in each section.
Small-scale caveat: accuracy studies use 8³–16³ voxel grids; timing studies use N=64–160. Anything labelled "estimate" is extrapolated.

## 1. What exists now (all tested against independent CPU oracles)

| crate | content | validation |
|---|---|---|
| `hasty-fft` | mixed-radix Stockham FFT (any 2·3·5·7·11·13 smooth length), pruned zero-padded fused convolution (`ConvPlan`), packed real/fp16/bf16 kernels, polynomial-in-coordinates and array diagonals, separate pruned forward/inverse (`PadFft3`) | rustfft f64 reference; matches cuFFT/VkFFT at the memory roofline |
| `hasty-nufft` | 1–3 D type 1/2 NUFFT (ES kernel, bin sort, GM + shared-memory spreading, multi-coil), FFT-order output, **chirp-aware NUFFT** (per-sample stencil absorbing the concomitant phase; POD+DEIM+least-squares stencil spectra) | NDFT f64: errors < 6·tol; adjointness; chirp: 1.3e-3 where the effect is 0.42 |
| `hasty-toeplitz` | Toeplitz kernel builder (type-1 NUFFT in FFT order → circulant → FFT), multi-field mixing engine, expansions: plain, Taylor-Hankel, commutator/Weyl, Tucker, CP | each expansion matches its own CPU series to 6e-7 (f32 level) and converges to the exact dense operator |
| `hasty-bench` | `fft_bench`, `conv_bench`, `nufft_bench`, `toeplitz_bench`, `chirp_bench`, `recon_sched` + C++ baselines (VkFFT, cuFINUFFT Release) | — |

## 2. Plain NUFFT vs cuFINUFFT (Release, same machine)
Type 1 is 1.7–1.9× faster at 10⁷ points (30.6 vs 52.8 ms uniform, 37.9 vs 73.6 ms radial, 128³ modes, tol 1e-6), 10–30% slower at 10⁶; type 2 is 10–40% slower at large M. Details and next steps: `baselines.md`.

## 3. Normal-operator families — measured cost and what limits them (N=128, one coil)
- Fused multi-field Z pass (`MixZPlan`: forward Z transforms of all fields, k-space mixing and inverse Z transforms in one shared-memory pass per connected group of fields) — `toeplitz_bench 128`, ms/coil (unfused in parentheses):

| operator | fields / kernels | fused | × plain fused conv (0.462 ms) | gain over unfused |
|---|---|---|---|---|
| plain through the engine | 1 / 1 | 0.53 (1.03) | 1.15 | 1.95× |
| commutator n=1 | 4 / 8 | 3.12 (4.89) | 6.8 | 1.57× |
| Taylor–Hankel P=1 | 5 / 15 | 4.77 (6.84) | 10.3 | 1.43× |
| Tucker L=4 | 4 / 10 | 3.58 (5.32) | 7.8 | 1.49× |
| CP L'=8 / L'=12 | 8 / 12 | 4.98 / 7.48 (8.94 / 13.4) | 10.8 / 16.2 | 1.79× |
| commutator n=2, Taylor P=2, Tucker L=6 | 10–15 | do not fit shared memory or tiles < 8 lines → unfused path: 14.9 / 26.6 / 8.85 | 32 / 58 / 19 | — |

  Fusion pays when the group fits and the tile is ≥ 8 lines wide; narrower tiles lose more to uncoalesced traffic than fusion saves, so `MixOp` falls back to the unfused path automatically (`set_fused` forces either, for comparison).
- **Memory is the real limit at N=320**: every field is a padded (2N)³ array (2 GB); kernel count grows fast (Weyl n=2: 36 kernels, n=3: 119, n=4: 322; Taylor P=2: 70). With 24 GB only ≲ 8 fields and a few kernels fit. First-order commutator (4 fields, 8 kernels) and CP with L' ≲ 8 are the ones that scale; higher orders need kernel streaming or recomputation.

## 4. Chirp-aware (modulated-row) NUFFT — lessons
1. **The stencil spectrum must be fitted by least squares on the FOV, not obtained by a DFT of a windowed chirp.** DFT projection needs S ≈ 8–10 cells for 1e-3 at 1 rad (5·10³ taps); the FOV least-squares fit reaches 1–2·10⁻³ with S=3–4 (`chirp_diag`): σ=2, |w_q|≤0.5 (Φ ≲ 2.5 rad corners): S=3 → 1.4e-3, S=4 → 1.1e-3; |w_q|≤1: 7e-3; σ=1.25 needs S=5 for 1e-2.
2. The family over a 4-D coefficient box needs a POD basis of L ≈ 20–60 functions (L=31 for |w|≤0.5 at 1e-3, 47–57 for |w|≤1); per sample this is L sin/cos, an L×L solve and an L×(2S+1)³ contraction, then a separable convolution with the ES weights (shared across coils).
3. Measured v1 cost (global memory, one block per sample): A^H A = 24–135 ms/coil at N=128 for K = 10⁶–3·10⁶ (`baselines.md`). That is 2–50× slower than the Toeplitz families at N=128 (K/N³ ≈ 0.5–1.4 samples per voxel: Toeplitz is K-independent). At N=320 with K=3·10⁶ (K/N³ ≈ 0.09) the model gives ≈ 100–135 ms/coil vs Toeplitz-CP L'=10 ≈ 111 ms/coil — a tie at Φ ≈ 2, favouring the stencil route for smaller K or larger Φ. Optimisation headroom (not done): shared-memory tile for the gather, tile-based scatter (the adjoint is atomics-bound: 3× slower than forward), joint ES+chirp least-squares stencils (≈ 1 tap/axis), time-adaptive stencil width, per-spoke structure.
4. It is the only route with zero stored kernels and an FFT count independent of Φ.
5. **Precomputation pays off here** (reconstructions are long-running; concomitant/GNL coefficients are scan-independent). (a) *Stored fp16 stencils* (`precompute_stencils`, 0.02–0.03 s, 4.0 GB at K=10⁶, S=3) + shared-memory tiled scatter: adjoint 17.1 → 11.3 ms/coil (1.5×); forward unchanged (gather is limited by per-coil streaming of the stencils). (b) *Sequence-trained basis tables*: training the POD on the actual w-samples of the trajectory instead of the full coefficient box reduces L from 31 to 13 at the same held-out row error (1.1e-3 vs 2.0e-3), `ChirpTables::save/load` persist them (build 5.0 s vs 2.7 s, once per sequence). Current best (N=128, K=10⁶, C=8, tol 1e-3, S=3): forward 6.9, adjoint 11.3, **A^H A 18.2 ms/coil** (24.4 before). Open: coil-blocked gather tiles / stencil compression (CP factors) for the forward.

## 5. Iteration-level fidelity schedules under P²nP (`recon_sched`, rewritten)
Setup: 16³ voxels, 3300 samples (110 spokes × 30), 4 coils with Gaussian sensitivity lobes, **no density compensation** (the preconditioner replaces it; DCW biases the problem), correlated coil noise Ψ handled by whitening the maps and data with Ψ^{-1/2} (S̃=Ψ^{-1/2}S, ỹ=Ψ^{-1/2}y so N = Σ_c S̃_c^H AᴴA S̃_c). Operator scaled to ‖N‖=1 (power method), step 1. Preconditioners as in the P²nP paper: none (PnP-ISTA), fixed F-1 (2−N), fixed F-Cheb (4−10/3 N, 2 operator applications per iteration), **dynamic ZMSHR1** (θ₁=2e-6, θ₂=200, δ=1e-8, secant pair from consecutive iterates and gradients, 1 application per iteration). Exact model = dense A including the concomitant phase; surrogates = the GPU `hasty-toeplitz` operators; exact cost = 39 plain-Toeplitz units/coil (measured: chirp A^H A with stored stencils, 18.2 ms vs 0.462 ms at N=128, K=10⁶). Denoiser: normalisation-equivariant Haar soft-threshold with threshold ∝ mean detail magnitude, but see caveat — the runs below use κ=0 (identity), i.e. they isolate the data-consistency/preconditioner part.

NRMSE vs truth, 100 iterations, Φ_max = 3 rad (Φ_max = 1.5 rad in parentheses where different):

| method | k=5 | k=20 | k=100 | exact evals | cost (plain-op units) |
|---|---|---|---|---|---|
| PnP-ISTA, exact | 0.766 | 0.535 | 0.313 | 100 | 15600 |
| P²nP-F-1, exact | 0.658 | 0.421 | 0.265 | 100 | 31200 |
| P²nP-F-Cheb, exact | 0.531 | 0.333 | 0.235 | 100 | 31200 |
| **P²nP-D (ZMSHR1), exact** | 0.580 | **0.291** | **0.213** (0.215) | 100 | 15600 |
| F-Cheb, exact grad + *plain* operator in the preconditioner | 0.531 | 0.333 | 0.235 | 100 | 16000 |
| P²nP-D, plain surrogate only | 0.580 | 0.301 | **1.29** (0.89) | 0 | 400 |
| P²nP-D, commutator-1 only | 0.580 | 0.291 | **NaN** (0.247) | 0 | 2680 |
| P²nP-D, commutator-2 only | 0.580 | 0.291 | **7.7e4** (0.216) | 0 | 12800 |
| P²nP-D, lagged plain, adaptive anchor τ=0.1 | 0.580 | 0.291 | 0.218 (0.215) | 12 | **2320** |
| P²nP-D, lagged plain, τ=0.05 | 0.580 | 0.291 | 0.214 (0.213) | 19 | 3440 |
| P²nP-D, lagged commutator-1, τ=0.1 | 0.580 | 0.291 | 0.213 (0.215) | 16 | 5605 |
| F-Cheb, lagged plain τ=0.05 + plain-operator preconditioner | 0.531 | 0.333 | 0.235 | 16 | 3360 |

Lagged correction (SVRG/control variate): `g_k = N_ℓ x_k − N_ℓ x_a + g_exact(x_a)`; anchor refreshed when `‖x − x_a‖ > τ‖x‖`.
Conclusions:
1. **Dynamic preconditioning is the right default**: one operator application per iteration, reaches PnP-ISTA's 100-iteration error in < 20 iterations and beats the fixed F-Cheb polynomial (which costs 2 applications) at every k ≥ 10 — same ordering as in the paper.
2. **Cheap-only gradients are unsafe with a quasi-Newton preconditioner.** The secant pairs learned from a mismatched operator are applied to an ill-conditioned problem (small eigenvalues of N far below the surrogate error), so even commutator-2 (3·10⁻⁴ operator error on typical images) blows up at 3 rad and plain/commutator-1 drift at 1.5 rad by k=100 (plain 0.89, commutator-1 0.25 vs 0.215 exact) although they track the exact curve to 3–4 digits at k=20. Operator error measured on typical images is not the relevant quantity (confirmed again); the lagged correction removes the bias at the fixed point and was stable in every configuration.
3. **Lagged plain surrogate + adaptive anchor: 4.5–6.7× cheaper than exact for the same error** (2320–3440 vs 15600 units, 12–19 exact applications in 100 iterations). Higher-order surrogates (commutator-1) do not pay: the extra 5.7 units per iteration outweigh the fewer refreshes. At the measured ratio R=39 the plain surrogate is the right choice; with R≈12 (N=320, K=3·10⁶) expect ≈ 2–2.5×.
4. The preconditioner's polynomial applications (F-1/F-Cheb) may use the plain operator with no visible change (4 digits), so the fixed preconditioners cost ~1 unit/coil there; with ZMSHR1 there is no extra application at all.
5. **Caveat**: tiny 16³ smooth phantom; with the Haar soft-threshold at κ ≥ 0.15 the denoiser's bias dominates and plateaus every strategy at 0.35–0.55, hiding the differences between gradient strategies (cheap-only variants even looked "better" at k=40, κ=0.6, which is noise from the plateau). The conclusions above are about the data-consistency + preconditioner machinery; a real learned/BM4D denoiser and N ≥ 64 with realistic coil maps are still to be tested.

## 6. Updated decision guide (replaces the earlier regime table where they differ)

| situation | what to run | evidence |
|---|---|---|
| any Φ, well-sampled frames (K/N³ ≳ 0.2), memory for few kernels | Toeplitz-family surrogate (plain / commutator-1 / CP L'≲8) with **exact refresh via the chirp NUFFT or a high-order expansion** | §3, §5 |
| undersampled frames (K/N³ ≲ 0.05), Φ ≳ 2 | chirp NUFFT as the main operator (no kernels; cost ∝ K), plain Toeplitz as preconditioner/surrogate | §4, §5 |
| Φ ≲ 0.5 | commutator n=1–2 or plain + one refresh | §3, round-1 numbers |
| memory-bound at N ≥ 320 | plain fused Toeplitz (1 field, 1 kernel) as surrogate; exact refresh by chirp NUFFT (no extra kernels) | §3, §4 |

## 7. Prioritised next steps
1. ~~Fuse the multi-field mixing~~ — done (1.4–1.95×). ~~Tile the chirp scatter, store stencils, sequence-train tables~~ — done (adjoint 1.5×, L 31→13). Still open for the chirp NUFFT: forward gather (coil-blocked tiles / stencil compression), time-adaptive width, Horner kernels.
2. Plain NUFFT type 2: tile gather/Horner to close the 10–40% gap to cuFINUFFT.
3. Put the solver loop on the GPU: ZMSHR1, lagged-correction anchor rule (τ ≈ 0.05–0.1), noise-whitened coil maps; run at N ≥ 64 with a real denoiser and real coil maps/non-synthetic data.
4. Precompute what is scan-independent (concomitant/GNL: stencil tables per sequence, kernels for the phase part) and keep off-resonance as the only subject-specific online ingredient.
5. Streaming of kernels and recomputation of forward transforms to reach N = 320 with higher-order expansions.
