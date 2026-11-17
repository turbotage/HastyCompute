# Velocity encoding (how many encodes, which moments) and the B0 / R2* / fat calibration scan as a prior (2026-11-17)

Questions (user):
1. VNR against the number of encodes at fixed scan time and at fixed readouts, including phase-wrap risk,
   per-encode undersampling (R ≈ 50 per frame and encode at 1 mm), gradient-moment cost and cross-talk between encodes.
2. Can 6 (or 5) encodes keep alias detection like the 7-point coprime scheme? Identifiability, conditioning,
   diminishing returns.
3. How to use the calibration scan (B0, R2*, fat) to regularise the flow reconstruction without bias, and how to handle
   motion between the two scans.
4. Ranked recommendations and pre-registered experiments.

Tags. References: **[V]** full text read, **[S]** full text through a summariser, **[A]** abstract / metadata only;
DOIs checked against Crossref (literature pass of 2026-11-17, delegated, downloaded text treated as data). Our numbers:
**[M]** computed here (scripts below), **[T]** derived here, **[E]** estimate. No GPU was used; all experiments are
small numpy models, not pipeline runs.

Scripts `research/experiments/encoding/`; outputs `data/hasty/seq/studies/encoding_2026-11-17/`:
- `enc_lib.py`: Fisher information, wrap geometry (sheets, margins, true aliases), union bound, ML decoders;
- `enc_search.py`: random search over integer designs (exact lattice margins by LLL);
- `enc_verify.py`: full enumeration of finalists and reference designs (`verify.json`);
- `enc_verify7.py`: the same for the 7-point lattice finalists (`verify7.log`);
- `enc_mc.py`, `enc_mc_grid.py`: Monte Carlo with exact complex noise and ML decoding (`mc.log`, `mc_grid.log`);
- `enc_bound.py`: tube-packing bound;
- `enc_order.py`: encode-order cross-talk;
- `enc_artifact.py`: 2-D undersampling toy (`artifact_R*.json`).

Conventions. V = the protected speed (3 m/s in the thorax); velocities in units of V. Phase of encode e:
φ_e = θ + k_e·v, k_e = γ M1_e. SNR_e = |ρ|/σ for one encode image; SNR_T = the SNR of one image made from all
readouts (so SNR_e = SNR_T/√K at fixed time). The figure of merit is the **time-normalised noise**
η = σ_v,rms · SNR_T / V (smaller is better; reference 4-point: η = 2/π = 0.64 at VENC = V).

## 0. Answer

1. **At fixed scan time the number of encodes does not enter the thermal VNR at all; only the spread of the first
   moments does** [T, §1.1]. With white noise, J_v = 2 SNR_T² Cov(k), Cov(k) the covariance of the encode moments.
   - So "more encodes" buys nothing by itself. It buys *room* for larger moments (lower VENC) while keeping wraps
     decodable: K − 4 redundant phases (r) act as parity checks.
   - A tube-packing bound (§1.3) gives the best achievable geometric-mean noise
     η_g ≳ c_K · SNR_T^{−r/3} at a fixed wrap-failure rate. The exponent is the whole story: r = 0 (4 encodes) gains
     nothing from SNR, r = 3 (7 encodes) gains SNR^{−1}. Our designs sit 1.2–2.3× above the bound.
2. **Measured design table (thorax, V = 3 m/s, P_wrap ≤ 1e-3 inside, ≤ 1e-2 at peak speed; Gaussian phase noise,
   enumeration-exact; §1.5)**:

   | design | K | η | SNR_T needed | max M1 vs today (p = 2 at V_ext = 3 m/s) |
   |---|---|---|---|---|
   | referenced 4-point (VENC 3.3) | 4 | 0.71 | 7.8 | 0.30 |
   | Hadamard 4-point, lattice decode | 4 | 0.57 | 9.3 | 0.27 |
   | FC + tetrahedron (Johnson–Markl 5-point) | 5 | 0.42–0.50 | 5.4–6.7 | 0.33–0.40 |
   | best integer 6-point found | 6 | 0.27 | 8.6 | 0.80 |
   | **coprime 7-point p = 2 (today's layout, 8 % slack)** | 7 | **0.29** | **7.9** | 0.92 |
   | 7-point lattice (search, all-axis moments) | 7 | 0.25 / 0.20 | 7.3 / 8.3 | 0.92 / 1.33 |
   | coprime 7-point p = 3 | 7 | 0.22 | 10.9 | 1.23 |
   | two tetrahedra, moments 3 : 2 ("8H") | 8 | 0.15–0.18 | 9.7 | 1.0–1.2 |

   - Monte Carlo with exact complex noise: failures are ≈ 3× the Gaussian bound at SNR_e = 3. That shifts every
     threshold by ≈ +10 % and leaves the ranking unchanged.
   - The current scheme is a sound default near SNR_T ≈ 8, about 1.8× above the packing bound.
   - **Today's sequence has zero range slack** (V_ext = CRT range = 3 m/s). If 3 m/s is the real peak speed, voxels
     near it alias through the cell boundary, and the requirement rises to SNR_T 14.3. If the true peak is lower, the
     slack is already there. Setting V_ext ≈ 3.26 m/s (VENC_lo 1.09 m/s) costs 8 % of
     VNR and removes this.
   - It is not Pareto-optimal within 7 encodes. Lattice designs that mix axes are 13 % quieter at a lower SNR
     requirement and the same maximum moment, and 1.45× quieter at +5 % SNR and 1.33× the moment.
3. **The undersampling regime changes which term matters** [M, toy §1.6]. In a 2-D toy at fixed total samples:
   - Fully sampled, the coronary-like vessels' velocity error follows η exactly: 4-point 0.030, coprime 7-point 0.011,
     8H 0.007 m/s.
   - At R_e = 12 per encode (7-point), with per-encode TV reconstruction, the order **reverses**: 4-point 0.056,
     6-point 0.081, coprime 0.083, 8H 0.112 m/s. That is 3–15× the thermal part. Static tissue keeps the thermal
     order (0.028 vs 0.022).
   - At R_e = 50 (our measured per-frame level) every design fails, as the practice baseline did. The most sensitive
     designs wrap the most in the aorta (41 % for 4-point, 79–84 % for coprime / 8H).
   - Reason (Prop. 4). When all encodes share one sampling pattern, the artifact enters the phase difference as the
     PSF side-lobes of ρ·(k_e − k₀)·v. Its velocity error is *independent of VENC*, while per-encode undersampling
     (R_e ∝ K) grows it. Lower VENC only reduces the thermal/incoherent part.
   - Sharing one pattern across encodes beat independent patterns for the coronaries in 5 of 6 designs (coprime
     0.083 vs 0.097 m/s), for static tissue in 6 of 6, for the aorta in 3 of 6, and in every wrap fraction at
     R_e = 50.
   - **This argument only holds for per-encode image reconstruction.** In the velocity-model reconstruction (the end
     goal, unknowns ρ and v per voxel), data per unknown do not depend on K. Per-encode undersampling then disappears
     as a concept, and the encoding enters only through Cov(k), the wrap geometry and the moments.
4. **Effective SNR decides between the candidates, and we do not know it yet** [E, §1.4].
   - Thermal SNR per (1 mm voxel, frame) is ≈ 0.4 (extrapolated from SNR 20 at 2 mm / 10 min), far below every
     threshold above.
   - Wraps must therefore be resolved on pooled quantities: the time-mean velocity, ROIs, temporal knots. That puts
     SNR_T at ≈ 2.5 per voxel for the time mean and ≈ 10 for a 20-voxel ROI.
   - Decision rule (E1): measure SNR_T,eff of the velocity-model output. Keep p = 2 (or the equally robust lattice A)
     below ≈ 9. Consider lattice B, p = 3 or 8H only above ≈ 11.
5. **Six encodes with alias detection exist and are as good as today's seven at the same scan time** [M, §2].
   - The 3-D Chinese-remainder theorem holds for lattices. The unaliased set is the cell of Λ = Λ_a ∩ Λ_b, and
     noise robustness is the dual of the integer parity-check lattice, 2π λ₁((ℤᴷ ∩ ker[1|k]ᵀ)*).
   - The best integer 6-point design found ([[0,0,0],[2,−2,0],[−1,−2,−1],[0,0,−2],[2,1,−3],[2,−1,−3]]·κ₀, two
     parity checks) reaches η 0.27 at SNR_T 8.6 with 0.8× today's moment, against 0.29 at 7.9 for coprime p = 2.
   - It needs non-orthogonal, mixed-magnitude moments on all three axes, a lattice/ML decoder, and a retrained
     (or encoding-agnostic) denoiser.
   - Five encodes (one parity check) are 1.45× noisier at equal robustness. The 5-point FC + tetrahedron design is
     already on the 5-point Pareto front. Four encodes have no detection at all: only range slack protects them.
6. **Diminishing returns.**
   - Thermal: each extra encode adds SNR_T^{−1/3} to the bound's scaling, so at SNR_T ≈ 8–10 going from 7 → 8 gains
     ≈ 1.6–1.7× in the bound and 1.2–1.5× in realised designs. The 8H design realises this, but only at SNR_T ≳ 10.
   - Per-encode undersampling: R_e ∝ K. In per-encode reconstruction the coronary error grew 1.5× from 4 to 7
     encodes and 2× to 8 in the toy. In the velocity model it does not grow.
   - Moments: +K costs at most ~0.2 ms of encode at 1 mm (≤ 1.5 % of TR, encode ∝ M1^{1/3}).
   - Cross-talk: the encode train is K·TR long. Under isotropic acceleration the best-order residual is
     0.04–0.05 m/s for coprime 7-point, 0.014 for 7-point lattice A and 0.010 for 8H. Sample-level time in the temporal operator models it
     anyway (§1.8).
7. **Calibration scan as a prior: physics maps transfer, edges are risky** [T, §3].
   - Velocity bias is first order only in *encode-differential* errors. Calibration maps act on all encodes alike, so
     they bias v only through partial volume and fat leakage. They are therefore safe to use with α-shrink and
     held-out selection.
   - Fat matters most: 77 % of static signal energy, and the coronaries sit in epicardial fat.
   - Structural (edge) priors are not safe from the calibration scan: misregistration of a few mm moves the edges of
     3 mm vessels.
   - **The flow scan's own time-mean (respiratory-resolved) image is a better guide.** It is ≈ 5× oversampled at
     1 mm, has the same contrast and the same pose, and has no inter-scan motion. Use directional TV with γ ≤ 0.9 on
     ρ only, never on v. The toy's joint-TV gain was small (≤ 13 %).
8. **Sequence-side, best first.**
   - (a) Embed a few constant-TR multi-echo "calibration TRs" in the flow scan, ~2–6 % of TRs. B0 needs ≈ 10⁴
     coefficients, so this is ample. There is then no inter-scan motion, B0 drift is tracked, and every respiratory
     and cardiac state is covered.
   - (b) Identical navigator modules in both sequences.
   - (c) Sandwich the separate high-resolution R2*/fat scan before and after the flow scan.
   - (d) Map transport by registration of a *synthesised* flow-contrast image, then per-region acceptance by held-out
     k-space.

## 1. VNR, the number of encodes and scan time

### 1.1 Fisher information [T]

One voxel, encode e: s_e = ρ e^{i(θ + k_e·v)} + n_e, n_e ~ CN(0, σ_e²). For complex Gaussian noise the phase parameters
decouple from |ρ|, and the Fisher matrix of (θ, v) is 2 Σ_e SNR_e² [1, k_eᵀ]ᵀ[1, k_eᵀ]. Profiling θ (Schur complement):

  J_v = 2 Σ_e SNR_e² (k_e − k̄)(k_e − k̄)ᵀ,  k̄ = SNR²-weighted mean.

**Proposition 1.**
- *Fixed scan time, white noise*, N/K readouts per encode: SNR_e² = SNR_T²/K, so J_v = 2 SNR_T² Cov(k), with
  Cov(k) = (1/K) Σ (k_e − k̄)(k_e − k̄)ᵀ. K enters only through the moment covariance.
- *Fixed readouts per encode* (time ∝ K): J_v = 2 SNR_1² K Cov(k). Extra encodes help like extra averages.
  This is not our case (fixed 8–12 min).

Examples at fixed time (η = σ_v,rms SNR_T / V):
- referenced 4-point at VENC = V: Cov⁻¹ = (4/κ²)(I + 11ᵀ), η = 2/π = 0.637;
- balanced (Hadamard) 4-point at the same pair moment: Cov⁻¹ = 4/κ² I, half the variance (Pelc et al. 1991 [A]).

Since Cov(k) scales as κ², VNR is unbounded without a constraint. **The design problem is the trade-off between
moment spread and decodability.**

### 1.2 Wrap geometry: sheets, margins, true aliases [T]

Let A = [1 | k] (K × 4), P = I − AA⁺, U = ker Aᵀ (dim r = K − 4). A wrap hypothesis n ∈ ℤᴷ (n₀ = 0 wlog) is the
flat A x + 2πn. Its closest approach to the true flat is at x_n = A⁺ 2πn:

  margin m_n = ‖P 2πn‖,  velocity error w_n = [A⁺ 2πn]_v.

- **True aliases** are n with m_n = 0. They are undetectable, and the unaliased radius is half the shortest such w_n.
- **False sheets** (m_n > 0) are detectable; the decoder confuses them only if noise exceeds m_n/2.
- With a speed prior |v| ≤ V, the distance to the *restricted* sheet is
  d_n(v)² = m_n² + min_{|v+u|≤V} (u − w_n)ᵀ F (u − w_n), F = kᵀ(I − 11ᵀ/K)k.
  This also covers r = 0 designs, which have no redundancy and are protected only by range slack.

**Proposition 2 (integer designs: 3-D Chinese remainder).** For k_e = κ₀ z_e, z_e ∈ ℤ³:
- the true-alias lattice is (2π/κ₀) M*, M = the lattice of differences z_e − z₀, so the unaliased radius is
  R = (π/κ₀) λ₁(M*);
- the minimal false-sheet margin is m_min = 2π λ₁(P_U ℤᴷ) = 2π λ₁((ℤᴷ ∩ U)*). The projection of ℤᴷ onto a rational
  subspace is the dual of the integer lattice in it; ℤᴷ ∩ U is the lattice of integer **parity checks** c (c·1 = 0,
  cᵀz = 0, so cᵀφ ≡ 0 mod 2π for noise-free data).
- For two sub-designs with alias lattices Λ_a, Λ_b, the combined design has Λ_a ∩ Λ_b. The 1-D CRT (coprime VENCs,
  range a·b·u) is the special case Λ_a = 2au ℤ, Λ_b = 2bu ℤ.

Check against `venc.rs`: coprime p = 2 has per-axis checks c = (1, 2, −3), so m_min = 2π/√14 = 1.68 rad. The
enumeration gives 1.687, with a small cross-axis correction from the shared reference. The union-bound statistic
equals venc.rs' σ_m formula.

Reading of the two standard ideas in this language:
- *Johnson–Markl 5-point* (FC + tetrahedron [V]): one check. The FC point makes the tetrahedron's wraps detectable.
- *Ma et al. triple-VENC* [V] is coprime p = 2 with the third "intrinsic VENC" vL·vH/(vH − vL), i.e. the CRT range.
- *Zwart & Pipe MDHM* [A] and *ICOSA6* [V]: redundancy from non-orthogonal directions with irrational (golden-ratio)
  relations. There are no exact checks, so P_U ℤᴷ is dense and margins shrink with range. In our runs, icosahedral
  designs had larger margins (good at low SNR) but 1.2–1.8× more noise than the lattice designs (§1.5).

### 1.3 A tube-packing bound [T]

The hypothesis set {A(θ, v) mod 2π : θ ∈ [0, 2π), |v| ≤ V} is a 4-dim piece of volume
K² √det Cov(k) · (8π²/3) V³ in the K-torus of volume (2π)ᴷ. ML decoding with margin d needs disjoint normal tubes of
radius d/2 (r normal dimensions), so vol₄ · ω_r (d/2)^r ≤ (2π)ᴷ. With Cov v = Cov(k)⁻¹/(2 SNR_e² K):

  **η_g := det(Cov v)^{1/6} SNR_T/V ≥ 2^{−1/2} [K² (8π²/3) ω_r (d/2)^r / (2π)ᴷ]^{1/3},  d/2 = z √K/(√2 SNR_T)**

Here z ≈ 3.3 for 1e-3, and edge effects are neglected. Consequences:
- η_g ∝ SNR_T^{−r/3}. Each redundant encode turns more SNR into lower VNR; with no redundancy (K = 4), SNR cannot be
  traded.
- Bound values at SNR_T = 8 / 10 / 15 (`enc_bound.py`):
  - K = 5: 0.31 / 0.29 / 0.26;
  - K = 6: 0.21 / 0.18 / 0.14;
  - K = 7: 0.13 / 0.11 / 0.07;
  - K = 8: 0.08 / 0.06 / 0.04.
- Designs at their own SNR requirement: η_g / bound = 1.24 (5-point search), 1.26 (Johnson–Markl), 1.30 (6-point
  search), 1.80 (coprime 7-point p = 2), 1.82 (p = 3), 2.0 (icosa 7-point), 2.3 (8H).
- The axis-aligned 7- and 8-point layouts therefore leave a factor 1.5–2 on the table. Searching all-axis 7-point
  lattices (|z| ≤ 3, 3.8·10⁵ designs, exact margins by LLL, finalists by full enumeration, `verify7.log`) recovered
  part of it:
  - η 0.251 at SNR_T 7.3 with the same moment as today (vs 0.287 at 7.9);
  - η 0.198 at 8.3 with 1.33× the moment;
  - η 0.161 at 10.5 with 1.6× the moment.

### 1.4 Failure probability; which SNR [M, E]

- Union bound P(v) ≈ Σ_n Q(d_n(v)/(2σ_φ)), σ_φ = 1/(√2 SNR_e). Evaluated on two populations:
  - *interior* (|v| ≤ V/2: most voxels, where only m_n matters);
  - *edge* (|v| = V: peak voxels, where range slack also matters).
- Requirement: P_int ≤ 1e-3, mean P_edge ≤ 1e-2.
- Slack: integer designs are scaled to f · (full CRT range). f = 0.92 (8 % slack) was the best compromise for every
  design: at f = 1 peak voxels alias through the cell boundary (coprime p = 2 needs SNR_T 14.3 at f = 1, 7.9 at 0.92).
- **Monte Carlo** (exact complex noise, ML decoding; interior population unless stated; 12–20 k draws per cell):

  | design (f) | SNR_e | MC | Gaussian bound | ratio |
  |---|---|---|---|---|
  | coprime p = 2 (0.92), class decoder | 3 / 4 | 2.6e-3 / 8e-5 | 9.5e-4 / 4.8e-6 | 2.7 / (1 event) |
  | same, edge population \|v\| = V | 3 / 4 | 3.0e-3 / 1.7e-4 | 6.5e-4 / 4.6e-6 | 4.6 / – |
  | coprime p = 2 (0.92), grid decoder | 3 / 4 | 3.4e-3 / 5e-5 | 9.5e-4 / 4.8e-6 | 3.5 / – |
  | coprime p = 3 (0.92) | 3 | 3.9e-2 | 2.7e-2 | 1.4 |
  | lattice A (0.92) | 3 / 4 | 9.5e-4 / 0 | 2.7e-4 / 6.6e-7 | 3.5 / – |
  | 6-point search b (0.92) | 3 / 4 | 1.4e-2 / 5.5e-4 | 5.5e-3 / 1.6e-4 | 2.6 / 3.5 |
  | 8H (1.2) | 3 / 4 | 1.45e-2 / 4.5e-4 | 6.1e-3 / 8.1e-5 | 2.4 / 5.6 |

  (`enc_mc.py` class decoder, `mc.log`; `enc_mc_grid.py` grid decoder, `mc_grid.log`. The class decoder needs
  > 10 GB for p ≥ 3 and 8H and was stopped there.)
  - The exact phase density has heavier tails than its Gaussian approximation, by 2.4–5.6× in failure rate (as
    `venc.rs` found, ≈ 3× at SNR 3).
  - The ratio is similar for all designs, so the ranking holds. Lattice A has the fewest failures at equal SNR_e.
  - The SNR thresholds in §0 are Gaussian; exact noise needs ≈ +10 % SNR.
- **Which SNR?** The thresholds are per-voxel SNR_T of the quantity being unwrapped [E]:
  - The navigator study used tissue SNR 20 per 2 mm voxel over 10 min. At 1 mm that is ÷8, and per frame (40 resp ×
    cardiac frames) ÷√40, giving SNR_T ≈ 0.4 per voxel and frame. Post-contrast blood may be 2× brighter.
  - The R1 per-frame reconstruction's wrapped phase error of 0.73 rad per encode corresponds to SNR_e ≈ 1 (with
    implicit regularisation).
  - So no design can be unwrapped voxel by voxel and frame by frame at 1 mm. Unwrapping has to act on pooled
    quantities:
    - the time-mean velocity: SNR_T ≈ 2.5 per voxel, ≈ 11 for 20 voxels;
    - dynamics relative to the mean. Coronary dynamics are 0.1–0.3 m/s, against VENC_lo = 1.09 m/s for p = 2, so
      they rarely wrap relative to a correct mean.
  - This is why the reconstruction (velocity model, temporal knots, spatial prior) matters more than the encoding.
    The encoding should be chosen for the SNR_T,eff that the reconstruction actually reaches, measured (E1).

### 1.5 Design table (full enumeration, `verify.json`; thorax V = 3 m/s) [M]

| design | K | r | f | η | SNR_T req. | m_min (rad) | max M1 / today (p = 2, f = 1) |
|---|---|---|---|---|---|---|---|
| referenced 4-point | 4 | 0 | 0.90 | 0.707 | 7.8 | – | 0.30 |
| Hadamard 4-point, lattice decode | 4 | 0 | 0.92 / 0.85 | 0.565 / 0.612 | 9.3 / 5.9 | – | 0.27 |
| FC + tetrahedron (J–M 5-point) | 5 | 1 | 1.2 / 1.0 | 0.419 / 0.503 | 6.7 / 5.4 | 1.41 | 0.40 / 0.33 |
| best 5-point (search) | 5 | 1 | 0.92 | 0.422 / 0.376 | 6.8 / 9.4 | 1.41 / 1.05 | 0.62 / 0.65 |
| icosahedral 6, no reference | 6 | 2 | 0.8 | 0.517 | 4.4 | 1.67 | 0.45 |
| 6-point search a / b / c | 6 | 2 | 0.92 | 0.334 / 0.273 / 0.225 | 6.7 / 8.6 / 10.8 | 1.68 / 1.30 / 1.06 | 0.80 / 0.80 / 1.33 |
| classic dual 1 : 2 | 7 | 3 | 0.92 | 0.422 | 5.5 | 2.63 | 0.61 |
| ICOSA6 + reference | 7 | 3 | 1.3 | 0.340 | 6.5 | 1.08 | 0.74 |
| coprime p = 2, today (no slack) | 7 | 3 | 1.00 | 0.264 | 14.3 | 1.69 | 1.00 |
| **coprime p = 2, 8 % slack** | 7 | 3 | 0.92 | **0.287** | **7.9** | 1.69 | 0.92 |
| coprime p = 3 | 7 | 3 | 0.92 | 0.215 | 10.9 | 1.23 | 1.23 |
| coprime p = 4 | 7 | 3 | 0.92 | 0.170 | 13.8 | 0.97 | 1.53 |
| coprime 8-point (two references) | 8 | 4 | 0.92 | 0.262 | 8.3 | 1.71 | 0.92 |
| 8H: tetrahedra × 3 and × 2 | 8 | 4 | 1.0 / 1.2 | 0.177 / 0.147 | 9.7 / 9.75 | 1.51 | 1.0 / 1.2 |
| 7-point lattice A: [[0,0,0],[0,−1,−1],[0,2,−2],[0,1,0],[2,1,1],[2,0,−1],[3,−1,−1]] | 7 | 3 | 0.92 | 0.251 | 7.3 | 1.76 | 0.92 |
| 7-point lattice B: [[0,0,0],[1,2,1],[1,−2,−1],[1,−1,2],[2,0,−2],[3,−3,2],[−2,−2,2]] | 7 | 3 | 0.92 | 0.198 | 8.3 | 1.49 | 1.33 |
| 7-point lattice C: [[0,0,0],[0,0,2],[1,−2,1],[−3,−2,−3],[−1,0,3],[−2,3,1],[3,1,−2]] | 7 | 3 | 0.92 | 0.161 | 10.5 | 1.25 | 1.59 |

Reading:
- Today's design is close to, but not on, the 7-point Pareto front. Lattice A dominates it: lower noise, lower SNR
  requirement, same moment. The two-reference 8-point of the coprime paper (Beghella
  Bartoli et al. 2025 [V]) is 9 % quieter at equal time, because the second reference adds moment spread.
- The 8H design (two balanced Hadamard sets at moment ratio 3 : 2) is the most efficient design tested: 1.6–2× lower
  noise than p = 2 for +23 % SNR requirement, and 1.2–1.5× lower than p = 3 at equal requirement.
- Classic 1 : 2 dual-VENC and icosahedral designs are the robust end (SNR_T 4.4–6.5), at 1.2–1.8× today's noise.
- The coprime paper's Eq. 6 combines the two VENCs with weights that are not inverse-variance [V]. Our `venc.rs`
  uses generalised least squares (inverse variance) and is correct.

### 1.6 Undersampling: per-encode artifacts are not noise [T, M]

**Proposition 4 (linearised artifact propagation).** Write the per-encode reconstruction as
x̂_e = x_e + n_e + (R_e − I)x_e, R_e the reconstruction's resolution/aliasing operator (exact for linear
reconstructions; local linearisation for TV/CS).
- *Shared sampling* (one pattern, R_e = R for all e; PC-VIPR style, our rotation is played for every encode): the
  phase-difference artifact is
  δφ_e − δφ₀ ≈ Im[(R − I)(ρ e^{iθ}(e^{ik_e·v} − e^{ik_0·v}))/x_e] ≈ Im[(R − I)(ρ e^{iθ} i(k_e − k_0)·v)/x_e].
  The common artifact (R − I)ρe^{iθ} cancels exactly. The rest is linear in k, so **the velocity error is
  VENC-independent**: it is the side-lobe of the flow-weighted image ρv (e.g. aortic flow aliasing into a coronary
  voxel). Only the nonlinear terms (k·v ≳ 1) and the thermal part depend on VENC.
- *Independent patterns*: the common artifact does not cancel. The phase error per encode is
  |(R_e − I)ρ|/|ρ|, VENC-independent in phase, so velocity error ∝ VENC, like noise.
- *Fixed total readouts*: R_e = K R₁. A linear reconstruction's artifact variance grows ∝ R_e like noise, and K
  cancels again (Prop. 1). A regularised reconstruction's error is mostly bias, which does not average over encodes.
  The measured view-sharing exponent (3× data → 0.76× error, recon_diag §8, an exponent of 0.25) puts us in the
  bias-dominated regime.

Toy (`enc_artifact.py`): 128² from a 4× finer grid (partial volume and intravoxel dephasing included), aorta 1.5 m/s,
six coronary-like vessels (r = 2.5 px, 0.15–0.35 m/s), single coil, variable-density random sampling, complex noise.
Total samples are fixed (equal scan time), so R_e = (R at K = 7) × K/7. Reconstructions: per-encode TV (FISTA), joint
TV across encodes, zero-filled. Velocity error: oracle wraps, rms in m/s over core voxels. Wrap fraction: share of
voxels where the ML decode (`GridDecoder`) is off by > 0.3 m/s.

Coronary velocity error (TV, shared sampling) at three undersampling levels:

| design | K | R₇ = 1 (thermal only) | R₇ = 12 | R₇ = 50 | static, R₇ = 12 |
|---|---|---|---|---|---|
| referenced, VENC 3.3 m/s | 4 | 0.030 | **0.056** | 0.176 | 0.028 |
| 5-point search | 5 | 0.017 | 0.068 | 0.177 | 0.025 |
| 6-point search b | 6 | 0.012 | 0.081 | **0.148** | **0.021** |
| coprime p = 2 (today) | 7 | 0.011 | 0.083 | 0.155 | 0.022 |
| coprime p = 3 | 7 | 0.009 | 0.087 | 0.163 | 0.022 |
| 8H (f 1.2) | 8 | **0.007** | 0.112 | 0.170 | 0.028 |

Other columns:
- Aorta error at R₇ = 12: 0.051 / 0.072 / 0.062 / 0.065 / 0.073 / 0.103.
- Aorta wrap fraction at R₇ = 50: 0.41 / 0.59 / 0.52 / 0.79 / 0.83 / 0.84.
- Joint TV vs TV, coronary at R₇ = 12: 0.053 / 0.064 / 0.072 / 0.080 / 0.084 / 0.108.
- Independent per-encode patterns, coronary at R₇ = 12: 0.075 / 0.092 / 0.083 / 0.097 / 0.080 / 0.113.
- Independent patterns at R₇ = 50: wrap fractions of 0.1–0.35 in the coronaries and static tissue, against
  0–0.04 shared.

Reading:
- **Fully sampled, the thermal ranking is exactly Prop. 1.** The coronary error ratios follow η: 4-point / coprime =
  2.7 (η ratio 2.5); 8H / coprime = 0.64 (0.51).
- **At R₇ = 12 the order reverses for small vessels.** Error grows monotonically with K, 0.056 → 0.112, and is 3–15×
  the thermal part: the undersampling-dominated regime. Static tissue still follows the thermal order.
- **At R₇ = 50** (our measured per-frame level) every design fails: coronary error ≈ speed, aortic wraps 40–85 %.
  The most sensitive designs wrap the most, so the toy reproduces the practice baseline (box_recon R1).
- Shared sampling beats independent patterns at R₇ = 12 for the coronaries in 5 of 6 designs, for static tissue in
  6 of 6 and for the aorta in 3 of 6, and in every wrap fraction at R₇ = 50. The common artifact cancels, as Prop. 4
  says. In the aorta the larger phases make the linearisation weaker.
- Joint TV helps 3–13 %.
- This is a 2-D single-coil toy with TV, not our 3-D multi-coil operator. It shows the mechanism (Prop. 4), not the
  size of the effect on our scans.
- **Consequence for us.** Per-encode (or per-encode-and-frame) image reconstruction is the regime where more encodes
  hurt. The velocity-parametrised reconstruction (research/velocity_model_sr.md) has 5 unknowns per voxel whatever K
  is, uses all K·N readouts for them, and has the same Fisher information as Prop. 1. Per-encode undersampling is
  then not a property of the data, only of the parametrisation.
  - Keep the encoding choice tied to the end-goal reconstruction.
  - Keep one rotation per encode set (shared sampling) for the per-encode baselines.

### 1.7 Gradient moments, TE, eddy currents, intravoxel dephasing [T, M]

- **Encode duration.**
  - For a slew-limited bipolar, M1 = 2 S r³ with duration 4r, so T_enc ∝ M1^{1/3}. The vencstudy confirms this:
    0.98 / 1.08 / 1.17 ms for p = 2 / 3 / 4 at full slew.
  - At 1 mm (flow_thorax_v2_1mm: PNS-derated encode slew 0.28, encode ≈ 1.43 ms, TR 15.17 ms):
    - 0.8× moment: −0.10 ms;
    - 1.2×: +0.09 ms;
    - 1.33×: +0.14 ms;
    - 1.53×: +0.22 ms (≤ 1.5 % of TR).
    Fewer readouts cost < 1 % in thermal VNR. The moment is not what limits the design.
  - Caution: the balanced layouts (and all-axis lattice designs) play bipolars on three axes at once, so PNS
    derating, which is a scalar per TR part, binds earlier. Run `seqdesign` before believing any of these.
- **Eddy currents and concomitant fields.**
  - Long-τ eddy phase per encode ∝ ΔM1/τ, so the velocity-equivalent error ∝ 1/τ is VENC-independent in m/s
    (WP1 finding). Its size *relative to noise* grows as VENC falls.
  - Bipolar Maxwell phase ∫G² ∝ S² r³ over ΔM1 ∝ S r³ is also VENC-independent in velocity units at fixed slew.
  - Both are modelled (played moments, concomitant operator), so they do not discriminate between designs. The
    prediction precision targets of WP1 (eddy amplitude ≈ 1 %) apply unchanged.
- **Intravoxel dephasing.**
  - Coronary of radius 1.5 mm with a 0.3 m/s parabolic peak: the wall shear gives Δv ≈ 0.4 m/s across a 1 mm voxel.
  - Phase spread |k|Δv: 1.2 rad for p = 2 (VENC_lo 1.09 m/s), 1.7 rad for p = 3, 2.1 rad for p = 4, which means
    signal attenuation of ≈ 0.88 / 0.78 / 0.65 at wall voxels.
  - That is a model error unless the operator carries sub-voxel velocity (a finer velocity grid, the SR flag).
    Prefer p ≤ 3 until it does.
- Acceleration (M2) errors scale with T_enc, i.e. ∝ M1^{1/3}: negligible between candidates.

### 1.8 Motion and cross-talk between encodes [M]

`enc_order.py`: with v(t) = v₀ + a t and all encodes of a rotation in one frame, LS gives v₀ + G a. The trace part of
G is a time stamp (absorbed by `velocity_time`). The rest mixes the other axes' accelerations in. Calibrated to the
measured 0.087 m/s of the natural coprime order under *isotropic* accelerations:

| design (balanced moments) | natural order | best order |
|---|---|---|
| coprime 7, p = 2 | 0.135 | 0.051 |
| coprime 7, p = 3 | 0.144 | 0.042 |
| 8H | 0.048 | **0.010** |
| 7-point lattice A / B | 0.075 / 0.083 | 0.014 / 0.017 |
| 6-point search b | 0.045 | 0.021 |
| 5-point search | 0.051 | 0.045 |

- The isotropic proxy does *not* reproduce the 8× of recon_diag_2026-11-13 §7 (0.087 → 0.011 with the measured,
  anisotropic ph0001 accelerations). With isotropic accelerations that order gives 0.071 m/s.
- The ranking of orders is therefore acceleration-field dependent. Choose the order on the phantom's acceleration
  covariance, not on this proxy.
- The symmetric 8H design is order-robust because each Hadamard set is antisymmetric.
- With sample-level time in the knot operator (temporal_operator.md §16–§19), the skew is modelled, so this matters
  only for bin-group per-frame baselines.
- The train length K·TR (106 ms for 7 at 15.2 ms) also sets the window for bulk motion between the first and last
  encode. The encode-line navigators (research/motion_navigators.md) see it at µm precision.

## 2. Six (and five) encodes with alias detection [T, M]

1. **What is needed.** Detection needs at least one integer parity check: r = K − 4 ≥ 1 and an *integer* (rational)
   structure of the moments. Irrational designs (icosahedral) have only approximate checks.
   - Range: R = (π/κ₀) λ₁(M*), set by the difference lattice.
   - Robustness: m_min = 2πλ₁(check lattice*).
   - Noise: Cov(k).
   - Prop. 2 turns the design into a lattice-packing problem. The packing bound (§1.3) says how well any K can do.
2. **A 6-point design exists that keeps detection with two checks** (search, verified by full enumeration):
   z = [[0,0,0], [2,−2,0], [−1,−2,−1], [0,0,−2], [2,1,−3], [2,−1,−3]], k = f·π·λ₁(M*)·z with λ₁(M*) = √3/2 (a BCC-type
   alias lattice, not cubic).
   - Checks: c₁ = (−1, 1, −2, 4, −2, 0), c₂ = (2, −1, −2, 1, −3, 3). Noise-free data satisfy c·φ ≡ 0 (mod 2π); a
     wrong wrap moves c·φ/2π by an integer.
   - Margin 1.30 rad, kissing number 2.
   - Balanced moments (half-integers in units of κ₀) are non-zero on all three axes in every encode. There is no
     "reference" and none is needed, since θ is a profiled nuisance; "reference-free" is automatic.
   - At f = 0.92: η 0.273, SNR_T 8.6, max moment 0.8× today. **Equal to coprime p = 2** (0.287 at 7.9) at the same
     scan time. Each encode gets 7/6 more readouts and the encode train is one TR shorter.
   - The other 6-point Pareto points: (a) η 0.33 at SNR 6.7; (c) η 0.225 at SNR 10.8, equal to coprime p = 3.
3. **Two VENCs on non-orthogonal direction sets with two magnitudes** (the user's idea) is what the 8H design and the
   6-point search designs are. The integer structure makes 3-D CRT checks across axes, which the axis-aligned
   coprime layout cannot. A 6-point "two triangles" design, (a e_i) ∪ (−b e_i), has two checks but a short alias
   vector along (1,1,1)/(a+b), so it is poor. Mixing axes is what makes 6 points work.
4. **Five encodes** (one check): the best is 1.45× noisier than coprime p = 2 at equal robustness (η 0.42 at SNR 6.8,
   or 0.376 at 9.4). The Johnson–Markl FC + tetrahedron is on the 5-point front (0.419 at 6.7). One check means one
   integer syndrome. Every wrap class must be a distinct integer, so the range grows only linearly with |c| while the
   margin falls as 1/|c|.
5. **Four encodes** have no detection (r = 0). Protection is range slack only. The best four-point is the Hadamard
   set with lattice (BCC) decoding, 1.25× better than referenced; Zhao et al. 2022 [V] make the same point with
   "perturbed" rows.
6. **Within seven encodes**, mixing axes also pays. Lattice A (§1.5) has three parity checks
   (3,−2,0,−1,−1,1,0), (2,0,1,−3,1,−1,0), (0,1,0,−2,1,2,−2), so no encode is a "reference" and no axis is singled out.
   It beats coprime p = 2 on noise (−13 %), SNR requirement (−8 %) and best-order cross-talk (0.014 vs 0.051 m/s) at
   the same maximum moment. It is the cheapest upgrade if the denoiser layout question is solved.
7. **When more encodes stop helping.**
   - (i) Thermal: in the bound, 7 → 8 gains a factor ∝ SNR_T^{−1/3}: 0.061 / 0.105 = 0.58 at SNR 10 (1.7×), and
     1.2–1.5× for realised designs (8H vs lattice B / p = 3). Only usable if SNR_T,eff ≳ 10.
   - (ii) Per-encode undersampling: R_e = K·R₁ for per-encode reconstructions. The toy's coronary error rose
     monotonically, 0.056 / 0.068 / 0.081 / 0.083 / 0.112 for K = 4 / 5 / 6 / 7 / 8 at R₇ = 12. None in the velocity
     model.
   - (iii) Moments: ≤ 1.5 % of TR.
   - (iv) Motion: train length +TR per encode. Modelled with sample-level time.
   - (v) Practical: the 7p denoiser layout and decoder code. Changing K needs a new decoder (the lattice/ML decoder
     in `enc_lib.Decoder` / `GridDecoder`; hasty_recon::flow already has an alias search) and either a retrained or
     encoding-agnostic denoiser. Velocity-space denoisers ((ρ, v) per voxel) are encoding-agnostic.
   - Net: **above 8 encodes nothing is worth it at our SNR**, and 6 is a legitimate alternative to 7 only for the
     per-encode reconstruction paths.

## 3. The calibration scan as a prior for the flow reconstruction [T]

### 3.1 What transfers

| from calibration | enters as | first-order effect on v | bias risk | use |
|---|---|---|---|---|
| B0 / ω (resp-resolved; χ model) | per-sample off-resonance in the operator | none (encode-common); fewer artifacts | map error > min(|ω|, 2.78/T) ≈ 40 Hz at T = 11 ms is worse than ignoring (model_robustness §1.2) | α-shrink + held-out |
| R2* | per-sample decay | none | small (weights only) | plug-in |
| ρ_w, ρ_f, fat spectrum | fat as a known/semi-known species | none directly; fat leakage biases v toward 0 in epicardial-fat-adjacent voxels | high near coronaries | fat image as nuisance with free smooth gain + motion; held-out per region |
| coil maps | S_c | none (encode-common) | coil motion with breathing (anterior array) | prefer flow-scan ESPIRiT / joint |
| anatomy edges | structural prior | depends on where the prior acts | high: misregistration moves edges | not from calibration (§3.3) |
| air / lung mask, χ | B0 model, support | none | low | yes |

**Principle (model_robustness §0.6, made exact here).** With x_e = ρ e^{i(θ + k_e·v)} and an encode-common operator
error E (A_θ̂ = A + E for every e), the error in each phase difference is a difference of the same linear map applied
to ρe^{iθ}(e^{ik_e·v} − e^{ik₀·v}). To first order it is linear in (k_e − k₀)·v, exactly as Prop. 4. Common errors
therefore bias v only by *spreading flow-weighted signal* (partial volume, fat or static tissue leaking into a vessel
voxel and pulling v toward the leaked velocity), never by a direct phase offset. So:
- calibration maps are low-risk for velocity bias and high-value for artifacts (fat at 440 Hz over a 9.5 ms readout is
  26 rad of evolution);
- the danger is regional: coronaries in epicardial fat, where a wrong fat map leaks zero-velocity signal.

### 3.2 What the calibration does *not* need to provide

- The flow scan's own data pooled over encodes and frames are ≈ 5× oversampled at 1 mm (20 k readouts vs 3.6 k
  Nyquist per volume). The time-mean magnitude, its edges, the static-tissue mask and a coil-map estimate therefore
  come from the flow scan itself, at the same pose (or per respiratory bin), contrast and resolution.
- That leaves the physics parameters (ω with χ, R2*, water/fat separation) as the calibration scan's unique content.
  The flow scan's single-echo-per-TR data cannot separate water and fat robustly; its 9.5 ms readout does carry B0
  information that can refine a smooth δω.

### 3.3 Structural priors without imposed edges

Setting: min_x ‖Ax − y‖² + λ R_g(x) with guide g. The linearised bias at x is
b = (AᴴA + λ∇²R)⁻¹ λ ∇R(x), as in denoiser_artifact_theory Thm 1. In well-determined directions it is suppressed;
in the null space the guide decides. Two failure modes:
1. **Suppression of non-shared edges.** For directional TV, R = Σ|(I − γξξᵀ)∇x| with ξ = ∇g/|∇g|_η, |ξ| < 1
   (Ehrhardt & Betcke 2016 [V]; γ is in later work). An edge of x that is absent from g pays the full TV price, so the
   prior is never worse than TV there. Schramm et al. 2018 [A/S] found small gains for guide-absent lesions.
2. **Transfer of guide-only edges.** Gradients along ξ cost (1 − γ) at a guide edge. Null-space content (artifacts,
   noise) can then form an edge there at no cost, and at high weight all priors transfer non-shared gradients
   (Schramm 2018 [S]). With R ≈ 50 the null space is huge, so this is real.

A misregistered guide combines both. The true edge is penalised and a ghost edge at the guide position is free. For a
3 mm coronary a 1–2 mm misregistration (respiratory state mismatch alone) is fatal. Rules:
- **Guide from the flow scan's own time-mean**, respiratory-resolved: no inter-scan motion, same contrast. Like
  PICCS (Chen et al. 2008 [A]) but used for edge *directions*, not intensities, to avoid the warm-start shrinkage of
  dynamics seen in R2 (box_recon.md).
- Act on ρ (magnitude/complex reference) only, never on v, and γ ≤ 0.9, so a guide-only edge still costs ≥ 10 % of
  TV. Joint sparsity across encodes (joint TV) is the encode-common special case. In the toy it gained 3–13 %
  (coronary 0.083 → 0.080 shared, 0.097 → 0.084 independent).
- **Tests** (the prediction-error test cannot see null-space content, denoiser_artifact_theory Prop. 10):
  - held-out k-space PE (model_robustness §3a);
  - the *guide-perturbation sensitivity*: reconstruct with g and with g shifted by ±1 voxel. A difference map
    concentrated at guide edges means edge transfer;
  - init-sensitivity;
  - accept only if PE improves and the sensitivity is below a pre-set fraction of the vessel's velocity.

### 3.4 Motion between calibration and flow: detect, compensate, decide

- **Detect.**
  - (i) Bellows: both scans record it. Respiratory-resolved maps are indexed by bellows state; the bellows→state map
    is recalibrated per scan from each scan's own k = 0 self-navigation.
  - (ii) Registration of a *synthesised* flow-contrast image: the calibration's ρ_w, ρ_f, R2*, ω through the flow
    sequence's signal equation (flip, TR, readout time). It is registered to the flow time-mean per respiratory bin:
    rigid → affine → B-spline. This is a monomodal problem, so a sum-of-squares metric is valid. Uncertainty comes
    from registering the two independent k-space halves of the flow data.
  - (iii) Navigators: the flow encode/tail lines and the B0 train's DC samples measure pose against each scan's own
    reference. Linking the scans needs a shared module (§3.5b).
  - (iv) Held-out consistency: transported maps (identity / rigid / deformable / α-shrunk) compared by paired held-out
    residual, the same machinery as model_robustness §3a.
- **Compensate.**
  - Rigid/affine motion is exact in k-space (rotate k, phase ramp, warp the coil maps; motion_navigators §2).
  - The B0 map does *not* move rigidly: susceptibility fields change with pose (≈ 10 Hz SD at 7 T for 5°/5 mm head
    motion, Liu et al. 2018 [A], ≈ 4 Hz at 3 T [E]), and respiration changes myocardial B0 significantly
    (Huang et al. 2026 [V partial]). Transport χ, not ω, and recompute the dipole field (b0_r2s_model_based option B).
  - Deformable motion: per respiratory bin, piecewise constant (model_robustness §0.8), uncertainty from the split
    halves. The map is shrunk toward "ignore" where the deformation uncertainty is high.
- **Decide** per region (heart / great vessels / liver dome / chest wall): accept a transported map component if the
  lower 95 % bound of the paired held-out residual gain is > 0 within that region. Otherwise fall back to the
  flow-data-only estimate (smooth δω from the flow readouts, fat as a free nuisance).

### 3.5 Sequence-side measures (best first)

- **(a) Embedded calibration TRs.** Replace a few TRs (every 16th–50th, 2–6 %) with a multi-echo bipolar train of the
  same TR, flip and net spoiler M0/M1, so the steady state and RF spoiling are unchanged. At 15.2 ms that is
  ≈ 12 lobes of 1.08 ms.
  - B0 needs ≈ 10⁴ coefficients, overdetermined 10⁴× even by a 30 s scan (b0_r2s_model_based §1), so 2 % of a
    10-min scan (12 s) is plenty for ω and its respiratory/cardiac/drift dynamics. There is zero inter-scan motion.
  - Costs: 2–6 % fewer flow readouts (thermal VNR −1–3 %), gradient-history changes in the next flow TR (modelled by
    `girf::played_moments` with history), and a scheduling slot.
  - The separate high-resolution R2*/fat scan remains for tissue properties.
- **(b) Shared navigator module.** Play the same short 3-axis line navigator (e.g. the flow encode curve of one encode)
  in both sequences, so poses are directly comparable across scans (motion_navigators §1).
- **(c) Sandwich.** Calibration 1.5 min before and after the flow scan. Agreement brackets the flow scan; disagreement
  flags a move and tells which half of the flow data each map belongs to.
- **(d) Same physiology handling.** Free breathing with bellows in both scans, respiratory-resolved maps, never one
  breath-hold map for a free-breathing flow scan.
- Not recommended: interleaving long calibration blocks (steady-state transitions, dummy TRs, no gain over (a)).

## 4. Ranked recommendations and pre-registered experiments

| # | recommendation | expected gain | risk | decision rule |
|---|---|---|---|---|
| 1 | Keep coprime 7-point p = 2 but with 8 % slack (V_ext 3 → 3.26 m/s in `seqdesign --v-ext`), shared sampling per rotation, GLS / lattice decode | peak-voxel aliasing threshold SNR_T 14.3 → 7.9 for −8 % VNR | none | – |
| 2 | Measure SNR_T,eff of the velocity-model output (E1) before any encoding change | sets 3–5 | none | below |
| 3 | Embedded calibration TRs (E4) | removes inter-scan motion for ω; drift tracking | −1–3 % VNR | adopt if ω held-out PE ≥ separate-scan PE and coronary error not worse |
| 4 | Structural prior from the flow time-mean on ρ, γ ≤ 0.9 (E5) | 3–13 % (toy) | edge transfer | PE gain > 0 and guide-shift sensitivity < 10 % of vessel speed |
| 5 | 7-point lattice A (same K, same max moment), with the lattice decoder (E2) | 1.15× VNR, −8 % SNR requirement, cross-talk 0.014 m/s best order | all-axis bipolars (PNS), 7p denoiser layout semantics | coronary ROI error −10 % with no rise in wrap fraction |
| 5b | If SNR_T,eff ≥ 11: 8H, lattice B/C or coprime p = 3 (E2) | 1.3–2× VNR | wraps, PNS, denoiser retrain | coronary ROI error −15 % with no rise in wrap fraction |
| 6 | 6-point lattice design only for per-encode reconstruction paths (E3) | equal VNR, R_e −14 %, one TR shorter train | new decoder + denoiser | coronary error −10 % at equal wrap fraction |
| 7 | Encode order from the phantom's acceleration covariance | cross-talk ÷2–8 in bin-group baselines | none | – |

Experiments (all at dev scale first: `prep_sim --src-step 2`, 32 coils, ph0001 train / ph9001 held out; simulator
physics and grid different from the reconstruction's, model_robustness §3e; one factor per run; config fingerprint
logged):
- **E1. Effective SNR.**
  - Two noise realisations of the same dev scan; velocity-model (or best available) reconstruction of both.
  - Per-voxel phase-difference variance between the realisations → SNR_e,eff. Also the frame-mean and 20-voxel-ROI
    versions, and the artifact part (noise-free run minus truth).
  - Question: what SNR_T,eff does the time-mean and the ROI reach?
  - Decision: < 9 keep p = 2; 9–11 keep p = 2 and revisit after the reconstruction improves; > 11 run E2.
- **E2. Encoding at equal scan time.**
  - coprime 7 p = 2 vs lattice A vs 8H vs coprime 7 p = 3: simphantom with `Scheme` added (hasty-seq has Coprime and
    Hadamard sets; 8H is `Coprime { layout: EightHadamard }`; lattice A needs a `Scheme::Lattice(z, κ₀)` variant),
    same TR count.
  - Scores: roi_score coronary ROI-mean |err|, correlation, wrap fraction (|error| > 0.5 VENC_lo), deep-lumen
    dynamics slope.
  - Decision: adopt if the coronary |err| falls ≥ 15 % (paired over the 20 frames, bootstrap 95 % CI excludes 0) and
    the wrap fraction does not rise.
- **E3. Sharing and K for per-encode baselines.**
  - framerecon with (a) one rotation per encode set (today) vs (b) independent rotations per encode;
  - then the 6-point design vs 7-point at equal readouts.
  - Decision: keep the shared order unless (b) is better by ≥ 10 %. 6-point only if ≥ 10 % better.
- **E4. Embedded calibration.**
  - Flow sim with 3 % calibration TRs vs separate calibration with a simulated 3 mm / 2° move between scans
    (needs plan_motion_sim).
  - Scores: ω error in the heart and held-out PE; coronary velocity error.
  - Decision as in table row 3.
- **E5. Structural prior.**
  - none / flow-time-mean dTV / calibration-water dTV (deliberately 2 mm misregistered as a negative control);
  - held-out PE, guide-shift sensitivity, coronary error.
  - Decision as row 4; the negative control must be flagged by the sensitivity test, or the test is not trusted.
- **E6. Decoder.** Replace roi_score's hierarchical decode (valid only to ±VENC_hi = 1.63 m/s) by the CRT/lattice
  decoder (hasty_recon::flow alias search) and report the voxels where they differ. No decision; a correctness check.

## 5. Literature (2026-11-17 pass)

- Pelc, Bernstein, Shimakawa, Glover, JMRI 1(4):405, 1991, 10.1002/jmri.1880010404 [A]: referenced vs balanced 4-point;
  six-point inefficient.
- Lee, Pike, Pelc, MRM 33:122, 1995, 10.1002/mrm.1910330119 [A]: aliased high-M1 + unaliased low-M1, 4× VNR for
  +50 % time.
- Johnson & Markl, MRM 63:349, 2010, 10.1002/mrm.22202 [V]: 5-point = FC + tetrahedron, VNR 1.63× theory, the 5th point
  at low resolution costs < 1 % time; unwrapping fails at strong background phase.
- Zwart & Pipe, MRM 69:1553, 2013, 10.1002/mrm.24390 [A]: multidirectional high-moment, unaliasing by redundancy,
  ≈ 1.7× dual-VENC VNR efficiency. Loecher & Ennis, MRM 80:42, 2018, 10.1002/mrm.26997 [A]: non-convex Laplacian
  regularised reconstruction allows 2–4× lower VENC.
- Haraldsson et al., MRM 79:1962, 2018, 10.1002/mrm.26853 [V]: ICOSA6 (FC + 6 icosahedral), integer-search unwrapping.
- Binter et al., MRM 69:1337, 2013, 10.1002/mrm.24370 [A]: Bayesian multipoint, robust to undersampling.
- Nett et al., JMRI 35:1462, 2012, 10.1002/jmri.23588 [V]: two 4-point PC-VIPR scans; in vivo high-VENC undersampling
  limited by artifacts. Schnell et al., JMRI 46:102, 2017, 10.1002/jmri.25595 [V]: 7-point shared reference.
- Callaghan et al., JMRI 43:352, 2016, 10.1002/jmri.24991 [A]: three VENCs, per-voxel selection.
- Carrillo, Osses, Uribe, Bertoglio, IEEE TMI 38:1263, 2019, 10.1109/TMI.2018.2882553 [A]: ODV, grid-search cosine
  cost, ratio ≈ 2/3. Franco et al., JMRI 57:763, 2023, 10.1002/jmri.28305 [A]: β = 1/2 most noise-robust. Herthum et
  al., MedIA 78:102416, 2022, 10.1016/j.media.2022.102416 [A]: N-encoding generalisation.
- Ma, Markl et al., MRM 83:505, 2020, 10.1002/mrm.27943 [V]: 7-point triple-VENC via the intrinsic VENC
  vL vH/(vH − vL), no extra time.
- Moersdorf et al., Magn Reson Imaging 61:73, 2019, 10.1016/j.mri.2019.05.015 [A]: undersampled multi-VENC 4D flow.
- Beghella Bartoli, Boccalini, Chechin, Boussel, Douek, Garcia, Sigovan, JCMR 27:101871, 2025,
  10.1016/j.jocmr.2025.101871 [V]: coprime p/(p+1), interleaved 8 points; Eq. 6 weights are not inverse-variance.
- Zhao, Ahmad, Potter, ISBI 2022, 10.1109/ISBI52829.2022.9761589, arXiv:2111.03990 [V]: the unambiguous set is a lattice
  cell; perturbed rows extend the range. Zhao et al. (PRoM) IEEE TMI 41:3712, 2022, 10.1109/TMI.2022.3193132 [V
  partial]: approximate ML over congruences.
- Loecher, Schrauben, Johnson, Wieben, JMRI 43:833, 2016, 10.1002/jmri.25045 [A]: 4-D Laplacian unwrapping.
- Undersampling and velocity: Bollache et al., MRM 79:195, 2018, 10.1002/mrm.26661 [A]; Pathrose et al., MRM 85:2174,
  2021, 10.1002/mrm.28561 [A]; Neuhaus et al., JCMR 21:65, 2019, 10.1186/s12968-019-0573-0 [A]. Peak velocity and
  flow are underestimated with CS acceleration. No paper separating artifact from thermal error versus K was found.
- Structural priors: Ehrhardt & Betcke, SIAM J Imaging Sci 9:1084, 2016, 10.1137/15M1047325 [V]; Ehrhardt et al., IEEE
  TMI 35:2189, 2016, 10.1109/TMI.2016.2549601 [A/S]; Ehrhardt et al., Inverse Problems 31:015001, 2015 [S]; Haldar et
  al., MRM 59:810, 2008, 10.1002/mrm.21536 [A]; Bilgic et al., MRM 66:1601, 2011, 10.1002/mrm.22956 [A]; Knoll et al.,
  IEEE TMI 36:1, 2017, 10.1109/TMI.2016.2564989 [A]; Chen, Tang, Leng, Med Phys 35:660, 2008, 10.1118/1.2836423 [A];
  Schramm et al., IEEE TMI 37:590, 2018, 10.1109/TMI.2017.2767940 [A/S]; Vunckx et al., IEEE TMI 31:599, 2012,
  10.1109/TMI.2011.2173766 [A].
- Motion / field: Kober et al., MRM 66:135, 2011, 10.1002/mrm.22797 [A]; Wallace et al., MRM 81:258, 2019,
  10.1002/mrm.27381 [A]; Liu et al., MRM 80:2538, 2018, 10.1002/mrm.27339 [A]; Huang et al., MRM 95:112, 2026,
  10.1002/mrm.70026 [V partial]; Dietrich et al., MRM 87:2621, 2022, 10.1002/mrm.29144 [A].
- Not found: a 4D-flow sequence with in-scan multi-echo B0 mapping (2-D PC-IDEAL/Dixon exists: Denecken et al., NMR
  Biomed 2025, 10.1002/nbm.70049 [A]), a reference-free six-point paper, or a 5/6-encode extended-range design.
