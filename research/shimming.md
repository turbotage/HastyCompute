# Shimming for the reconstruction: what matters, what helps, experiments (2026-11-08)

## 1. What the reconstruction needs from the field

The segment count is set by the off-resonance span over the signal-carrying voxels times the readout length,
L ≈ 1.3 Δf·T_ro + 3.5. On the sim_point field the optimal (SVD) basis needs about half a segment less than fixed-time
segments; the rank bound Δf·T holds for any separable basis.

Some parts of the field cost the operator nothing; only the rest needs segments:

| field part | in the reconstruction | cost |
|---|---|---|
| constant ω0 | per-sample phase φ0 += ω0 τ_j | free, exact |
| linear g·r | per-sample k-shift k_j += g τ_j | free, exact |
| quadratic, degree 3–4 | per-sample polynomial phase in the stencil (ΔC_j += H τ_j; order-3 slot with u_j = τ_j) | free while \|χ\| ≲ 0.7 rad at the end of the readout (Taylor), wider with least-squares stencils |
| local structure (heart–lung interface, cardiac veins, chest wall / liver) | time segments | ∝ L |

Consequence:
- **Global spherical-harmonic shims (degree ≤ 2, even ≤ 3) cannot lower L**, because the reconstruction can absorb
  the same polynomials per sample. They still matter physically: intravoxel dephasing / T2*, fat suppression, and they
  keep the polynomial part small enough for the Taylor stencil.
- **Only local field control lowers L**: local shim coils, or an integrated receive / shim ("AC/DC", iPRES) array.
  Our 32-loop receive array is the natural candidate in simulation.
- An earlier measurement (concomitant_segmented_first_order.md) found little gain from moving the field's polynomial
  part into the trajectory: deg 1 3.2e-3 → 2.7e-3, deg 2 worse, from the Taylor floor. That field was already
  2nd-order shimmed, so this is the same statement.

## 2. Literature (3 T heart)

- After 2nd-order volume shimming at 3 T the remaining off-resonance over the heart is typically **40–60 Hz SD, or
  about 90–200 Hz peak-to-peak**:
  - LV 176 ± 30 Hz and RV 121 ± 31 Hz peak-to-peak with localized linear + 2nd-order shim;
  - an algorithm tailored to the heart geometry: 235 → 86 Hz;
  - B0 / B1 mapped in one breath-hold, localized 2nd-order shim + F0: 302 → 194 Hz.
  [Huang 2026; Schär 2008; Hock 2021]
- **Shim volume:** manual shim boxes often include liver, fat and chest wall with large deviations. Image-based
  shimming over a heart contour beats the box. [Huang 2026]
- **Motion-adaptive (respiratory-state) shimming** at 3 T: shim currents computed per respiratory state from
  motion-resolved B0 maps, then selected in real time from the respiratory sensor.
  - End-expiration SD: 56.9 Hz (scanner volume shim) → 39.8 Hz (customised and motion-adaptive).
  - End-inspiration: 57.0 → 38.1 Hz.
  - The field is consistent over the cardiac cycle. [Huang 2026; van Gelderen 2007 for the brain]
- **7 T heart:** cardiac-phase-specific shimming in simulation; dynamic shim updates need a dynamic shim interface,
  pre-emphasis and dedicated amplifiers. [Hock 2021]
- **Local shim coils (UNIC, ISMRM 2020):** myocardium 62.9 ± 57.3 Hz → 30.3 ± 40.5 Hz. This is the size of change
  that would move L.
- **Integrated receive / B0-shim arrays (AC/DC, iPRES):**
  - Introduced 2013. Built as 8- and 31-channel shim systems inside 32-channel 3 T receive arrays, and a 31-channel
    7 T array.
  - Improve brain B0 homogeneity without hurting RF performance.
  - Add little size or weight compared with higher-order spherical-harmonic inserts.
  - No cardiac/body version found in this search: an open niche. [Han / Stockmann / Truong; Sci Rep 2020]

## 3. Our phantom's shim is idealised

- `tools/phantom/prep_sim.py` (realistic / factory phantoms) fits **10 monomials**
  (1, x, y, z, x², y², z², xy, xz, yz) over a tissue box.
- That set contains x² + y² + z², which is not harmonic: no shim coil produces it. It also has no current limits.
- The simulated field is therefore shimmed slightly better than any scanner can. `hasty-phantom::b0::shim` (scene
  path) uses the 9 harmonic terms correctly.
- To fix later, so as not to change the running simulation: the harmonic basis, scanner current limits, and a
  heart-ROI shim volume.

## 4. Library (written 2026-11-08, not yet compiled or run)

- `hasty_phantom::shimfit`:
  - bases: `harmonic_basis` (solid harmonics ≤ 3rd order, per-degree limits) and `loop_basis` (DC B_z of receive
    loops, Hz/A, current limit), combined with `join`;
  - `fit`: weighted p-norm shim (IRLS for p > 2, which targets the span), box-constrained by coordinate descent,
    ridge;
  - `field_stats` (weighted SD, 1–99 % span, peak-to-peak) and `segments_for(span, T)`;
  - unit test: exact removal, limits, p > 2 lowers the peak-to-peak.
- `hasty_nufft::build_segments_svd_weighted(..., voxel_w, time_w, ...)`:
  - voxel weights (the C++ `L2Energy` / `Count` weightings);
  - readout-time weights: the k-space data energy per ADC time, see §6.

## 5. Experiments (CPU only, minutes; run when compute is allowed)

1. **Baseline:** realistic phantom field before shim, with harmonic shims of degree 1 / 2 / 3, scanner-like limits,
   over (a) the current tissue box, (b) a heart + great-vessel ROI, (c) signal-energy weights.
   Report SD, span and L for T = 3.6 ms.
2. **Span-targeted objective:** p = 2 against p = 8 for the same hardware: L against SD.
3. **AC/DC loop shim:** the 32 receive loops as DC elements (limits 1–3 A), alone and on top of the 2nd-order
   harmonics. Report the heart-ROI span and L. Check that the loop fields stay small in the coil-sensitivity sense
   (they are the same loops; DC does not change B1⁻).
4. **Respiratory states:** the B0 scan's breathing bins give per-state fields. Report the shim per state and how much
   of the span is respiratory. The reconstruction already uses per-bin field maps, so this mainly tells whether a
   motion-adaptive shim would also lower L.
5. **Basis weighting:** `build_segments_svd_weighted` with signal voxel weights and data-energy time weights, against
   the plain SVD and fixed times. Measure the **data-space** model error ‖(E − Ẽ)x‖ / noise on the realistic scan, not
   the image-space error.

## 6. Notes on non-separable bases

- Any method whose channels are full-grid FFTs needs at least the numerical rank of [e^{−iω(r)τ}] channels, so
  separable SVD is optimal in that class.
- Non-separable components that are cheap:
  - (i) the polynomial part, absorbed per sample (§1);
  - (ii) weighting the fit by what the data actually see. A centre-out readout carries most of its energy early,
    while the accumulated phase, and hence the rank, is largest late. The data-norm model error weights late τ by
    the data energy there, which shortens the effective T in the rank bound.
- Local sub-box channels for the field's tails (spatially compact near air) cut FFT work but not gather work: each
  sample still gathers W³ taps per channel. Gain ≲ 15 % for considerable complexity; not pursued.

Sources:
- [Huang et al., Enhancing cardiac MRI reliability at 3 T using motion-adaptive B0 shimming, MRM 2026 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12892406/)
- [Huang et al., MRM 2026 (Wiley)](https://onlinelibrary.wiley.com/doi/10.1002/mrm.70026)
- [Hock et al., B0 shimming of the human heart at 7T, MRM 2021](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.28423)
- [Schär et al., Simultaneous B0/B1 map acquisition for localized shim, frequency and RF power in the heart at 3 T](https://www.researchgate.net/publication/41124711_Simultaneous_B-0-_and_B-1-Map_Acquisition_for_Fast_Localized_Shim_Frequency_and_RF_Power_Determination_in_the_Heart_at_3_T)
- [UNIC local shim coils, ISMRM 2020 abstract 2183](https://cds.ismrm.org/protected/20MProceedings/PDFfiles/2183.html)
- [van Gelderen et al., Real-time shimming to compensate for respiration-induced B0 fluctuations, MRM 2007](https://onlinelibrary.wiley.com/doi/10.1002/mrm.21136)
- [Stockmann et al., A 31-channel integrated AC/DC B0 shim and RF receive array coil for 7T (PubMed)](https://pubmed.ncbi.nlm.nih.gov/34632626/)
- [Integrated RF/shim coil array for parallel reception and localized B0 shimming in the human brain (NeuroImage)](https://www.sciencedirect.com/science/article/abs/pii/S1053811914007940)
- [iPRES with multiple shim loops per RF coil element (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5107366/)
- [An integrated RF-receive/B0-shim array coil boosts whole-brain MRSI at 7 T, Sci Rep 2020](https://www.nature.com/articles/s41598-020-71623-5)

## 7. First results (factory_arms/ph0001, 3 T, 2026-11-09)

Unshimmed χ-dipole field (`tools/phantom/shim_field.py`), shims fitted with `shimstudy` at 4 mm. Spans are 1–99 %;
L comes from the PD-weighted signal span, T_ro = 3.56 ms.

**Realism check**, myocardium only (32.6 k voxels at 2 mm):

| shim | SD | span | peak-to-peak |
|---|---|---|---|
| none | 61.5 Hz | 332 Hz | 982 Hz |
| SH ≤ 1 fitted on the myocardium | 52.4 Hz | 305 Hz | 882 Hz |
| SH ≤ 2 fitted on the myocardium | **46.5 Hz** | 269 Hz | 841 Hz |

This matches the literature (40–60 Hz SD after a 2nd-order shim), so the phantom's field is realistic.

**Shim configurations** (heart ROI = heart labels plus tissue within 12 mm of the myocardium):

| configuration | heart SD / span (Hz) | signal SD / span (Hz) | L |
|---|---|---|---|
| no shim | 81.6 / 482 | 86.4 / 418 | 6 |
| SH ≤ 2, prep_sim box | 80.4 / 469 | 82.8 / 410 | 6 |
| SH ≤ 2, signal-weighted | 80.6 / 474 | 82.0 / 415 | 6 |
| SH ≤ 2, heart | 74.4 / 460 | 210 / 1055 | **9** |
| SH ≤ 3, heart | 69.8 / 445 | 727 / 3850 | 22 |
| SH ≤ 2 + 32 loops (±2 A), heart | 71.0 / 452 | 325 / 1844 | 13 (25 loops at the limit) |
| SH ≤ 2 + 32 loops, signal | 78.3 / 469 | 75.7 / 393 | 6 |

Findings:
- The field is dominated by local susceptibility structure. Global shims change the signal-weighted SD by ≤ 5 %.
- **A heart-targeted shim makes everything else worse.** The segment basis has to cover all tissue the readout
  sees, so heart-only shimming raises L from 6 to 9 (3rd order: 22).
- For the reconstruction, the shim objective should be the whole signal support. That gives about nothing.
- The heart-targeted shim helps physics inside the heart (T2*, fat suppression), not the operator.
- The receive loops at ±2 A are too weak: 25 of 32 sit at the limit for the heart ROI. Local shim coils would need more
  current or more loops close to the heart, and loops on the body surface are 6–10 cm from the coronaries.
- p = 8 lowers the heart peak-to-peak (1248 → 1028 Hz) but raises SD and span. The 1–99 % span that sets L is not
  the maximum.

## 8. Target scanner: GE SIGNA Premier 3T (correction, 2026-11-09)

The earlier practical notes assumed Siemens. The likely scanner is a GE SIGNA Premier 3T.
- **Hardware:** linear X/Y/Z (gradient offsets) plus 5 second-order shim coils (XY, ZX, ZY, Z2, X2−Y2) integrated in
  the gradient coil; GE describes high-order shim terms "for B0 inhomogeneity correction across the entire body".
- **Workflow:**
  - Auto prescan does the centre frequency and a volume shim (prescribed shim volume).
  - High-order shim (HOS) computes the 2nd-order currents from a field map.
  - Manual prescan gives access to the transceiver settings. Entering 2nd-order currents by hand is not documented
    publicly (operator manual or research agreement).
  - autoHOS (Premier, 3T and 7T) removes the manual shim-volume placement.
- **Custom shims / dynamic shimming:** EPIC sequence code (research agreement). Pulseq sequences run through the
  community pge2 interpreter, whose EPIC source GE now hosts for research sites.
- §1–§7 are unchanged: the shim basis on the Premier is the same 9 terms (≤ 2nd order), so the study's
  "SH ≤ 2" rows apply directly.

Sources:
- [GE SIGNA Premier spec sheet (5 second-order shim coils)](https://nyspi.org/sites/default/files/inline-files/SIGNA%20Premier_Spec%20Sheet_FDA%20Cleared_Final%20071717.pdf)
- [GE SIGNA Premier product page](https://www.gehealthcare.com/en-us/products/magnetic-resonance-imaging/3t-mri-scanners/signa-premier-wide-bore-mri-scanner)
- [Automated High-Order Shimming for Neuroimaging Studies (autoHOS, Premier)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10748357/)
- [Manual B0 shimming guide on GE (Premier, manual prescan)](http://mpijournal.org/pdf/2023-01/MPI-2023-01-p081.pdf)
- [HarmonizedMRI/pge2 (Pulseq → GE)](https://github.com/HarmonizedMRI/pge2)
- [Pulseq on GE](https://toppemri.github.io/)
- [Pulseq stack-of-stars for the SIGNA Premier](https://github.com/uw-rad-mitrp/Pulseq-Stack-of-Stars)
