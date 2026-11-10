# Is a phase-contrast reconstruction a biased velocity estimator? (plan, 2026-11-07)

Question (user, 2026-11-07): with the LBM beat library as truth, does the reconstruction give a biased estimate of the
velocity, and can we explain the bias theoretically?

## 1. What one voxel measures

Noise-free, fully sampled, one encode with first-moment sensitivity k (rad per m/s, k = π / VENC), the voxel signal
of encode e relative to the reference is

    S_e / S_0 = Σ_j w_j s_j e^{i k_e·u_j} / Σ_j w_j s_j

over the spins j in the voxel (point-spread weights w_j, relative M_z s_j). Write p(u) ∝ Σ_j w_j s_j δ(u − u_j), the
signal-weighted velocity distribution, and φ(k) = E_p[e^{i k·u}] its characteristic function. The measured velocity
along ê is v̂ = arg φ(k ê) / k. The cumulant expansion of log φ gives

    arg φ(k ê) = k κ1 − k³ κ3 / 6 + k⁵ κ5 / 120 − …   (κ_n: cumulants of ê·u under p)

so

    v̂ = κ1 − (k² / 6) κ3 + O(k⁴)    with  κ1 = E_p[ê·u],  κ3 = E_p[(ê·u − κ1)³].

Three separate biases against the volume-mean velocity ū = E_vol[u]:
1. **Signal weighting:** κ1 = E_p[u] ≠ E_vol[u] when s (inflow, saturation) or w (partial volume, PSF) correlates
   with u. To first order, κ1 − ū = cov_vol(s, u) / E_vol[s] — exactly the stored `c_su` / `mz` of the library.
2. **Skewness:** −(π / VENC)² κ3 / 6. Jets, shear layers and recirculation are skewed; the lower the VENC, the larger
   the bias (it grows as VENC⁻²). With dual VENC the low-VENC phase is unwrapped against the high one, so the
   low-VENC skew bias enters the estimate.
3. **Magnitude loss** |φ| = exp(−k² σ² / 2 + …): intravoxel dephasing lowers the SNR of the phase but not its mean;
   at |φ| → 0 (σ ≳ VENC) the phase becomes meaningless (signal void).

Gated reconstructions add a fourth: a cardiac bin mixes several heartbeats and a window of phases. The complex data
average, not the velocities: p becomes the mixture over the bin's beats and phases, so beat-to-beat variability
enters κ3 (and κ1 through the signal weights).

The reconstruction itself adds (5) regularisation bias (LLR / learned prior shrink small structures, smear the lumen
edge), (6) the decoder (unwrapping errors are not small biases but outliers), (7) model errors (GIRF, B0,
concomitant fields when not modelled).

## 2. What the simulator gives us

simphantom with `--beats` draws per covered 1 mm cell six sub-sources matching mean, covariance and the axial third
moments of the cell's distribution in that beat and bin, each carrying s through the regression on u (exact E[s],
E[s u]). Hence for a voxel made of such cells it reproduces terms 1–4 to the order the moments allow (κ3 along the
principal axes exact, the full third-moment tensor not).

## 3. Measurement design

On the reconstruction grid, per frame, three truths from the library (over the scan's heartbeat sequence, the beats
in each frame's phase window, the 1 mm cells in each voxel with their coverage):
- `ū` volume mean (what sim_truth gives now, with `--chamber-flow lbm:<run>`);
- `κ1` signal-weighted mean (s from the spin history);
- `v̂_ideal` = arg of the noise-free PC signal of the six-point sources per encode (the ideal, infinitely sampled,
  unregularised measurement).

Then reconstruction − v̂_ideal = reconstruction bias (5–7); v̂_ideal − ū = measurement physics (1–4), split into
κ1 − ū (signal weighting) and v̂_ideal − κ1 (skew and mixing). Report against VENC (1.0 vs 1.5 m/s pairs), against
SNR (noise off / on), and by region (jets, recirculation, laminar great vessels). The theory predicts the skew part
∝ VENC⁻² and the signal-weighting part independent of VENC — a direct test.

Status (2026-11-07): `rustlib/tools/phantom/bias_truth.py <phantom> <sim_dir>` writes the three truths
(`truth_bias.npz`). First numbers, ph0001 dev scans (VENC 1.0 / 1.5, 2.5 mm): skew + mixing p95 0.8 cm/s, signal
weighting (gadolinium, T1 0.39 s) p95 0.46 cm/s — small. Larger effects are expected for native blood (ph0103), low
VENC, larger voxels and stenosis jets (ph0102); the reconstruction part needs scans of minutes.
