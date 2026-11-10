# From simulation to a usable 4D-flow pipeline (plan, 2026-11-09)

Goal: accurate coronary flow from a real scan ([[coronary-flow-goal]]). Target scanner, if any: GE SIGNA Premier 3T
(Pulseq through the pge2 interpreter). Everything below must work without labels, truth maps or simulator outputs:
every input the reconstruction uses has to come from the scan session itself.

## What is still oracle or missing

| input | now | needed for a real scan |
|---|---|---|
| trajectory, ADC timing | simulator sample tables | our own sequence export (Pulseq → pge2) + the same tables written at design time; GIRF-predicted played waveforms |
| geometry (FOV centre, rotation, table) | meta.json | GE raw-data header → scanner frame |
| coil noise Ψ | simulated | noise prescan / noise-only ADC |
| coil maps | ESPIRiT (done), true maps as oracle | ESPIRiT from the scan's own calibration region (done); joint maps later |
| B0 / R2* / fat | recon_b0 on our B0 scan (done) | the same B0 sequence on the scanner, per respiratory state |
| concomitant fields | from the waveforms (done) | GIRF-filtered measured waveforms |
| gradient nonlinearity | simulator's model | the Premier's gradwarp coefficients (GE, research agreement) or a phantom calibration |
| cardiac timing | simulated ECG / PPG | GE physio logs (PPG; delay to the R-wave) |
| respiration | **ignored in the flow reconstruction** | bellows / self-gating: soft gating or motion compensation |
| support, weights | warm start (data-derived, done) | same |
| prior | LLR / learned denoiser trained on phantoms | the denoiser validated on held-out phantoms, then on real data |

## Work packages, in order

**P1 — real-data path (blocking any scan)**
1. Sequence export: hasty-seq → Pulseq .seq → `pge2.check` / `pge2.validate`.
   - GE raster times (4 µs gradients) and the Premier's limits (80 mT/m, 200 T/m/s) instead of the pypulseq example.
   - GE's PNS model / per-scanner PNS weights.
   - The flow sequence and the B0 sequence both go this way.
2. Raw-data reader: GE ScanArchive / P-file → our scan format (`SimData`-compatible: readouts, data, Ψ, geometry,
   physio).
   - The trajectory is ours (design-time tables), not the scanner's.
   - kload / KStore then stream it unchanged.
3. Physio import: PPG triggers and bellows from GE's physio files.
   - Cardiac phase from PPG with a fitted pulse delay (the simulator already writes PPG and bellows as a scanner
     records them).
4. GNL: gradwarp coefficients for the Premier, or a one-off phantom calibration fitted to a spherical-harmonic model.
5. Mock end-to-end test: write a simulated scan in the GE raw format and run the whole chain from that file.

**P2 — respiratory motion in the flow reconstruction (largest accuracy gap for coronaries)**
6. Soft gating: bellows / self-gating amplitude → per-readout weights (end-expiration favoured) in the data term.
   - Cheap: per-sample weights already exist (`WeightedSets`).
7. Respiratory-binned field maps: recon_b0 bins → per-readout field by respiratory state (FieldMaps per bin exists;
   wire it per readout in the segments).
8. Motion-compensated flow reconstruction: per-bin affine or deformable warps (recon_b0 `--moco` has the machinery)
   applied in the flow operator.

**P3 — automatic, robust reconstruction settings**
9. Segments:
   - `svdw` by default once the 10-min evaluation confirms one segment fewer at equal error;
   - choose L from a data-space error target with a worst-voxel guard;
   - weights `count` or mild `mass`.
10. Linear B0 part into the per-sample k-shift: exact; a guard against heart-targeted or poor shims.
11. Iterations / regularisation from the discrepancy principle (noise level known after whitening) instead of fixed
    counts.

**P4 — prior**
12. Train the denoiser on the realistic phantoms (diffeomorphic variants, held-out ph9001) and evaluate it on the
    10-min scan against LLR.

**P5 — outputs**
13. Flow quantification: vessel tool / centrelines → flow rate and net forward volume through planes, with
    uncertainty (bias study, research/velocity_bias.md).
14. Export: NIfTI / DICOM-compatible 4D velocity for external tools.

**P6 — throughput**
15. Reconstruction time of the 10-min scan (the evaluation's smoke measures it); multi-GPU set partitioning if
    needed (the stores are already shareable).

## Validation along the way

- Every P1 item gets a simulated counterpart: the simulator writes GE-format raw data, physio logs and a GNL model, so
  the real-data path is tested before any scanner time.
- The 10-min evaluation (tools/recon/eval_10min.sh) is the reference: oracle vs estimated fields / coil maps, then
  each P2/P3 change is measured against it.
