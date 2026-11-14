# PVS / ePVS segmentation model survey (inference-only), 2026-11-13

V = verified from fetched README/paper; I = inferred / not checked.

| Model | Input | Res/preproc | Arch / framework | Weights | Licence (code / weights) | Accuracy (Dice) | Export |
|---|---|---|---|---|---|---|---|
| SHIVA-PVS (pboutinaud/SHIVA_PVS) | T1 (v4/v5), T1+FLAIR (v4); legacy v2 T2 via inversion (V) | 1 mm iso, crop 160x214x176, min-max to 99th pct; v5 works with/without skull-strip (V) | 3D U-Net, TF/Keras 3 SavedModel (TF>=2.17) (V) | Not in repo; cloud.efixia.com (V) | code not stated; weights CC BY-NC-SA 4.0 (V); SHiVAi wrapper AGPLv3+ (V, search snippet) | v0 paper: voxel 0.51 WM / 0.66 BG; cluster 0.64/0.71; >0.9 for PVS >10 mm3 (V) | tf2onnx likely feasible (I) |
| Pham nnUNet-Perivascular-Spaces (wpham17) | T1, T1+FLAIR, T2, T2+FLAIR (separate models) (V) | voxel-spacing-agnostic; optional NLM+AHE; custom reader/normalisation files to copy into nnU-Net (V) | nnU-Net v2, ResEnc-L plans, 3d_fullres (V) | Google Drive folder, zip for nnUNetv2_install_pretrained_model_from_zip (V) | repo MIT, models CC BY-NC-SA 4.0 (V, README ambiguous) | T1 79.2, T1+FLAIR 75.6, T2 84.7, T2+FLAIR 77.5 (V; metric definition unchecked) | nnU-Net to ONNX non-trivial, run native (I) |
| MedNet-PVS (iBrain-Lab) | T1 or T2 (V) | `_0000.nii.gz` naming, little preproc stated (V) | MedNeXt-L-k5; one nnU-Net T2 model; PyTorch (V) | Google Drive via Releases (V) | Apache-2.0 (V; weights licence not stated) | T1 WM 0.51 vox/0.64 count; T2 HCP-Aging 0.88-0.89 (WM only) (V) | PyTorch, TorchScript/ONNX plausible (I) |
| DRIPS (CIR-FAU) | T1 and T2, any anisotropy (domain randomisation) (V) | native res, synthetic voxel sizes 0.5-4 mm (V) | 3D U-Net, Keras/TF, Docker image (V) | Not mentioned in paper (V); presumably in Docker (I) | not verified | median voxel Dice 0.39-0.64 across cohorts (V) | I |
| PINGU / mc-PVSnet | T1 | - | nnU-Net | Not checked | not checked | - | - |
| VALDO Task 1 (MICCAI 2021) | T2 (V) | - | nnU-Net/MaskRCNN/UNet entries (V) | winner not identified; most teams released docker containers (V) | per team | - | - |
| Frangi/FSL-style | T1/T2 | - | non-DL | n/a | varies | lower, parameter sensitive (I) | n/a |
| SynthSeg / WMH-SynthSeg | context only; no PVS class (I) | | | | | | |

Not checked: Frontiers 3T-T2 Parkinson model repo (reported DSC 0.702, weights on GitHub per search snippet), 7T-specific models, other multi-contrast variants, GPU memory figures (not stated by any source).

## Ranking
1. **Default: Pham nnU-Net family.** Covers T1, T1+FLAIR and T2 with one runtime, best reported Dice, updated Nov 2025, voxel-spacing agnostic. Cost: nnU-Net v2 runtime plus custom reader/normaliser files; Google Drive weights (mirror them into the repo data dir); weights non-commercial.
2. **Alternate: SHIVA-PVS v5 (T1).** Most validated (cohort-scale, histology-adjacent), stable preprocessing contract, mono-T1. Needs TensorFlow 2.17+ (or convert to ONNX), fixed 160x214x176 crop at 1 mm.
3. **Alternate: MedNet-PVS** (Apache-2.0 code, PyTorch, T2 WM Dice ~0.88). Best licence story for code, but T1 accuracy is low and T2 is WM-only. Confirm weights licence before any commercial use.
   DRIPS is worth keeping as a robustness fallback for anisotropic clinical T2 if its Docker weights can be extracted.

## Red flags
- Every high-accuracy model has weights under CC BY-NC-SA 4.0 (SHIVA, Pham). Research use only; share-alike may apply to derived/fine-tuned weights. MedNet-PVS is the only Apache-2.0 option, and that covers code only as far as I verified.
- Weights hosted on Google Drive and cloud.efixia.com: link rot risk; mirror locally.
- Pham repo licence is internally inconsistent (MIT sidebar vs CC BY-NC-SA in text); treat as non-commercial.
- SHIVA README mentions SWI/T2GRE in its step-by-step section, which contradicts T1/FLAIR inputs; test before relying on it.
- Dice numbers are not comparable across papers (voxel vs lesion/count, different cohorts).
- No source gave GPU memory or an official ONNX/TorchScript export; all export feasibility is inferred.

Sources: github.com/pboutinaud/SHIVA_PVS; github.com/pboutinaud/SHiVAi; github.com/wpham17/nnUNet-Perivascular-Spaces; github.com/iBrain-Lab/MedNet-PVS; PMC12633610 (DRIPS); arxiv.org/abs/2208.07167 (VALDO); arxiv.org/abs/2411.19564 (nnU-Net PVS framework); arxiv.org/abs/2405.08337 (PINGU).
