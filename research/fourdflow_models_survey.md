# Pretrained 4D-flow denoising / super-resolution models (survey 2026-11-13)

Scope: models for velocity-field denoising / SR with weights you can download today. Verified = read from the repo/figshare
API/file itself; (S) = from a paper or search snippet only. Licences are for what is stated; weights usually carry none of their own.

## Weights you can download now

| Model | What | Weights | Licence | Framework / format | Notes |
|---|---|---|---|---|---|
| **4DFlowNet** (Ferdian 2020, EdwardFerdian/4DFlowNet) | spatial SR ×2 (any int; ×1 = denoise), residual 3D CNN, 3.34 M params | figshare 12253424, `4DFlowNet.h5` 13.6 MB (verified, downloaded) | code MIT, figshare **CC BY 4.0** | Keras 2.x HDF5 with full `model_config` | trained on aortic CFD; input patch 24³, u,v,w (venc-normalised) + 3 magnitudes |
| **Cerebrovascular 4DFlowNet** (2022) | same net, retrained on intracranial CFD | figshare 19158122, `4DFlowNet_cerebro.zip` (h5 + optimizer.pkl; pickle not opened) | CC BY 4.0 | same | closest to our brain / neurofluid use; v2.0 code changes: no divergence loss, linear last activation (lets phase alias) |
| **Ensemble4DFlowNet** (Ericsson, leonericsson/ensemble4dflownet) | baseline + bagging/stacking ensembles, domain nets aorta / cardiac / cerebro / combined | SharePoint folder (link from the upstream repo's README): https://kise-my.sharepoint.com/:f:/g/personal/david_marlevi_ki_se/El1e02h1-HVAjBKaH_ZgewgB5wgWD6RANBKVeeykSDUYPA?email=david.marlevi%40ki.se&e=0EtbtT (401 to curl; in a browser it needs a KI login, Umu identity fails; **dropped 2026-11-13**) | code MIT, weights unstated | TF 2.6 `.h5` | cardiac net is the only open cardiac-trained weights I found |
| **Temporal4DFlowNet** (PiaaCaa/Temporal4DFlowNet) | temporal SR ×2 (time patches 16), same ResNet style, 3.22 M params | in repo `pretrained/Temporal4DFlowNet-best.h5` 13 MB + yaml (verified, downloaded) | MIT | TF 2.9.1 HDF5 | in-silico CFD training; input HDF5 `u,v,w (t,x,y,z)`, mask, venc |
| 4DFlowNetv2 (dlon450) | Res / Dense / CSP variants | no weights listed; training data on request (S) | MIT | TF | not usable as-is |
| Aortic **segmentation** (ETH, gitlab ibt-cmr 4dflowmri-segmentation) | not SR; time-resolved aorta segmentation, synthetically trained | pretrained S-full available (S) | unchecked | - | candidate for a later `model_run` segmentation model |

Downloaded copies: `data/hasty/external/flow_weights/` (62 MB: orig, cerebro, temporal + config). `optimizer.pkl` never loaded.

## Method papers, no weights found
- DAF-FlowNet (arXiv 2604.00205): per-case unsupervised divergence-free vector-potential net + phase unwrapping; **code public**, nothing to pretrain (94 s/frame on A100 vs 1 s for 4DFlowNet) (S via page summary).
- 4D-MRAR (arXiv 2611.04053): encoder + ConvLSTM + coordinate decoder, spatial+temporal SR, pressure; no code/weights statement.
- PINGS-X (2511.11048, Gaussian splatting), SMURF (2505.12494, JAX, unsupervised recon), neural fields for 2D PC (2509.25388), div-mDCSRN-Flow (neurofluid CSF, physics-guided), attention-guided carotid SR (2609.04891), intracranial multi-cohort transfer learning (2610.35396: weights "upon publication"): none with downloadable weights confirmed.
- No PyTorch SR/denoiser with weights, no diffusion/transformer model with weights found in two search rounds (absence of evidence only).

## Fit for this project
- The three downloadable nets are plain Conv3D + LeakyReLU residual blocks with mirror padding and a bilinear (reshape-resize) upsample, so **ONNX export via my own PyTorch re-implementation is easy** (same approach as `export_fastsurfer.py`; the `model_config` JSON in the h5 gives the exact topology; weights are data, no code executed). Hyper-parameters read: 8 low-res res-blocks (64 ch), upsample, 4 hi-res res-blocks, 3 heads with tanh (u,v,w). Residual scale `mul_k` constants must be read from the config.
- Domain gap: trained on ≤1 mm-scale CFD-synthetic aorta/cerebral patches at a fixed low-res→high-res ratio, magnitude as extra input, venc-normalised velocities; our recon outputs (own phantom/CFD pipeline, patch denoiser plan in `denoiser_design.md`) are the intended comparison: use these as baselines, not as the product.
- Server integration would need a new recipe (`patch4d`: 3 velocity + 3 magnitude channels, patch 24³ overlap-blend, venc scaling, optional time loop) rather than `patch3d` label output: output is a float velocity tensor. Not started; ask before building.
- Evaluation plan if wanted: run orig/cerebro/temporal on the synthetic ph0001 flow (known truth) at several noise levels and compare against our denoiser: gives the baseline curve.

## Update 2026-11-13: deeper search + ONNX exports
- **Deeper search (GitHub/Zenodo/HF APIs): nothing new with downloadable weights.** Found: DSR (wenxy123/DSR, PyTorch SR, MIT, no weights, only training scripts + patch .pt files), INRs-4DFlowMRI (Saitta; per-case implicit neural representation, MIT, nothing pretrained), CWLS phase unwrapping/denoising (classical, CC BY-NC-SA code), hinrah aorta segmentation (4D nnU-Net style; weights on Zenodo 22255537 **restricted**, CC BY-NC-ND), DianaM228 aorta segmentation (code only), Bazgh PINN scripts (no weights/licence). ETH GitLab (4dflowmri-segmentation) blocked by an anti-bot page; unchecked.
- **Ensemble4DFlowNet weights: SharePoint link returns HTTP 401 to curl** (needs a browser session). Not downloaded. If you can download the folder in a browser, put it in `data/hasty/external/flow_weights/ensemble/` (aorta/cardiac/cerebro/combined `.h5`; the same exporter handles them).
- **Exported to ONNX** (own PyTorch re-implementation, `rustlib/tools/models/export_4dflownet.py`, strict weight load, h5 `model_config` read as JSON, no pickle):
  - `flow_4dflownet_orig` (aortic CFD; head tanh; patch 16 in config) and `flow_4dflownet_cerebro` (head linear; patch 12), 13 MB each, spatial ×2 (trilinear align_corners=True, equal to the original chained bilinear resizes). Dynamic patch size (checked 16 and 24).
  - `flow_temporal4dflownet`, 13 MB, temporal ×2 (half-pixel linear along T), no magnitude input, linear head, residual scale 1 assumed from the source default.
  - ONNX vs torch: 3e-6 (orig, cerebro), 6e-9 (temporal).
- **Plausibility, not a validation** (`tools/models/check_4dflownet.py`; no TensorFlow reference available): smooth analytic flow + noise. orig: noise σ=0.1 (venc-normalised) → 0.066 error vs HR truth, but a ~0.059 floor at low noise (domain gap: my field is not a vessel); cerebro similar; temporal: σ=0.02 → 0.0154 (better than input), but σ≥0.05 gets worse (0.134 at 0.05) – it looks trained for noise-free temporal interpolation, not denoising. Real validation needs the aortic CFD dataset (figshare 24424888, 8.7 GB, CC BY 4.0; not downloaded) or our own phantom flow.
- Server: models with `kind: flow_sr` are listed by `model_list` but hidden from the Run Model node (needs a new recipe `patch4d_sr`; not built).
