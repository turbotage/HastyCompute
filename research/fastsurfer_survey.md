# FastSurfer segmentation survey (2026-11-13)

Source: github.com/Deep-MI/FastSurfer, branch `stable`, release v2.6.1. Raw base: https://raw.githubusercontent.com/Deep-MI/FastSurfer/stable/
V = verified from fetched source, I = inferred. Checkpoints were NOT downloaded or loaded; the state-dict key layout is unverified.

## 1. Versions and architectures
- V: Default model is **FastSurferVINN**, v2.0.0 checkpoints (`FastSurferCNN/config/checkpoint_paths.yaml`). It is still **3 separate networks**, one per plane (axial, coronal, sagittal). Each is a 2D net with the same architecture and its own weights. VINN's "single network" property is resolution independence, not view independence. The old FastSurferCNN (v1.0.0 and v2.0.0 `aparc_cnn_*` files) is still hosted.
- V: Config (`FastSurferCNN/config/FastSurferVINN_{axial,coronal,sagittal}.yaml`): NUM_FILTERS 71, NUM_FILTERS_INTERPOL 32, **NUM_CHANNELS 7** (3 slices each side, middle one is segmented), 3x3 kernels, pool 2, HEIGHT = WIDTH = 256, BASE_RES 1.0, bilinear interpolation, crop top_left, OUT_TENSOR 320x320, PADDED_SIZE 320.
- V: **Classes: axial and coronal 79, sagittal 51** (hemisphere-merged). The README's "95 classes" is the post-processed output after re-lateralisation. LUT: https://raw.githubusercontent.com/Deep-MI/FastSurfer/stable/FastSurferCNN/config/FastSurfer_ColorLUT.tsv (79 rows, columns ID, LabelName, R, G, B, A). The net's output index i maps to LUT row i, so the label ID is `LUT.ID[argmax]` (`data_utils.map_label2aparc_aseg`, 838-857).
- V: Checkpoints are torch pickles with `weights_only=False`, a dict with key `"model_state"` (also `optimizer_state`, `scheduler_state`) (`inference.py:187-220`, `utils/checkpoint.py:255-322`). Loading them needs a torch pickle reader that does not execute code, since they are pickles. I: the export script must use a trusted environment or a restricted unpickler.

## 2. Weights and licence
- V: `checkpoint_paths.yaml` URLs: primary https://b2share.fz-juelich.de/api/files/a423a576-220d-47b0-9e0c-b5b32d45fc59 , mirror https://zenodo.org/records/10390573/files (append `/<name>`). The paths are `checkpoints/aparc_vinn_{axial,coronal,sagittal}_v2.0.0.pkl`.
- V: Zenodo record 10390573 "FastSurferVINN Checkpoints", sizes in bytes: aparc_vinn_axial_v2.0.0.pkl 22,399,769; aparc_vinn_coronal_v2.0.0.pkl 22,399,783; aparc_vinn_sagittal_v2.0.0.pkl 22,375,598. Old CNN files: aparc_cnn_{axial,coronal,sagittal}_v1.0.0.pkl 21.7/21.7/21.7 MB, aparc_cnn_*_v2.0.0.pkl 19.4 MB.
- V: Code is Apache-2.0 (`LICENSE`). The Zenodo record metadata also says `apache2.0` for the weights. The README adds a research-use, not-clinical disclaimer (line 217). The FreeSurfer licence applies only to the surface pipeline, not to segmentation.

## 3. Preprocessing (`FastSurferCNN/data_loader/conform.py`)
- V: `conform()` (1016-1165) mimics `mri_convert -c`. Orientation is **LIA** (axis0 toward Left, axis1 toward Inferior, axis2 toward Anterior), strict LIA with the FreeSurfer centre convention. Interpolation is linear (order 1) on float data, then rescale, then `np.rint`, then uint8.
- V: Target voxel size and size (`conformed_vox_img_size`, 1535-1643, called with `vox_size="min"`, `threshold_1mm=0.95` from `run_prediction.main`): vox = min of the 3 native voxel sizes, capped at 1.0 mm; if above 0.95 it becomes 1.0. So **sub-mm inputs are processed at native min voxel size** (VINN high-res, down to about 0.7 mm). The grid is a cube: 256 for 1 mm, otherwise the FOV-derived size, and at least 256.
- V: Intensity rescale (`getscale`, 848-930; `scalecrop`; call in `conform` ~1140-1150). The scale is computed on the **original, un-resampled** data:
  - f_low = 0, so src_min is the data minimum.
  - f_high = 0.999: 1000-bin histogram over [min, max]; upper_cutoff = N_total - int(0.001 * N_nonzero); src_max = bin_edges[first index where cum_hist >= upper_cutoff, minus 2].
  - scale = 255 / (src_max - src_min), out = clip(scale * (x - src_min), 0, 255).
  - Voxels where the resampled data is ~0 (`isclose(.., 0)`) are forced to 0.
  - Then rint, then uint8.
- V: The net input is `img/255` clipped to [0,1] as float32 (`data_loader/augmentation.py` ToTensorTest, 28-62).
- V: **View feeding** (`dataset.py` MultiScaleOrigDataThickSlices, 34-87; `data_utils.transform_axial` 510, `transform_sagittal` 535). The conformed volume V is in LIA (L, I, A):
  - coronal: V as is, slices along axis2 (A), 2D image = (L, I).
  - axial: `moveaxis(V,[0,1,2],[1,2,0])` gives (A, L, I), slices along I, 2D image = (A, L).
  - sagittal: `moveaxis(V,[0,1,2],[2,1,0])` gives (A, I, L), slices along L, 2D image = (A, I).
  - Slab (`data_utils.get_thick_slices`, 561-590): np.pad edge mode with 3 on each side of the slice axis, sliding window of 7. The 7 slices are the channels, in ascending index order. The image is `(H, W, 7)`, transposed to (7, H, W).
- V: The scale_factor input is `BASE_RES / zoom_2d` (the in-plane voxel size for that view; `dataset.py:89-102`). It is 1.0 for 1 mm data, which makes the interpolation layers identity for the 256 input.

## 4. Inference and aggregation (`FastSurferCNN/inference.py`, `run_prediction.py`)
- V: Order in `get_prediction` (354-405): the input is reoriented "soft LIA" (pure axis permutation/flip); the three views run in order; each view's **raw logits** are added into a shared fp16 buffer (N, 256, 256, 256, C=79) with weights **axial 0.4, coronal 0.4, sagittal 0.2** (`inference.py:127, 371`; `alpha = {"sagittal": 0.2}`, default 0.4). No softmax. Argmax over the class axis.
- V: Sagittal 51 logits are expanded to 79 by index gather (`map_prediction_sagittal2full`, `data_utils.py:1315-1354`): `idx = [0] + range(5,14) + range(1,4) + [14,15,4] + range(16,19) + range(5,51) + [20,22,27] + range(29,32) + [33,34] + range(38,43) + [45]` (79 entries; `logits_full = sag[:, idx]`).
- V: Output permutation back to LIA (`inference.py:128-132`) on the net output (B, C, H, W): axial `permute(3,0,2,1)`, coronal `permute(2,3,0,1)`, sagittal `permute(0,3,2,1)`, each giving (L, I, A, C). Crop to the image size, which is the identity at 256.
- V: Post-processing:
  1. `native_to_lia.inverse(pred, order=0)` (nearest-neighbour back to the input orientation).
  2. LUT map (79 to FreeSurfer IDs).
  3. `data_utils.split_cortex_labels` (998-): re-lateralises cortex by centroid of the largest connected components of WM 2 and 41. The 1xxx labels become 2xxx for the right hemisphere. This produces the 95 classes.
  4. `reduce_to_aseg.clean_segmentation` (rta.py:274): brain mask by `create_mask(seg,5,4)`, remove labels outside the mask, `flip_wm_islands` (180) flips disconnected WM islands to the other hemisphere.
  5. `reduce_to_aseg` (105): cortex 1xxx becomes 3, 2xxx becomes 42, for the aseg-only output.
- V: Output to the native grid: the conformed grid is the "native" frame of the net, and the written segmentation uses the conformed affine (`async_save_img(..., orig_img)`), so the output is NOT resampled back to the raw input grid by default. I: to get native voxels, apply `conform`'s vox2vox inverse with nearest-neighbour.

## 5. Code map (stable, `FastSurferCNN/`)
- conform: `data_loader/conform.py` `conform` 1016-1165, `getscale` 848-930, `scalecrop` 947, `conformed_vox_img_size` 1535, `Reorientation.from_target_orientation` 327-424, `apply_vox2vox` 745.
- views and slab: `data_loader/dataset.py` 34-140, `data_loader/data_utils.py` `get_thick_slices` 561, `transform_axial` 510, `transform_sagittal` 535.
- aggregation: `inference.py` `Inference.__init__` 82-138 (alpha, permute_order), `eval` 300-381; `run_prediction.py` `RunModelOnData.get_prediction` 354-405, `main` 583-740.
- label handling: `data_utils.py` `map_prediction_sagittal2full` 1315, `map_label2aparc_aseg` 838, `split_cortex_labels` 998. Cleanup: `reduce_to_aseg.py`.
- model: `models/networks.py` (FastSurferVINN forward ~255-310), `models/sub_module.py` (blocks), `models/interpolation_layer.py` (Zoom2d, `_interpolate` 418-485).

## 6. ONNX export feasibility (I, from reading the model code)
- Ops used: Conv2d, BatchNorm2d (fold at eval), PReLU, `torch.maximum` (maxout), MaxPool2d with indices plus MaxUnpool2d (encoder to decoder). ONNX has MaxPool with indices and MaxUnpool, but the torch exporter support for `return_indices` combined with unpool is the known weak spot (opset 11+; test it). Fallback is the TorchScript path already planned.
- Resolution-normalisation: Zoom2d loops over the batch in Python, with crop/pad/F.interpolate and scale-dependent shapes. This is the dynamic-shape trouble. For the 1 mm case (scale 1.0, 256x256 input) it is an identity, so **export with the scale fixed to 1.0 and a static 256x256 input** (batch dynamic is fine). Sub-mm (VINN) needs a second export per voxel-size bucket or a rewrite of Zoom2d with static shapes.
- Inputs: (N, 7, 256, 256) float32 and scale_factor (N, 2). Output: logits (N, 79 or 51, 256, 256). The three view networks are exported separately, and the Rust recipe does conform, slabs, the logit-sum with 0.4/0.4/0.2, the sagittal gather, argmax and the LUT.
- Size is about 22 MB per view (fp32), so about 66 MB in total.

## 7. Recommendation
Implement **FastSurferVINN v2.0.0 at 1 mm (256^3, scale 1.0)** first: 3 static-shape nets, 79/79/51 classes, logit-weighted sum 0.4/0.4/0.2, and the LIA conform with the 0.999 histogram rescale. Skip the old FastSurferCNN (same recipe, older weights, no gain). Postprocessing order: LUT map, then optional split_cortex_labels/clean_segmentation (needs connected components and the brain-mask code; for aseg-only use, skip). High-res/sub-mm VINN is a later flag.

## Differences from plan_models.md section 5
- Classes are 79 net outputs (95 only after re-lateralisation), not about 95.
- Slab is 7 channels, edge-padded (matches the plan). Weights per view are 0.4/0.4/0.2 summed logits, not "weighted average of softmax".
- Rescale uses the 0.999 histogram percentile, with src_min the data minimum.
- Weights are Apache-2.0 per Zenodo metadata.
