# Server-side model inference (segmentation first)

Status 2026-11-13: steps 1–5 of §8 implemented (see 'Implemented' at the end); FastSurfer recipe/export and PVS models open. First model: FastSurfer (aseg). Formats: TorchScript/AOTI now, ONNX in
the first round (not later), others behind the same trait.

## 0. Current state (read from repo)

- Only the 4D-flow denoiser loads a model: `hasty-recon/src/learned.rs` dlopens `libhasty_denoise_torch.so`
  (own cargo workspace `hasty-denoise-torch`, C ABI `hdt_*`, denoiser-shaped). libtorch never enters the main build.
- `hasty-server/src/commands.rs`: `Registry` of sync closures `(Ctx, JSON opts, inputs) -> (msg, outputs)`;
  `Ctx` = `Bank` only. No model cache, no GPU worker, no progress reporting beyond `bank.notify("log", …)`.
- NodeEditor `NiftiLoad` parses in the browser and `put`s a tensor; `parseNifti`/`buildNifti1` drop spacing + affine.

## 1. Principle: NIfTI is one source, not the contract

The contract is a **tensor + geometry metadata** in the bank. NIfTI load, recon output, DICOM import later all fill
the same metadata; `model_run` never sees a file format.

Entry metadata (JSON, already supported by the bank):
```
{ "axes": ["x","y","z"],            // already written by NiftiLoad
  "spacing_mm": [sx,sy,sz],         // required for models; fallback from recon's own spacing
  "affine": [[4x4]],                // optional; voxel -> scanner/world (RAS). absent => axis-aligned from spacing
  "labels": {"0":"Unknown",…} }     // on segmentation outputs
```
Missing affine = identity direction (documented fallback; the result then also has none). Orientation handling is
done by permuting/flipping to the model's convention using the affine when present, else assumed already in RAS.

## 2. Model directory

`data/hasty/models/<name>/` (inside repo, weights gitignored) with `model.json`:
```
{ "kind": "segmentation",
  "recipe": "slab2d_views" | "patch3d" | ...,   // Rust pre/post pipeline
  "backend": "torch" | "onnx",
  "files": {"axial": "axial.onnx", …},          // recipe-defined roles
  "spacing_mm": [1,1,1], "orientation": "LIA", "conform": [256,256,256],
  "norm": {…}, "labels": {…}, "license": "…", "source": "…" }
```
Backend is chosen per directory; one recipe can run on either backend (needed to compare ONNX vs TorchScript output).

## 3. Crates

- **hasty-infer** (new, torch-free, ort-free; unit-testable on CPU):
  - `trait Backend { fn open(spec) ; fn forward(&mut self, role, input: &TensorF32) -> TensorF32 ; fn device() }`.
  - geometry: reorient / conform / resample (trilinear for images, nearest for labels), inverse mapping.
  - recipes: `patch3d` (nnU-Net-like: tile, Gaussian weights, overlap, optional flip TTA) and `slab2d_views`
    (FastSurfer: see §5). Recipes are plain Rust fns over `Backend`, so a mock backend tests them.
  - `ModelSpec` parse/validate of `model.json`.
- **ONNX backend**: `ort` crate with `load-dynamic` (dlopen `libonnxruntime.so`, same philosophy as the torch
  plugin: runtime lib stays out of the build, path via `HASTY_ORT` env + default under `target/`). CUDA EP when
  present, CPU EP fallback. Lives in its own feature `onnx` of hasty-infer (or tiny crate `hasty-infer-onnx`).
  Check whether ort's build pulls anything heavy into the main workspace; if it does, make it a plugin like torch.
- **Torch backend**: generic ABI in the existing torch plugin workspace: `hmi_open(path, cuda)`,
  `hmi_forward(h, in, shape, ndim, out, out_cap, out_shape)`, `hmi_close`, `hmi_abi`. Same libtorch, same dlopen
  pattern as `learned.rs`; denoiser `hdt_*` stays untouched.
- Later formats (safetensors + hand-written net, TensorRT engines, GGUF…) = another `Backend` impl, no other change.

## 4. Server

- `Ctx` gains `models: Arc<ModelCache>`: lazy load, one resident GPU model by default, LRU eviction, size-aware.
  Mutex per device slot until the per-device worker exists (do not run next to training/sim — gpu pacing rule).
- Commands:
  - `model_list` -> message JSON: name, kind, recipe, backend, spacing, labels count, license.
  - `model_run`: options `{model, device: "cuda"|"cpu", overlap, tta, output: "labels"|"prob"|"both",
    backend?: override}`; input `[image]`; outputs `[labels (u8/u16), prob? (f16)]`; message = label table +
    timings. Output metadata copies geometry from input, adds `labels`.
  - Per-step progress via `bank.notify("log", …)` ("axial 120/256").
- Cancellation: check a flag between tiles/slices (cheap, matches the resumable-runs preference at granularity of
  one call; whole-volume results are seconds–minutes, no checkpointing needed).

## 5. FastSurfer specifics (verify against the version's repo before coding — from memory)

- Input conformed: 1 mm isotropic, 256³, LIA orientation, intensities rescaled to uint8-like 0–255 (percentile
  based), as `mri_convert --conform` would do. Recipe `slab2d_views` does this conform itself in `hasty-infer`.
- Three 2D networks (coronal, axial, sagittal), each fed a thick slab of adjacent slices (7 in the versions I
  remember); per-view softmax maps are permuted back to a common grid and combined (weighted average), then argmax.
  The newer VINN variant is a single resolution-independent net; support is a different recipe flag, not a rewrite.
- Output classes: aseg (+ DKT cortical parcels, ~95 classes); labels table shipped in `model.json`.
- Weights: Apache-2.0 (verify), export script `rustlib/tools/models/export_fastsurfer.py` writes both `.ts` and
  `.onnx` plus `model.json`; PyTorch reference output saved as golden.
- Not in scope: surface recon, cortical thickness, FreeSurfer-compatible stats.

## 6. NodeEditor

1. IO fix first: `parseNifti` fills `spacing_mm` (pixdim) and `affine` (sform, else qform quaternion, else
   spacing diag); `buildNifti1` writes them back and supports u8/u16 labels. Round-trip test.
2. `ModelRunNode` (category "Models"): dropdown from `model_list`, widgets overlap / TTA / device / output; outputs
   `labels` (+ `prob`); calls `Execute("model_run")`. Verify `_nodeFingerprint` covers widget values (cache).
3. Other sources need no node change: any node that yields a tensor uuid with spacing metadata works (recon output,
   upload). Add a small "Set geometry" node (spacing/affine override) for data without headers.
4. Viewer: label overlay in the orthoslicer using `labels` for names and a stable colour table.
5. Downstream: labels -> `flow_make` mask input (vessel/brain masks), ROI stats node later.

## 7. Tests

- Mock backend: tiling/aggregation identity (tiled == whole away from borders), view-permutation round-trip,
  conform -> inverse conform returns the input grid for axis-aligned and flipped/permuted affines.
- Torch vs ONNX of the same exported tiny conv net agree to 1e-4; FastSurfer export agrees with PyTorch on a
  fixture volume (Dice > 0.99 vs reference labels, not bitwise).
- Affine round-trip NIfTI load -> save.
- CPU and CUDA execution providers agree on the fixture.

## 8. Order

1. Editor NIfTI geometry (affine/spacing) + round-trip test.
2. hasty-infer: spec, geometry, mock backend, `patch3d` + `slab2d_views` recipes.
3. ONNX backend (dlopen, CPU first, then CUDA EP).
4. Server `ModelCache`, `model_list`, `model_run`.
5. `ModelRunNode` + overlay.
6. FastSurfer export script + golden; Torch generic ABI (`hmi_*`).
7. PVS model when chosen (likely `patch3d`).

## Open

- Does ort `load-dynamic` keep the main build free of ONNX Runtime binaries (expected yes)? Check before step 3.
- Which PVS model (SHIVA-PVS style nnU-Net?) and its licence.
- Large outputs: probabilities for 95 classes at 256³ in f16 = ~3 GB; default `output: labels`, probabilities only
  on request and/or top-k.

## Implemented (2026-11-13)

- `plotting_website/node_editor/src/nifti.js` (+ `nifti.test.mjs`, `package.json` type=module for node tests):
  sform/qform/pixdim <-> `spacing_mm` + `affine`; `parseNifti` returns geometry, NiftiLoad stores it in the entry
  metadata, NiftiSave writes it back. Run tests: `node --test plotting_website/node_editor/src/nifti.test.mjs`.
- `rustlib/crates/hasty-infer` (workspace member): `geom` (affine, resample, `covering_geometry`, `conform_geometry`),
  `spec` (model.json + `scan`), `backend::Backend` + `Mock`, `recipe::segment_patch3d` (normalise, Gaussian-blended
  tiling, mirror TTA, argmax, labels back on the input grid), feature `onnx` = `onnx::OnnxBackend` (`ort`
  2.0.0-rc.15 `load-dynamic`, CUDA EP with CPU fallback). Tests: `tests/patch3d.rs`, `tests/onnx.rs`.
- `hasty-server/src/models.rs`: `model_list`, `model_run` (one resident model, serialised by a mutex, progress as
  `log` events). Geometry comes through the options (`spacing_mm`/`affine`) because commands do not see entry
  metadata; the node editor passes it. Models dir: `$HASTY_MODELS` or `data/hasty/models`. Server feature `onnx` is on
  by default; runtime lib: `$HASTY_ORT` / `$ORT_DYLIB_PATH` / `rustlib/target/ort/libonnxruntime.so`.
- Editor node "Models/Run Model" (`ModelRunNode`): model dropdown from `model_list`, writes geometry + label table to
  the output metadata. Not exercised in a browser yet.
- `rustlib/tools/models/make_test_onnx.py`: tiny pointwise ONNX net used by the tests.

Dev setup used for the tests (all under `rustlib/target/ort/`, not tracked): `onnxruntime-gpu` wheel unpacked
(`wheelx/onnxruntime/capi/libonnxruntime.so.1.31.1`, symlink `libonnxruntime.so`), venv with `onnx numpy`, test
models in `models/pointwise/`. Run:
```
cd rustlib/crates && HASTY_MODELS=$PWD/../target/ort/models HASTY_ORT=$PWD/../target/ort/libonnxruntime.so \
  HASTY_ORT_TEST_MODEL=$PWD/../target/ort/models/pointwise CARGO_TARGET_DIR=$PWD/../target/infer-check \
  cargo test -p hasty-infer --features onnx; cargo test -p hasty-server --test models
```

## PVS models (survey: research/pvs_model_survey.md)

- Default: Pham et al. nnU-Net family (T1, T1+FLAIR, T2, T2+FLAIR; nnU-Net v2 ResEnc 3d_fullres). Best reported Dice,
  maintained (Nov 2025). Weights CC BY-NC-SA (README) — research only. Needs: ONNX export of the 3d_fullres network
  with the plans' patch size, target spacing and normalisation copied into `model.json` (`patch3d` fits), and
  **multi-channel input** (T1+FLAIR) which `model_run` rejects today (`in_channels` must be 1).
- Alternates for the library: SHIVA-PVS v5 (T1 only, TF SavedModel -> tf2onnx, 1 mm, crop 160×214×176, CC BY-NC-SA,
  weights on a private host), MedNet-PVS (Apache-2.0 code, T2 white-matter PVS, weights licence unstated).
- Not verified by the survey: ONNX export of any of them, GPU memory, SHIVA's SWI/T2GRE contradiction.

## PVS T1 model installed (2026-11-13)

`data/hasty/models/pvs_t1_wmbg/` (model.onnx 408 MB, model.json; gitignored with the rest of data/hasty): Pham et al.
Dataset703 (T1w, WM + BG PVS), nnU-Net v2 ResEnc-M 3d_fullres, **fold_all single model** (their script ensembles
folds 0–4). Licence CC BY-NC-SA 4.0 (research use; user OK'd for now, no commercial plans).

- Source zip (4.5 GB, five folds + fold_all): `rustlib/target/pvs/weights/T1_WM-BG-PVS.zip` (Google Drive folder
  14PAgatsxO2wSLsXohG7ladxEcYgz4Acg; other files there: T1_HCPVS, T1_MBPVS, T1FLAIR_PVS_WMH, T2_WM-BG-PVS, T2FLAIR_PVS_WMH).
- Export: `rustlib/tools/models/export_nnunet.py <folder> all <out> --crop-nonzero --license … --source …` run with
  `rustlib/.venv` (torch + nnunetv2; I added `onnx` to that venv). Checkpoints are loaded with
  `tools/models/safe_load.py` (allowlisted unpickler) — never `torch.load(weights_only=False)` on downloaded files.
- Their image reader (`AgnoSITKIO`) forces spacing 1 mm, i.e. **no resampling**: model.json has
  `orientation: "native"` (run on the stored grid) and `crop_nonzero`; `norm: t1w_pvs` = clip, percentile rescale,
  NLM denoise, CLAHE, implemented in Rust (`hasty-infer/src/pvsnorm.rs`) and verified against
  `tools/models/pvs_t1_ref_norm.py` (skimage 0.26 + dipy 1.12.1, written from their algorithm; their code is not run).
- Parity (Stanford HARDI T1, 2 mm — real but coarse, so compared on probabilities, not PVS labels):
  Rust normalisation vs Python: mean |Δ| 4.4e-4 on [0,1] (max 0.075, CLAHE bin flips); ONNX vs torch logits max
  2.8e-3 at logit scale 62; tiling + softmax vs nnU-Net's `predict_sliding_window_return_logits`: mean prob diff
  2.7e-8 (plain) / 6.9e-9 (8-fold mirror), max 1e-2; full pipeline foreground probability vs Python-normalised
  reference: mean 6e-7. Server path (`model_run` through the registry) works, 4 s for 76×105×81 on the 4090.
- Known approximations: dipy's *classic* NLM (default before dipy 1.12; the new "blockwise" default differs from it
  noticeably, corr 0.95 after CLAHE — which one the authors used is unknown, they trained Aug–Sep 2025); fold_all
  instead of the 5-fold ensemble; Rust pads small volumes like nnU-Net (centred).
- Not done: T2 (needs `T2wPVSNormalization`, read their file first), T1+FLAIR / T2+FLAIR (2 channels: `model_run`
  rejects `in_channels != 1`), regional MB/HP models, 5-fold ensemble (average the logits of several `files` roles),
  a real 1 mm T1 with PVS for an end-to-end sanity check of the labels.

## Generic nnU-Net import, multi-channel, ensembles, thorax and brain models (2026-11-13, later)

**Infrastructure** (`hasty-infer`): `model_run` takes one input tensor per model channel (same grid); `norm` may be a
per-channel list (`zscore` with optional `nonzero_mask`, `ct`, `t1w_pvs`, `t2w_pvs`, `percentile`, `scale_max`);
`ensemble` lists roles whose raw outputs are averaged (nnU-Net folds); `patch3d` now follows nnU-Net's order
(crop to non-zero box → resample → normalise → tile → argmax → resample back → paste), with `orientation: "native"` +
`spacing_mm` meaning "resample on the stored axes" (stock nnU-Net) and `orientation: "RAS"` + `spacing_mm`
"reorient then resample" (TotalSegmentator). Recipes are chosen by `recipe` (`patch3d`, `slab2d_views`) through
`recipe::segment`. Tests: `tests/patch3d.rs` (10, incl. 2-channel, ensemble cancellation, native resampling, crop).

**Importer**: `rustlib/tools/models/export_nnunet.py <plans folder> <folds> <out> [--native|--orientation RAS]
[--step-size] [--crop-nonzero]` derives normalisation (ZScore / CT / PVS schemes), target spacing, patch, labels and
the fold ensemble from `plans.json` / `dataset.json`; also accepts the legacy plan format of TotalSegmentator's
weights. Parity fixture `ref_x/ref_y.npy` + generic test `HASTY_MODEL_PARITY=<model dir>` (ONNX vs torch).

**Installed (data/hasty/models):**
| name | what | licence | check |
|---|---|---|---|
| `pvs_t1_wmbg` | Pham T1 PVS (see above) | CC BY-NC-SA | parity vs nnU-Net sliding window |
| `totalseg_mr_3mm` | TotalSegmentator `total_mr` fast (Dataset852, 3 mm, 50 classes: heart 22, aorta 23, IVC 24, lungs, brain 50…) | Apache-2.0 (survey: research/totalsegmentator_survey.md) | ONNX vs torch 4e-3 at logit scale 294; brain-only T1 gives a 1.55 L brain |
| `totalseg_mr_organs_1p5mm` | Dataset850 (1.5 mm organs: heart, aorta, IVC, iliac vessels, 29 classes) | Apache-2.0 | export only |
| `fastsurfer_vinn_1mm` | FastSurfer VINN v2 (aseg + DKT, 79 classes) | Apache-2.0 (code and weights; research only) | see below |

TotalSegmentator MR has no open-licence chamber / pulmonary-artery / coronary model (those are CT-only and
licence-gated); the 3 mm model gives aorta, IVC and a single "heart" label — use as ROI / prior. Only fold_0 exists.
Their pipeline casts the resampled image to int32 (survey): we resample in float, z-score is scale-invariant.

**FastSurfer**: `slab2d_views` recipe in `hasty-infer/src/fastsurfer.rs` (conform to 256³ 1 mm LIA, FreeSurfer robust
intensity scale, 7-slice slabs, three views, logits 0.4/0.4/0.2, sagittal 51→79 gather, class → FreeSurfer id).
The VINN network is **re-implemented** in `tools/models/export_fastsurfer.py` from the published architecture (no
FastSurfer code is executed; strict state-dict load; MaxUnpool replaced by scatter because the ONNX exporter lacks it).
Not implemented: cortex re-lateralisation (`split_cortex_labels`), clean-up, sub-mm VINN, native-grid output.

### Correction: the CUDA execution provider was not active in the earlier runs

The `ort` CUDA provider is registered *silently* by default, so everything reported as "cuda active" before
2026-11-13 ~11:20 (PVS parity runs, TotalSegmentator smoke test, first FastSurfer run: 127 s) actually ran on the CPU.
Fixed: the provider is requested with `error_on_failure` (+ `with_tf32(false)` for fp32 fidelity) and `OnnxBackend::cuda`
now tells the truth. Two environment requirements, both in `rustlib/tools/models/ort_env.sh` (source it before starting
`hasty-serve` or running the tests):
1. `libonnxruntime_providers_shared.so` and `libonnxruntime_providers_cuda.so` must sit next to `libonnxruntime.so`
   (symlinks in `rustlib/target/ort/`); ORT resolves them relative to the path it was loaded from.
2. `LD_LIBRARY_PATH` must contain the CUDA 13 runtime + cuDNN of the repo venv's torch wheels
   (`.venv/lib/python3.12/site-packages/nvidia/{cu13,cudnn}/lib`); the system toolkit libs abort with
   "Cannot load symbol cublasLtCreate".
Timings on the RTX 4090 after the fix: FastSurfer 2-D net 12 ms/slice (CPU 350 ms) → full FastSurfer run 16.8 s
(forward 18.6 s total of which axial 3.6, coronal 6.5, sagittal 8.5 incl. accumulation overhead; accumulation of the half-precision
logits ~8 s, mostly the strided sagittal scatter — a candidate for chunking), PVS 0.6 s, TotalSegmentator 3 mm 2.8 s.
Parity ONNX(CUDA, fp32) vs torch: totalseg 5e-4 @ logit scale 294, FastSurfer axial 4e-4, coronal 0.6 / sagittal 0.25 at
logit scale 60/46 on a clamped-noise fixture (max-pool tie-breaking differs between kernels; the brain run gives the
same structure volumes as the CPU run: WM 272/275 mL, hippocampus 4.5/4.6, thalamus 8.3/7.6, lateral ventricles 6.9/7.1).

FastSurfer cortex re-lateralisation (`split_cortex_labels` port: 26-connected components vs WM centroids, Gaussian split
for 1011/1019/1026/1029) is on (`"relateralize": true`) and unit-tested on a synthetic two-hemisphere volume; it was
not compared against FastSurfer's own output (no reference run available).

## Example data and PVS T2 (2026-11-13)
- Examples (compressed, CC0, in `data/hasty/examples`, see README there): HCP-Aging T2w + WM-PVS masks (OpenNeuro ds005595, 3 subjects), UCLA 1 mm T1w (ds000030).
- `pvs_t2_wmbg` (Pham Dataset705 fold_all) exported. On HCP-Aging T2w vs the dataset's WM-PVS masks (single model, no ensemble): Dice WM 0.74 / 0.50 / 0.72 (sub-001/002/003); sub-002 under-segments (2.5k vs 7.1k voxels). Reference masks are refined deep-learning labels, not independent truth.
- UCLA T1 (CPU-equivalent counts verified on GPU earlier): FastSurfer WM 192/194 mL, L/R DKT parcels symmetric (cortex split works), TotalSegmentator brain 1.44 L, PVS T1 66 WM + 138 BG voxels (young brain, expected few).
- Start: `rustlib/tools/models/serve.sh`; node default device is now cpu (GPU shared with recon runs).
- Tests: `pvs_t2_hcp_dice` (HASTY_HCP), `case_models` (HASTY_CASE, `tools/models/nii_to_npy.py`).

### Not-done items closed (2026-11-13)
- T2 normalisation vs Python reference (skimage+dipy classic, HCP-Aging sub-001): mean |Δ| 7e-5, max 0.011 (`HASTY_PVS_CONTRAST=t2`).
- dipy classic vs blockwise: blockwise became the default in dipy 1.12.0 (2026-03-15); Pham's code is dated 2025-11-04 (dipy ≤ 1.11, released 2025-03-16) → classic is what training used. Keep classic.
- `pvs_t1flair_wmbg_wmh` exported (2 channels: input 1 = T1w, input 2 = FLAIR, same grid; classes bg/WM/BG/WMH). ONNX vs torch (CPU) 3.0e-3 at logit scale 64. No FLAIR example volume downloaded, so no end-to-end run.
- `pvs_t2_wmbg` ONNX vs torch (CPU) 2.9e-3 at logit scale 59.
- `pvs_t2_wmbg_5fold`: published 5-fold ensemble (2 GB ONNX). Results below.
- Weight zips deleted for T2 and T1+FLAIR (T1 zip kept for a T1 5-fold export).
- 4D-flow pretrained model survey: `research/fourdflow_models_survey.md`.
