"""Export folds of an nnU-Net v2 trained-model folder to ONNX + a hasty model.json (recipe patch3d).

usage: python export_nnunet.py <trained_model_folder> <folds> <out_dir> [options]
  folds        "all", "0", or a comma list "0,1,2,3,4" (several -> roles fold_<i>, averaged by the recipe)
  --native               image reader does not resample / reorient (e.g. Pham's AgnoSITKIO): orientation "native", no spacing
  --orientation RAS      reorient the input to this orientation before resampling (TotalSegmentator: RAS); with
                         neither flag the plans' spacing is applied on the stored axes (stock nnU-Net behaviour)
  --step-size 0.5        nnU-Net tile_step_size (overlap = 1 - step); TotalSegmentator total_mr uses 0.8
  --crop-nonzero         nnU-Net crop to the non-zero bounding box before normalisation
  --norm JSON            override the normalisation derived from the plans (object, or list per channel)
  --labels JSON --license TXT --source TXT --description TXT

The network is exported for the fixed training patch size (nnU-Net always feeds that size), eval mode, deep
supervision off, logits out. nnU-Net array axes are (z, y, x) of the stored file; hasty-infer's `patch` and
`spacing_mm` are [x, y, z], so they are reversed. Run with a Python that has torch + nnunetv2 + onnx (rustlib/.venv).
Checkpoints come from the internet: they are read with safe_load (allowlisted unpickler), never torch.load with
weights_only=False, and no code of the model's repository is imported.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from safe_load import load_state  # noqa: E402
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans  # noqa: E402
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("folds")
ap.add_argument("out")
ap.add_argument("--native", action="store_true")
ap.add_argument("--orientation", default=None)
ap.add_argument("--step-size", type=float, default=0.5)
ap.add_argument("--crop-nonzero", action="store_true")
ap.add_argument("--norm", default=None)
ap.add_argument("--labels", default=None, help="JSON {id: name}; default from dataset.json")
ap.add_argument("--license", default="")
ap.add_argument("--source", default="")
ap.add_argument("--description", default="")
a = ap.parse_args()

folder = Path(a.folder)
plans = json.load(open(folder / "plans.json"))
dataset_json = json.load(open(folder / "dataset.json"))
pm = PlansManager(plans)
cm = pm.get_configuration("3d_fullres")
nin = len(dataset_json["channel_names"])
nout = max(v for v in dataset_json["labels"].values() if isinstance(v, int)) + 1

# --- normalisation from the plans -------------------------------------------------------------------------------
fg = plans.get("foreground_intensity_properties_per_channel", {})
schemes = cm.configuration["normalization_schemes"]
use_mask = cm.configuration.get("use_mask_for_norm", [False] * nin)


def norm_of(c):
    s = schemes[c]
    if s == "ZScoreNormalization":
        return {"type": "zscore", "nonzero_mask": bool(use_mask[c])}
    if s == "CTNormalization":
        p = fg[str(c)]
        return {"type": "ct", "lo": p["percentile_00_5"], "hi": p["percentile_99_5"], "mean": p["mean"], "std": p["std"]}
    if s == "T1wPVSNormalization":
        return {"type": "t1w_pvs"}
    if s == "T2wPVSNormalization":
        return {"type": "t2w_pvs"}
    if s == "NoNormalization":
        return {"type": "none"}
    raise SystemExit(f"channel {c}: normalisation scheme {s!r} is not implemented in hasty-infer; pass --norm")


norm = json.loads(a.norm) if a.norm else [norm_of(c) for c in range(nin)]
if isinstance(norm, list) and len(norm) > 1 and all(n == norm[0] for n in norm):
    norm = norm[0]
elif isinstance(norm, list) and len(norm) == 1:
    norm = norm[0]

# --- grid ---------------------------------------------------------------------------------------------------------
sp_zyx = list(cm.spacing)
tf = list(pm.transpose_forward)
if tf != [0, 1, 2] and not a.native:
    raise SystemExit(f"transpose_forward {tf} is not supported (axes would be permuted)")
if a.native:
    spacing, orientation = None, "native"
else:
    spacing, orientation = [float(v) for v in sp_zyx[::-1]], (a.orientation or "native")

# --- networks -----------------------------------------------------------------------------------------------------
folds = a.folds.split(",")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
patch = [int(p) for p in cm.patch_size]
torch.manual_seed(0)
x = torch.randn(1, nin, *patch)
files, ref = {}, None
for f in folds:
    net = get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs, cm.network_arch_init_kwargs_req_import,
                                 nin, nout, allow_init=False, deep_supervision=False)
    net.load_state_dict(load_state(str(folder / f"fold_{f}" / "checkpoint_final.pth")))
    net.eval()
    role = "main" if len(folds) == 1 else f"fold_{f}"
    fn = "model.onnx" if len(folds) == 1 else f"{role}.onnx"
    with torch.no_grad():
        y = net(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        if ref is None:
            ref = y.detach().numpy()
        torch.onnx.export(net, x, str(out / fn), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    files[role] = fn

labels = json.loads(a.labels) if a.labels else {str(v): k for k, v in dataset_json["labels"].items() if isinstance(v, int)}
spec = {
    "kind": "segmentation", "recipe": "patch3d", "backend": "onnx", "files": files, "ensemble": list(files),
    "spacing_mm": spacing, "orientation": orientation, "crop_nonzero": a.crop_nonzero,
    "patch": patch[::-1], "overlap": round(1.0 - a.step_size, 4), "norm": norm, "activation": "softmax", "in_channels": nin,
    "labels": labels, "license": a.license, "source": a.source, "description": a.description,
    "nnunet": {"dataset": pm.dataset_name, "configuration": "3d_fullres", "folds": folds, "plans": pm.plans_name,
               "trainer_note": "mirroring is the caller's choice (tta option); TotalSegmentator trains with NoMirroring"},
}
json.dump(spec, open(out / "model.json", "w"), indent=2)
np.save(out / "ref_x.npy", x.numpy())  # parity fixture (first fold) for the ONNX backend
np.save(out / "ref_y.npy", ref)
print("exported", out, "patch(z,y,x)", patch, "classes", ref.shape[1], "folds", folds, "norm", norm, "spacing", spacing, orientation,
      "onnx MB", sum(os.path.getsize(out / f) for f in files.values()) / 1e6)
