"""Reference logits of an exported nnU-Net fold, computed with nnU-Net's own sliding-window code (torch), for the parity
tests of hasty-infer. usage: python pvs_ref_predict.py <trained_model_folder> <fold> norm.npy out_prefix
norm.npy: normalised image [z,y,x] float32. Writes <out_prefix>_mirror.npy / _plain.npy: logits [C,z,y,x].
Checkpoints are read with safe_load (allowlisted unpickler)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from safe_load import load_state  # noqa: E402
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans  # noqa: E402
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402

folder, fold, norm, prefix = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
pm = PlansManager(json.load(open(folder / "plans.json")))
cm = pm.get_configuration("3d_fullres")
dj = json.load(open(folder / "dataset.json"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs, cm.network_arch_init_kwargs_req_import,
                             len(dj["channel_names"]), len(dj["labels"]), allow_init=False, deep_supervision=False)
net.load_state_dict(load_state(str(folder / f"fold_{fold}" / "checkpoint_final.pth")))
net.to(dev).eval()
x = torch.from_numpy(np.load(norm).astype(np.float32))[None]
for name, axes in (("mirror", (0, 1, 2)), ("plain", None)):
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=axes is not None, perform_everything_on_device=True, device=dev, verbose=False)
    p.network, p.configuration_manager, p.allowed_mirroring_axes = net, cm, axes
    p.label_manager = SimpleNamespace(num_segmentation_heads=len(dj["labels"]))
    with torch.no_grad():
        lg = p.predict_sliding_window_return_logits(x).cpu().numpy()
    np.save(f"{prefix}_{name}.npy", lg.astype(np.float32))
    print(name, lg.shape, np.bincount(lg.argmax(0).ravel(), minlength=lg.shape[0]))
