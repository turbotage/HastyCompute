"""Export FastSurfer v2 (FastSurferVINN, aparc_vinn_{axial,coronal,sagittal}_v2.0.0.pkl) to ONNX + model.json for the
hasty-infer `slab2d_views` recipe.

The network is re-implemented here from reading FastSurfer's published architecture (Apache-2.0, Deep-MI/FastSurfer
v2.6.1, FastSurferCNN/models/{networks,sub_module}.py); nothing of that repository is imported or executed, and the
checkpoints are read with safe_load (allowlisted unpickler, no code execution). Hyper-parameters are inferred from the
checkpoint's tensor shapes; the load is strict, so a wrong re-implementation fails loudly. Only the 1 mm case is
exported: with scale factor 1.0 and a 256x256 plane the VINN resolution interpolation layers are the identity.

usage: python export_fastsurfer.py <dir with aparc_vinn_*_v2.0.0.pkl> <out_dir> [--lut FastSurfer_ColorLUT.tsv]
Run with rustlib/.venv (torch + onnx).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from safe_load import load_state  # noqa: E402


class DenseBlock(nn.Module):
    """CompetitiveDenseBlock (kind='mid'), CompetitiveDenseBlockInput ('input') or OutputDenseBlock ('output')."""

    def __init__(self, kind, cin, nf, k, nf_last=None, outblock=False):
        super().__init__()
        p = k // 2
        self.kind = kind
        self.outblock = outblock
        nl = nf_last or nf
        self.conv0 = nn.Conv2d(cin, nf, k, 1, p)
        self.conv1 = nn.Conv2d(nf, nf, k, 1, p)
        self.conv2 = nn.Conv2d(nf, nf, k, 1, p)
        self.conv3 = nn.Conv2d(nf, nl, k, 1, p)
        names = ["bn1", "bn2", "bn3", "bn4"] if kind == "mid" else (["bn0", "bn1", "bn2", "bn3", "bn4"] if kind == "input" else ["gn1", "gn2", "gn3", "gn4"])
        for n in names:
            setattr(self, n, nn.BatchNorm2d(nl if n.endswith("4") else nf) if n != "bn0" else nn.BatchNorm2d(cin))
        self.prelu = nn.PReLU()

    def forward(self, x, skip=None):
        pr = self.prelu
        if self.kind == "mid":
            x1 = torch.maximum(x, self.bn1(self.conv0(pr(x))))
            x2 = torch.maximum(self.bn2(self.conv1(pr(x1))), x1)
            x3 = torch.maximum(self.bn3(self.conv2(pr(x2))), x2)
            out = self.conv3(pr(x3))
            return out if self.outblock else self.bn4(out)
        if self.kind == "input":
            x1 = self.bn1(self.conv0(self.bn0(x)))
            x2 = torch.maximum(self.bn2(self.conv1(pr(x1))), x1)
            x3 = torch.maximum(self.bn3(self.conv2(pr(x2))), x2)
            return self.bn4(self.conv3(pr(x3)))
        # output block: concat(decoder, input-block skip)
        c = torch.cat((x, skip), dim=1)
        x1 = self.gn1(self.conv0(pr(c)))
        x2 = torch.maximum(x1, self.gn2(self.conv1(pr(x1))))
        x3 = torch.maximum(self.gn3(self.conv2(pr(x2))), x2)
        return self.gn4(self.conv3(pr(x3)))


class InputDense(nn.Module):
    """InputDenseBlock of FastSurferVINN (batch norms are called gn*)."""

    def __init__(self, cin, nf, k):
        super().__init__()
        p = k // 2
        self.conv0, self.conv1, self.conv2, self.conv3 = (nn.Conv2d(cin if i == 0 else nf, nf, k, 1, p) for i in range(4))
        self.bn0 = nn.BatchNorm2d(cin)
        self.gn1, self.gn2, self.gn3, self.gn4 = (nn.BatchNorm2d(nf) for _ in range(4))
        self.prelu = nn.PReLU()

    def forward(self, x):
        pr = self.prelu
        x1_gn = self.gn1(self.conv0(self.bn0(x)))
        x1 = self.conv1(pr(x1_gn))
        x2_max = torch.maximum(self.gn2(x1), x1_gn)
        x2 = self.conv2(pr(x2_max))
        x3_max = torch.maximum(self.gn3(x2), x2_max)
        return self.gn4(self.conv3(pr(x3_max)))


class Enc(nn.Module):
    def __init__(self, kind, cin, nf, k):
        super().__init__()
        self.block = DenseBlock(kind, cin, nf, k)
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):
        o = self.block(x)
        p, idx = self.maxpool(o)
        return p, o, idx


class Dec(nn.Module):
    def __init__(self, nf, k):
        super().__init__()
        self.block = DenseBlock("mid", nf, nf, k)

    def forward(self, x, skip, idx):
        n, c, h, w = skip.shape  # max-unpool by scatter (aten::max_unpool2d has no ONNX export): idx is flat per channel
        up = torch.zeros(n, c, h * w, dtype=x.dtype).scatter(2, idx.flatten(2), x.flatten(2)).reshape(n, c, h, w)
        return self.block(torch.maximum(up, skip))


class VINN(nn.Module):
    """FastSurferVINN at scale factor 1 (interpolation layers = identity)."""

    def __init__(self, channels, nf, nfi, k, classes):
        super().__init__()
        self.inp_block = InputDense(channels, nfi, k)
        self.encode1 = Enc("input", nfi, nf, k)
        self.encode2, self.encode3, self.encode4 = (Enc("mid", nf, nf, k) for _ in range(3))
        self.bottleneck = DenseBlock("mid", nf, nf, k)
        self.decode4, self.decode3, self.decode2, self.decode1 = (Dec(nf, k) for _ in range(4))
        self.outp_block = DenseBlock("output", nf + nfi, nf, k)
        self.classifier = nn.Module()
        self.classifier.conv = nn.Conv2d(nf, classes, 1)

    def forward(self, x):
        s0 = self.inp_block(x)
        e1, s1, i1 = self.encode1(s0)
        e2, s2, i2 = self.encode2(e1)
        e3, s3, i3 = self.encode3(e2)
        e4, s4, i4 = self.encode4(e3)
        d = self.bottleneck(e4)
        d = self.decode4(d, s4, i4)
        d = self.decode3(d, s3, i3)
        d = self.decode2(d, s2, i2)
        d = self.decode1(d, s1, i1)
        return self.classifier.conv(self.outp_block(d, s0))


def remap(sd):
    """checkpoint keys -> this module's names (blocks of the encoders / decoders are wrapped in `.block`)"""
    out = {}
    for k, v in sd.items():
        for pre in ("encode1", "encode2", "encode3", "encode4", "decode1", "decode2", "decode3", "decode4"):
            if k.startswith(pre + ".") and not k.startswith(pre + ".block."):
                k = pre + ".block." + k[len(pre) + 1:]
                break
        out[k] = v
    return out


def build(sd, classes_hint=None):
    ch = sd["inp_block.conv0.weight"].shape[1]
    nfi = sd["inp_block.conv0.weight"].shape[0]
    nf = sd["encode1.conv0.weight"].shape[0]
    k = sd["encode1.conv0.weight"].shape[-1]
    classes = sd["classifier.conv.weight"].shape[0]
    return VINN(ch, nf, nfi, k, classes), (ch, nf, nfi, k, classes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--lut", default=None, help="FastSurfer_ColorLUT.tsv (ID, name columns): class index -> FreeSurfer label")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files, classes = {}, {}
    ref = {}
    torch.manual_seed(0)
    for view in ("axial", "coronal", "sagittal"):
        sd = load_state(str(Path(a.src) / f"aparc_vinn_{view}_v2.0.0.pkl"), "model_state")
        net, hp = build(sd)
        net.load_state_dict(remap(sd), strict=True)
        net.eval()
        x = torch.randn(1, hp[0], 256, 256).clamp(0, 1)
        with torch.no_grad():
            y = net(x)
            torch.onnx.export(net, x, str(out / f"{view}.onnx"), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False,
                              dynamic_axes={"x": {0: "n"}, "y": {0: "n"}})
        np.save(out / f"ref_{view}_x.npy", x.numpy())
        np.save(out / f"ref_{view}_y.npy", y.numpy())
        files[view], classes[view] = f"{view}.onnx", hp[4]
        print(view, "hyper-parameters (channels, filters, interp filters, kernel, classes):", hp)
    lut_ids, lut_names = None, {}
    if a.lut:
        rows = list(csv.reader(open(a.lut), delimiter="\t"))
        hdr = rows[0]
        idc = hdr.index("ID")
        nc = hdr.index("LabelName") if "LabelName" in hdr else 1
        lut_ids = [int(r[idc]) for r in rows[1:]]
        lut_names = {str(int(r[idc])): r[nc] for r in rows[1:]}
    spec = {
        "kind": "segmentation", "recipe": "slab2d_views", "backend": "onnx", "files": files, "ensemble": list(files),
        "spacing_mm": [1.0, 1.0, 1.0], "orientation": "LIA", "grid": [256, 256, 256], "slab": 7,
        "view_weights": {"axial": 0.4, "coronal": 0.4, "sagittal": 0.2},
        "view_classes": classes, "full_classes": classes["axial"],
        # sagittal logits (hemisphere-merged) -> full class order, FastSurferCNN/data_loader/data_utils.py map_prediction_sagittal2full
        "sagittal_gather": [0] + list(range(5, 14)) + list(range(1, 4)) + [14, 15, 4] + list(range(16, 19)) + list(range(5, 51)) + [20, 22, 27] + list(range(29, 32)) + [33, 34] + list(range(38, 43)) + [45],
        "class_ids": lut_ids, "norm": {"type": "fastsurfer_conform"}, "activation": "none", "in_channels": 1,
        "labels": lut_names, "license": "Apache-2.0 (code and weights; research use, not for clinical decisions)",
        "source": "Deep-MI/FastSurfer v2.6.1, aparc_vinn_*_v2.0.0 (zenodo.org/records/10390573)",
        "description": "FastSurfer VINN (1 mm): aseg + DKT cortical parcels, 3 view networks, logits combined 0.4/0.4/0.2. No cortex re-lateralisation or cleanup.",
    }
    assert len(spec["sagittal_gather"]) == classes["axial"], len(spec["sagittal_gather"])
    json.dump(spec, open(out / "model.json", "w"), indent=2)
    print("exported", out, "onnx MB", sum(os.path.getsize(out / f) for f in files.values()) / 1e6)
