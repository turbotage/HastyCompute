"""Export 4DFlowNet-family Keras HDF5 weights (spatial super-resolution / denoising of 4D flow MRI) to ONNX.

Network re-implemented here from reading the published architecture (EdwardFerdian/4DFlowNet, MIT, src/Network/SR4DFlowNet.py)
and the `model_config` JSON stored inside the .h5 (layer wiring, residual scales, resize sizes). Weights are plain arrays read
with h5py; nothing from that repository is imported or executed and no pickle is loaded.

usage: python export_4dflownet.py <weights.h5> <out_dir> [--name NAME] [--license TXT] [--source TXT]
Run with rustlib/.venv (torch, onnx, h5py).

Network (all convs 3x3x3 on SYMMETRIC-padded input = replicate-edge pad of 1, except one 1x1x1):
  inputs u,v,w (velocity / venc, per component) and u_mag,v_mag,w_mag (magnitude / 4095), patch (N,1,X,Y,Z) each
  speed = |(u,v,w)|, mag = |(u_mag,v_mag,w_mag)|, pcmr = mag*speed
  phase branch: conv(u,v,w -> 64, relu) x2 ; pc branch: conv([pcmr,mag,speed] -> 64, relu) x2 ; concat -> 1x1 conv relu -> 3x3 conv relu
  8 low-res residual blocks, trilinear x`res_increase` (align_corners=True; = 2 chained bilinear resizes of the original),
  4 high-res residual blocks, 3 heads conv(64, relu) -> conv(1) [tanh in the published h5 files] -> (u,v,w) / venc output.
  residual block: y = leaky(conv_b(leaky(conv_a(x))) * s + x)  (leaky slope 0.2; first conv of a block has no activation of its own)
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Conv(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.c = nn.Conv3d(cin, cout, k)
        self.p = k // 2

    def forward(self, x):
        return self.c(F.pad(x, [self.p] * 6, mode="replicate")) if self.p else self.c(x)


class Net(nn.Module):
    def __init__(self, scales_lo, scales_hi, head_act, res_increase=2, ch=64, temporal=False):
        super().__init__()
        self.res_increase, self.head_act, self.temporal = res_increase, head_act, temporal
        self.scales = list(scales_lo) + list(scales_hi)
        self.nlo = len(scales_lo)
        # conv list in Keras creation order: conv3d, conv3d_1, ...  (index = suffix)
        self.convs = nn.ModuleList()
        if temporal:  # no magnitude: phase branch only (u,v,w -> 64 -> 64), then 1x1, 3x3
            self.convs.append(Conv(3, ch))
            self.convs.append(Conv(ch, ch))
            self.convs.append(Conv(ch, ch, 1))
            self.convs.append(Conv(ch, ch))
        else:
            self.convs.append(Conv(3, ch))      # 0  pc branch
            self.convs.append(Conv(ch, ch))     # 1
            self.convs.append(Conv(3, ch))      # 2  phase branch
            self.convs.append(Conv(ch, ch))     # 3
            self.convs.append(Conv(2 * ch, ch, 1))  # 4
            self.convs.append(Conv(ch, ch))     # 5
        for _ in range(2 * (len(scales_lo) + len(scales_hi))):
            self.convs.append(Conv(ch, ch))  # 6..
        base = len(self.convs)
        for h in range(3):
            self.convs.append(Conv(ch, ch))  # head a (relu)
            self.convs.append(Conv(ch, 1))   # head b

    def act(self, x):
        return F.leaky_relu(x, 0.2)

    def block(self, x, ia, ib, s):
        y = self.act(self.convs[ia](x))
        y = self.convs[ib](y) * s
        return self.act(x + y)

    def forward(self, u, v, w, um=None, vm=None, wm=None):
        r = torch.relu
        if self.temporal:
            x = torch.cat([u, v, w], 1)
            for j in range(4):
                x = r(self.convs[j](x))
            return self.tail(x, 4)
        speed = (u ** 2 + v ** 2 + w ** 2) ** 0.5
        mag = (um ** 2 + vm ** 2 + wm ** 2) ** 0.5
        pc = torch.cat([mag * speed, mag, speed], 1)
        ph = torch.cat([u, v, w], 1)
        pc = r(self.convs[1](r(self.convs[0](pc))))
        ph = r(self.convs[3](r(self.convs[2](ph))))
        x = r(self.convs[4](torch.cat([ph, pc], 1)))
        x = r(self.convs[5](x))
        return self.tail(x, 6)

    def tail(self, x, i):
        r = torch.relu
        for k, s in enumerate(self.scales):
            if k == self.nlo:  # the first hi-res block takes the upsampled tensor directly, no activation
                if self.temporal:  # tf.image.resize bilinear (half-pixel centres) along T only (axis 2 here)
                    x = F.interpolate(x, scale_factor=(float(self.res_increase), 1.0, 1.0), mode="trilinear", align_corners=False)
                else:
                    x = F.interpolate(x, scale_factor=float(self.res_increase), mode="trilinear", align_corners=True)
            x = self.block(x, i, i + 1, s)
            i += 2
        outs = []
        for h in range(3):
            y = r(self.convs[i](x))
            y = self.convs[i + 1](y)
            outs.append(torch.tanh(y) if self.head_act == "tanh" else y)
            i += 2
        return torch.cat(outs, 1)


def read_keras(path):
    h = h5py.File(path, "r")
    cfg = json.loads(h.attrs["model_config"])["config"]
    w = {}
    for name, grp in h["model_weights"].items():
        for k in grp:
            sub = grp[k]
            for wn in sub:
                w[(name, wn.split(":")[0])] = sub[wn][()]
    return cfg, w


def parse(cfg):
    layers = {l["config"]["name"]: l for l in cfg["layers"]}
    muls = sorted([n for n in layers if n.startswith("mul_") and n != "mul"], key=lambda n: int(n.split("_")[1]))
    scales = [float(layers[n]["config"]["constants"]["1"]) for n in muls]
    conv_names = sorted([n for n, l in layers.items() if l["class_name"] == "Conv3D"], key=lambda n: -1 if n == "conv3d" else int(n.split("_")[1]))
    last_act = layers[conv_names[-1]]["config"]["activation"]
    resize = [layers[n]["config"]["constants"]["1"] for n in layers if n.startswith("ResizeBilinear")]
    inp = [l["config"]["batch_input_shape"] for l in layers.values() if l["class_name"] == "InputLayer"]
    # structure checks
    assert len(conv_names) == 6 + 2 * len(scales) + 6, (len(conv_names), len(scales))
    assert all(layers[n]["config"]["strides"] == [1, 1, 1] for n in conv_names)
    pads = [l["config"]["constants"]["1"] for n, l in layers.items() if l["config"]["name"].startswith("MirrorPad")]
    return conv_names, scales, last_act, resize, inp, pads


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--name", default="4DFlowNet")
    ap.add_argument("--res-increase", type=int, default=2, help="temporal model only (from its yaml)")
    ap.add_argument("--license", default="")
    ap.add_argument("--source", default="")
    a = ap.parse_args()
    cfg, w = read_keras(a.weights)
    temporal = not any(l["class_name"] == "InputLayer" and l["config"]["name"] == "u_mag" for l in cfg["layers"])
    if temporal:  # Temporal4DFlowNet (PiaaCaa): layers are TFOpLambda, so structure comes from its published source instead of constants
        conv_names = sorted([l["config"]["name"] for l in cfg["layers"] if l["class_name"] == "Conv3D"], key=lambda n: -1 if n == "conv3d" else int(n.split("_")[1]))
        assert len(conv_names) == 4 + 2 * 12 + 6, len(conv_names)
        scales, last_act, pads = [1.0] * 12, "linear", []  # resnet_block scale=1 (source default; not read from the h5)
        inp = [l["config"]["batch_input_shape"] for l in cfg["layers"] if l["class_name"] == "InputLayer"]
        patch, res_increase, nlo = inp[0][1:4], a.res_increase, 8
    else:
        conv_names, scales, last_act, resize, inp, pads = parse(cfg)
        patch = inp[0][1:4]
        res_increase = resize[0][0] // patch[1]
    nlo = 8  # low-res blocks: scales before the upsample; read from wiring: count blocks whose input precedes ResizeBilinear
    # blocks in wiring order: mul_1..mul_k; the upsample sits after the block whose output feeds reshape_bx
    if not temporal:
        layers = {l["config"]["name"]: l for l in cfg["layers"]}
        src = layers["reshape_bx"]["inbound_nodes"][0][0][0]  # leaky_re_lu_{2*nlo-1}
        nlo = (int(src.split("_")[-1]) + 1) // 2 if src.split("_")[-1].isdigit() else 1
    net = Net(scales[:nlo], scales[nlo:], last_act, res_increase, temporal=temporal)
    sd = {}
    nobias = {}
    for idx, n in enumerate(conv_names):
        k, b = w[(n, "kernel")], w.get((n, "bias"))
        nobias[idx] = b is None
        sd[f"convs.{idx}.c.weight"] = torch.from_numpy(np.ascontiguousarray(k.transpose(4, 3, 0, 1, 2)))
        if b is not None:
            sd[f"convs.{idx}.c.bias"] = torch.from_numpy(b)
    for idx, nb in nobias.items():
        if nb:
            net.convs[idx].c.bias = None
    net.load_state_dict(sd, strict=True)  # shapes must agree: a wrong wiring/size fails here
    net.eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    p = int(patch[0])
    names = ["u", "v", "w"] if temporal else ["u", "v", "w", "u_mag", "v_mag", "w_mag"]
    x = [torch.rand(1, 1, p, p, p) * 2 - 1 for _ in range(3)] + ([] if temporal else [torch.rand(1, 1, p, p, p) for _ in range(3)])
    with torch.no_grad():
        y = net(*x)
        torch.onnx.export(net, tuple(x), str(out / "model.onnx"), input_names=names, output_names=["y"], opset_version=17, dynamo=False,
                          dynamic_axes={n: {0: "n", 2: "x", 3: "y", 4: "z"} for n in names} | {"y": {0: "n", 2: "x", 3: "y", 4: "z"}})
    np.savez(out / "ref.npz", **{n: t.numpy() for n, t in zip(names, x)}, y=y.numpy())
    spec = {
        "kind": "flow_sr", "recipe": "patch4d_sr", "axis_layout": "(N,C,T,Y,Z): res_increase applies to T" if temporal else "(N,C,X,Y,Z): res_increase applies to all three spatial axes", "backend": "onnx", "files": {"main": "model.onnx"},
        "res_increase": int(res_increase), "patch": [p, p, p], "inputs": names, "output": "u,v,w (normalised by venc)",
        "norm": {"velocity": "divide by venc (per component, array units of the phase image)", "magnitude": None if temporal else "divide by 4095"},
        "residual_scales": scales, "head_activation": last_act, "padding": "symmetric (replicate edge)", "temporal": temporal,
        "license": a.license, "source": a.source, "description": a.name,
    }
    json.dump(spec, open(out / "model.json", "w"), indent=2)
    print("exported", out, "patch", p, "res_increase", res_increase, "low blocks", nlo, "high blocks", len(scales) - nlo, "scales", set(scales), "head", last_act,
          "pads", sorted({json.dumps(q) for q in pads}), "onnx MB", os.path.getsize(out / "model.onnx") / 1e6)
