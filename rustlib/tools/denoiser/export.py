"""Export a trained denoiser for inference outside Python (Rust: crates/hasty-denoise via tch-rs).

    python -I export.py <ckpt.pt | --random> [--out DIR] [--pt2] [--aoti] [--ref-case PATH]

Writes to DIR (default <ckpt dir>/export, or data/hasty/denoiser/test_export for --random):
  model.ts     TorchScript (traced, frozen weights; dynamic B, T, X, Y, Z), float32, CPU — move to CUDA after loading
  model.json   config, pad_multiple, receptive field, I/O layout and phase convention
  model.pt2    (--pt2) torch.export ExportedProgram with dynamic shapes; (--aoti) AOTInductor package for this machine
  ref/         reference volume in the Rust layout (frame-major, encode-minor list of x-fastest complex64 images,
               phase convention x_e ∝ exp(+i kv v)) and the Python `denoise_volume` output of it — the Rust test
               (`cargo test -p hasty-denoise --features torch`) must reproduce it through its own layout conversion.
--random: untrained model with random head and conditioning (non-trivial map), for tests only.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: E402,F401  (caches inside the repo; before torch)
import torch  # noqa: E402
from dataset import REPO, TRAIN, _Case, layout  # noqa: E402
from model import Denoiser7p, denoise_volume, receptive_field  # noqa: E402

args = sys.argv[1:]


def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default


torch.manual_seed(0)
if args[0] == "--random":
    model = Denoiser7p(widths=(1, 2, 2), group="Oh")
    with torch.no_grad():
        for prm in model.head.kern.params:
            prm.normal_(0, 0.05)
        for prm in model.cond[-1].parameters():
            prm.normal_(0, 0.1)
    out = Path(opt("--out", REPO / "data/hasty/denoiser/test_export"))
    src = "random (test)"
else:
    ck = torch.load(args[0], map_location="cpu", weights_only=False)  # our own training checkpoint (run.json, versions)
    cfg = ck["cfg"]
    model = Denoiser7p(widths=cfg["widths"], group=cfg["group"], t_levels=cfg["t_levels"], t_pad="circular", blocks=cfg["blocks"])
    model.load_state_dict(ck["model"])
    out = Path(opt("--out", Path(args[0]).parent / "export"))
    src = str(Path(args[0]).resolve())
model = model.float().eval()
model.cfg["t_pad"] = "circular"  # inference is on full cardiac cycles
for m in model.modules():
    if hasattr(m, "pad"):
        m.pad = "circular"
rs, rt = receptive_field(model)
model.freeze()
out.mkdir(parents=True, exist_ok=True)

L = model.pad_multiple
ex = (torch.randn(2, 7, 2, 4, 2 * L + 1, 2 * L + 1, 2 * L + 1), torch.tensor([[0.0, np.pi, np.pi / 2.5] + [np.pi, np.pi / 2.5] * 2] * 2), torch.tensor([2e-3] * 2))  # batch 2: no 0/1 specialisation
with torch.no_grad():
    ts = torch.jit.trace(model, ex, check_trace=False)
ts.save(str(out / "model.ts"))
info = {
    "source": src,
    "torch": torch.__version__,
    "cfg": model.cfg,
    "pad_multiple": L,
    "rf_space": rs,
    "rf_time": rt,
    "inputs": {
        "x": "float32 [B, 7, 2, T, X, Y, Z]: (re, im) of [ref, ax0_lo, ax0_hi, ax1_lo, ax1_hi, ax2_lo, ax2_hi]; spatial sizes ≡ 1 mod pad_multiple; full cardiac cycle (circular in T)",
        "kv": "float32 [B, 7]: π/VENC per slot (0 for the reference)",
        "h": "float32 [B]: isotropic voxel size, m",
    },
    "output": "float32 [B, 7, 2, T, X, Y, Z], same layout",
    "phase_convention": "x_e ∝ m e^{iθ} exp(+i kv_e v_axis(e)), encode axes = tensor spatial axes; for exp(−i…) data apply x_e → conj(x_e) x_ref/conj(x_ref) before and after",
    "scale": "any (positively homogeneous); the training data had p99 |x_ref| ≈ 1",
}
(out / "model.json").write_text(json.dumps(info, indent=1))

if "--pt2" in args or "--aoti" in args:
    from torch.export import Dim

    b, t = Dim("b", min=1, max=64), Dim("t", min=2, max=128)
    nx, ny, nz = Dim("nx", min=1, max=256), Dim("ny", min=1, max=256), Dim("nz", min=1, max=256)
    dyn = {"x": {0: b, 3: t, 4: L * nx + 1, 5: L * ny + 1, 6: L * nz + 1}, "kv": {0: b}, "h": {0: b}}
    try:
        ep = torch.export.export(model, ex, dynamic_shapes=dyn)
        torch.export.save(ep, str(out / "model.pt2"))
        print(f"torch.export: {out / 'model.pt2'}")
        if "--aoti" in args:
            pkg = torch._inductor.aoti_compile_and_package(ep, package_path=str(out / "model_aoti.pt2"))
            print(f"AOTInductor package: {pkg}")
    except Exception as e:  # keep the TorchScript export usable
        print(f"torch.export failed: {type(e).__name__}: {str(e)[:400]}")

# reference volume in the Rust layout: a test-split RACLETTE crop with sizes not ≡ 1 mod L (exercises padding)
case = opt("--ref-case", "7p_2Venc/VirtualSubject_n001")
meta = json.loads((TRAIN / case / "case.json").read_text())
order, kv = layout(meta["encodings"])
c = np.array(meta["dims"]) // 2
d = np.array([23, 19, 13])
T = 6
sl = tuple(slice(int(c[a] - d[a] // 2), int(c[a] - d[a] // 2 + d[a])) for a in range(3))
cs = _Case(TRAIN / case)  # case.hv (chunked, lazy) or legacy npy
v = np.asarray(cs.box(cs.x, [q.start for q in sl], [q.stop for q in sl], list(range(T))), np.float32)[:, :, :, :, order]  # [X, Y, Z, T, 7, 2]  (native grid: test of plumbing)
vc = v[..., 0] + 1j * v[..., 1]
xin = torch.from_numpy(np.ascontiguousarray(np.transpose(v, (4, 5, 3, 0, 1, 2))))  # [7, 2, T, X, Y, Z]
kvt = torch.from_numpy(kv.astype(np.float32))
with torch.no_grad():
    y = denoise_volume(model, xin, kvt, 2e-3).numpy()  # [7, 2, T, X, Y, Z]
yc = y[:, 0] + 1j * y[:, 1]  # [7, T, X, Y, Z]
(out / "ref").mkdir(exist_ok=True)
# Rust layout: images[frame * 7 + e], each x-fastest (Fortran) [X, Y, Z]
def rust_layout(a):  # a [7, T, X, Y, Z] complex -> bytes
    imgs = [np.asfortranarray(a[e, t]).ravel(order="F") for t in range(a.shape[1]) for e in range(7)]
    return np.concatenate(imgs).astype(np.complex64).tobytes()


(out / "ref" / "volume_in.c8").write_bytes(rust_layout(np.transpose(vc, (4, 3, 0, 1, 2))))
(out / "ref" / "volume_out.c8").write_bytes(rust_layout(yc))
(out / "ref" / "ref.json").write_text(json.dumps({"dims": d.tolist(), "frames": T, "kv": kv.tolist(), "h": 2e-3, "case": case,
                                                   "layout": "complex64, images[frame*7 + e] x-fastest, e in [ref, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]"}, indent=1))
print(f"exported {src} -> {out}: model.ts, model.json, ref/ (dims {d.tolist()}, {T} frames); receptive field ±{rs} / ±{rt}")
