"""Sanity check of an exported 4DFlowNet ONNX: smooth analytic flow + noise -> model; error vs truth must drop below the noisy input's,
and ONNX must reproduce the stored torch reference. usage: python check_4dflownet.py <model dir> [patch=16]"""
import sys
import numpy as np
import onnxruntime as ort

d = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
s = ort.InferenceSession(d + "/model.onnx", providers=["CPUExecutionProvider"])
ref = np.load(d + "/ref.npz")
names = ["u", "v", "w", "u_mag", "v_mag", "w_mag"]
y = s.run(None, {k: ref[k] for k in names})[0]
print("onnx vs torch ref max|diff|", np.abs(y - ref["y"]).max())

def field(c):  # c: coordinate in LR-index units per axis; smooth divergence-free-ish field, venc-normalised (<0.6)
    x, yy, z = np.meshgrid(*c, indexing="ij")
    k = 2 * np.pi / 24
    return np.stack([0.5 * np.sin(k * yy) * np.cos(k * z), 0.4 * np.sin(k * z) * np.cos(k * x), 0.3 * np.sin(k * x) * np.cos(k * yy)])[:, None, ...][:, 0]

rng = np.random.default_rng(0)
lr = np.arange(n, dtype=np.float64)
hr = np.arange(2 * n) * ((n - 1) / (2 * n - 1))
t_lr, t_hr = field([lr] * 3), field([hr] * 3)
for sigma in (0.02, 0.05, 0.1):
    noisy = t_lr + rng.normal(0, sigma, t_lr.shape)
    feed = {k: noisy[i][None, None].astype(np.float32) for i, k in enumerate(names[:3])} | {k: np.full((1, 1, n, n, n), 0.5, np.float32) for k in names[3:]}
    out = s.run(None, feed)[0][0]
    e_in = np.sqrt(((noisy - t_lr) ** 2).mean())
    e_out = np.sqrt(((out - t_hr) ** 2).mean())
    print(f"patch {n} sigma {sigma}: rms err input {e_in:.4f} -> output {e_out:.4f} (out shape {out.shape})")
