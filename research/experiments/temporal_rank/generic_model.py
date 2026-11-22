"""Run 7: a phantom-free model of the velocity-phase rank (research/temporal_rank.md).

An ensemble of voxels, each x(t) = exp(-i pi a_v W(t - tau_v) / VENC) (one encode with VENC = the pair's
VENC: phase pi at v = VENC), with W a unit-peak waveform, a_v uniform in [0, v_max] (voxels across the lumen profile
and across vessels), tau_v uniform in [0, spread]. T = 200 phases. K at the error levels from the Gram. Waveforms:
the network's ascending-aorta flow (systolic jet), a smooth sinusoid-like venous waveform (SVC) and the coronary
diastolic-dominant shape with a ramp of 60 / 30 / 15 ms. Also the magnitude analogue: a moving edge (partial volume
ramps between two levels as a wall of amplitude A mm passes a 2 mm voxel).
"""
import json

import numpy as np

from common import LEVELS, OUT, PH, k_at
from regimes import coronary_shape

T = 200
RR = 0.8528221101760362
rng = np.random.default_rng(0)


def waveform(name):
    z = np.load(PH / "flow/network_flow.npz")
    t = np.arange(T) / T * RR
    if name in ("aorta", "svc"):
        names = json.loads((PH / "graph.json").read_text())["edge_names"]
        from hastycompute import value_format as vf
        eo = np.asarray(vf.open_lazy(PH / "vessels.hv")["edge_offsets"][:])
        nm = "Ascending aorta" if name == "aorta" else "Superior vena cava"
        e = names.index(nm)
        p = (eo[e] + eo[e + 1]) // 2
        c = z["q"][p]
        Q = np.real(c[0] + 2 * (c[None, 1:] * np.exp(1j * z["omega"][None, 1:] * t[:, None])).sum(1))
        return Q / np.abs(Q).max()
    ramp = {"cor60": 60, "cor30": 30, "cor15": 15}[name]
    s = coronary_shape(np.arange(T) / T, ramp * 1e-3 / RR, 2.5)
    return s / s.max()


def K_of(X):
    G = X.conj().T @ X
    w = np.sort(np.linalg.eigvalsh(G))[::-1]
    return k_at(np.maximum(w, 0), LEVELS)


def ensemble(W, vmax_over_venc, spread_ms, V=4000):
    a = rng.uniform(0, vmax_over_venc, V)
    tau = rng.uniform(0, spread_ms * 1e-3 / RR * T, V)
    tt = np.arange(T)
    out = np.empty((V, T), np.complex128)
    Wf = np.fft.fft(W)
    f = np.fft.fftfreq(T) * T
    for i in range(V):
        Ws = np.real(np.fft.ifft(Wf * np.exp(-2j * np.pi * f * tau[i] / T)))
        out[i] = np.exp(-1j * np.pi * a[i] * Ws)
    return out


def edge(amp_mm, ramp_frac=None, V=4000, vox_mm=2.0):
    """partial-volume ramps: a wall moving with displacement amp_mm * (1 - cos 2 pi t)/2 (+ random offsets) through a
    voxel of vox_mm; the voxel's blood fraction is the clipped overlap; signal = 0.3 + 0.7 * fraction"""
    t = np.arange(T) / T
    pos = amp_mm * (1 - np.cos(2 * np.pi * t)) / 2
    off = rng.uniform(-vox_mm, amp_mm + vox_mm, V)
    frac = np.clip((pos[None] - off[:, None]) / vox_mm + 0.5, 0, 1)
    return (0.3 + 0.7 * frac).astype(np.complex128)


if __name__ == "__main__":
    res = {"levels": LEVELS, "velocity": [], "edge": []}
    for wn in ["aorta", "svc", "cor60", "cor30", "cor15"]:
        W = waveform(wn)
        for r in [0.25, 0.5, 1.0, 1.5, 2.0]:
            for sp in [0, 50, 100]:
                k = K_of(ensemble(W, r, sp))
                res["velocity"].append(dict(waveform=wn, vmax_over_venc=r, spread_ms=sp, K=k))
                print(f"{wn:6s} vmax/VENC {r:4.2f} spread {sp:3d} ms  K@{LEVELS} = {k}", flush=True)
    for amp in [2, 5, 10, 15]:
        k = K_of(edge(amp))
        res["edge"].append(dict(amp_mm=amp, K=k))
        print(f"edge amplitude {amp} mm: K = {k}")
    (OUT / "generic_model.json").write_text(json.dumps(res, indent=1))
