"""k-resolved energy of pooled images (warm_full.hv, encode 0) and, when <run>/synth_pd.hv exists, of their error
against the inverse-crime object, in shells of the normalised 3D radius r = |k| / k_Nyquist per axis (r > 1: the cube
corners outside the k-crop ball). Usage: err_spec.py <run dir> [<run dir> ..]"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/lib"))
from hastycompute import value_format as vf

EDGES = [0, 0.125, 0.25, 0.5, 0.75, 0.9, 1.0, 1.2, 1.8]


def shells(shape):
    f = [np.fft.fftfreq(n) * 2 for n in shape]  # in units of the Nyquist frequency
    g = np.meshgrid(*f, indexing="ij")
    return np.sqrt(sum(a * a for a in g))


for d in map(Path, sys.argv[1:]):
    # stored (E, Z, Y, X) in C order ("order F" of the logical (X, Y, Z, E)); shells are per-axis normalised
    x = np.asarray(vf.read(d / "warm_full.hv")["array"])[0]
    r = shells(x.shape)
    ref = None
    if (d / "synth_pd.hv").exists():
        ref = np.asarray(vf.read(d / "synth_pd.hv")["array"]).reshape(x.shape)
    X = np.fft.fftn(x)
    tot = np.sum(np.abs(X) ** 2)
    row = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        m = (r >= a) & (r < b)
        s = f"{np.sum(np.abs(X[m]) ** 2) / tot:.3f}"
        if ref is not None:
            E = np.fft.fftn(x - ref)
            R = np.fft.fftn(ref)
            s += f"/{np.sqrt(np.sum(np.abs(E[m]) ** 2) / max(np.sum(np.abs(R[m]) ** 2), 1e-30)):.2f}"
        row.append(s)
    print(f"{d.name:16s} shape {x.shape} energy share per shell {list(zip(EDGES[:-1], row))}")
