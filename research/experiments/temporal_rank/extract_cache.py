"""Fine-field cache of the 100-frame cardiac truth over the heart box (common.CBOX): magnitude m (spoiled GRE at TE with
the phantom's tissue table and texture, blood x inflow gain), voxel-mean velocity v, label, blood fraction, 1 mm.

    rustlib/.venv/bin/python -I extract_cache.py [workers]

One truth4d frame decodes in ~50 s (unchunked stream), so frames are decoded in parallel workers that write into
memory-mapped .npy files in cache/. Resumable: frames whose done-marker exists are skipped.
"""
import sys
import time
from multiprocessing import Pool

import numpy as np

from common import CACHE, CBOX, NF, PH, magnitude, label_tables, read_frame

SRC = PH / "simblood_lib/truth4d.hv"
(z0, z1), (y0, y1), (x0, x1) = CBOX
SH = (z1 - z0, y1 - y0, x1 - x0)


def arrays(mode):
    from numpy.lib.format import open_memmap
    a = {}
    for name, dt, sh in [("m", np.float16, (NF,) + SH), ("v", np.float16, (NF, 3) + SH), ("lab", np.uint8, (NF,) + SH), ("blood", np.uint8, (NF,) + SH)]:
        p = CACHE / f"{name}.npy"
        a[name] = open_memmap(p, mode=mode, dtype=dt, shape=sh) if mode == "w+" else np.load(p, mmap_mode=mode)
    return a


def work(f):
    done = CACHE / f"done_{f:03d}"
    if done.exists():
        return f, 0.0
    from hastycompute import value_format as vf
    t0 = time.time()
    T = vf.open_lazy(SRC)
    tissues = list(T["tissues"])
    rows = label_tables(tissues)
    bx = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    lab = read_frame(T["labels"], f)[bx]
    bl = read_frame(T["blood"], f)[bx]
    vel = read_frame(T["vel"], f)[(slice(None),) + bx]
    tex = read_frame(T["tex"], f)[(slice(None),) + bx]
    m = magnitude(lab, tex[0], tex[1], bl, tissues, rows)
    a = arrays("r+")
    a["m"][f] = m.astype(np.float16)
    a["v"][f] = vel
    a["lab"][f] = lab
    a["blood"][f] = bl
    for x in a.values():
        x.flush()
    done.touch()
    return f, time.time() - t0


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    if not (CACHE / "m.npy").exists():
        arrays("w+")
    todo = [f for f in range(NF) if not (CACHE / f"done_{f:03d}").exists()]
    print(f"{len(todo)} frames to extract with {nw} workers, box {CBOX} -> {SH}", flush=True)
    t0 = time.time()
    with Pool(nw) as p:
        for f, dt in p.imap_unordered(work, todo):
            print(f"frame {f} {dt:.1f} s (elapsed {time.time() - t0:.0f} s)", flush=True)
    print("done", flush=True)
