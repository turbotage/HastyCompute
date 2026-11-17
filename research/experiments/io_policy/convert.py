"""Convert existing .npy arrays to policy-compressed .hv files (research/io_policy.md §2). Dry run by default.

    rustlib/.venv/bin/python -I research/experiments/io_policy/convert.py [--apply] [--delete-source] [--min-mb 64]
        [--out DIR] [--log FILE] PATH...

PATH: files or directories (recursed). Every .npy is classified like audit.py (k-space, per-sample records, image,
integer); the .hv is written next to it (`x.npy` -> `x.hv`), read back and checked against the policy's error bound on
the whole array (k-space / images: max |err| ≤ step/2 (+ float32 rounding); records: per field and readout), and
only then, with --delete-source, the .npy is removed. Readers resolve `x.npy` -> `x.hv` (hasty_value::store::resolve,
hastycompute.store.load_any), so paths in scripts and meta files keep working.

Run only after the readers of the files are migrated (research/io_policy.md §4) and never on a directory a running job
is writing (simulations write `*.part.npy` and `state_*.hv`, which are skipped).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import store  # noqa: E402

args = sys.argv[1:]
apply_, delete = "--apply" in args, "--delete-source" in args
VALUED = ("--min-mb", "--out", "--log")
opt = lambda k, d=None: args[args.index(k) + 1] if k in args else d
min_mb = float(opt("--min-mb", 64.0))
# --out DIR: write the .hv files there (same names) instead of next to the sources (verification copies); the source
# directory's other files are not copied
out_dir = Path(opt("--out")) if opt("--out") else None
log = open(opt("--log"), "a") if opt("--log") else None
paths = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in VALUED)]


def say(m):
    print(m, flush=True)
    if log:
        log.write(m + "\n")
        log.flush()


# per-sample records: 24-bit signed integers per field and readout (= simphantom's sched_save)
RECORD_BITS = 24
# directories whose arrays are read by code without .hv support (hasty-cfd cases, hasty-phantom subjects, models,
# training data, external downloads): never converted
EXCLUDE = ("/lbm/", "/subjects/", "/models/", "/external/", "/train/", "/totalseg/", "/denoiser/", "/pip-cache/")


def classify(p, a):
    n = p.name
    if a.dtype == np.complex64 and a.ndim == 3 and (n.startswith("data_clean") or n == "data.npy"):
        return "kspace"
    if (n.startswith("sched_") or n.startswith("samples")) and a.dtype == np.float32 and a.ndim == 3:
        return "traj"
    if a.dtype in (np.float32, np.float64, np.complex64, np.complex128):
        return "image"
    return "lossless"


def policy(p, a, cls):
    if cls == "kspace":
        meta = p.parent / "meta.json"
        sigma = json.loads(meta.read_text()).get("noise_sigma", 0.0) if meta.exists() else 0.0
        ss, n = 0.0, 0
        for o in range(0, a.shape[0], 256):
            x = np.asarray(a[o:o + 256]).view(np.float32).astype(np.float64)
            ss, n = ss + float(np.dot(x.ravel(), x.ravel())), n + x.size
        rms_c = (2 * ss / max(n, 1)) ** 0.5
        return dict(policy=("abs", store.kspace_step(sigma, rms_c)), rows=1), {"sigma": sigma, "rms": rms_c}
    if cls == "traj":
        return dict(policy=("bits", RECORD_BITS), field_chunks=True, delta=True), {}
    if cls == "image":
        return dict(policy="image"), {}
    return dict(policy="lossless"), {}


def check(a, out, cls):
    """max error over the whole array against the policy bound, piece by piece along the stored axis 0 (bounded
    memory: the source is memory-mapped, the .hv read box by box)"""
    from hastycompute import value_format as vf

    tree = vf.open_lazy(out)
    t, order = tree["array"], tree.get("order", "C")
    c = a.T if order == "F" else a  # C-order view of the stored tensor (no copy)
    rec = store.load_meta(out)
    step, mbits = rec.get("step"), rec.get("mantissa_bits")
    row = max(1, int(np.prod(c.shape[1:])) * c.dtype.itemsize)
    per = max(1, (64 << 20) // row)
    worst = 0.0
    for o in range(0, c.shape[0], per):
        x = np.asarray(c[o:o + per])
        y = t[o:o + per]
        if x.dtype.kind not in "fc" or (step is None and mbits is None and cls != "traj"):
            if not np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"):
                return False, "lossless array differs"
            continue
        fl = np.float32 if x.dtype in (np.float32, np.complex64) else np.float64
        xs, ys = x.view(fl).astype(np.float64), y.view(fl).astype(np.float64)
        fin = np.isfinite(xs)
        if not np.array_equal(fin, np.isfinite(ys)):
            return False, "non-finite pattern differs"
        d = np.where(fin, np.abs(xs - ys), 0.0)
        worst = max(worst, float(d.max()))
        if cls == "traj":
            # Bits(B) per field and readout: step = max|field| / (2^(B−1) − 1)
            lim = np.abs(np.where(fin, xs, 0)).max(axis=1, keepdims=True) / ((1 << (RECORD_BITS - 1)) - 1) / 2
        elif mbits is not None:
            lim = np.abs(np.where(fin, xs, 0)) * 0.5 ** (mbits + 1)
        else:
            lim = step / 2
        if np.any(d > lim * (1 + 1e-9) + np.abs(np.where(fin, xs, 0)) * 2.0 ** -23):
            return False, f"error {float(d.max()):.3e} exceeds the bound"
    return True, f"max err {worst:.3e}" + (f" (step/2 {step / 2:.3e})" if step else f" (pointwise ≤ 2^-{mbits + 1})" if mbits else "")


def open_paths():
    """files currently open or memory-mapped by any process of this user (/proc/*/fd, /proc/*/maps)"""
    got = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    got.add(os.path.realpath(f"/proc/{pid}/fd/{fd}"))
                except OSError:
                    pass
            with open(f"/proc/{pid}/maps") as m:
                for line in m:
                    parts = line.split(None, 5)
                    if len(parts) == 6 and parts[5].startswith("/"):
                        got.add(parts[5].strip())
        except (OSError, PermissionError):
            continue
    return got


def files(ps):
    for p in map(Path, ps):
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*.npy") if q.is_file())
        elif p.suffix == ".npy":
            yield p


tot_in = tot_out = 0
links = []
for p in files(paths):
    if p.is_symlink():   # links are re-pointed to the converted target afterwards (relink below)
        links.append(p)
        continue
    if not p.exists():
        continue
    if p.name.endswith(".part.npy") or p.stat().st_size < min_mb * (1 << 20) or any(e in str(p.resolve()) + "/" for e in EXCLUDE):
        continue
    if any(q.name.endswith(".part.npy") for q in p.parent.iterdir()):
        say(f"skip {p}: in-progress simulation in this directory (*.part.npy)")
        continue
    if apply_ and str(p.resolve()) in open_paths():
        say(f"skip {p}: open by a running process")
        continue
    hv = (out_dir / p.name if out_dir else p).with_suffix(".hv")
    if hv.exists():
        print(f"skip {p}: {hv.name} exists")
        continue
    a = np.load(p, mmap_mode="r")
    cls = classify(p, a)
    kw, info = policy(p, a, cls)
    if not apply_:
        print(f"[dry] {p} ({p.stat().st_size / 1e9:.2f} GB) -> {hv.name}: {cls} {kw['policy']} {info}")
        continue
    t0 = time.time()
    chunks = [1, a.shape[1], 1] if kw.pop("field_chunks", False) else None
    out = store.save_array(hv, a, kw["policy"], rows=kw.get("rows"), chunks=chunks, delta=kw.get("delta", False), drop_stale=False)
    ok, msg = check(a, out, cls)
    si, so = p.stat().st_size, out.stat().st_size
    say(f"{'OK  ' if ok else 'FAIL'} {p} -> {out}: {si / 1e9:.3f} -> {so / 1e9:.3f} GB ({so / si:.3f}), {cls}, {msg}, {time.time() - t0:.0f} s")
    if not ok:
        out.unlink()
        continue
    tot_in, tot_out = tot_in + si, tot_out + so
    if delete:
        del a
        os.remove(p)
if apply_:
    say(f"total {tot_in / 1e9:.2f} -> {tot_out / 1e9:.2f} GB")
    # symlinks X.npy -> T.npy whose target was converted: X.hv -> T.hv (relative, as before), drop the stale X.npy link
    for l in links:
        if l.exists():
            continue   # target not converted: the link still works
        th = Path(os.readlink(l)).with_suffix(".hv")
        if (l.parent / th).exists() and not l.with_suffix(".hv").exists():
            os.symlink(th, l.with_suffix(".hv"))
            os.remove(l)
            say(f"relink {l} -> {th}")
