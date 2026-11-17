"""Storage audit of data/hasty under the I/O precision policy (research/io_policy.md). Read-only.

    rustlib/.venv/bin/python -I research/experiments/io_policy/audit.py [--out rustlib/target/iopolicy] [--sample-mb 384]

Every regular file ≥ 1 MB is classified (k-space, per-sample records, image/volume, integer, existing .hv tree,
checkpoint, LBM beat library, other); every file ≥ MEASURE_MIN in a measured class is sampled with
rustlib/target/release/examples/ioaudit (chunks spread evenly over the file, real codec + policy, decoded and compared),
smaller ones get the size-weighted ratio of their class. Hard links count once; sizes are on-disk (st_blocks).
Writes audit.jsonl (per measured file) and audit.md (tables) to --out.
"""
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data/hasty"
TOOL = REPO / "rustlib/target/release/examples/ioaudit"
MEASURE_MIN = 64 << 20

args = sys.argv[1:]
out = Path(args[args.index("--out") + 1]) if "--out" in args else REPO / "rustlib/target/iopolicy"
smb = args[args.index("--sample-mb") + 1] if "--sample-mb" in args else "384"
out.mkdir(parents=True, exist_ok=True)


def npy_info(p):
    with open(p, "rb") as f:
        pre = f.read(12)
        if pre[:6] != b"\x93NUMPY":
            return None
        hl, off = (int.from_bytes(pre[8:10], "little"), 10) if pre[6] == 1 else (int.from_bytes(pre[8:12], "little"), 12)
        f.seek(off)
        h = f.read(hl).decode("latin1")
    descr = h.split("'descr':")[1].split("'")[1]
    shape = tuple(int(x) for x in h.split("'shape':")[1].split("(")[1].split(")")[0].split(",") if x.strip())
    return descr, shape


def classify(p: Path):
    n, ext = p.name, p.suffix
    if ext == ".npy":
        info = npy_info(p)
        if info is None:
            return "other", None
        descr, shape = info
        if descr[-2:] in ("c8",) and len(shape) == 3 and (n.startswith("data_clean") or n == "data.npy"):
            return "kspace", "kspace"
        if (n.startswith("sched_") or n.startswith("samples")) and descr.endswith("f4") and len(shape) == 3:
            return "traj", "traj"
        if descr[-2:] in ("f4", "f8", "c8") or descr.endswith("c16"):
            return "image", "image"
        return "integer/f16 npy", "lossless"
    if ext == ".hv":
        return ("checkpoint (.hv)", "checkpoint") if n.startswith("state_") else ("hv tree", "hvtree")
    if ext == ".hsk":
        return "checkpoint (.hsk raw)", "raw"
    if ext == ".bin" and p.parent.name == "beats":
        return "LBM beat library", "beats"
    if ext == ".npz":
        return "npz (numpy zip)", None
    return "not ours (models, pdf, meshes, ...)", None


files = []
seen = set()
for root, _, names in os.walk(DATA):
    for nm in names:
        p = Path(root) / nm
        try:
            st = p.lstat()
        except OSError:
            continue
        if not p.is_file() or p.is_symlink() or st.st_size < (1 << 20) or (st.st_dev, st.st_ino) in seen:
            continue
        seen.add((st.st_dev, st.st_ino))
        cls, tool = classify(p)
        files.append({"path": str(p.relative_to(REPO)), "class": cls, "tool": tool, "size": st.st_size, "disk": st.st_blocks * 512, "sparse": st.st_blocks * 512 < 0.9 * st.st_size})

done = {}
jl = out / "audit.jsonl"
if jl.exists():
    for line in jl.read_text().splitlines():
        j = json.loads(line)
        done[j["file"]] = j
with open(jl, "a") as fo:
    for f in sorted(files, key=lambda f: -f["size"]):
        if f["tool"] is None or f["size"] < MEASURE_MIN or f["sparse"] or f["path"] in done:
            continue
        r = subprocess.run(["nice", "-n", "10", "systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=16G", "-p", "MemorySwapMax=0", "env", "HASTY_IO_THREADS=10", str(TOOL), f["tool"], f["path"], "--sample-mb", smb], cwd=REPO, capture_output=True, text=True,
                           env={**os.environ, "HASTY_IO_THREADS": "10"})
        if r.returncode != 0:
            print("FAILED", f["path"], r.stderr[-400:], file=sys.stderr)
            continue
        j = json.loads(r.stdout)
        done[j["file"]] = j
        fo.write(json.dumps(j) + "\n")
        fo.flush()
        print(f"{f['size'] / 1e9:8.2f} GB  ratio {j['result']['ratio']:.3f}  {f['path']}", file=sys.stderr)

# per class: measured files (ratio from the file), the rest at the class's size-weighted ratio
cls = defaultdict(lambda: {"n": 0, "disk": 0, "proj": 0.0, "meas_disk": 0, "meas_proj": 0.0, "maxrel": 0.0, "rmsrel": 0.0, "unmeasured": 0})
for f in files:
    c = cls[f["class"]]
    c["n"] += 1
    c["disk"] += f["disk"]
    j = done.get(f["path"])
    if j is not None:
        pr = f["disk"] * j["result"]["ratio"]
        f["proj"] = pr
        c["meas_disk"] += f["disk"]
        c["meas_proj"] += pr
        c["maxrel"] = max(c["maxrel"], j["result"].get("max_err_rel_rms", 0.0) or 0.0)
        c["rmsrel"] = max(c["rmsrel"], j["result"].get("rms_err_rel", 0.0) or 0.0)
for k, c in cls.items():
    ratio = c["meas_proj"] / c["meas_disk"] if c["meas_disk"] else None
    c["ratio"] = ratio
    for f in files:
        if f["class"] == k and "proj" not in f:
            f["proj"] = f["disk"] * (ratio if ratio is not None else 1.0)
            c["unmeasured"] += f["disk"]
    c["proj"] = sum(f["proj"] for f in files if f["class"] == k)

tot_disk = sum(c["disk"] for c in cls.values())
tot_proj = sum(c["proj"] for c in cls.values())
lines = ["| class | files | on disk (GB) | measured (GB) | ratio | projected (GB) | saved (GB) | worst rel err (max/rms) | rms rel err |", "|---|---|---|---|---|---|---|---|---|"]
for k, c in sorted(cls.items(), key=lambda kv: -kv[1]["disk"]):
    r = f"{c['ratio']:.3f}" if c["ratio"] is not None else "—"
    lines.append(f"| {k} | {c['n']} | {c['disk'] / 1e9:.2f} | {c['meas_disk'] / 1e9:.2f} | {r} | {c['proj'] / 1e9:.2f} | {(c['disk'] - c['proj']) / 1e9:.2f} | {c['maxrel']:.2e} | {c['rmsrel']:.2e} |")
lines.append(f"| **total** | {len(files)} | **{tot_disk / 1e9:.2f}** | | **{tot_proj / tot_disk:.3f}** | **{tot_proj / 1e9:.2f}** | **{(tot_disk - tot_proj) / 1e9:.2f}** | | |")
lines += ["", "| file | class | on disk (GB) | ratio | projected (GB) | max err / rms | rms err / rms | enc / dec MB/s |", "|---|---|---|---|---|---|---|---|"]
for f in sorted(files, key=lambda f: -f["disk"]):
    j = done.get(f["path"])
    if j is None or f["disk"] < (512 << 20):
        continue
    r = j["result"]
    tp = f"{r.get('enc_MBps', 0):.0f} / {r.get('dec_MBps', 0):.0f}" if "dec_MBps" in r else "—"
    lines.append(f"| {f['path'].replace('data/hasty/phantom/', '')} | {f['class']} | {f['disk'] / 1e9:.2f} | {r['ratio']:.3f} | {f['proj'] / 1e9:.2f} | {r.get('max_err_rel_rms', 0):.2e} | {r.get('rms_err_rel', 0):.2e} | {tp} |")
(out / "audit.md").write_text("\n".join(lines) + "\n")
(out / "audit_files.json").write_text(json.dumps(files))
print("\n".join(lines[: len(cls) + 3]))
