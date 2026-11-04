"""Tier-1 denoiser training cases from RACLETTE (research/cfd_training_data.md §2), streamed: download a subject's Data,
convert, delete the raw download (raw data present before the run is kept).

    python -I build_training.py [--variants 7p_2Venc,13p_2Venc_ICOSA,7p_2Venc_Breathing,7p_2Venc_Super]
                                [--subjects VirtualSubject_n001,..] [--subs bin_00,High,..|all] [--limit N] [--workers 3]
                                [--keep-raw] [--min-free-gb 60] [--index-only]

Writes data/hasty/train/raclette/<variant>/<subject>[/<sub>]/ (sub = breathing bin_XX or Super Low/Mid/High):
  x.npy       float16 C-order [X, Y, Z, F, E, 2]   complex encodes (re, im), divided by case.json "scale"
  mask.npy    uint8   C-order [X, Y, Z, F]         lumen mask per frame (walls move)
  smaps.npy   float16 C-order [X, Y, Z, C, 2]      RACLETTE coil sensitivities (as provided)
  motion_<name>.npy                                breathing variant: point arrays of breathing_motion.pv (cropped)
  case.json   geometry, encodings, frame times, scale, provenance; written last (= case complete)
and index.json (all complete cases, split by background patient).

Conventions (checked on n001 against the ground-truth CFD): x_e ∝ m e^{iθ} exp(+i k_e · v), with k_e and v in the
*grid* basis (voxel axes i, j, k). RACLETTE encodes along scanner xyz; k_grid = Rᵀ k_xyz, R = normalised columns of
ijk_to_xyz (= xyz_to_mps ᵀ, i.e. the grid is the logical MPS frame). hasty-mrsim uses the opposite phase sign.
"""
import glob
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyvista_zstd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch  # noqa: E402

REPO = fetch.REPO
RAW = fetch.OUT
TRAIN = REPO / "data/hasty/train/raclette"
ORDER = ["7p_2Venc", "13p_2Venc_ICOSA", "7p_2Venc_Breathing", "7p_2Venc_Super"]
MARGIN = 4  # voxels kept around the body bounding box
# image sub-cases converted by default (disk): 4 of 12 breathing bins (motion fields of all 12), Super Low + Mid (High is
# 1 mm / 48 frames, ≈ 3.6 GB per subject); --subs a,b,.. or --subs all overrides
DEFAULT_SUBS = {"7p_2Venc_Breathing": {"bin_00", "bin_03", "bin_06", "bin_09"}, "7p_2Venc_Super": {"Low", "Mid"}}
args = sys.argv[1:]
log_lock = threading.Lock()
stop = threading.Event()


def opt(name, default):
    return args[args.index(name) + 1] if name in args else default


def log(msg):
    with log_lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def load(f):
    img = pyvista_zstd.read(str(f))
    d = tuple(img.dimensions)
    return img, d


def complex_field(img, d):
    n = img["real"].shape[1] if img["real"].ndim == 2 else 1
    return (img["real"] + 1j * img["imag"]).astype(np.complex64).reshape(d + ((n,) if n > 1 else ()), order="F")


def fd(img, k):
    return np.asarray(img.field_data[k])


def save_motion(bm, out, d=None, sl=None):
    """point arrays of a breathing_motion.pv (displacement fields), cropped like the case when on its grid"""
    mi, dm = load(bm)
    out.mkdir(parents=True, exist_ok=True)
    motion = {}
    for k in mi.array_names:
        if k in mi.point_data:
            arr = np.asarray(mi.point_data[k])
            arr = arr.reshape(dm + arr.shape[1:], order="F")
            if sl is not None and dm == d:
                arr = arr[sl]
            np.save(out / f"motion_{k}.npy", arr.astype(np.float16))
            motion[k] = list(arr.shape)
    motion["field_data"] = {k: fd(mi, k).tolist() for k in mi.field_data.keys() if fd(mi, k).dtype.kind not in "US"}
    motion["dims"] = list(dm)
    motion["cropped"] = sl is not None and dm == d
    return motion


def subject_meta(v, s, sub):
    sd = RAW / v / s
    name = {"High": "meta_high.json", "Mid": "meta_mid.json", "Low": "meta_low.json"}.get(sub, "meta.json")
    return json.loads((sd / name).read_text())


def convert(v, s, sub, data_dir, out):
    """one case: frames in data_dir (Data or Data/<sub>); coil maps in data_dir or Data/"""
    frames = sorted(data_dir.glob("*_4DFlow_*.pv"))
    img, d = load(frames[0])
    E = img["real"].shape[1]
    F = len(frames)
    x = np.empty(d + (F, E), np.complex64)
    mask = np.empty(d + (F,), np.uint8)
    times = []
    for t, f in enumerate(frames):
        im, dt = load(f)
        assert dt == d, f"{f}: dims {dt} != {d}"
        x[..., t, :] = complex_field(im, d)
        mask[..., t] = (im["mask"].reshape(d, order="F") > 0)
        times.append(float(fd(im, "dtime").ravel()[0]))
    sens_f = data_dir / "sensitivity.pv"
    if not sens_f.exists():
        sens_f = RAW / v / s / "Data" / "sensitivity.pv"
    sens, ds = load(sens_f)
    smaps = complex_field(sens, ds)
    if smaps.ndim == 3:
        smaps = smaps[..., None]
    assert ds == d, f"coil-map grid {ds} != image grid {d}"

    # crop to the body (time-max reference magnitude) + margin
    a0 = np.abs(x[..., 0]).max(axis=-1)
    hi = float(np.percentile(a0, 99))
    body = a0 > 0.02 * hi
    lo_i = [max(int(np.flatnonzero(body.any(axis=tuple(b for b in range(3) if b != q))).min()) - MARGIN, 0) for q in range(3)]
    hi_i = [min(int(np.flatnonzero(body.any(axis=tuple(b for b in range(3) if b != q))).max()) + MARGIN + 1, d[q]) for q in range(3)]
    sl = tuple(slice(lo_i[q], hi_i[q]) for q in range(3))
    x, mask, smaps = x[sl], mask[sl], smaps[sl]
    scale = float(np.percentile(np.abs(x[..., 0])[body[sl][..., None].repeat(F, -1)], 99))

    out.mkdir(parents=True, exist_ok=True)
    x16 = np.empty(x.shape + (2,), np.float16)
    err = ref = 0.0
    for t in range(F):  # frame by frame: Super/High would need ~5 GB for a float32 copy
        xs = np.stack([x[..., t, :].real, x[..., t, :].imag], -1) / scale
        x16[..., t, :, :] = xs
        err += float(np.sum((x16[..., t, :, :].astype(np.float32) - xs) ** 2))
        ref += float(np.sum(xs ** 2))
    rt = (err / ref) ** 0.5
    np.save(out / "x.npy", x16)
    np.save(out / "mask.npy", mask)
    np.save(out / "smaps.npy", np.stack([smaps.real, smaps.imag], -1).astype(np.float16))
    bm = data_dir / "breathing_motion.pv"
    motion = save_motion(bm, out, d, sl) if bm.exists() else {}

    ijk_to_xyz = fd(img, "ijk_to_xyz").reshape(4, 4).astype(np.float64)
    R = ijk_to_xyz[:3, :3] / np.linalg.norm(ijk_to_xyz[:3, :3], axis=0)
    kvs = fd(img, "kvs").ravel().astype(np.float64)
    dirs = fd(img, "enc_dirs").reshape(-1, 3).astype(np.float64)
    vencs = fd(img, "vencs").ravel().astype(np.float64)
    k_xyz = kvs[:, None] * dirs
    meta = subject_meta(v, s, sub)
    msp = fd(img, "ms_parameters").ravel().tolist() if "ms_parameters" in img.field_data.keys() else []
    case = {
        "variant": v,
        "subject": s,
        "sub": sub,
        "group": meta.get("source_background"),
        "geometry": meta.get("source_geometry"),
        "stenosis_grade": meta.get("hemodynamics", {}).get("stenosis_grade"),
        "peak_velocity_mps": meta.get("hemodynamics", {}).get("peak_velocity_mps"),
        "dims": list(x.shape[:3]),
        "n_frames": F,
        "n_enc": E,
        "n_coils": int(smaps.shape[-1]),
        "crop_lo": lo_i,
        "orig_dims": list(d),
        "spacing_m": [float(h) for h in img.spacing],
        "origin_pv": [float(o) for o in img.origin],
        "ijk_to_xyz": ijk_to_xyz.tolist(),
        "rot_grid_to_xyz": R.tolist(),
        "xyz_to_raf": fd(img, "xyz_to_raf").reshape(4, 4).tolist(),
        "xyz_to_mps": fd(img, "xyz_to_mps").reshape(4, 4).tolist(),
        "encodings": [{"venc": float(vencs[e]), "kv": float(kvs[e]), "dir_xyz": dirs[e].tolist(), "k_xyz": k_xyz[e].tolist(), "k_grid": (k_xyz[e] @ R).tolist()} for e in range(E)],
        "phase_convention": "x_e ∝ exp(+i k_e·v), k_grid and v in grid basis, k in rad/(m/s)",
        "frame_times_s": times,
        "scale": scale,
        "fp16_roundtrip_rel": rt,
        "lumen_voxels_per_frame": [int(m) for m in mask.reshape(-1, F).sum(0)],
        "ms_parameters": dict(zip(msp[0::2], msp[1::2])),
        "motion": motion,
        "layout": {"x": "float16 C [X,Y,Z,F,E,2] / scale", "mask": "uint8 C [X,Y,Z,F]", "smaps": "float16 C [X,Y,Z,C,2]"},
        "source": f"RACLETTE {v}/{s}{'/' + sub if sub else ''} (Dirix, Buoso, Kozerke; CC BY-NC 4.0; DOI 10.3929/ethz-c-000799752)",
    }
    (out / "case.json").write_text(json.dumps(case, indent=1))
    return case


def subs_of(v, s, c):
    """sub-cases of a subject: [(sub, remote Data subdir)]"""
    if v == "7p_2Venc_Breathing" or v == "7p_2Venc_Super":
        return sorted(it["name"].rstrip("/").split("/")[-1] for it in c.ls(f"/{v}/{s}/Data", detail=True) if it["type"] == "directory")
    return [""]


def process(v, s, keep_raw, min_free, only_subs):
    if stop.is_set():
        return
    c = fetch.client()
    sel = only_subs if only_subs else DEFAULT_SUBS.get(v, set())
    all_subs = subs_of(v, s, c)
    subs = [u for u in all_subs if "all" in sel or not sel or u in sel or u == ""]
    todo = [u for u in subs if not (TRAIN / v / s / u / "case.json").exists()]
    raw_sd = RAW / v / s / "Data"
    pre_existing = raw_sd.exists()
    if v == "7p_2Venc_Breathing":  # displacement fields of every bin, also of bins whose images are not converted
        for u in all_subs:
            out = TRAIN / v / s / u
            if u in subs or (out / "motion.json").exists() or (out / "case.json").exists():
                continue
            r = f"/{v}/{s}/Data/{u}/breathing_motion.pv"
            n = next(it["content_length"] for it in c.ls(f"/{v}/{s}/Data/{u}", detail=True) if it["name"].endswith("breathing_motion.pv"))
            fetch.get(c, r, raw_sd / u / "breathing_motion.pv", n)
            (out / "motion.json").parent.mkdir(parents=True, exist_ok=True)
            (out / "motion.json").write_text(json.dumps(save_motion(raw_sd / u / "breathing_motion.pv", out), indent=1))
            if not keep_raw and not pre_existing:
                shutil.rmtree(raw_sd / u, ignore_errors=True)
    if not todo:
        if not keep_raw and not pre_existing:
            shutil.rmtree(raw_sd, ignore_errors=True)
        return
    for u in todo:
        if stop.is_set():
            return
        free = shutil.disk_usage(REPO).free / 1e9
        if free < min_free:
            log(f"STOP: {free:.0f} GB free < --min-free-gb {min_free}")
            stop.set()
            return
        remote = f"/{v}/{s}/Data" + (f"/{u}" if u else "")
        fl = fetch.files(c, remote)
        if u:  # top-level files of Data (the breathing bins share one set of coil maps)
            fl += [(it["name"], it["content_length"]) for it in c.ls(f"/{v}/{s}/Data", detail=True) if it["type"] == "file"]
        t0 = time.time()
        nbytes = 0
        for r, n in fl:
            rel = r.split(f"{v}/{s}/Data/", 1)[1]
            nbytes += n if fetch.get(c, r, raw_sd / rel, n) else 0
        t1 = time.time()
        data_dir = raw_sd / u if u else raw_sd
        try:
            case = convert(v, s, u, data_dir, TRAIN / v / s / u)
        except Exception as e:  # keep going; the case stays incomplete (no case.json)
            log(f"FAIL {v}/{s}/{u}: {e!r}")
            continue
        t2 = time.time()
        log(f"{v}/{s}/{u or '-'}: {case['dims']}×{case['n_frames']}f×{case['n_enc']}e, lumen {case['lumen_voxels_per_frame'][0]}, "
            f"rt {case['fp16_roundtrip_rel']:.1e}, dl {nbytes / 1e6:.0f} MB in {t1 - t0:.0f} s, conv {t2 - t1:.0f} s, free {free:.0f} GB")
        if not keep_raw and not pre_existing and u:
            shutil.rmtree(data_dir, ignore_errors=True)
    if not keep_raw and not pre_existing:
        shutil.rmtree(raw_sd, ignore_errors=True)


def build_index():
    cases = []
    for f in sorted(TRAIN.glob("*/*/case.json")) + sorted(TRAIN.glob("*/*/*/case.json")):
        c = json.loads(f.read_text())
        cases.append({k: c[k] for k in ["variant", "subject", "sub", "group", "geometry", "stenosis_grade", "peak_velocity_mps", "dims", "n_frames", "n_enc", "spacing_m"]} | {"path": str(f.parent.relative_to(TRAIN))})
    # split by background patient over *all* variants (local metadata of every subject, not only converted ones)
    groups = set()
    for m in glob.glob(str(RAW / "*" / "*" / "meta*.json")):
        groups.add(json.loads(Path(m).read_text()).get("source_background"))
    groups = sorted(g for g in groups if g)
    held = json.loads((RAW / "7p_2Venc/VirtualSubject_n001/meta.json").read_text())["source_background"]  # operator-validation case
    rng = np.random.default_rng(0)
    rest = [g for g in groups if g != held]
    perm = [rest[i] for i in rng.permutation(len(rest))]
    n_test, n_val = max(round(0.15 * len(groups)) - 1, 1), max(round(0.10 * len(groups)), 1)
    split = {held: "test"} | {g: "test" for g in perm[:n_test]} | {g: "val" for g in perm[n_test:n_test + n_val]} | {g: "train" for g in perm[n_test + n_val:]}
    for c in cases:
        c["split"] = split.get(c["group"], "train")
    cnt = {sp: sum(c["split"] == sp for c in cases) for sp in ["train", "val", "test"]}
    (TRAIN / "index.json").write_text(json.dumps({"groups": split, "counts": cnt, "cases": cases}, indent=1))
    log(f"index: {len(cases)} cases {cnt}, {len(groups)} background groups (test {sorted(g for g, s in split.items() if s == 'test')})")


def main():
    TRAIN.mkdir(parents=True, exist_ok=True)
    if "--index-only" in args:
        build_index()
        return
    variants = opt("--variants", ",".join(ORDER)).split(",")
    only_subj = set(opt("--subjects", "").split(",")) - {""}
    only_subs = set(opt("--subs", "").split(",")) - {""}
    limit = int(opt("--limit", "0"))
    workers = int(opt("--workers", "3"))
    min_free = float(opt("--min-free-gb", "60"))
    keep_raw = "--keep-raw" in args
    c = fetch.client()
    jobs = []
    for v in variants:
        subs = sorted(it["name"].rstrip("/").split("/")[-1] for it in c.ls(f"/{v}", detail=True) if it["type"] == "directory")
        subs = [s for s in subs if not only_subj or s in only_subj]
        jobs += [(v, s) for s in (subs[:limit] if limit else subs)]
    log(f"{len(jobs)} subjects, {workers} workers, raw kept: {keep_raw}, min free {min_free} GB -> {TRAIN}")
    with ThreadPoolExecutor(workers) as ex:
        for fut in [ex.submit(process, v, s, keep_raw, min_free, only_subs) for v, s in jobs]:
            fut.result()
    build_index()


if __name__ == "__main__":
    main()
