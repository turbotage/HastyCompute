"""Patch dataset for the complex 4D-flow denoiser on RACLETTE tier-1 cases (data/hasty/train/raclette, built by
tools/raclette/build_training.py). Design: research/denoiser_design.md, research/cfd_training_data.md §2.

Fixed 7-point dual-VENC layout [ref, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi] (x_e ∝ m e^{iθ} exp(+i k_e·v), k_e = kv_e ê_axis).
RACLETTE encodes along scanner xyz, ~20° oblique to its voxel grid, so every patch is resampled (cubic B-spline) onto
an isotropic grid whose axes *are* the encoding axes. On that grid the octahedral group O_h acts exactly on the channels:
  - an axis permutation permutes the (lo, hi) pairs;
  - an axis flip (v_a → −v_a) maps x_a → conj(x_a) · x_ref / conj(x_ref), i.e. conj of the reference-relative feature
    c_a = x_a x̄_ref/|x_ref|. Exact when the background phase is uniform inside a voxel; the noise is drawn after the
    transform, so its statistics are exact.

Two stages, so that the resampling can run on the training device:
  Tier1 (DataLoader worker, IO only): picks case, cyclic window of T frames, patch centre (lumen-biased), spacing h,
      g ∈ O_h, gain, SNR, and reads the native block covering the rotated patch.
  Patches (any device): B-spline prefilter (separable FIR) + tricubic evaluation as 8 trilinear fetches
      (Sigg & Hadwiger, GPU Gems 2 ch. 20), O_h action, gain, noise.
Output of Patches, per batch:
  clean, noisy  complex64 [B, 7, T, P, P, P]   target scale ≈ 1 (p99 |x_ref| of the case)
  kv            float32   [B, 7]               |k_e| = π/VENC_e, rad/(m/s) (0 for the reference)
  valid         bool      [B, P, P, P]         inside the acquired FOV (loss only there)
  mask          bool      [B, T, P, P, P]      lumen
  sigma         float32   [B, P, P, P]         noise std per voxel (complex: E|n|² = σ²)
  h, snr        float32   [B]
The network should be fully convolutional (translation-equivariant) and the loss restricted to the patch interior
(`interior`), so patch-trained weights apply unchanged to whole volumes or overlap-tiled inference.
"""
import json
from pathlib import Path

import numpy as np
import _env  # noqa: F401  (caches inside the repo; before torch)
import torch
import torch.nn.functional as Fn

REPO = Path(__file__).resolve().parents[3]
TRAIN = REPO / "data/hasty/train/raclette"
# our own phantom cases (rustlib/tools/phantom/to_training.py: moving atlas phantoms with LBM chamber flow, the same
# layout): joined to RACLETTE when HASTY_TRAIN_PHANTOM is not "0"; their weight in the case draw by --phantom-weight
PHANTOM = REPO / "data/hasty/train/phantom"
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402


class _Case:
    """x [X, Y, Z, F, E, 2] f16 and mask [X, Y, Z, F] of one case: case.hv (zstd, chunked [16,16,16,1,…]: only the
    chunks a patch touches are decoded, by the Rust reader when built) or the legacy npy files"""

    def __init__(self, d: Path):
        if (d / "case.hv").exists():
            t = vf.open_lazy(d / "case.hv")
            self.x, self.mask, self.lazy = t["x"], t["mask"], True
        else:
            self.x, self.mask, self.lazy = np.load(d / "x.npy", mmap_mode="r"), np.load(d / "mask.npy", mmap_mode="r"), False

    @staticmethod
    def _runs(frames):
        """consecutive (cyclic) frame indices -> contiguous [a, b) runs"""
        runs, a = [], frames[0]
        for i in range(1, len(frames) + 1):
            if i == len(frames) or frames[i] != frames[i - 1] + 1:
                runs.append((a, frames[i - 1] + 1))
                if i < len(frames):
                    a = frames[i]
        return runs

    def box(self, arr, lo, hi, frames):
        """arr[lo:hi spatial][:, :, :, frames]"""
        if not self.lazy:
            return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][:, :, :, frames])
        tail = list(arr.shape[4:])
        parts = [arr.read_box([*lo, a] + [0] * len(tail), [*hi, b] + tail) for a, b in self._runs(list(frames))]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=3)

    def blood_level(self, ref, frames=(0, 6, 12, 18)):
        """mean |x_ref| (encode index `ref`) in the lumen over a few frames: the case's signal level for the noise model"""
        acc, n = 0.0, 0
        for f in frames:
            f = f % self.mask.shape[3]
            m = self.mask_frame(f) > 0
            if not m.any():
                continue
            xr = np.asarray(self.x[:, :, :, f, ref, :], np.float32)
            a = np.hypot(xr[..., 0], xr[..., 1])[m]
            acc, n = acc + float(a.sum()), n + a.size
        return acc / n if n else 1.0

    def mask_frame(self, f):
        return self.mask[:, :, :, f] if self.lazy else np.asarray(self.mask[..., f])
GRADE_CLASS = {"Healthy": 0, "Sclerosis": 0, "Mild": 1, "Moderate": 2, "Severe_LowGradient": 3, "Severe_HighGradient": 3, "Critical": 4}
SEVEN_POINT = ["7p_2Venc", "7p_2Venc_Breathing", "7p_2Venc_Super", "phantom_7p"]
MARGIN = 8  # native voxels around the rotated patch: B-spline prefilter edge error ~ |z1|^8 ≈ 3e-5
Z1 = np.sqrt(3.0) - 2.0
PREFILTER = torch.tensor([np.sqrt(3.0) * Z1 ** abs(k) for k in range(-12, 13)], dtype=torch.float32)  # inverse of (1, 4, 1)/6


def layout(enc):
    """indices of RACLETTE encodes in the order [ref, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi] and kv per slot"""
    d = np.array([e["dir_xyz"] for e in enc])
    kv = np.array([e["kv"] for e in enc])
    order = [int(np.flatnonzero(kv == 0)[0])]
    for a in range(3):
        on = np.flatnonzero((np.abs(d[:, a]) > 0.99) & (kv > 0))
        assert len(on) == 2 and np.all(d[on, a] > 0), f"axis {a}: encodes {on}"
        order += sorted(on.tolist(), key=lambda e: -kv[e])  # lo VENC (large kv) first
    return order, kv[order]


class Tier1(torch.utils.data.Dataset):
    def __init__(self, split="train", patch=32, frames=12, h=2.0e-3, length=10000, variants=SEVEN_POINT, snr=(5.0, 60.0),
                 lumen_frac=0.7, oh=True, balance_grades=True, seed=0, offset=0, roots=None, phantom_weight=0.2):
        if roots is None:
            import os

            roots = [TRAIN] + ([PHANTOM] if os.environ.get("HASTY_TRAIN_PHANTOM", "1") != "0" and (PHANTOM / "index.json").exists() else [])
        self.cases = []
        for r in roots:
            idx = json.loads((Path(r) / "index.json").read_text())
            self.cases += [dict(c, path=str(Path(r) / c["path"])) for c in idx["cases"]
                           if c["split"] == split and c["variant"] in variants and c["n_frames"] >= (frames or 1) and c["n_enc"] == 7]
        assert self.cases, f"no cases for split {split}"
        cls = np.array([GRADE_CLASS.get(c["stenosis_grade"], 2) for c in self.cases])
        w = 1.0 / np.bincount(cls)[cls] if balance_grades else np.ones(len(cls))
        # the phantom cases share `phantom_weight` of the draw (few cases, many patches each), RACLETTE the rest
        ph = np.array([c["variant"] == "phantom_7p" for c in self.cases])
        if ph.any() and (~ph).any():
            w = np.where(ph, w / w[ph].sum() * phantom_weight, w / w[~ph].sum() * (1 - phantom_weight))
        self.p = w / w.sum()
        self.P, self.T = int(patch), frames
        self.h = (h, h) if np.isscalar(h) else tuple(h)
        self.length, self.snr, self.lumen_frac, self.oh, self.seed = length, snr, lumen_frac, oh, seed
        self.offset = offset  # global index of sample 0: sample k depends only on (seed, k) — resumable streams
        self._meta = {}
        self._cases = {}
        self._levels = {}

    def __len__(self):
        return self.length

    def meta(self, path):
        if path not in self._meta:
            self._meta[path] = json.loads((TRAIN / path / "case.json").read_text())  # absolute paths: TRAIN / p = p
        return self._meta[path]

    def symmetry(self, rng):
        """g ∈ O_h as (axis permutation, flips): patch axis a' = source axis perm[a'], then flipped"""
        if not self.oh:
            return np.arange(3), np.zeros(3, bool)
        return rng.permutation(3), rng.random(3) < 0.5

    def choose(self, rng, m, mk):
        F, dims = m["n_frames"], np.array(m["dims"])
        T = self.T or F
        frames = (int(rng.integers(F)) + np.arange(T)) % F
        if rng.random() < self.lumen_frac:
            lum = np.argwhere(mk.mask_frame(int(frames[T // 2])) > 0)
            centre = lum[rng.integers(len(lum))] if len(lum) else dims // 2
        else:
            centre = np.array([rng.integers(d) for d in dims])
        h = float(np.exp(rng.uniform(np.log(self.h[0]), np.log(self.h[1]))))
        return frames, centre, h

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i + self.offset])
        c = self.cases[rng.choice(len(self.cases), p=self.p)]
        m = self.meta(c["path"])
        if c["path"] not in self._cases:
            self._cases[c["path"]] = _Case(TRAIN / c["path"])  # absolute path (roots joined above)
            self._levels[c["path"]] = self._cases[c["path"]].blood_level(layout(m["encodings"])[0][0])
        mk = self._cases[c["path"]]
        frames, centre, h = self.choose(rng, m, mk)
        perm, flip = self.symmetry(rng)
        # patch index p -> native fractional index: centre + Rᵀ h (p − (P−1)/2) / s, axes of the patch = xyz = encode axes
        P, R, s, dims = self.P, np.array(m["rot_grid_to_xyz"]), np.array(m["spacing_m"]), np.array(m["dims"])
        A = (R.T * h) / s[:, None]
        corners = np.array([[u, v, w] for u in (0, P - 1) for v in (0, P - 1) for w in (0, P - 1)]) - (P - 1) / 2
        q = centre + corners @ A.T
        lo = np.maximum(np.floor(q.min(0)).astype(int) - MARGIN, 0)
        hi = np.minimum(np.ceil(q.max(0)).astype(int) + MARGIN + 1, dims)
        order, kv = layout(m["encodings"])
        blk = mk.box(mk.x, lo, hi, frames)[:, :, :, :, order]  # [bx,by,bz,T,7,2] f16
        mblk = mk.box(mk.mask, lo, hi, frames)  # [bx,by,bz,T]
        return {
            "block": torch.from_numpy(np.ascontiguousarray(np.transpose(blk, (4, 3, 0, 1, 2, 5)))),  # [7, T, bx, by, bz, 2]
            "mblock": torch.from_numpy(np.ascontiguousarray(np.transpose(mblk, (3, 0, 1, 2)))),  # [T, bx, by, bz]
            "A": torch.from_numpy(A.astype(np.float64)),
            "b": torch.from_numpy((centre - lo - A @ np.full(3, (P - 1) / 2)).astype(np.float64)),
            "vol": torch.from_numpy(np.stack([-lo, dims - 1 - lo]).astype(np.float64)),  # native volume in block coords
            "kv": torch.from_numpy(kv.astype(np.float32)),
            "h": h,
            "perm": torch.from_numpy(perm.copy()),
            "flip": torch.from_numpy(flip.copy()),
            "gain": complex(np.exp(1j * rng.uniform(0, 2 * np.pi)) * np.exp(rng.uniform(np.log(0.5), np.log(2.0)))),
            "snr": float(np.exp(rng.uniform(*np.log(self.snr)))),
            "level": self._levels[c["path"]],  # case blood level (σ = level · gain / SNR: a property of the acquisition, not of the patch)
            "noise_seed": int(rng.integers(2**62)),
            "case": c["path"],
        }


def collate(batch):
    """blocks differ in size: keep lists"""
    return {k: [b[k] for b in batch] for k in batch[0]}


def prefilter(c):
    """cubic B-spline coefficients of samples c [C, n0, n1, n2] (separable FIR, replicate edges)"""
    k = PREFILTER.to(c.device, c.dtype).view(1, 1, -1)
    for a in (1, 2, 3):
        y = c.movedim(a, -1)
        sh = y.shape
        y = Fn.conv1d(Fn.pad(y.reshape(-1, 1, sh[-1]), (12, 12), mode="replicate"), k)
        c = y.reshape(sh).movedim(-1, a)
    return c


def sample(coef, q, cubic=True):
    """coef [C, n0, n1, n2], q [P, P, P, 3] fractional index -> [C, P, P, P]; cubic B-spline = 8 trilinear fetches"""
    n = torch.tensor(coef.shape[1:], device=q.device, dtype=q.dtype)

    def fetch(pos):  # trilinear at fractional index pos [..., 3]
        g = (2 * pos / (n - 1) - 1).flip(-1)  # grid_sample: last dim (x, y, z) = (n2, n1, n0)
        return Fn.grid_sample(coef[None], g[None].to(coef.dtype), mode="bilinear", padding_mode="border", align_corners=True)[0]

    if not cubic:
        return fetch(q)
    i = torch.floor(q)
    t = q - i
    w0, w1 = (1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6
    w2, w3 = (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6
    g = (w0 + w1, w2 + w3)
    hp = (i - 1 + w1 / g[0], i + 1 + w3 / g[1])
    out = 0
    for s in [(a, b_, c) for a in (0, 1) for b_ in (0, 1) for c in (0, 1)]:
        pos = torch.stack([hp[s[d]][..., d] for d in range(3)], -1)
        wt = g[s[0]][..., 0] * g[s[1]][..., 1] * g[s[2]][..., 2]
        out = out + wt.to(coef.dtype) * fetch(pos)
    return out


def flip_pair(x, a):
    """v_a → −v_a on the 7p layout x [7, ...]: conj of the reference-relative feature of both encodes on axis a"""
    ref = x[0]
    nz = ref.abs() > 0
    rot = torch.where(nz, ref / torch.where(nz, ref.conj(), torch.ones_like(ref)), torch.ones_like(ref))
    x = x.clone()
    for e in (1 + 2 * a, 2 + 2 * a):
        x[e] = x[e].conj() * rot
    return x


class Patches:
    """batch of worker outputs -> training tensors on `device`"""

    def __init__(self, P, cubic=True, sigma_profile=0.5, device="cpu"):
        self.P, self.cubic, self.sigma_profile, self.device = P, cubic, sigma_profile, torch.device(device)
        ax = torch.arange(P, dtype=torch.float64)
        self.grid = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).to(self.device)  # [P, P, P, 3]

    def one(self, b, i):
        P, dev = self.P, self.device
        blk = b["block"][i].to(dev).float()  # [7, T, bx, by, bz, 2]
        E, T = blk.shape[:2]
        sh = blk.shape[2:5]
        q = self.grid @ b["A"][i].to(dev).T + b["b"][i].to(dev)  # [P, P, P, 3]
        vol = b["vol"][i].to(dev)
        valid = ((q >= vol[0]) & (q <= vol[1])).all(-1)
        re_im = blk.permute(0, 1, 5, 2, 3, 4).reshape(E * T * 2, *sh)
        coef = prefilter(re_im) if self.cubic else re_im
        v = sample(coef, q, self.cubic).reshape(E, T, 2, P, P, P)
        x = torch.complex(v[:, :, 0], v[:, :, 1]) * valid
        mask = (sample(b["mblock"][i].to(dev).float(), q, cubic=False) > 0.5) & valid
        perm, flip = b["perm"][i].tolist(), b["flip"][i].tolist()
        x = x.permute(0, 1, *(p + 2 for p in perm))
        sel = [0] + [1 + 2 * perm[a] + u for a in range(3) for u in (0, 1)]
        x, kv = x[sel], b["kv"][i][sel]
        mask, valid = mask.permute(0, *(p + 1 for p in perm)), valid.permute(*perm)
        for a in range(3):
            if flip[a]:
                x, mask, valid = x.flip(a + 2), mask.flip(a + 1), valid.flip(a)
                x = flip_pair(x, a)
        x = x * b["gain"][i]
        # noise: blood SNR relative to the case's blood level (not this patch's: air patches would get σ ≈ 0)
        gen = torch.Generator(device=dev).manual_seed(b["noise_seed"][i] % (2**63))
        level = abs(b["gain"][i]) * b["level"][i]
        sigma = torch.full((P, P, P), float(level) / b["snr"][i], device=dev)
        if self.sigma_profile > 0:  # smooth log-normal spatial profile (coil geometry / g-factor-like), periodic Gaussian blur
            g = torch.randn((P, P, P), generator=gen, device=dev)
            f = torch.fft.fftfreq(P, device=dev)
            k2 = f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2
            g = torch.fft.ifftn(torch.fft.fftn(g) * torch.exp(-2 * (np.pi * P / 4) ** 2 * k2)).real
            sigma = sigma * torch.exp(self.sigma_profile * g / (g.std() + 1e-12))
        n = torch.complex(torch.randn(x.shape, generator=gen, device=dev), torch.randn(x.shape, generator=gen, device=dev)) * (sigma / np.sqrt(2))
        return x, x + n * valid, kv.to(dev), valid, mask, sigma

    def __call__(self, b):
        out = [self.one(b, i) for i in range(len(b["block"]))]
        keys = ["clean", "noisy", "kv", "valid", "mask", "sigma"]
        r = {k: torch.stack([o[j] for o in out]) for j, k in enumerate(keys)}
        r["h"] = torch.tensor(b["h"], dtype=torch.float32, device=self.device)
        r["snr"] = torch.tensor(b["snr"], dtype=torch.float32, device=self.device)
        r["case"] = b["case"]
        return r


def interior(P, T, margin=(4, 1)):
    """loss weight: patch interior (spatial margin, temporal margin) — the borders see zero padding / window edges"""
    w = torch.zeros(T, P, P, P, dtype=torch.bool)
    ms, mt = margin
    w[mt:T - mt or None, ms:P - ms, ms:P - ms, ms:P - ms] = True
    return w
