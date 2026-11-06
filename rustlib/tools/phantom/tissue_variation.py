"""Per-phantom tissue appearance: jittered MR property table, contrast state, and resolution-free textures.

Textures are random band-limited Fourier fields in the atlas (reference) frame, so they move with the tissue and can be
evaluated at any point / grid: per tissue class t, field_t(X) = Σ_m A_m cos(k_m · X + φ_m) with M modes, wavelengths
log-uniform in [λ_min, λ_max] of the class, isotropic directions, A_m = σ_t √(2 / M) (standard deviation σ_t). Two
independent fields per class: `a` modulates proton density, `b` modulates T2* (and, weaker, T1) — see `apply`.
The modes are generated deterministically from (texture seed, tissue index), so a phantom only stores the seed and the
per-class (σ_a, σ_b, λ range).
"""
import numpy as np

# per class: (σ_a of PD, σ_b of T2*, λ_min mm, λ_max mm) — base values, scaled per phantom by a global gain
TEXTURE = {
    "fat": (0.03, 0.05, 3, 20), "skin": (0.04, 0.05, 1, 4), "muscle": (0.05, 0.08, 1, 10), "bone_cortical": (0.10, 0.10, 2, 10),
    "bone_marrow": (0.10, 0.15, 1, 6), "cartilage": (0.04, 0.05, 2, 8), "liver": (0.06, 0.10, 2, 12), "gallbladder": (0.02, 0.02, 4, 20),
    "spleen": (0.05, 0.08, 2, 10), "thymus": (0.08, 0.10, 2, 8), "stomach_wall": (0.06, 0.08, 1, 6), "gi_content": (0.15, 0.20, 2, 15),
    "pancreas": (0.08, 0.08, 1, 6), "bowel_wall": (0.06, 0.08, 1, 6), "thyroid": (0.05, 0.05, 1, 5), "lung": (0.15, 0.20, 1, 8),
    "myocardium": (0.04, 0.06, 2, 8), "valve": (0.03, 0.03, 1, 4), "vessel_wall": (0.03, 0.03, 1, 4), "kidney": (0.06, 0.08, 2, 8),
}
BLOOD = ("blood_LA", "blood_LV", "blood_RA", "blood_RV", "blood_artery", "blood_vein", "blood_pulm_artery", "blood_pulm_vein")
ARTERIAL = ("blood_LA", "blood_LV", "blood_artery", "blood_pulm_vein")
GD_UPTAKE = ("skin", "muscle", "liver", "spleen", "thymus", "stomach_wall", "pancreas", "bowel_wall", "thyroid", "myocardium", "kidney", "vessel_wall", "valve")


def sample_tissue(rng, table):
    """per-phantom tissue parameters: jittered table rows, contrast state, texture gain and seed"""
    cols = table["columns"]
    rows = {r[0]: dict(zip(cols, r)) for r in table["tissues"]}
    contrast = str(rng.choice(["native", "gadolinium", "ferumoxytol"], p=[0.4, 0.35, 0.25]))
    out = {}
    for n, r in rows.items():
        r = dict(r)
        if r["pd"] > 0:
            r["pd"] *= float(np.exp(rng.normal(0, 0.05)))
            r["t1_s"] *= float(np.exp(rng.normal(0, 0.08)))
            r["t2_s"] *= float(np.exp(rng.normal(0, 0.10)))
            r["t2s_s"] = min(r["t2_s"], r["t2s_s"] * float(np.exp(rng.normal(0, 0.10))))
            if r["ff"] > 0:
                r["ff"] = float(np.clip(r["ff"] + rng.normal(0, 0.05), 0, 0.95))
        out[n] = r
    if contrast == "gadolinium":
        f = float(rng.uniform(0.45, 0.7))
        for n in GD_UPTAKE:
            if n in out:
                out[n]["t1_s"] *= f
        tb = float(rng.uniform(0.25, 0.45))
        for n in BLOOD:
            out[n]["t1_s"], out[n]["t2_s"] = tb, out[n]["t2_s"] * 0.8
    elif contrast == "ferumoxytol":
        tb, t2s = float(rng.uniform(0.10, 0.20)), float(rng.uniform(0.012, 0.030))
        for n in BLOOD:
            out[n]["t1_s"], out[n]["t2_s"], out[n]["t2s_s"] = tb, min(out[n]["t2_s"], 0.06), t2s
    gain = float(rng.uniform(0.5, 1.5))
    tex = {n: [v[0] * gain, v[1] * gain, v[2], v[3]] for n, v in TEXTURE.items()}
    return {"contrast": contrast, "texture_gain": gain, "texture_seed": int(rng.integers(1 << 31)), "texture": tex, "table": out}


def modes(seed, tissue_index, sigma, lam_mm, M=48, which=0):
    """k vectors [M, 3] (rad/m), amplitudes [M], phases [M] of one class field (deterministic)"""
    rng = np.random.default_rng([int(seed), int(tissue_index), int(which)])
    lam = np.exp(rng.uniform(np.log(lam_mm[0]), np.log(lam_mm[1]), M)) * 1e-3
    d = rng.normal(size=(M, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    k = d * (2 * np.pi / lam)[:, None]
    return k, np.full(M, sigma * np.sqrt(2.0 / M)), rng.uniform(0, 2 * np.pi, M)


def apply(table_row, tex_a, tex_b):
    """effective PD, T1, T2* for texture values (vectorised): PD (1 + a), T2* (1 + b), T1 (1 + b / 3), clipped"""
    pd = table_row["pd"] * np.clip(1 + tex_a, 0.2, 2.0)
    t2s = table_row["t2s_s"] * np.clip(1 + tex_b, 0.2, 2.0)
    t1 = table_row["t1_s"] * np.clip(1 + tex_b / 3, 0.5, 1.5)
    return pd, t1, t2s
