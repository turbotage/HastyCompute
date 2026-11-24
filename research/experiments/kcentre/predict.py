"""Physics predictions of the k = 0 signal per coil at the out-centre (τ = 1.600 ms) and in-centre (τ = 9.578 ms) samples.

s_c(τ) = Σ_x S_c(x) W(x) [pw e^{-r_w τ} + pf F(τ) e^{-r_f τ}] e^{-i 2π Δf τ} e^{-i (w·q(x) + m1·v(x))}
(the simulator's convention: signal ∝ e^{-iφ}, φ = k·u + m1·v + w·q + ω τ; fat six-peak F(τ) = Σ α_p e^{-i ω_p τ}).

Variants (voxels: the truth_ratemap_bc mask, 2 mm grid):
  static_pd   : static truth maps (truth_pd_water/fat, truth_r2s, field_ppm_recon), weight W = 1 (PD only), no slab
  static_ss   : the same with the steady-state spoiled-GRE weight W (T1 from the source list, flip 12°, slab profile)
  state       : per breathing × cardiac state (truth_ratemap_bc, 120 states), SS weight, no concomitant / flow phase
  conc / flow / full (readout subset): state + concomitant phase w·q, + flow phase m1·v, + both; w, m1 per readout from
                sched_nominal at samples 0 and 3989 (physical frame = logical, rot_presc = I).
Approximations: W from template positions (not moved); blood steady state (the sim uses the LBM spin history);
voxel-mean velocity on the whole voxel; slab from the small-tip FT of the Hamming sinc; no intra-voxel dephasing.
"""
import json
import sys
import time
from pathlib import Path

import numexpr as ne
import numpy as np

sys.path.insert(0, "python/lib")
from hastycompute import value_format as vf  # noqa: E402
from hastycompute.store import load_any  # noqa: E402

SIM = Path("data/hasty/phantom/factory_arms/ph0001/sim_flow_10min")
SCAN = SIM / "scan"
BC = SCAN / "truth_ratemap_bc"
OUT = Path("data/hasty/phantom/factory_arms/ph0001/kcentre")
TAU = np.array([0.0016, 0.009578])
GBAR = 2.6752218744e8 / (2 * np.pi)
B0 = 3.0
FAT_PPM = np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])
FAT_A = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
FAT_A = FAT_A / FAT_A.sum()
GROUP_STEP = 16  # readout subset: every 16th rotation group, all 7 encodes
NB = 24


def fatF(t):
    return np.sum(FAT_A * np.exp(-1j * 2 * np.pi * GBAR * B0 * FAT_PPM * 1e-6 * t))


def slab_profile(meta):
    """small-tip slab profile α(z) (rad) of the Hamming-windowed sinc, TBW 6, 0.8 ms, BW 7500 Hz, thk 175.5 mm"""
    tbw, dur, thk = 6.0, 0.8e-3, 0.17550909805297832
    flip = np.deg2rad(meta["flip_deg"])
    x = np.linspace(-1, 1, 801)
    b = np.sinc(tbw / 2 * x) * (0.54 + 0.46 * np.cos(np.pi * x))
    t = x * dur / 2
    hz_per_m = tbw / dur / thk
    zz = np.linspace(-0.2, 0.2, 2001)
    f = hz_per_m * zz
    p = np.abs(np.exp(-1j * 2 * np.pi * np.outer(f, t)) @ b) / np.abs(b.sum())
    return zz, flip * p, hz_per_m


def ss(alpha, e1):
    return np.sin(alpha) * (1 - e1) / np.maximum(1 - np.cos(alpha) * e1, 1e-12)


def main():
    t00 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((SCAN / "meta.json").read_text())
    dims = np.array(meta["recon_dims"])
    res = meta["res"][0]
    center = np.array(meta["presc_center_subject"])
    iso = np.array(meta["iso_subject"])
    tr = meta["tr"]
    mask = np.load(BC / "mask.npy")
    lin = np.flatnonzero(mask.ravel(order="F"))
    nv = lin.size
    ijk = np.stack(np.unravel_index(lin, tuple(dims), order="F"), 1)
    xs = (ijk - dims // 2) * res + center - iso  # position relative to isocentre (m)
    q = np.stack([xs[:, 2] ** 2, xs[:, 0] ** 2 + xs[:, 1] ** 2, xs[:, 0] * xs[:, 2], xs[:, 1] * xs[:, 2]], 1).astype(np.float32)
    print(f"mask voxels {nv}", flush=True)

    # coil maps at the mask voxels
    sm = load_any(SCAN / "smaps_true.hv")
    S = np.ascontiguousarray(sm.reshape(-1, sm.shape[-1], order="F")[lin]).astype(np.complex64)
    del sm
    nc = S.shape[1]
    print(f"smaps {S.shape} {time.time() - t00:.0f} s", flush=True)

    # T1 weight from the source list (template positions): pd-weighted mean E1 per voxel, water / fat separately
    tree = vf.open_lazy(SIM / "sim_input.hv")
    e1w = np.zeros(int(np.prod(dims)))
    e1f = np.zeros_like(e1w)
    pw_ = np.zeros_like(e1w)
    pf_ = np.zeros_like(e1w)
    srcs = [tree["tissue"], tree["blood"][0]]
    for sdct in srcs:
        pos = sdct["pos"].read().astype(np.float64)
        pd = sdct["pd"].read().astype(np.float64)
        e1 = np.exp(-tr / np.maximum(sdct["t1"].read().astype(np.float64), 1e-4))
        fat = sdct["shift_ppm"].read() != 0 if "shift_ppm" in sdct else np.zeros(pd.size, bool)
        idx = np.rint((pos - center) / res + dims // 2).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < dims), 1)
        li = idx[:, 0] + dims[0] * (idx[:, 1] + dims[1] * idx[:, 2])
        for sel, ea, pa in ((ok & ~fat, e1w, pw_), (ok & fat, e1f, pf_)):
            ea += np.bincount(li[sel], pd[sel] * e1[sel], ea.size)
            pa += np.bincount(li[sel], pd[sel], pa.size)
        del pos, pd, e1, fat, idx, li
    e1w = np.where(pw_ > 0, e1w / np.maximum(pw_, 1e-30), np.nan)[lin]
    e1f = np.where(pf_ > 0, e1f / np.maximum(pf_, 1e-30), np.nan)[lin]
    e1w = np.where(np.isfinite(e1w), e1w, np.nanmedian(e1w))
    e1f = np.where(np.isfinite(e1f), e1f, np.nanmedian(e1f))
    del pw_, pf_
    zz, prof, hz_per_m = slab_profile(meta)
    slab_dz = xs[:, 2] - (np.array(tree["presc"]["slab"]["center"])[2] - iso[2])
    print(f"E1 water median {np.median(e1w):.4f} (T1 {-tr / np.log(np.median(e1w)) * 1e3:.0f} ms), fat {np.median(e1f):.4f} "
          f"(T1 {-tr / np.log(np.median(e1f)) * 1e3:.0f} ms); {time.time() - t00:.0f} s", flush=True)

    def weights(om_hz):
        """steady-state weights (water, fat) with the slab displaced by the off-resonance (fat also by its shift)"""
        zw = slab_dz + om_hz / hz_per_m
        zf = slab_dz + (om_hz + GBAR * B0 * -3.4e-6) / hz_per_m
        aw, af = np.interp(zw, zz, prof, 0, 0), np.interp(zf, zz, prof, 0, 0)
        return ss(aw, e1w), ss(af, e1f)

    F = np.array([fatF(t) for t in TAU])

    def base(pw, pf, rw, rf, om, Ww, Wf):
        """[2, nv] complex64 voxel signal before coil weighting at the two τ"""
        out = np.empty((2, nv), np.complex64)
        for j, t in enumerate(TAU):
            out[j] = (Ww * pw * np.exp(-rw * t) + Wf * pf * F[j] * np.exp(-rf * t)) * np.exp(-1j * 2 * np.pi * om * t)
        return out

    def clean(a):
        a = np.asarray(a, np.float64)
        return np.where(np.isfinite(a), a, 0.0)

    csum = (np.abs(S) ** 2).sum(1)  # coil-weight per voxel

    def truth_stats(pw, pf, rw, rf, om, Ww, Wf):
        """signal-weighted (Σ_c |S_c|² × |m W| at τ_out) mean decay rate and off-resonance"""
        aw = Ww * pw * np.exp(-rw * TAU[0]) * csum
        af = Wf * pf * np.exp(-rf * TAU[0]) * csum
        tot = (aw + af).sum()
        return np.array([(aw * rw + af * rf).sum() / tot, ((aw + af) * om).sum() / tot, af.sum() / tot])

    res_out = {}
    # static truth maps
    tm = lambda n: clean(np.load(SCAN / n, mmap_mode="r").ravel(order="F")[lin])  # noqa: E731
    pw, pf, r2, fld = tm("truth_pd_water.npy"), tm("truth_pd_fat.npy"), tm("truth_r2s.npy"), tm("field_ppm_recon.npy") * GBAR * B0 * 1e-6
    one = np.ones(nv)
    pall = np.nansum(np.load(SCAN / "truth_pd_water.npy")) + np.nansum(np.load(SCAN / "truth_pd_fat.npy"))
    print(f"truth PD inside the ratemap mask: {(pw.sum() + pf.sum()) / pall:.4f}", flush=True)
    for name, (Ww, Wf) in (("static_pd", (one, one)), ("static_ss", weights(fld))):
        b = base(pw, pf, r2, r2, fld, Ww, Wf)
        res_out[name] = (b @ S).T  # [nc, 2]
        b = base(pw, 0 * pf, r2, r2, fld, Ww, Wf)
        res_out[name + "_water"] = (b @ S).T
        res_out[name + "_truth"] = truth_stats(pw, pf, r2, r2, fld, Ww, Wf)
    print(f"static done {time.time() - t00:.0f} s", flush=True)

    # per-state
    K, Q = 6, 20
    rstates = np.array(json.loads((BC / "meta.json").read_text())["states"]["r"])
    mm = {n: np.load(BC / f"{n}.npy", mmap_mode="r") for n in ("pd_water", "pd_fat", "r_water", "r_fat", "omega_hz")}
    ro = np.load(SCAN / "readouts.npy")
    resp = np.load(SCAN / "resp.npy")
    kst = np.abs(resp[:, :1] - rstates[None]).argmin(1)
    qst = np.rint(ro[:, 4] * Q).astype(int) % Q
    state = qst * K + kst
    nro = ro.shape[0]
    groups = np.arange(0, nro // 7, GROUP_STEP)
    sub = (groups[:, None] * 7 + np.arange(7)[None]).ravel()
    sched = load_any(SCAN / "sched_nominal.hv", lazy=True)
    par = np.zeros((sub.size, 2, 7), np.float32)  # [m1 (3), w (4)] at samples 0 and 3989
    for n, i in enumerate(sub):
        a = sched[int(i)]
        par[n, 0] = np.r_[a[0, 3:6], a[0, 9:13]]
        par[n, 1] = np.r_[a[-1, 3:6], a[-1, 9:13]]
    print(f"subset {sub.size} readouts, sched read {time.time() - t00:.0f} s", flush=True)
    vel = load_any(SCAN / "truth_vel_logical.hv", lazy=True)  # (254, 144, 114, 3, 20)

    P0 = np.zeros((K * Q, nc, 2), np.complex64)
    Pc = np.zeros((sub.size, nc, 2), np.complex64)
    Pv = np.zeros_like(Pc)
    Pf = np.zeros_like(Pc)
    tstat = np.zeros((K * Q, 3))
    sub_state = state[sub]
    for qq in range(Q):
        v = np.asarray(vel[..., qq]).reshape(-1, 3, order="F")[lin].astype(np.float32)
        vs = np.flatnonzero(np.abs(v).max(1) > 1e-3)
        Vv, Sv = v[vs], S[vs]
        for kk in range(K):
            s = qq * K + kk
            pw, pf, rw, rf, om = (clean(mm[n][s]) for n in ("pd_water", "pd_fat", "r_water", "r_fat", "omega_hz"))
            Ww, Wf = weights(om)
            b = base(pw, pf, rw, rf, om, Ww, Wf)
            P0[s] = (b @ S).T
            tstat[s] = truth_stats(pw, pf, rw, rf, om, Ww, Wf)
            act = np.flatnonzero(np.abs(b[0]) > 0)
            ba, qa = b[:, act], q[act]
            Sa = S[act]
            pos_in_act = np.searchsorted(act, vs)
            okv = (pos_in_act < act.size) & (act[np.minimum(pos_in_act, act.size - 1)] == vs)
            vs2, pv = vs[okv], pos_in_act[okv]
            Vv2, Sv2 = Vv[okv], Sv[okv]
            rows = np.flatnonzero(sub_state == s)
            for r0 in range(0, rows.size, NB):
                rr = rows[r0:r0 + NB]
                for j in range(2):
                    w = par[rr, j, 3:]  # [nb, 4]
                    m1 = par[rr, j, :3]
                    psic = w @ qa.T  # [nb, na]
                    bj = ba[j][None]
                    E = ne.evaluate("bj * exp(-1j * psic)").astype(np.complex64)
                    Pc[rr, :, j] = E @ Sa
                    dv = np.exp(-1j * (m1 @ Vv2.T)) - 1  # [nb, nvs]
                    Pv[rr, :, j] = P0[s][:, j][None] + (ba[j][pv][None] * dv) @ Sv2
                    Pf[rr, :, j] = Pc[rr, :, j] + (E[:, pv] * dv) @ Sv2
            del ba, Sa, qa
        print(f"cardiac frame {qq} done {time.time() - t00:.0f} s", flush=True)
    np.savez(OUT / "pred.npz", P0=P0, tstat=tstat, sub=sub, Pconc=Pc, Pflow=Pv, Pfull=Pf, par=par, state=state,
             **{k: np.asarray(v) for k, v in res_out.items()}, tau=TAU)
    print("done", time.time() - t00, flush=True)


if __name__ == "__main__":
    main()
