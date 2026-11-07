"""Figures of a final sequence design: the chosen 4D-flow TR (`seqdesign` run, e.g. data/hasty/seq/flow_thorax) or the
B0 / R2* / fat train (`b0seq` run, data/hasty/seq/b0_multiecho). Both write final.json + the same arrays.
    rustlib/.venv/bin/python -I rustlib/tools/seq/plot_final.py <run>
Writes <run>/figs/final_*.png:
  01_gradients   one TR per variant (encode / TE shift): G_x, G_y, G_z, |G| on physical axes, TR parts shaded
  02_slew        the same TRs: per-axis slew and |dG/dt| against S_max and the PNS-derated cap of each part
  03_sequence    consecutive TRs of the played sequence: G, |dG/dt|, SAFE PNS per axis and combined vs the limit
  04_pns_tr      one TR of that window (steady state) in detail
  05_moments     |k|/kmax and the first moment as velocity phase γM1 (rad per m/s) per axis, from the RF centre
  06_spokes_3d   readouts in 3D k-space (rotated spokes / lobes) with the kmax sphere, two views + |k|(t)
  07_girf        nominal vs played (illustrative GIRF) gradients and the accumulated k error
  08_spectrum    amplitude spectrum of the sequence window per axis (where the GIRF / mechanical resonances act)
  09_echoes      (B0 train) echo times of both TE shifts, ADC time per echo, velocity phase per echo
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

AX = ["#2a78d6", "#eb6834", "#1baf7a"]  # x, y, z (fixed order)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
PART_FILL = {1: "#f3ecdc", 2: "#e4eefa", 3: "#ececea"}  # encode, readout (ADC), tail / spoiler
GAMMA = 2.6752218744e8
GBAR = GAMMA / (2 * np.pi)


def style(ax, xlabel=None, ylabel=None):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)


def shade(ax, t, part, names):
    """background bands per TR part (codes 1 encode, 2 readout, 3 tail)"""
    if part is None:
        return
    i = 0
    n = len(part)
    while i < n:
        j = i
        while j + 1 < n and part[j + 1] == part[i]:
            j += 1
        c = int(part[i])
        if c in PART_FILL:
            ax.axvspan(t[i], t[min(j + 1, n - 1)], color=PART_FILL[c], lw=0, zorder=0)
        i = j + 1


def part_legend(names):
    return [Patch(color=PART_FILL[c], label=names[c]) for c in (1, 2, 3) if c < len(names) and names[c] != "-"]


def main():
    run = Path(sys.argv[1])
    F = json.loads((run / "final.json").read_text())
    figs = run / "figs"
    figs.mkdir(exist_ok=True)
    dt, kmax, gmax, smax = F["dt"], F["kmax"], F["gmax"], F["smax"]
    names = F["parts"]
    var = F["variants"]
    sg = np.load(run / "seq_grad.npy")
    sp = np.load(run / "seq_part.npy")
    pns = np.load(run / "seq_pns.npy")
    trg = np.load(run / "tr_grad.npy")
    trm = np.load(run / "tr_moments.npy")
    ks = np.load(run / "k_spokes.npy")
    gi = np.load(run / "girf.npy")
    nv, L = trg.shape[0], trg.shape[1]
    t_tr = np.arange(L) * dt * 1e3
    is_b0 = F["kind"] == "b0_multiecho"
    title = F["title"]
    sf = F["slew_factors"]
    caps = {2: sf.get("readout"), 3: sf.get("tail", sf.get("spoiler")), 1: sf.get("encode")}

    def tr_part(v):
        o = v * (L - 1) if is_b0 else 0
        return sp[o:o + L]

    def te_marks(v):
        if is_b0:
            sh = F["te_shift"][v % len(F["te_shift"])]
            return [(F["t_rf"] + t + sh) * 1e3 for t in F["te"]]
        return [(F["t_rf"] + t) * 1e3 for t in F["te"]]

    # ── 01 gradients ──
    fig, axs = plt.subplots(nv, 1, figsize=(13, 1.9 * nv + 1.2), sharex=True, squeeze=False)
    for v in range(nv):
        ax = axs[v, 0]
        style(ax, ylabel=f"{var[v]}\nmT/m")
        shade(ax, t_tr, tr_part(v), names)
        for a in range(3):
            ax.plot(t_tr, trg[v, :, a] * 1e3, color=AX[a], lw=1.1, label="xyz"[a])
        ax.plot(t_tr, np.linalg.norm(trg[v], axis=1) * 1e3, color=INK, lw=0.9, label="|G|")
        ax.axvline(F["t_rf"] * 1e3, color=MUTED, lw=0.8, ls=":")
        for te in te_marks(v)[: (1 if is_b0 else None)]:
            ax.axvline(te, color=INK, lw=0.8, ls="--")
    axs[0, 0].legend(handles=axs[0, 0].get_lines()[:4] + part_legend(names), fontsize=7.5, ncol=8, frameon=False, loc="upper right")
    axs[-1, 0].set_xlabel("ms from TR start (dotted: RF centre, dashed: first echo)", color=INK, fontsize=9)
    fig.suptitle(f"{title}\nphysical-axis gradients, one TR per {'TE shift' if is_b0 else 'encode'} (first rotation)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "final_01_gradients.png", dpi=110)
    plt.close(fig)

    # ── 02 slew ──
    fig, axs = plt.subplots(nv, 1, figsize=(13, 1.9 * nv + 1.2), sharex=True, squeeze=False)
    tm = (t_tr[1:] + t_tr[:-1]) / 2
    for v in range(nv):
        ax = axs[v, 0]
        style(ax, ylabel=f"{var[v]}\nT/m/s")
        shade(ax, t_tr, tr_part(v), names)
        s = np.diff(trg[v], axis=0) / dt
        for a in range(3):
            ax.plot(tm, s[:, a], color=AX[a], lw=0.9, label=f"dG{'xyz'[a]}/dt")
        ax.plot(tm, np.linalg.norm(s, axis=1), color=INK, lw=0.9, label="|dG/dt|")
        ax.axhline(smax, color=INK, lw=0.8, ls="--")
        ax.axhline(-smax, color=INK, lw=0.8, ls="--")
        pp = tr_part(v)
        for c, f in caps.items():
            if f is None:
                continue
            m = pp[:-1] == c
            if m.any():
                ax.plot(np.where(m, tm, np.nan), np.full_like(tm, f * smax), color=MUTED, lw=1.4, ls=(0, (2, 2)))
        ax.set_ylim(-1.1 * smax, 1.1 * smax)
    axs[0, 0].legend(fontsize=7.5, ncol=4, frameon=False, loc="lower right")
    axs[-1, 0].set_xlabel("ms from TR start (dashed: S_max per axis; dotted grey: PNS-derated vector cap of the part)", color=INK, fontsize=9)
    fig.suptitle(f"{title}\nslew rates (derating: {', '.join(f'{k} {v:.2f}' for k, v in sf.items() if v is not None)} × S_max)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "final_02_slew.png", dpi=110)
    plt.close(fig)

    # ── 03 sequence window + 04 one TR ──
    def seq_panels(i0, i1, fname, label):
        t = np.arange(len(sg))[i0:i1] * dt * 1e3
        fig, axs = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True)
        style(axs[0], ylabel="G (mT/m)")
        style(axs[1], ylabel="|dG/dt| (T/m/s)")
        style(axs[2], xlabel="ms", ylabel="SAFE PNS (% of threshold)")
        for ax in axs:
            shade(ax, t, sp[i0:i1], names)
        for a in range(3):
            axs[0].plot(t, sg[i0:i1, a] * 1e3, color=AX[a], lw=0.8, label="xyz"[a])
        s = np.linalg.norm(np.diff(sg, axis=0) / dt, axis=1)
        axs[1].plot(t[:-1], s[i0:i1 - 1], color=INK, lw=0.7)
        axs[1].axhline(smax, color=INK, lw=0.8, ls="--")
        j1 = min(i1, len(pns))
        for a in range(3):
            axs[2].plot(t[: j1 - i0], pns[i0:j1, a], color=AX[a], lw=0.8, label="xyz"[a])
        axs[2].plot(t[: j1 - i0], pns[i0:j1, 3], color=INK, lw=1.0, label="combined")
        axs[2].axhline(F["pns_limit"], color=INK, lw=0.9, ls="--", label=f"design limit {F['pns_limit']:.0f} %")
        axs[2].axhline(100, color=MUTED, lw=0.8, ls=":")
        axs[0].legend(handles=axs[0].get_lines()[:3] + part_legend(names), fontsize=7.5, ncol=7, frameon=False, loc="upper right")
        axs[2].legend(fontsize=7.5, ncol=5, frameon=False, loc="upper right")
        axs[2].set_ylim(0, 110)
        fig.suptitle(f"{title}\n{label}; SAFE model {F['pns_hw']} (illustrative hardware, not a real scanner); max {pns[i0:j1, 3].max():.1f} %", color=INK, fontsize=10)
        fig.tight_layout()
        fig.savefig(figs / fname, dpi=110)
        plt.close(fig)

    ntr_win = (len(sg) - 1) // (L - 1)
    seq_panels(0, len(sg), "final_03_sequence.png", f"{ntr_win} consecutive TRs of the played sequence")
    last = (ntr_win - 1) * (L - 1)
    seq_panels(last, last + L, "final_04_pns_tr.png", "last TR of that window (PNS in steady state)")

    # ── 05 moments ──
    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    style(axs[0], ylabel="|k| / kmax")
    style(axs[1], xlabel="ms from TR start", ylabel="γ M1 (rad per m/s)")
    shade(axs[0], t_tr, tr_part(0), names)
    shade(axs[1], t_tr, tr_part(0), names)
    ls = ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1))]
    for v in range(nv):
        axs[0].plot(t_tr, np.linalg.norm(trm[v, :, :3], axis=1) / kmax, color=INK if v == 0 else MUTED, lw=1.0, ls=ls[v % len(ls)], label=var[v])
        for a in range(3):
            axs[1].plot(t_tr, GAMMA * trm[v, :, 3 + a], color=AX[a], lw=0.9, ls=ls[v % len(ls)], label=f"{'xyz'[a]}" if v == 0 else None)
    axs[0].axhline(1.0, color=INK, lw=0.8, ls="--")
    for te in te_marks(0):
        for ax in axs:
            ax.axvline(te, color=MUTED, lw=0.5, ls=":")
    axs[0].legend(fontsize=7, ncol=4, frameon=False, loc="upper right")
    axs[1].legend(fontsize=7.5, ncol=3, frameon=False, loc="upper left", title="axis (line style = variant)", title_fontsize=7)
    fig.suptitle(f"{title}\nk-space excursion and first moment from the RF centre (dotted: echo times of the first variant)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "final_05_moments.png", dpi=110)
    plt.close(fig)

    # ── 06 spokes in 3D ──
    fig = plt.figure(figsize=(15, 5.6))
    u, w = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    sph = [kmax * np.cos(u) * np.sin(w), kmax * np.sin(u) * np.sin(w), kmax * np.cos(w)]
    if is_b0:
        nl = ks.shape[0] // 2
        cols = [AX[0]] * nl + [AX[1]] * nl
        lab = ["TR 1 lobes"] + [None] * (nl - 1) + ["TR 2 lobes"] + [None] * (nl - 1)
    else:
        pal = AX + ["#8a5cd1", "#c2417a", "#7a6a2f"]
        cols = pal[: ks.shape[0]]
        lab = [f"rotation {i}" for i in range(ks.shape[0])]
    for p, (el, az) in enumerate([(22, -60), (75, 20)]):
        ax = fig.add_subplot(1, 3, p + 1, projection="3d")
        ax.plot_wireframe(*sph, color=GRID, lw=0.4)
        for i in range(ks.shape[0]):
            ax.plot(ks[i, :, 0], ks[i, :, 1], ks[i, :, 2], color=cols[i], lw=0.9 if is_b0 else 0.8, label=lab[i])
        ax.view_init(elev=el, azim=az)
        ax.set_box_aspect((1, 1, 1))
        for setl in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setl(-kmax, kmax)
        ax.set_xlabel("kx (1/m)", fontsize=8)
        ax.set_ylabel("ky (1/m)", fontsize=8)
        ax.set_zlabel("kz (1/m)", fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)
        if p == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax = fig.add_subplot(1, 3, 3)
    style(ax, xlabel="ms from readout start", ylabel="|k| / kmax")
    tk = np.arange(ks.shape[1]) * dt * 1e3
    for i in range(min(ks.shape[0], 6)):
        ax.plot(tk, np.linalg.norm(ks[i], axis=1) / kmax, color=cols[i], lw=1.0)
    ax.axhline(1, color=INK, lw=0.8, ls="--")
    fig.suptitle(f"{title}\n{'every lobe of two consecutive TRs (pairs (d, −d) along one diameter, a new diameter per pair)' if is_b0 else 'the base readout under the first rotations of the acquisition order (sphere: kmax)'}",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "final_06_spokes_3d.png", dpi=110)
    plt.close(fig)

    # base spoke coloured by time (flow) / one lobe pair (B0)
    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    k0 = ks[0] if not is_b0 else np.concatenate([ks[0], ks[1][1:]])
    tt = np.arange(len(k0)) * dt * 1e3
    ax.plot_wireframe(*sph, color=GRID, lw=0.3)
    sc = ax.scatter(k0[:, 0], k0[:, 1], k0[:, 2], c=tt, cmap="viridis", s=3)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="ms from readout start")
    ax.set_box_aspect((1, 1, 1))
    for setl in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setl(-kmax, kmax)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title(("first lobe pair (out along d, back, out along −d, back)" if is_b0 else "one readout, coloured by time"), color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "final_06b_spoke_time.png", dpi=110)
    plt.close(fig)

    # ── 07 GIRF ──
    t = gi[:, 0] * 1e3
    gn, ga = gi[:, 1:4], gi[:, 4:7]
    dk = np.cumsum((ga - gn) * dt, axis=0) * GBAR
    fig, axs = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
    style(axs[0], ylabel="G (mT/m)")
    style(axs[1], ylabel="played − nominal (mT/m)")
    style(axs[2], xlabel="ms from TR start", ylabel="accumulated Δk (1/m)")
    p0 = tr_part(0)
    for ax in axs:
        shade(ax, t, p0, names)
    for a in range(3):
        axs[0].plot(t, gn[:, a] * 1e3, color=AX[a], lw=1.0, label=f"{'xyz'[a]} nominal")
        axs[0].plot(t, ga[:, a] * 1e3, color=AX[a], lw=1.0, ls="--", label=f"{'xyz'[a]} played")
        axs[1].plot(t, (ga[:, a] - gn[:, a]) * 1e3, color=AX[a], lw=0.9)
        axs[2].plot(t, dk[:, a], color=AX[a], lw=0.9, label="xyz"[a])
    axs[2].text(0.01, 0.92, f"Nyquist cell 1/FOV at FOV 0.32 m = {1 / 0.32:.1f} 1/m", transform=axs[2].transAxes, color=MUTED, fontsize=8)
    axs[0].legend(fontsize=7, ncol=6, frameon=False, loc="upper right")
    fig.suptitle(f"{title}\nthrough the illustrative GIRF ({F['girf']}: 2–3 µs axis delays, 8 µs low-pass, 0.2 %/3 ms + 0.1 %/50 µs eddy currents)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "final_07_girf.png", dpi=110)
    plt.close(fig)

    # ── 08 spectrum ──
    fig, ax = plt.subplots(figsize=(11, 4.2))
    style(ax, xlabel="frequency (kHz)", ylabel="|G(f)| (mT/m · ms)")
    nfft = int(2 ** np.ceil(np.log2(len(sg) * 4)))
    f = np.fft.rfftfreq(nfft, dt) / 1e3
    gmx = 0.0
    for a in range(3):
        G = np.abs(np.fft.rfft(sg[:, a], nfft)) * dt * 1e6
        gmx = max(gmx, G[f <= 6].max())
        ax.plot(f, G, color=AX[a], lw=0.8, label="xyz"[a])
    ax.axvspan(1.0, 1.5, color="#f3ecdc", lw=0, zorder=0, label="typical mechanical resonance band (assumed)")
    ax.set_xlim(0, 6)
    ax.set_yscale("log")
    ax.set_ylim(gmx * 1e-5, gmx * 2)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{title}\namplitude spectrum of the {ntr_win}-TR window per physical axis", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "final_08_spectrum.png", dpi=110)
    plt.close(fig)

    # ── 09 echoes (B0) ──
    if is_b0 and (run / "echoes.json").exists():
        E = json.loads((run / "echoes.json").read_text())
        fig, axs = plt.subplots(1, 2, figsize=(13, 4))
        style(axs[0], xlabel="echo time after the RF centre (ms)", ylabel="")
        for v, te in enumerate(E["te_ms"]):
            axs[0].scatter(te, np.full(len(te), v), color=AX[v], s=22, edgecolors="white", linewidths=0.6, zorder=3, label=var[v])
        axs[0].set_yticks(range(len(E["te_ms"])))
        axs[0].set_yticklabels(var, fontsize=8)
        axs[0].set_ylim(-0.7, len(E["te_ms"]) - 0.3)
        fat_period = 1e3 / (GBAR * 2.89 * 3.4e-6)
        axs[0].set_title(f"{len(E['te_ms'][0])} echoes per TR, interleaved by the TE shift (fat period {fat_period:.2f} ms at 2.89 T)", color=INK, fontsize=9)
        style(axs[1], xlabel="echo index", ylabel="γ |M1| at the echo (rad per m/s)")
        e = np.arange(len(E["flow_phase_rad_per_m_s"]))
        axs[1].bar(e, E["flow_phase_rad_per_m_s"], color=[AX[0] if i % 2 == 0 else AX[1] for i in e], width=0.7)
        axs[1].set_title("even echoes first-order flow compensated (M1 = 0), odd echoes carry one lobe's M1", color=INK, fontsize=9)
        fig.suptitle(title, color=INK, fontsize=10)
        fig.tight_layout()
        fig.savefig(figs / "final_09_echoes.png", dpi=110)
        plt.close(fig)
    print("wrote", figs)


if __name__ == "__main__":
    main()
