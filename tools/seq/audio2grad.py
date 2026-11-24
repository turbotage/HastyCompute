#!/usr/bin/env python3
"""Audio -> gradient waveforms: which x/y/z gradients make the scanner play a given sound (research/acoustic_sim.md).

Inverse of `acousticsim`. Forward model (the same LTI model, 3 axes): p = sum_a h_a * g_a with the modal transfer function H_a(f)
of the model JSON (and the linear-interpolation sinc^2 of the 20 us gradient points). Given the target pressure t(t) (the audio,
scaled to a requested level), find g_x, g_y, g_z

    min  || W (A g - t) ||^2 + eps ||g||^2   s.t.  |g| <= Gmax,  |dg/dt| <= Smax        (W: band mask fmin..fmax)

Start: per-frequency minimum-norm 3-axis solution G_a = conj(H_a) T / (sum_b |H_b|^2 + eps)  (every frequency is played mostly on the
axis whose resonance is nearest: bass on one axis, treble on another, no hand-made axis split). Then preconditioned Landweber
iterations g <- P(g + W (t - A g)) with P = amplitude clip and slew limit (average of the lower and upper Lipschitz envelopes,
vectorised). The target level is the requested broadband rms; when the limits cannot reach it, the level of the result is lower,
and reported. Written as one pypulseq .seq (extended trapezoids, 20 us points), `acousticsim --seq` renders it.

    data/hasty/venvs/pypulseq/bin/python tools/seq/audio2grad.py SONG.wav|mp3|... --out DIR [--level-db 90] [--start 0] [--dur 10]
        [--model model.json] [--fmin 80] [--fmax 4000] [--gmax 38] [--slew 110] [--eps 3e-3] [--iters 30] [--verify]

Writes DIR/{target.wav, achieved.wav, roundtrip_acousticsim.wav (--verify), naive.wav (audio used directly as gradient, as MRI_music), grad.seq, grad.npy, report.json, compare.png}.
--verify renders grad.seq with the Rust `acousticsim` (independent implementation of the forward model; reverb off) and reports
the match against the target. Levels: calibrated to nothing (toy model), so the level is "toy-model dB SPL at the ear".
"""
import argparse
import json
import subprocess as sp
import sys
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.fft import next_fast_len, rfft, irfft, rfftfreq

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "rustlib/target/acoustic/release/acousticsim"
P0 = 20e-6


def load_audio(path, start, dur):
    """mono float array + sample rate; wav via wave/ffmpeg fallback for anything else"""
    path = str(path)
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", path, "-t", str(dur), "-ac", "1", "-ar", "48000", "-f", "f32le", "-"]
    raw = sp.run(cmd, capture_output=True)
    if raw.returncode != 0:
        sys.exit("ffmpeg failed: " + raw.stderr.decode()[:300])
    return np.frombuffer(raw.stdout, dtype="<f4").astype(float), 48000


def H_axes(model, f, step):
    """complex transfer (Pa per T/m) of the 3 axes on the frequency grid f, incl. delay and linear-interpolation sinc^2"""
    out = []
    for a in range(3):
        h = np.zeros_like(f, dtype=complex)
        for m in model["axes"][a]:
            u = f / m["f0"]
            jq = 1j * u / m["q"]
            h += m["peak"] * jq / (1 - u * u + jq)
        h *= model["b0"] / model["b0_ref"]
        h *= np.exp(-2j * np.pi * f * model.get("delay_s", [0, 0, 0])[a]) * np.sinc(f * step) ** 2
        out.append(h)
    return np.array(out)


def lip_project(g, delta):
    """nearest-ish delta-Lipschitz function (slew limit): mean of the lower and upper Lipschitz envelopes (running min, vectorised)"""
    i = np.arange(len(g)) * delta

    def minor(x):  # largest delta-Lipschitz function below x: min_j x_j + delta |i-j|
        fwd = np.minimum.accumulate(x - i) + i
        bwd = np.minimum.accumulate((x + i)[::-1])[::-1] - i
        return np.minimum(fwd, bwd)

    return 0.5 * (minor(g) - minor(-g))


def project(g, gmax, delta):
    g = np.clip(g, -gmax, gmax)
    out = np.empty_like(g)
    for a in range(3):
        out[a] = lip_project(g[a], delta)
    out[:, 0] = out[:, -1] = 0.0
    return np.clip(out, -gmax, gmax)


def forward(H, g):
    return irfft((H * rfft(g, axis=1)).sum(0), g.shape[1])


def level_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x**2)) / P0 + 1e-30)


def shape_match(t, p):
    """correlation and shape-SNR (dB) after the best scalar gain, and that gain"""
    n = min(len(t), len(p))
    t, p = t[:n], p[:n]
    gain = np.dot(p, t) / max(np.dot(p, p), 1e-30)
    r = t - gain * p
    return float(np.dot(p, t) / np.sqrt(np.dot(p, p) * np.dot(t, t) + 1e-30)), float(10 * np.log10(np.dot(t, t) / (np.dot(r, r) + 1e-30))), float(gain)


def write_wav(path, x, gain):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(np.clip(x * gain, -1, 1).__mul__(32767).astype("<i2").tobytes())


def write_seq(path, g, dt, name):
    """one block, one extended trapezoid per axis (g: 3 x n, T/m, points dt apart); returns (timing ok, errors)"""
    import pypulseq as pp_

    s = pp_.Opts(max_grad=40, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s", grad_raster_time=4e-6, rf_raster_time=2e-6, adc_raster_time=2e-6, block_duration_raster=4e-6)
    seq = pp_.Sequence(s)
    times = np.round(np.arange(g.shape[1]) * dt / 4e-6) * 4e-6
    seq.add_block(*[pp_.make_extended_trapezoid(channel=ch, amplitudes=g[k] * s.gamma, times=times, system=s) for k, ch in enumerate("xyz")])
    ok, errs = seq.check_timing()
    seq.set_definition("Name", name)
    seq.write(str(path))
    return ok, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--level-db", type=float, default=90.0, help="requested broadband rms (dB SPL at the ear, toy model)")
    ap.add_argument("--fmin", type=float, default=80.0)
    ap.add_argument("--fmax", type=float, default=4000.0)
    ap.add_argument("--gmax", type=float, default=38.0, help="mT/m")
    ap.add_argument("--slew", type=float, default=110.0, help="T/m/s")
    ap.add_argument("--eps", type=float, default=3e-3, help="Tikhonov, relative to the largest sum |H|^2 in band")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--step", type=float, default=20e-6, help="gradient point spacing (multiple of 4 us)")
    ap.add_argument("--pad", type=float, default=0.25, help="s of silence before/after (gradients start/end at 0, resonances can be damped)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-seq", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.model:
        model = json.loads(Path(a.model).read_text())
    else:
        sp.run([str(BIN), "--dump-model", "--out", str(out)], check=True, capture_output=True)
        model = json.loads((out / "model_toy_3t.json").read_text())

    x, sr = load_audio(a.audio, a.start, a.dur)
    dt = a.step
    fr = Fraction(round(1 / dt), sr).limit_denominator(1000)
    xt = signal.resample_poly(x, fr.numerator, fr.denominator)
    fade = int(0.02 / dt)
    xt[:fade] *= np.linspace(0, 1, fade)
    xt[-fade:] *= np.linspace(1, 0, fade)
    npad = int(a.pad / dt)
    n0 = len(xt) + 2 * npad
    N = next_fast_len(n0)
    t = np.zeros(N)
    t[npad : npad + len(xt)] = xt
    f = rfftfreq(N, dt)
    # band mask, raised-cosine edges (15 % of the corner frequency)
    def edge(fx, f0, w, up):
        s = np.clip((fx - f0) / w, 0, 1)
        s = 0.5 - 0.5 * np.cos(np.pi * s)
        return s if up else 1 - s
    band = edge(f, a.fmin, a.fmin * 0.3, True) * edge(f, a.fmax, a.fmax * 0.15, False) * (f <= a.fmax * 1.2)
    T = rfft(t) * band
    t = irfft(T, N)
    seg = slice(npad, npad + len(xt))
    t *= 10 ** (a.level_db / 20) * P0 / np.sqrt(np.mean(t[seg] ** 2))
    T = rfft(t)

    H = H_axes(model, f, dt)
    s2 = (np.abs(H) ** 2).sum(0)
    eps = a.eps * s2[band > 0.5].max()
    W = np.conj(H) / (s2 + eps) * band
    G = W * T
    g = np.array([irfft(G[k], N) for k in range(3)])
    gmax, delta = a.gmax * 1e-3, a.slew * dt
    g0 = g.copy()
    for it in range(a.iters):
        g = project(g, gmax, delta)
        r = t - forward(H, g)
        g = g + np.array([irfft(W[k] * rfft(r), N) for k in range(3)])
    g = project(g, gmax, delta)
    p = forward(H, g)

    # naive baseline: the audio is the gradient on one axis (MRI_music): low-passed, scaled to the same limits
    xn = irfft(rfft(t) * band, N)
    gn = np.zeros_like(g)
    gn[0] = xn
    gn[0] *= min(gmax / np.abs(gn[0]).max(), a.slew / (np.abs(np.diff(gn[0])).max() / dt))
    pn = forward(H, gn)

    seg_t = t[seg]
    c, snr, gain = shape_match(seg_t, p[seg])
    cn, snrn, gainn = shape_match(seg_t, pn[seg])
    rep = {
        "audio": a.audio, "start_s": a.start, "dur_s": len(xt) * dt, "requested_level_db": a.level_db,
        "achieved_level_db": float(level_db(p[seg])), "naive_level_db": float(level_db(pn[seg])),
        "match_corr": c, "match_snr_db": snr, "naive_corr": cn, "naive_snr_db": snrn,
        "peak_g_mT_m": (np.abs(g).max(axis=1) * 1e3).tolist(), "peak_slew_T_m_s": (np.abs(np.diff(g, axis=1)).max(axis=1) / dt).tolist(),
        "rms_g_mT_m": (np.sqrt((g**2).mean(axis=1)) * 1e3).tolist(),
        "limits": {"gmax_mT_m": a.gmax, "slew_T_m_s": a.slew}, "fmin": a.fmin, "fmax": a.fmax, "eps_rel": a.eps, "iters": a.iters,
    }

    # audio files at 48 kHz, shared gain
    r48 = lambda y: signal.resample_poly(y, 48, round(1 / dt / 1000))  # 48 kHz from 1/dt (multiple of 1 kHz)
    tt48, pp48, pn48 = (r48(y[seg]) for y in (t, p, pn))
    gw = 0.9 / max(np.abs(tt48).max(), np.abs(pp48).max())
    write_wav(out / "target.wav", tt48, gw)
    write_wav(out / "achieved.wav", pp48, gw)
    write_wav(out / "naive.wav", pn48, gw)
    np.save(out / "grad.npy", g.astype(np.float32))

    if not a.no_seq:
        ok, errs = write_seq(out / "grad.seq", g, dt, f"audio2grad {Path(a.audio).name}")
        rep["timing_ok"] = bool(ok)
        if not ok:
            rep["timing_errors"] = [str(e) for e in errs[:3]]

    if a.verify and not a.no_seq:
        vd = out / "verify"
        # acousticsim windows are limited to 120 s and a single-block .seq is one "TR": verify in 100 s chunks of grad (+-0.6 s overlap,
        # 5 ms fades at the cut ends), keep the middle part of each render
        C, ov = int(100.0 / dt), int(0.6 / dt)
        n48c, n48o = int(C * dt * 48000), int(ov * dt * 48000)
        parts = []
        for k in range(int(np.ceil(N / C))):
            i0, i1 = max(0, k * C - ov), min(N, (k + 1) * C + ov)
            gc = g[:, i0:i1].copy()
            fd = int(0.005 / dt)
            if i0 > 0:
                gc[:, :fd] *= np.linspace(0, 1, fd)
            if i1 < N:
                gc[:, -fd:] *= np.linspace(1, 0, fd)
            wd = vd / f"win{k}"
            wd.mkdir(parents=True, exist_ok=True)
            write_seq(wd / "chunk.seq", gc, dt, f"verify chunk {k}")
            sp.run([str(BIN), "--seq", str(wd / "chunk.seq"), "--tr-ms", "1e6", "--tr-start", "0", "--ntr", "1", "--lead", "0", "--reverb", "off", "--out", str(wd)], check=True, capture_output=True)
            pc = np.load(next(wd.glob("w00_*_p.npy")))
            lo = 0 if i0 == 0 else n48o
            hi = len(pc) if i1 == N else lo + n48c
            parts.append(pc[lo:hi])
            (wd / "chunk.seq").unlink()
        pv = np.concatenate(parts)
        tv = signal.resample_poly(t, 48, round(1 / dt / 1000))
        n = min(len(pv), len(tv))
        npad48 = int(a.pad * 48000)
        sl = slice(npad48, min(n, npad48 + int(len(xt) * dt * 48000)))
        cv, snrv, gv = shape_match(tv[sl], pv[sl])
        write_wav(out / "roundtrip_acousticsim.wav", pv[sl], gw)  # what the Rust simulator makes of grad.seq
        rep["acousticsim_verify"] = {"corr": cv, "snr_db": snrv, "level_db": float(level_db(pv[sl])), "gain_vs_python_forward": gv}

    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("limits",)}, indent=1))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(11, 9))
        sd = max(1, len(seg_t) // 200000)  # plot decimation for long files
        tt = (np.arange(len(seg_t)) * dt)[::sd]
        ax[0].plot(tt, seg_t[::sd], "k", lw=0.5, label="target")
        ax[0].plot(tt, p[seg][::sd], "tab:orange", lw=0.5, alpha=0.8, label=f"achieved (SNR {snr:.1f} dB)")
        ax[0].plot(tt, pn[seg][::sd], "tab:blue", lw=0.5, alpha=0.6, label=f"naive direct drive (SNR {snrn:.1f} dB)")
        ax[0].legend(fontsize=7, ncol=3)
        ax[0].set_ylabel("Pa")
        for k, c_ in enumerate(["tab:red", "tab:green", "tab:blue"]):
            ax[1].plot((np.arange(N) * dt)[::sd], g[k][::sd] * 1e3, c_, lw=0.4, label="xyz"[k])
        ax[1].set_ylabel("gradient mT/m")
        ax[1].legend(ncol=3, fontsize=7)
        ff = rfftfreq(len(seg_t), dt)
        sp_t, sp_p, sp_n = (20 * np.log10(np.abs(rfft(y * np.hanning(len(y)))) / len(y) * 4 / P0 + 1e-9) for y in (seg_t, p[seg], pn[seg]))
        ax[2].semilogx(ff[1:], np.convolve(sp_t, np.ones(9) / 9, "same")[1:], "k", lw=0.7, label="target")
        ax[2].semilogx(ff[1:], np.convolve(sp_p, np.ones(9) / 9, "same")[1:], "tab:orange", lw=0.7, label="achieved")
        ax[2].semilogx(ff[1:], np.convolve(sp_n, np.ones(9) / 9, "same")[1:], "tab:blue", lw=0.7, label="naive")
        ax[2].set_xlim(50, 8000)
        ax[2].set_ylabel("dB SPL / bin")
        ax[2].set_xlabel("Hz")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "compare.png", dpi=120)
    except Exception as e:  # plots are optional
        print("plot failed:", e)


if __name__ == "__main__":
    main()
