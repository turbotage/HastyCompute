#!/usr/bin/env python3
"""Acoustic transfer function (ATF) calibration for `acousticsim` (research/acoustic_sim.md).

Protocol after OPTIKS (McCready et al., IEEE TMI 2025: sinusoids every 10 Hz, 200-3130 Hz, 160 ms each, a 1 kHz
reference, sound pressure recorded at the ear position) and MR-RADIO (ISMRM 2022 abstract 2774: resonances from the peaks
of the audio spectrum). One sequence per axis: a gradient sinusoid (Hann-tapered 5 ms ends) of frequency f_j per
period P = 160 ms, amplitude limited by the slew rate; the first period is the 1 kHz reference.

    make   OUTDIR [--step 20] [--fmin 200] [--fmax 3130] [--dur 0.12]
           → OUTDIR/atf_{x,y,z}.seq + atf.json (frequencies, amplitudes, period)
    fit    OUTDIR AXIS RECORDING.npy [--fs 48000] [--offset S] [--model-in model.json] [--out model_fit.json]
           magnitude of H(f_j) = |O(f_j)| / |I(f_j)| from the recording (the part of every burst after the taper; timing
           offsets only matter to ±10 ms, phase is not used), then a least-squares fit of the modal model
           (f0, Q, peak per mode; log-magnitude residual) → JSON axis modes, merged into --model-in
    eval   model_a.json model_b.json   dB difference of two models' |H| per axis, 300-3000 Hz

Synthetic round trip (what tests the chain, no scanner needed): see tools/seq/run_acoustic_atf_check.sh.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

AXES = "xyz"
RASTER = 4e-6
SLEW = 120.0  # T/m/s, as the demo sequences
AMAX = 15e-3  # T/m


def stim_freqs(fmin, fmax, step):
    return np.arange(fmin, fmax + 1e-9, step)


def amp_for(f):
    """amplitude (T/m) of the burst at f: slew-limited with margin for the taper"""
    return float(min(AMAX, 0.6 * SLEW / (2 * np.pi * f)))


def burst(f, amp, dur, dt):
    """gradient samples (T/m) at multiples of dt, Hann-tapered 5 ms ends"""
    t = np.arange(0, dur + 1e-12, dt)
    w = np.ones_like(t)
    r = 5e-3
    w[t < r] = 0.5 - 0.5 * np.cos(np.pi * t[t < r] / r)
    w[t > dur - r] = 0.5 - 0.5 * np.cos(np.pi * (dur - t[t > dur - r]) / r)
    return t, amp * w * np.sin(2 * np.pi * f * t)


def make(a):
    import pypulseq as pp

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    freqs = [1000.0] + list(stim_freqs(a.fmin, a.fmax, a.step))
    period = a.dur + 0.04
    meta = {"freqs": freqs, "amps": [amp_for(f) if i else 10e-3 for i, f in enumerate(freqs)], "dur": a.dur, "period": period, "taper": 5e-3}
    for ax in AXES:
        s = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=SLEW, slew_unit="T/m/s", grad_raster_time=RASTER, rf_raster_time=2e-6, adc_raster_time=2e-6, block_duration_raster=RASTER)
        seq = pp.Sequence(s)
        for f, amp in zip(freqs, meta["amps"]):
            # piecewise-linear points: >= 14 per cycle, on the gradient raster
            step = max(RASTER, np.floor(1 / (14 * f) / RASTER) * RASTER)
            n = int(round(a.dur / step))
            t = np.arange(n + 1) * step
            t[-1] = np.round(a.dur / RASTER) * RASTER
            _, g = burst(f, amp, a.dur, step)
            g = np.interp(t, np.arange(len(g)) * step, g)
            g[0] = g[-1] = 0.0
            ev = pp.make_extended_trapezoid(channel=ax, amplitudes=g * s.gamma, times=t, system=s)
            seq.add_block(ev)
            seq.add_block(pp.make_delay(float(np.ceil((period - a.dur) / RASTER) * RASTER)))
        ok, errs = seq.check_timing()
        seq.set_definition("Name", f"ATF sweep {ax}")
        seq.write(str(out / f"atf_{ax}.seq"))
        print(f"atf_{ax}.seq: {len(freqs)} bursts, {len(freqs) * period:.1f} s, timing ok: {ok}")
        if not ok:
            print("   ", errs[:3])
    (out / "atf.json").write_text(json.dumps(meta, indent=1))


def sine_amp(x, f, fs):
    n = len(x)
    k = np.exp(-2j * np.pi * f * np.arange(n) / fs)
    w = np.hanning(n)
    return 2 * abs(np.sum(x * w * k)) / np.sum(w)


def measure(meta, rec, fs, offset):
    """|H|(f_j) (Pa per T/m) from the recording: each burst's response after the leading taper"""
    P, D = meta["period"], meta["dur"]
    f, h = [], []
    for j, (fj, aj) in enumerate(zip(meta["freqs"], meta["amps"])):
        t0 = offset + j * P
        i0, i1 = int((t0 + 0.04) * fs), int((t0 + D - 0.005) * fs)
        if i1 > len(rec):
            break
        _, g = burst(fj, aj, D, 1 / fs)
        gi0, gi1 = int(0.04 * fs), int((D - 0.005) * fs)
        h.append(sine_amp(rec[i0:i1], fj, fs) / max(sine_amp(g[gi0:gi1], fj, fs), 1e-30))
        f.append(fj)
    return np.array(f), np.array(h)


def modal(params, f, kinds):
    out = np.zeros_like(f, dtype=complex)
    for (f0, q, pk) in params:
        u = f / f0
        jq = 1j * u / q
        out += pk * jq / (1 - u * u + jq)
    return out


def fit_axis(f, h, n_narrow=4, floor=((900.0, 1.3, 110.0), (1800.0, 1.5, 90.0))):
    from scipy.optimize import least_squares

    order = np.argsort(f)
    f, h = f[order], h[order]
    sm = np.convolve(h, np.ones(3) / 3, mode="same")
    pk = [i for i in range(1, len(f) - 1) if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1]]
    pk = sorted(pk, key=lambda i: -sm[i])
    chosen = []
    for i in pk:
        if all(abs(f[i] - f[c]) > 150 for c in chosen):
            chosen.append(i)
        if len(chosen) == n_narrow:
            break
    init = [(f[i], 20.0, h[i]) for i in chosen] + list(floor)
    nn = len(chosen)

    def unpack(p):
        return [(p[3 * k], np.exp(p[3 * k + 1]), np.exp(p[3 * k + 2])) for k in range(len(init))]

    def res(p):
        return 20 * np.log10(np.abs(modal(unpack(p), f, None)) + 1e-12) - 20 * np.log10(h + 1e-12)

    p0, lo, hi = [], [], []
    for k, (f0, q, pkv) in enumerate(init):
        p0 += [f0, np.log(q), np.log(pkv)]
        if k < nn:
            lo += [f0 * 0.9, np.log(4.0), np.log(pkv * 0.05)]
            hi += [f0 * 1.1, np.log(80.0), np.log(pkv * 20)]
        else:
            lo += [f0 * 0.5, np.log(0.8), np.log(5.0)]
            hi += [f0 * 1.5, np.log(3.0), np.log(1000.0)]
    sol = least_squares(res, p0, bounds=(lo, hi))
    r = res(sol.x)
    return [(float(a), float(b), float(c)) for a, b, c in unpack(sol.x)], float(np.sqrt(np.mean(r**2)))


def fit(a):
    meta = json.loads((Path(a.outdir) / "atf.json").read_text())
    rec = np.load(a.recording).astype(float)
    f, h = measure(meta, rec, a.fs, a.offset)
    ax = AXES.index(a.axis)
    modes, rms = fit_axis(f[1:], h[1:])  # [0] is the 1 kHz reference
    print(f"axis {a.axis}: {len(f) - 1} frequencies, fit rms {rms:.2f} dB (log magnitude); modes (f0 Hz, Q, peak Pa/(T/m)):")
    for m in modes:
        print("   %8.1f  %5.1f  %7.1f" % m)
    base = json.loads(Path(a.model_in).read_text()) if a.model_in else None
    if base is None:
        sys.exit("--model-in required (model JSON to update; see `acousticsim --dump-model`)")
    base["axes"][ax] = [{"f0": m[0], "q": m[1], "peak": m[2]} for m in modes]
    base["name"] = f"fitted from ATF sweep ({a.axis} updated)"
    Path(a.out).write_text(json.dumps(base, indent=1))
    print("wrote", a.out)


def h_of(model, ax, f):
    return np.abs(modal([(m["f0"], m["q"], m["peak"]) for m in model["axes"][ax]], f, None)) * model["b0"] / model["b0_ref"]


def evaluate(a):
    A, B = (json.loads(Path(p).read_text()) for p in (a.model_a, a.model_b))
    f = np.arange(300, 3001, 10.0)
    for ax in range(3):
        d = 20 * np.log10(h_of(B, ax, f) / h_of(A, ax, f))
        print(f"axis {AXES[ax]}: |H| difference rms {np.sqrt(np.mean(d**2)):.2f} dB, max {np.abs(d).max():.2f} dB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("outdir")
    m.add_argument("--step", type=float, default=20.0)
    m.add_argument("--fmin", type=float, default=200.0)
    m.add_argument("--fmax", type=float, default=3130.0)
    m.add_argument("--dur", type=float, default=0.12)
    f_ = sub.add_parser("fit")
    f_.add_argument("outdir")
    f_.add_argument("axis", choices=list(AXES))
    f_.add_argument("recording")
    f_.add_argument("--fs", type=float, default=48000.0)
    f_.add_argument("--offset", type=float, default=0.0)
    f_.add_argument("--model-in")
    f_.add_argument("--out", default="model_fit.json")
    e = sub.add_parser("eval")
    e.add_argument("model_a")
    e.add_argument("model_b")
    a = ap.parse_args()
    {"make": make, "fit": fit, "eval": evaluate}[a.cmd](a)
