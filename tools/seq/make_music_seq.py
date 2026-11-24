#!/usr/bin/env python3
"""A melody played through the gradient system, as in MRI_music (Q. Zhang, github.com/qinweizhang/MRI_music_pub, Philips 3 T;
Case Western's Bach cello suite; RESON'INT note tables): the audio waveform IS the gradient waveform on one axis
(low-passed at 2 kHz, scaled to the slew limit), pitch = gradient frequency, no equalisation. Rendering it with
`acousticsim` colours it with the scanner's acoustic transfer function: a realism test that needs no scanner
(compare with the published recordings of MRI music).

    data/hasty/venvs/pypulseq/bin/python tools/seq/make_music_seq.py OUTDIR [--axis x]
writes OUTDIR/music_in.wav (the synthetic melody), OUTDIR/music_{axis}.seq
"""
import argparse, struct, wave
from pathlib import Path
import numpy as np
import pypulseq as pp

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--axis", default="x")
ap.add_argument("--slew", type=float, default=100.0, help="peak slew of the played waveform, T/m/s")
ap.add_argument("--gmax", type=float, default=30e-3)
a = ap.parse_args()
out = Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)

# Ode to Joy, 15 notes: E E F G G F E D C C D E E D D (equal temperament, A4 = 440 Hz)
def hz(semi_from_a4): return 440.0 * 2 ** (semi_from_a4 / 12)
E4, F4, G4, D4, C4 = hz(-5), hz(-4), hz(-2), hz(-7), hz(-9)
notes = [E4, E4, F4, G4, G4, F4, E4, D4, C4, C4, D4, E4, E4, D4, D4]
dur = [0.4] * 14 + [0.8]
fs = 48000
audio = []
for f, d in zip(notes, dur):
    t = np.arange(int(d * fs)) / fs
    env = np.minimum(1, np.minimum(t / 0.01, (d - t) / 0.08)) ** 1
    audio.append(env * (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t) + 0.15 * np.sin(6 * np.pi * f * t)))
audio = np.concatenate([np.zeros(int(0.1 * fs)), np.concatenate(audio), np.zeros(int(0.5 * fs))])
audio /= np.abs(audio).max()
with wave.open(str(out / "music_in.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs)
    w.writeframes((audio * 0.9 * 32767).astype("<i2").tobytes())

# gradient: audio low-passed at 2 kHz (rectangular, as MRI_music), resampled to 20 us points, scaled to the limits
step = 20e-6
t_out = np.arange(0, len(audio) / fs, step)
X = np.fft.rfft(audio)
f = np.fft.rfftfreq(len(audio), 1 / fs)
X[f > 2000] = 0
x = np.fft.irfft(X, len(audio))
g = np.interp(t_out, np.arange(len(x)) / fs, x)
g[0] = g[-1] = 0.0
sl = np.abs(np.diff(g)).max() / step
g *= min(a.gmax / np.abs(g).max(), a.slew / sl)
print(f"peak {np.abs(g).max()*1e3:.1f} mT/m, peak slew {np.abs(np.diff(g)).max()/step:.1f} T/m/s, {t_out[-1]:.1f} s")
s = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s", grad_raster_time=4e-6, rf_raster_time=2e-6, adc_raster_time=2e-6, block_duration_raster=4e-6)
seq = pp.Sequence(s)
tt = np.round(t_out / 4e-6) * 4e-6
seq.add_block(pp.make_extended_trapezoid(channel=a.axis, amplitudes=g * s.gamma, times=tt, system=s))
ok, errs = seq.check_timing()
seq.set_definition("Name", "music via gradients")
seq.write(str(out / f"music_{a.axis}.seq"))
print("timing ok:", ok, errs[:2] if not ok else "")
