"""Amplitude spectrum of the v2 readout as played on GE (4 µs raster, rotation-averaged vector magnitude) and of the
whole segmented TR train: where resonances hurt (the band the pass-2 targets come from)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qafigs as qf
d = qf.PROTO
ro = np.load(os.path.join(d, 'flowtr_ro.npy'))
dt = 4e-6
nfft = 1 << 20
f = np.fft.rfftfreq(nfft, dt)
S = np.sqrt(sum(np.abs(np.fft.rfft(ro[:, a], nfft)) ** 2 for a in range(3))) * dt
S /= S.max()
m = (f > 50) & (f < 5000)
fpk = f[m][np.argmax(S[m])]
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(f[m], S[m], color=qf.C[0])
ax.axvspan(400, 1050, color=qf.GRID, alpha=0.6, lw=0)
ax.text(410, 0.92, 'girf_sensitivity: vulnerable band', fontsize=8, color=qf.INK2)
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('|G(f)| (vector, rel.)')
ax.set_title(f'v2 readout on the GE raster: main lobe {fpk:.0f} Hz')
qf.save(fig, 'v2_readout_spectrum.png')
lev = {int(x): float(np.interp(x, f, S)) for x in (300, 500, 688, 1000, 1500, 2000, 3000)}
print('peak', fpk, lev)
json.dump(dict(peak_hz=float(fpk), rel=lev), open(os.path.join(qf.FIGS, 'v2_readout_spectrum.json'), 'w'), indent=1)
