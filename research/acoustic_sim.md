# Acoustic noise simulator (`hasty-seq::acoustic`)

Goal: Pulseq `.seq` (or `Grad` waveforms) + scanner acoustic model → SPL plots, harmonics table, playable `.wav`.

## Model

Per axis $a\in\{x,y,z\}$ (physical axes, after rotation; use the *played* gradient, i.e. after GIRF, when a GIRF is given):

$$F_a(t)\propto B_0\,G_a(t),\qquad p_a(f)=H_a(f)\,\hat G_a(f),\qquad p=\sum_a p_a\ (+\text{cross terms }H_{ab})$$

- Lorentz force on the gradient coil is linear in current, so the coil drive is $G_a$ itself, scaled by $B_0$. The SPL scales as $20\log_{10}(B_0)$ and with the gradient amplitude.
- $H_a(f)$: modal sum of 2nd-order resonances
  $H_a(f)=g_a\sum_k \dfrac{A_k\,(jf/f_k)^{m}}{1-(f/f_k)^2+j\,f/(Q_kf_k)}$ with $m=1$ default (pressure ∝ velocity at low f, tunable), plus a high-pass for radiation efficiency below the first mode.
- Ear transfer: hearing protection attenuation curve $L_{\rm hp}(f)$ (dB, table) applied last.
- Absolute level: one calibration scalar per axis, $g_a$, set so a reference trapezoid (or the chirp measurement) gives the measured dB(A). Uncalibrated runs print "relative dB".

## Outputs

1. `acoustic.wav` (48 kHz, mono; stereo option = per-axis split), normalized with a stated reference level; optional real-time level scaling so the file is loudness-true up to a chosen headroom.
2. Spectrogram (STFT, 20 ms window), dB(A)(t) with $L_{\rm eq}$, $L_{\rm peak}$, $L_{\rm max}$, 1/3-octave band levels.
3. Harmonics table: peaks of $|p(f)|$, matched to their cause (TR rate $1/{\rm TR}$ multiples, readout lobe rate, spoiler), and the resonance each lies on (distance to $f_k$ in units of $f_k/Q_k$).
4. Optional: per-block contribution list (which `.seq` blocks dominate $L_{\rm eq}$).

All figures + arrays written to `--out` directory (house rule).

## Implementation steps

1. `acoustic.rs` in `hasty-seq`: `AcousticModel` (per-axis resonance list, $g_a$, $B_0$, hearing-protection table, cross-coupling 3×3 optional), TOML/JSON loader following `girfmeas.rs` conventions.
2. Core: waveform → resample to 96 kHz (Grad is piecewise linear, evaluate exactly, no aliasing from the raster) → per-axis FFT filter, in chunks (overlap-save) so a 10-min sequence does not allocate a full-length buffer. Time-domain biquad bank is the alternative; take FFT blocks first (also gives non-parametric tables later).
3. Metrics: A-weighting, 1/3-octave, $L_{\rm eq}$ etc.; WAV writer (16-bit PCM, hand-written, no new dependency).
4. Bin `acousticsim` in `hasty-bench`: `--seq file.seq --model scanner.toml [--girf ...] --out dir`. Loops over a repeated TR if the `.seq` is long: simulate N TRs + report periodic structure; `--full` for whole sequence.
5. Plots via the repo's existing plotting route (check what `girfqa`/`seqdesign` use; python helper if that is how they do it).
6. Tests: single sine gradient → single spectral line at expected amplitude; resonance peak at $f_k$ with correct $Q$ (−3 dB width); linearity in $G$ and $B_0$; WAV round trip; chunked == one-shot filtering.
7. Defaults: a generic "GE SIGNA Premier 3T, generic" model with resonance bands from the literature, clearly labelled *uncalibrated*.

## Calibration (later work package)

- `girfqa`-style chirp sequence per axis (log sweep 100 Hz–5 kHz at moderate amplitude), mic (phone or measurement mic) at the ear position / bore centre; $H_a(f)=P(f)/\hat G(f)$, fit modal sum (LM, shares the Rust LM lib from the qfit plan).
- Repeat with heated gradient coil (as in GIRF QA) to get $f_k(T)$ shift.
- Cross-axis: drive pairs of axes at the same frequency, fit $H_{ab}$.

## Hooks into the designer

- Penalty/constraint in `robust.rs`/`tr.rs`: TR harmonics and readout lobe rates away from resonance bands (shares the resonance list with GIRF-robust TR).
- `optim.rs`: report $L_{\rm eq}$ alongside PNS.

## Open questions (user)

- Do we have any real acoustic data for the Premier (service report, vendor noise figures, a measurement)? Without it the first version is relative-only.
- Mono at the ear vs per-axis stereo wav as default?
- Include table/patient-bore position dependence? (Proposal: no; one ear position.)

## Status (2026-11-23): v1 built

Files (new only, no shared files touched except one `pub mod` line in `hasty-seq/src/lib.rs`):
`hasty-seq/src/acoustic.rs`, `hasty-seq/src/bin/acousticsim.rs`, `tools/seq/plot_acoustic.py`.
Build with a private target dir to avoid lock contention: `CARGO_TARGET_DIR=rustlib/target/acoustic cargo build --release -j 4 -p hasty-seq --bin acousticsim`.

Decisions (user, 2026-11-23): no measured data → toy model (3 axes × 3 resonances 0.5–2.7 kHz, Q 16–32, + 1 broad mode; ~100 dB(A) on a flow sequence, hearing-protection table −20…−42 dB optional);
mono WAV 48 kHz; `--ntr` TRs per window; windows by `--at` seconds / `--tr-start` TR index / automatic events (start, loudest stretch, signature changes).

Implementation vs plan:
- no overlap-save: windows are a few hundred ms to seconds; one transform per window (limit 120 s) with 0.2 s zero guard and a 0.15 s lead-in of earlier TRs for the ringing.
- the gradient is evaluated at its raster (must divide 1 ms: 4 µs GE and 10 µs both fine), sinc² of the linear interpolation corrected, resampling = spectrum truncation at 24 kHz.
- TRs: split at RF blocks; sequences without RF (GIRF QA) need `--tr-ms`.
- Event finder: median octave-band/dB(A) signature (TR repeated periodically) of `--median` TRs after vs before; on `h1_flow.seq` the sound drifts slowly with the readout rotation, so the automatic events are weak (3–5 dB band changes), manual `--at` is the reliable route there. A step-versus-drift test would be the next improvement.
- WAV gain: shared across the run (loudest peak 0.9 FS) so loudness compares between windows; `--each`, `--fs-db`.
- Optional `--girf` plays the gradients through a measured GIRF first.
- Cross-axis coupling and calibration still open.

Example: `acousticsim --seq data/hasty/seq/girf_qa/protocol/h1_flow.seq --tr-ms 14.104 --out data/hasty/seq/studies/acoustic/h1_flow` (37 s: signatures of 49 810 TRs) and with `--at 300,500 --ntr 48 --protection` (0.2 s).
First finding (toy model): the strongest lines of h1_flow are harmonics 8–11 of the 71 Hz TR rate at 570–780 Hz, i.e. on the first x/y/z modes.

## Demo sequences for scoring realism (2026-11-23)

pypulseq 1.5.0 installed in `data/hasty/venvs/pypulseq` (venv, inside the repo). `tools/seq/make_acoustic_demos.py` writes
`data/hasty/seq/acoustic_demo/{se_2d,gre_2d,ute_3d,epi_ms}.seq` (GE-like rasters 4/2 µs, 40 mT/m, 120 T/m/s; all pass `check_timing`).
Rendered with the toy model, one shared absolute scale (`--fs-db 126 --also-norm`) into `data/hasty/seq/studies/acoustic/demo/listen/`
(`*_true_level.wav` comparable loudness, `*_norm.wav` for timbre) + README table; per-sequence figures in `demo/<name>/figs/`.

Toy-model results (dB(A), uncalibrated): SE 71 (4 ms readout ticks every 0.5 s), GRE 86, UTE 86, EPI 85 (1/(2·ESP) = 727 Hz fundamental
from the alternating readout, 3rd harmonic 2.2 kHz), flow 99. Expected real-world order: EPI loudest (~110+ dB(A) at 3 T), SE and UTE
quieter, so the toy gain under-states EPI/GRE relative to the flow sequence: a calibration point to take from the listening score.

## Update 2026-11-23 (later): literature pass, 10 s windows

See `research/acoustic_literature.md`. Model now: literature-informed toy (resonances 0.7/1.25/1.7/2.4 kHz x,y; z 1.275 kHz;
broad floor modes), per-axis delay, reverb (default 0.4 s / −12 dB), IEC / SMRT limit flags, ATF sweep + fit tools
(synthetic round trip: |H| error 3.2 dB rms → 0.13 dB, noise-free, same model class: optimistic), music test.
`tools/seq/run_acoustic_demos.sh` regenerates the demo sequences and the 10 s listen/ folder (6 files).
Beating (not an artefact of patching TRs: each window is one continuous waveform through one transform): sidebands of the TR rate
from TR-to-TR changes (golden-angle direction 137.5°/spoke in UTE → lines at k·200 ± 76.4 Hz, envelope 47.2 Hz as measured),
slice bursts (EPI 10 Hz), rotation sets (flow, ≈ 2 Hz) that land inside one resonance's bandwidth.
