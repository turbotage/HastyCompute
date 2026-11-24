# Acoustic noise of MRI gradients: literature notes (2026-11-23) and what the simulator took from them

Searched with WebSearch / WebFetch; numbers below are what the fetched pages said (several full texts were not available, so
resonance tables from vendors / papers are missing). Model code: `rustlib/crates/hasty-seq/src/acoustic.rs`; plan `research/acoustic_sim.md`.

## Model structure (all agree)

- Sound = gradient waveform filtered by a linear time-invariant transfer function per axis, summed:
  Ŷ(f) = Σ_a G_a(f) H_a(f) (Hedeen & Edelstein, MRM 1997; Wu, Kim, Khoo, Nayak, MRM 2013 PMC3795844; OPTIKS Eq. 9–10).
  Force ∝ B0 · current (Lorentz), so the pitch of the sound is the pitch of the gradient waveform (no slew / derivative in
  between). Hedeen: noise varies ≥ 10 dB with position along the bore, +1–3 dB with a patient. Wu: transfer functions
  differ strongly between subjects (30 % prediction error) and 5 cm microphone shifts (45 %), especially above 1 kHz.
- Wu 2013 (GE 3T HDxt, 40 mT/m, 150 T/m/s): transfer functions plotted re 0.1 Pa/(mT/m) = 0 dB; most acoustic energy
  500–3000 Hz; **the axes must be time-aligned** before they are combined (error 32 % → 3.9 % after correcting the shift;
  a 2-sample (0.04 ms) mismatch took < 3 % to > 10 %). → per-axis `delay` in the model.
- "MRI music" (Q. Zhang, github.com/qinweizhang/MRI_music_pub, Philips 3 T; also Case Western Bach cello suite; RESON'INT
  note table; MRF-Music, ISMRM 2016 0036): the audio waveform itself, low-passed at 2–4 kHz and scaled to 20–40 mT/m, is
  played as the gradient on x(/y), with no equalisation, and is recognisable music. So the chain is linear in G and the
  pitch is preserved; the scanner only colours the timbre. (`tools/seq/make_music_seq.py` renders this through our model.)

## Resonances and levels

- Dillinger et al. MRM 2022 (research 3 T): mechanical resonances x 1725 Hz, z 1275 Hz.
- High-performance gradient coil vibration paper: modes near 1400, 1900, 2400 Hz; x and y alike (asymmetric coils), z
  different (symmetric); a vendor allows at most 5 lock-out bands.
- Philips 3T Quiet-EPI paper: clinical EPI frequencies 1121–1580 Hz; acoustic response peak near 1800 Hz.
- Ravicz, Melcher, Kiang JASA 2000: EPI readout at 1 kHz (1.5 T, 115 dB SPL spectral peak) and 1.4 kHz (3 T, 131 dB);
  Lpk 123 / 138 dB; 2-s level 97 / 114 dB SPL; a 700 Hz component at switching transients (possible mechanical resonance);
  **noise persists 300–500 ms after the gradients stop** (coil resonance + room, reverberation time ≈ 0.5 s). → reverb.
- Measured at 3 T: peak 125.7–130.7 dB, LAeq 110–115.8 dB(A) (head insert); Foster 2000 EPI 122–131 dB SPL; shifting
  an EPI readout frequency by 200 Hz changed noise by 12 dB (McJury review 2022).
- Limits: IEC 60601-2-33 — hearing protection above 99 dB(A) rms or 140 dB peak (unweighted); ISMRM/SMRT guideline —
  protection should bring the eardrum level below 85 dB(A); earplugs 10–30 dB(A); NEMA MS 4 is the measurement procedure.

## Measurement / use in design

- OPTIKS (McCready et al., IEEE TMI 2025, arXiv 2505.07117, GE UHP / Premier): resonances from 120 ms sinusoids every 10 Hz,
  50–2000 Hz, per axis (back-EMF or field camera); ATF from 160 ms sinusoids every 10 Hz, 200–3130 Hz with a 1 kHz reference,
  two calibrated microphones at the ears, least-squares H = ΣI*O/Σ|I|², standing waves above ~1 kHz so the larger of the two
  mics was used; Y-axis ATF has strong peaks and troughs below 1 kHz. Acoustic loss λ3‖A(f)·F{g}‖²; quiet spiral on the Premier
  −5.8 / −9.2 dBA (x / y) vs time-optimal. No numeric resonance tables in the text.
- MR-RADIO (ISMRM 2022 abstract 2774): resonances (0–2 kHz) from the peaks of a phone-recorded audio spectrum; TR rule: good
  TRs are (2n−1)/(2 f_res), otherwise irrational multiples of 1/f_res; EPI frequency kept far from resonances (beat
  amplitude ∝ 1/(f_EPI² − f_res²), the even/odd-line field oscillation, not the sound).

## Taken into the code

| literature | implementation |
|---|---|
| LTI per axis + sum, ∝ B0 | `AcousticModel::h`, `simulate` (done earlier) |
| transfer-function scale 0.1 Pa/(mT/m) ref; peaks +10–20 dB | toy peaks 200–900 Pa/(T/m), valleys ≥ −15 dB via two broad floor modes |
| resonances x ≈ y ≈ 0.7/1.25/1.7/2.4 kHz, z 0.7/1.275/2.0/2.9 kHz | `toy_3t()` (x 1725, z 1275 from Dillinger) |
| axis time alignment (Wu) | `delay` per axis (s), JSON `delay_s` |
| noise outlasts the gradients 300–500 ms | `reverb` (RT60, level), CLI default 0.4 s / −12 dB, `--reverb off` |
| IEC 99 dB(A) / 140 dB; SMRT 85 dB(A) | `limits` in summary.json, flags in the console |
| OPTIKS sweep protocol, MR-RADIO peak picking | `tools/seq/acoustic_atf.py` (make / fit / eval), round trip `tools/seq/run_acoustic_atf_check.sh` |
| MRI music | `tools/seq/make_music_seq.py` |

Not taken: the vibro-acoustic FEM models (4 T insert, LMS Sysnoise), active noise cancellation, coil design for quiet
gradients; subject / microphone position dependence stays an uncertainty (±30–45 %, above 1 kHz), not modelled.
