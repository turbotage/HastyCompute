"""Valve disease of a phantom (research/flow_phantom.md, valve disease): one spec read by the 1D network (root
waveforms), the cardiac model (atrioventricular flows, chamber volumes) and the LBM case builder (valve geometry,
velocity scale, outlet kinds), so the three stay consistent.

Spec: comma-separated `<valve>:<lesion>=<area cm²>` or the short forms
    mr / tr / ar / pr = <EROA cm²>     regurgitation (mitral, tricuspid, aortic, pulmonary)
    ms / ts / as / ps = <area cm²>     stenosis: the effective open orifice area (continuity / Gorlin, as graded)
    bav = <offset>                     bicuspid aortic valve: two cusps, coaptation line at the offset (units of R)
e.g. "mr=0.3", "as=0.9,ar=0.15", or a JSON file / string {"mitral": {"regurgitation": 0.3}, ...}.
Severity guide (adult, ASE/EACVI): MR/AR EROA mild < 0.2, moderate 0.2–0.39, severe ≥ 0.4 (AR ≥ 0.3) cm²; AS area
mild > 1.5, moderate 1.0–1.5, severe < 1.0 cm²; MS area mild > 1.5, moderate 1.0–1.5, severe < 1.0 cm².

Kinematic model (the LBM is driven by the wall motion, it has no physiological pressures):
- regurgitant flow Q(t) = 0.8 EROA · √(2 ΔP(t) / ρ) (EROA is the peak orifice, its cycle mean ≈ 0.8 of it) with ΔP the driving pressure from simple chamber / artery pressure
  waveforms (ventricle − atrium over isovolumic contraction, ejection and isovolumic relaxation for MR / TR; artery −
  ventricle over diastole for AR / PR), from the phantom's mean systemic / pulmonary pressures;
- the forward stroke stays the network's SV, the ventricle ejects SV + RVol (MR/TR: the excess goes back to the atrium
  in systole, the ventricle fills SV + RVol; AR/PR: the root flow reverses in diastole);
- AV stenosis: the E wave decelerates with the pressure half-time PHT = 220 ms / area (Hatle), DT = PHT / 0.29;
- semilunar stenosis: geometry only (the jet follows from the orifice and the prescribed ejection).
The LBM case opens the leaflets to θ with πR²(1 − cos θ) = area / 0.75 (the geometric orifice; the closed strip fan
tiles the disk, a strip turned by θ projects cos θ of its area), cuts the closed valve's hole of the anatomic ROA = EROA / 0.75 (contraction), and gives
the affected side's arterial tree no pressure outlet: every cut prescribed, so mass conservation forces the
regurgitant volume through the hole.
"""
import json
from pathlib import Path

import numpy as np

MMHG = 133.322
RHO = 1060.0
SHORT = {"mr": ("mitral", "regurgitation"), "tr": ("tricuspid", "regurgitation"), "ar": ("aortic", "regurgitation"),
         "pr": ("pulmonary", "regurgitation"), "ms": ("mitral", "stenosis"), "ts": ("tricuspid", "stenosis"),
         "as": ("aortic", "stenosis"), "ps": ("pulmonary", "stenosis"), "bav": ("aortic", "bicuspid")}
VALVES = ("mitral", "tricuspid", "aortic", "pulmonary")
CC = 0.75  # contraction coefficient: anatomic orifice = effective / CC
# EROA is the peak (PISA at peak velocity); the orifice varies through the cycle, its mean is lower: with 0.8 the
# grading thresholds line up (MR EROA 0.4 cm² ↔ RVol ≈ 60 mL, 0.2 ↔ ≈ 30 mL at 70 bpm, RVol / EROA ≈ 150 cm = the
# usual CW VTI)
DYN = 0.8


def parse(spec):
    """{valve: {lesion: area_cm2}} from a spec string, a JSON string / file, a dict or None"""
    if spec is None or spec == "" or spec == {}:
        return {}
    if isinstance(spec, dict):
        d = spec
    else:
        s = str(spec).strip()
        if s.startswith("{"):
            d = json.loads(s)
        elif s.endswith(".json") and Path(s).exists():
            d = json.loads(Path(s).read_text())
        else:
            d = {}
            for item in s.split(","):
                k, v = item.split("=")
                k = k.strip().lower()
                valve, lesion = SHORT[k] if k in SHORT else tuple(k.split(":"))
                d.setdefault(valve, {})[lesion] = float(v)
    out = {}
    for valve, les in d.items():
        assert valve in VALVES, f"valve disease: unknown valve {valve}"
        for lesion, a in les.items():
            assert lesion in ("regurgitation", "stenosis", "bicuspid"), f"valve disease: unknown lesion {lesion}"
            assert lesion == "bicuspid" or float(a) > 0, f"valve disease: {valve} {lesion} area must be > 0"
            assert lesion != "bicuspid" or (valve == "aortic" and abs(float(a)) < 0.9), "bicuspid: aortic, coaptation offset |y| < 0.9 R"
            out.setdefault(valve, {})[lesion] = float(a)
    return out


def side(valve):
    return "left" if valve in ("mitral", "aortic") else "right"


def regurgitant_sides(d):
    return sorted({side(v) for v, les in d.items() if "regurgitation" in les})


def _bump(t, t0, t1, T):
    """smooth 0→1→0 over [t0, t1] (periodic in T, t0 may be negative)"""
    x = ((t - t0) % T) / max(t1 - t0, 1e-9)
    return np.where(x < 1, np.sin(np.pi * np.clip(x, 0, 1)) ** 2, 0.0)


def pressures(t, T, tej, p_mean_art, left=True):
    """ventricle, artery and atrium pressure waveforms (mmHg) at times t (t = 0: onset of ejection)"""
    t = np.asarray(t, float)
    pp = 40.0 if left else 14.0  # pulse pressure
    p_sys, p_dia = p_mean_art + pp * 2 / 3, p_mean_art - pp / 3
    p_atr = 8.0 if left else 4.0
    ivc, ivr = 0.05, 0.08
    tt_ = t % T
    # artery: ejection rise to systolic, then exponential diastolic decay to p_dia at the next ejection
    ej = (tt_ >= 0) & (tt_ < tej)
    art = np.where(ej, p_dia + (p_sys - p_dia) * np.sin(np.pi * np.clip(tt_ / tej, 0, 1)) ** 0.7, 0.0)
    p_ic = p_dia + (p_sys - p_dia) * 0.75  # dicrotic notch
    tau = (T - tej) / np.log(p_ic / p_dia)
    art = np.where(ej, art, p_ic * np.exp(-(tt_ - tej) / tau))
    # ventricle: isovolumic contraction (end of the cycle) rises from the atrium's to the artery's, follows the artery
    # in ejection, isovolumic relaxation falls to the atrium's
    ven = np.full_like(tt_, p_atr)
    ic = tt_ > T - ivc
    ven = np.where(ic, p_atr + (p_dia - p_atr) * np.sin(0.5 * np.pi * (tt_ - (T - ivc)) / ivc) ** 2, ven)
    ven = np.where(ej, art, ven)
    ir = (tt_ >= tej) & (tt_ < tej + ivr)
    ven = np.where(ir, p_atr + (p_ic - p_atr) * np.cos(0.5 * np.pi * (tt_ - tej) / ivr) ** 2, ven)
    return ven, art, np.full_like(tt_, p_atr)


def regurgitant_flow(valve, eroa_cm2, t, T, tej, p_mean_art):
    """regurgitant flow (m³/s, ≥ 0, backwards through the valve) at times t"""
    ven, art, atr = pressures(t, T, tej, p_mean_art, left=side(valve) == "left")
    dp = np.clip(ven - atr, 0, None) if valve in ("mitral", "tricuspid") else np.clip(art - ven, 0, None)
    if valve in ("aortic", "pulmonary"):
        tt_ = np.asarray(t, float) % T
        dp = np.where((tt_ >= tej) & (tt_ < T - 0.05), dp, 0.0)  # closed valve: diastole (not isovolumic contraction)
    return DYN * eroa_cm2 * 1e-4 * np.sqrt(2 * dp * MMHG / RHO)


def regurgitant_volume(valve, eroa_cm2, T, tej, p_mean_art, n=2048):
    t = np.arange(n) * T / n
    return float(regurgitant_flow(valve, eroa_cm2, t, T, tej, p_mean_art).sum() * T / n)


def pht_s(area_cm2):
    """mitral / tricuspid pressure half-time (Hatle: 220 ms / area), 60–400 ms"""
    return float(np.clip(0.220 / area_cm2, 0.06, 0.40))


def tej_left(T):
    return 0.30 * (T / 0.857) ** 0.5  # as network_flow.py / cardiac_motion.py


def tej_right(T):
    return 0.33 * (T / 0.857) ** 0.5


def totals(d, T, p_sys, p_pul):
    """regurgitant volumes (m³ per beat) per valve"""
    out = {}
    for v, les in d.items():
        if "regurgitation" in les:
            left = side(v) == "left"
            out[v] = regurgitant_volume(v, les["regurgitation"], T, tej_left(T) if left else tej_right(T), p_sys if left else p_pul)
    return out


def open_angle(area_cm2, radius_m, theta_normal):
    """leaflet opening angle for an open orifice area (strip fan: πR²(1 − cos θ)), at most the normal one"""
    f = area_cm2 * 1e-4 / (np.pi * radius_m**2)
    return float(min(np.arccos(np.clip(1 - f, -1, 1)), theta_normal))


def closed_hole(valve, eroa_cm2, radius_m, yc=0.2):
    """the closed valve's regurgitant orifice in valve-local units of R: centre (x, y), semi-axes (a, b). Mitral: a slit
    along the coaptation line (y = yc), aspect 3 (the central A2–P2 jet); the others central, aspect 1.5"""
    A = eroa_cm2 * 1e-4 / CC / radius_m**2  # anatomic area / R²
    asp = 3.0 if valve == "mitral" else 1.5
    b = np.sqrt(A / (np.pi * asp))
    return [0.0, yc if valve == "mitral" else 0.0, float(asp * b), float(b)]


def describe(d, T=None, p_sys=93.0, p_pul=15.0):
    if not d:
        return "no valve disease"
    parts = []
    rv = totals(d, T, p_sys, p_pul) if T else {}
    for v, les in d.items():
        for lesion, a in les.items():
            if lesion == "bicuspid":
                parts.append(f"{v} bicuspid (coaptation offset {a:g} R)")
                continue
            parts.append(f"{v} {lesion} {a:g} cm²" + (f" (RVol {rv[v] * 1e6:.0f} mL)" if lesion == "regurgitation" and v in rv else ""))
    return ", ".join(parts)


if __name__ == "__main__":
    import sys

    d = parse(sys.argv[1])
    T = 60 / float(sys.argv[2]) if len(sys.argv) > 2 else 0.857
    print(json.dumps(d), "→", describe(d, T))
