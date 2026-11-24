#!/usr/bin/env python3
"""Standard demo sequences for the acoustic simulator (research/acoustic_sim.md), written with pypulseq:

  se_2d    2D spin echo, T1-like: TR 500 ms, TE 20 ms, 256 x 64, 5 mm, readout 4.1 ms
  ute_3d   3D radial UTE (centre-out, ADC from the ramp start): TR 5 ms, 3000 spokes, golden-spiral directions
  epi_ms   multi-slice GRE-EPI, 64 x 64, FOV 220 mm, 16 slices, 100 ms per slice, 8 volumes, echo spacing ~0.7 ms
  gre_2d   2D spoiled GRE: TR 8 ms, TE 3.5 ms, 256 x 128, 16 repetitions of the 128 lines

System: GE-like rasters (4 µs gradient / block, 2 µs RF / ADC), 40 mT/m, 120 T/m/s (clinical use of a 3 T whole-body
coil, below the 80 mT/m / 200 T/m/s hardware limit).

    data/hasty/venvs/pypulseq/bin/python tools/seq/make_acoustic_demos.py data/hasty/seq/acoustic_demo
"""
import sys
from pathlib import Path

import numpy as np
import pypulseq as pp

SYS = dict(
    max_grad=40, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s",
    rf_ringdown_time=60e-6, rf_dead_time=100e-6, adc_dead_time=40e-6,
    grad_raster_time=4e-6, rf_raster_time=2e-6, adc_raster_time=2e-6, block_duration_raster=4e-6,
)


def system():
    return pp.Opts(**SYS)


def ras(t, r=4e-6):
    return float(np.ceil(t / r - 1e-9) * r)


def finish(seq, path, name, extra=None):
    ok, errs = seq.check_timing()
    seq.set_definition("Name", name)
    if extra:
        for k, v in extra.items():
            seq.set_definition(k, v)
    seq.write(str(path))
    print(f"{path.name}: {len(seq.block_events)} blocks, {sum(seq.block_durations.values()) if hasattr(seq, 'block_durations') else seq.duration()[0]:.2f} s, timing ok: {ok}")
    if not ok:
        for e in errs[:5]:
            print("   ", e)


def se_2d(out):
    s = system()
    seq = pp.Sequence(s)
    fov, nx, ny, thk = 240e-3, 256, 64, 5e-3
    TR, TE = 500e-3, 20e-3
    rf90, gz90, gz90r = pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=3e-3, slice_thickness=thk, apodization=0.5, time_bw_product=4, system=s, return_gz=True, delay=s.rf_dead_time, use="excitation")
    rf180, gz180, _ = pp.make_sinc_pulse(flip_angle=np.pi, duration=3e-3, slice_thickness=thk, apodization=0.5, time_bw_product=4, phase_offset=np.pi / 2, system=s, return_gz=True, delay=s.rf_dead_time, use="refocusing")
    dk = 1 / fov
    gx = pp.make_trapezoid("x", flat_area=nx * dk, flat_time=4.096e-3, system=s)
    adc = pp.make_adc(nx, duration=gx.flat_time, delay=gx.rise_time, system=s)
    gx_pre = pp.make_trapezoid("x", area=gx.area / 2 + dk / 2 * 0, duration=1.5e-3, system=s)
    gz_cr = pp.make_trapezoid("z", area=4 * abs(gz90.area), duration=3e-3, system=s)  # crusher around the 180
    c90 = rf90.delay + pp.calc_rf_center(rf90)[0]
    c180 = rf180.delay + pp.calc_rf_center(rf180)[0]
    d90 = pp.calc_duration(rf90, gz90)
    d180 = pp.calc_duration(rf180, gz180)
    dpre = max(pp.calc_duration(gz90r), pp.calc_duration(gx_pre), 1.5e-3)
    dcr = pp.calc_duration(gz_cr)
    dly1 = ras(TE / 2 - (d90 - c90) - dpre - dcr - c180)
    dly2 = ras(TE / 2 - (d180 - c180) - dcr - (gx.rise_time + gx.flat_time / 2))
    assert dly1 >= 0 and dly2 >= 0, (dly1, dly2)
    gz90r.delay = 0
    for i in range(ny):
        gy = pp.make_trapezoid("y", area=(i - ny / 2) * dk, duration=1.5e-3, system=s)
        gy_re = pp.make_trapezoid("y", area=-(i - ny / 2) * dk, duration=1.5e-3, system=s)
        t0 = 0.0
        seq.add_block(rf90, gz90); t0 += d90
        seq.add_block(gz90r, gx_pre, gy); t0 += dpre
        seq.add_block(pp.make_delay(dly1)); t0 += dly1
        seq.add_block(gz_cr); t0 += dcr
        seq.add_block(rf180, gz180); t0 += d180
        seq.add_block(gz_cr); t0 += dcr
        seq.add_block(pp.make_delay(dly2)); t0 += dly2
        seq.add_block(gx, adc); t0 += pp.calc_duration(gx, adc)
        gz_sp = pp.make_trapezoid("z", area=4 * abs(gz90.area), duration=3e-3, system=s)
        seq.add_block(gz_sp, gy_re); t0 += max(pp.calc_duration(gz_sp), pp.calc_duration(gy_re))
        seq.add_block(pp.make_delay(ras(TR - t0)))
    finish(seq, out / "se_2d.seq", "SE 2D T1 TR500 TE20 256x64")


def ute_3d(out):
    s = system()
    seq = pp.Sequence(s)
    n, TR = 3000, 5e-3
    A, ft = 12e-3, 0.7e-3
    A_hz = A * s.gamma
    rt = ras(A / 100.0)  # 100 T/m/s ramp
    rf = pp.make_block_pulse(flip_angle=np.deg2rad(10), duration=80e-6, system=s, use="excitation")
    dwell = 4e-6
    adc_delay = s.adc_dead_time
    nadc = int(np.floor((rt + ft - adc_delay) / dwell / 2) * 2)
    adc = pp.make_adc(nadc, dwell=dwell, delay=adc_delay, system=s)
    area_ro = A_hz * (rt + ft)
    a_sp = 20e-3 * s.gamma
    rt_sp = ras(20e-3 / 100.0)
    ft_sp = ras(1.5 * area_ro / a_sp - rt_sp)
    ga = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z * z)
        d = np.array([r * np.cos(i * ga), r * np.sin(i * ga), z])
        seq.add_block(rf)
        seq.add_block(
            *[pp.make_trapezoid(ch, amplitude=a * d[k], rise_time=rt, flat_time=ft, fall_time=rt, system=s) for k, (ch, a) in enumerate(zip("xyz", [A_hz] * 3))],
            adc,
        )
        sp = [pp.make_trapezoid(ch, amplitude=-a_sp * d[k], rise_time=rt_sp, flat_time=ft_sp, fall_time=rt_sp, system=s) for k, ch in enumerate("xyz")]
        seq.add_block(*sp)
        used = pp.calc_duration(rf) + (rt * 2 + ft) + (rt_sp * 2 + ft_sp)
        seq.add_block(pp.make_delay(ras(TR - used)))
    finish(seq, out / "ute_3d.seq", "UTE 3D radial TR5 3000 spokes")


def epi_ms(out):
    s = system()
    seq = pp.Sequence(s)
    fov, nx, ny, thk, nsl, nvol = 220e-3, 64, 64, 4e-3, 16, 8
    tr_slice = 100e-3
    rf, gz, gzr = pp.make_sinc_pulse(flip_angle=np.deg2rad(80), duration=3e-3, slice_thickness=thk, apodization=0.5, time_bw_product=4, system=s, return_gz=True, delay=s.rf_dead_time, use="excitation")
    dk = 1 / fov
    gx = pp.make_trapezoid("x", flat_area=nx * dk, flat_time=3.84e-4, system=s)
    adc = pp.make_adc(nx, duration=gx.flat_time, delay=gx.rise_time, system=s)
    pre = 8e-4
    gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=pre, system=s)
    gy_pre = pp.make_trapezoid("y", area=-ny / 2 * dk, duration=pre, system=s)
    gzr2 = pp.make_trapezoid("z", area=-gz.area / 2, duration=pre, system=s)
    blip0 = pp.make_trapezoid("y", area=dk, system=s)
    line_dur = pp.calc_duration(gx)
    blip_delay = ras(line_dur - pp.calc_duration(blip0))
    gz_sp = pp.make_trapezoid("z", area=4 * abs(gz.area), duration=3e-3, system=s)
    print(f"  EPI: Gx {gx.amplitude / s.gamma * 1e3:.1f} mT/m, echo spacing {line_dur * 1e6:.0f} µs ({1 / line_dur:.0f} Hz)")
    for v in range(nvol):
        for sl in range(nsl):
            rf.freq_offset = gz.amplitude * thk * (sl - (nsl - 1) / 2)
            t0 = pp.calc_duration(rf, gz)
            seq.add_block(rf, gz)
            seq.add_block(gzr2, gx_pre, gy_pre)
            t0 += pre
            for i in range(ny):
                g = pp.scale_grad(gx, 1 if i % 2 == 0 else -1)
                blocks = [g, adc] if i < ny - 1 else [g, adc]
                if i < ny - 1:
                    gb = pp.make_trapezoid("y", area=dk, delay=blip_delay, system=s)
                    blocks.append(gb)
                seq.add_block(*blocks)
                t0 += line_dur
            seq.add_block(gz_sp)
            t0 += pp.calc_duration(gz_sp)
            seq.add_block(pp.make_delay(ras(tr_slice - t0)))
    finish(seq, out / "epi_ms.seq", "EPI multislice 64x64 16sl")


def gre_2d(out):
    s = system()
    seq = pp.Sequence(s)
    fov, nx, ny, thk = 240e-3, 256, 128, 5e-3
    TR, TE, nrep = 8e-3, 3.5e-3, 16
    rf, gz, gzr = pp.make_sinc_pulse(flip_angle=np.deg2rad(15), duration=1.2e-3, slice_thickness=thk, apodization=0.5, time_bw_product=4, system=s, return_gz=True, delay=s.rf_dead_time, use="excitation")
    dk = 1 / fov
    gx = pp.make_trapezoid("x", flat_area=nx * dk, flat_time=3.072e-3, system=s)
    adc = pp.make_adc(nx, duration=gx.flat_time, delay=gx.rise_time, system=s)
    gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3, system=s)
    gx_sp = pp.make_trapezoid("x", area=gx.area * 0.6, duration=1e-3, system=s)
    c = rf.delay + pp.calc_rf_center(rf)[0]
    d_exc = pp.calc_duration(rf, gz)
    for rep in range(nrep):
        for i in range(ny):
            gy = pp.make_trapezoid("y", area=(i - ny / 2) * dk, duration=1e-3, system=s)
            gy_re = pp.make_trapezoid("y", area=-(i - ny / 2) * dk, duration=1e-3, system=s)
            seq.add_block(rf, gz)
            seq.add_block(pp.make_trapezoid("z", area=gzr.area, duration=1e-3, system=s), gx_pre, gy)
            dly = ras(TE - (d_exc - c) - 1e-3 - (gx.rise_time + gx.flat_time / 2))
            assert dly >= 0, dly
            seq.add_block(pp.make_delay(dly))
            seq.add_block(gx, adc)
            seq.add_block(gx_sp, gy_re)
            used = d_exc + 1e-3 + dly + pp.calc_duration(gx, adc) + 1e-3
            seq.add_block(pp.make_delay(ras(TR - used)))
    finish(seq, out / "gre_2d.seq", "GRE 2D TR8 TE3.5 256x128")


def _gre_common(s, fov, nx, thk, flip, rf_dur, tbw, flat):
    rf, gz, gzr = pp.make_sinc_pulse(flip_angle=np.deg2rad(flip), duration=rf_dur, slice_thickness=thk, apodization=0.5, time_bw_product=tbw, system=s, return_gz=True, delay=s.rf_dead_time, use="excitation")
    dk = 1 / fov
    gx = pp.make_trapezoid("x", flat_area=nx * dk, flat_time=flat, system=s)
    adc = pp.make_adc(nx, duration=gx.flat_time, delay=gx.rise_time, system=s)
    return rf, gz, gzr, gx, adc, dk


def bssfp_2d(out):
    """balanced SSFP (FIESTA-like): TR ~4.4 ms, every gradient lobe rewound within the TR, 256 x 192, 12 repetitions"""
    s = system()
    seq = pp.Sequence(s)
    ny, nrep = 192, 20
    rf, gz, gzr, gx, adc, dk = _gre_common(s, 240e-3, 256, 5e-3, 50, 0.8e-3, 2, 1.024e-3)
    gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=0.8e-3, system=s)
    gx_post = pp.make_trapezoid("x", area=-gx.area / 2, duration=0.8e-3, system=s)
    gzr = pp.make_trapezoid("z", area=gzr.area, duration=0.8e-3, system=s)
    gz_post = pp.make_trapezoid("z", area=gzr.area, duration=0.8e-3, system=s)
    k = 0
    for rep in range(nrep):
        for i in range(ny):
            ph = (i - ny / 2) * dk
            gy = pp.make_trapezoid("y", area=ph, duration=0.8e-3, system=s)
            gy_re = pp.make_trapezoid("y", area=-ph, duration=0.8e-3, system=s)
            rf.phase_offset = np.pi * (k % 2)
            adc.phase_offset = np.pi * (k % 2)
            k += 1
            seq.add_block(rf, gz)
            seq.add_block(gzr, gx_pre, gy)
            seq.add_block(gx, adc)
            seq.add_block(gz_post, gx_post, gy_re)
    finish(seq, out / "bssfp_2d.seq", "bSSFP 2D 256x192 5 mm")


def _spgr_tr(seq, s, rf, gz, gzr, gx, adc, gx_pre, gy_amp, gz_amp, dk, ro_spoil, tr_target, phase_k):
    """one spoiled-GRE TR: exc | pre + phase encodes | readout | rewind + spoilers, padded to tr_target"""
    gy = pp.make_trapezoid("y", area=gy_amp, duration=0.8e-3, system=s)
    gyr = pp.make_trapezoid("y", area=-gy_amp, duration=0.8e-3, system=s)
    d = [pp.calc_duration(rf, gz)]
    seq.add_block(rf, gz)
    b = [pp.make_trapezoid("z", area=gzr.area, duration=0.8e-3, system=s), gx_pre, gy]
    if gz_amp is not None:
        b[0] = pp.make_trapezoid("z", area=gzr.area + gz_amp, duration=0.8e-3, system=s)
    seq.add_block(*b)
    d.append(max(pp.calc_duration(x) for x in b))
    seq.add_block(gx, adc)
    d.append(pp.calc_duration(gx, adc))
    zre = pp.make_trapezoid("z", area=-(gz_amp or 0) + 4 * abs(gzr.area), duration=1.0e-3, system=s)
    b2 = [ro_spoil, gyr, zre]
    seq.add_block(*b2)
    d.append(max(pp.calc_duration(x) for x in b2))
    pad = tr_target - sum(d)
    assert pad > -1e-9, pad
    if pad > 4e-6:
        seq.add_block(pp.make_delay(ras(pad)))
    return tr_target


def fspgr_3d(out):
    """3D fast spoiled GRE (FSPGR): TR 6 ms, TE ~2.2 ms, 256 x 96 x 24 (ky, kz looped), 1 slab"""
    s = system()
    seq = pp.Sequence(s)
    ny, nz = 96, 24
    rf, gz, gzr, gx, adc, dk = _gre_common(s, 256e-3, 256, 24e-3, 12, 1.0e-3, 4, 1.536e-3)
    gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=0.8e-3, system=s)
    ro_spoil = pp.make_trapezoid("x", area=gx.area * 0.75, duration=1.0e-3, system=s)
    for kz in range(nz):
        for ky in range(ny):
            _spgr_tr(seq, s, rf, gz, gzr, gx, adc, gx_pre, (ky - ny / 2) * dk, (kz - nz / 2) * dk * 4, dk, ro_spoil, 6e-3, None)
    finish(seq, out / "fspgr_3d.seq", "FSPGR 3D TR6 256x96x24")


def ir_fspgr_3d(out):
    """inversion-prepared FSPGR (BRAVO-like): per shot a 180 inversion + crusher, TI 450 ms, 64 spoiled-GRE TRs (6 ms), shot period 1.2 s, 10 shots"""
    s = system()
    seq = pp.Sequence(s)
    nshot, nline, ny, nz = 10, 64, 96, 24
    shot_t, ti = 1.2, 450e-3
    rf, gz, gzr, gx, adc, dk = _gre_common(s, 256e-3, 256, 24e-3, 12, 1.0e-3, 4, 1.536e-3)
    gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=0.8e-3, system=s)
    ro_spoil = pp.make_trapezoid("x", area=gx.area * 0.75, duration=1.0e-3, system=s)
    inv = pp.make_block_pulse(flip_angle=np.pi, duration=1.0e-3, system=s, use="inversion")  # non-selective stand-in for an adiabatic pulse
    crush = [pp.make_trapezoid(ch, area=8e-3 * s.gamma * 3e-3, duration=3e-3, system=s) for ch in "xyz"]
    for sh in range(nshot):
        t = pp.calc_duration(inv)
        seq.add_block(inv)
        seq.add_block(*crush)
        t += pp.calc_duration(crush[0])
        seq.add_block(pp.make_delay(ras(ti - t - nline * 6e-3 / 2)))  # TI to the centre of the train
        t0 = 0.0
        for ln in range(nline):
            idx = sh * nline + ln
            ky, kz = idx % ny, (idx // ny) % nz
            t0 += _spgr_tr(seq, s, rf, gz, gzr, gx, adc, gx_pre, (ky - ny / 2) * dk, (kz - nz / 2) * dk * 4, dk, ro_spoil, 6e-3, None)
        used = ras(ti - pp.calc_duration(inv) - 3e-3 - nline * 6e-3 / 2) + pp.calc_duration(inv) + 3e-3 + t0
        seq.add_block(pp.make_delay(ras(shot_t - used)))
    finish(seq, out / "ir_fspgr_3d.seq", "IR-FSPGR TI450 64 TR/shot")


def pcvipr_balanced(out, venc=0.8, nproj=800):
    """PC-VIPR 4D flow, balanced velocity encoding: every played encode is a row of +-1 (no zero-moment point is played; the reference is
    formed in reconstruction by subtracting one row from the others, as in Johnson & Markl, MRM 2010 / Pelc 1991):
    rows (-,-,-) (+,+,-) (+,-,+) (-,+,+), bipolar on all three axes at once, |M1| per axis = 1/(2 venc) (phase pi at venc per axis).
    Centre-out radial readout (ADC from the ramp start) along 3D golden-means directions (Chan et al., MRM 2009), 10 degree
    non-selective pulse, 4 consecutive TRs per projection, TR ~ 4.5 ms."""
    s = system()
    seq = pp.Sequence(s)
    m1 = 1.0 / (2.0 * venc)  # Hz/m * s^2 -> s/m
    G0 = 25e-3 * s.gamma
    rt = ras(25e-3 / 100.0)
    tf = ras(0.5 * (-3 * rt + np.sqrt(rt * rt + 4 * m1 / G0)))  # (tf+rt)(tf+2rt) = m1/G0
    amp = m1 / ((tf + rt) * (tf + 2 * rt))
    tt = np.array([0, rt, rt + tf, 2 * rt + tf, 3 * rt + tf, 3 * rt + 2 * tf, 4 * rt + 2 * tf])
    shape = np.array([0, 1, 1, 0, -1, -1, 0.0])
    bip_dur = tt[-1]
    print(f"  PC-VIPR: bipolar amp {amp / s.gamma * 1e3:.1f} mT/m, lobe {tf * 1e6:.0f}+2x{rt * 1e6:.0f} us, duration {bip_dur * 1e3:.2f} ms, venc {venc} m/s")
    A, ft = 14e-3, 0.9e-3
    A_hz = A * s.gamma
    rt_ro = ras(A / 100.0)
    rf = pp.make_block_pulse(flip_angle=np.deg2rad(10), duration=80e-6, system=s, use="excitation")
    adc_delay = s.adc_dead_time
    nadc = int(np.floor((rt_ro + ft - adc_delay) / 4e-6 / 2) * 2)
    adc = pp.make_adc(nadc, dwell=4e-6, delay=adc_delay, system=s)
    area_ro = A_hz * (rt_ro + ft)
    a_sp = 20e-3 * s.gamma
    rt_sp = ras(20e-3 / 100.0)
    ft_sp = ras(1.5 * area_ro / a_sp - rt_sp)
    enc = [(-1, -1, -1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]
    for i in range(nproj):
        z = 2 * ((i * 0.4656) % 1.0) - 1
        az = 2 * np.pi * ((i * 0.6823) % 1.0)
        r = np.sqrt(1 - z * z)
        d = np.array([r * np.cos(az), r * np.sin(az), z])
        for e in enc:
            seq.add_block(rf)
            if any(e):
                seq.add_block(*[pp.make_extended_trapezoid(channel=ch, amplitudes=amp * e[k] * shape, times=tt, system=s) for k, ch in enumerate("xyz")])
            else:
                seq.add_block(pp.make_delay(bip_dur))
            seq.add_block(*[pp.make_trapezoid(ch, amplitude=A_hz * d[k], rise_time=rt_ro, flat_time=ft, fall_time=rt_ro, system=s) for k, ch in enumerate("xyz")], adc)
            seq.add_block(*[pp.make_trapezoid(ch, amplitude=-a_sp * d[k], rise_time=rt_sp, flat_time=ft_sp, fall_time=rt_sp, system=s) for k, ch in enumerate("xyz")])
    tr = pp.calc_duration(rf) + bip_dur + (2 * rt_ro + ft) + (2 * rt_sp + ft_sp)
    print(f"  PC-VIPR TR {tr * 1e3:.3f} ms, {nproj * 4} TRs = {nproj * 4 * tr:.1f} s")
    finish(seq, out / "pcvipr_balanced.seq", f"PC-VIPR balanced +-1 venc {venc} m/s, {nproj} golden-means projections")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for f in (se_2d, ute_3d, epi_ms, gre_2d, bssfp_2d, fspgr_3d, ir_fspgr_3d, pcvipr_balanced):
        f(out)
