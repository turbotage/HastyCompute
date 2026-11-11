"""Round trip of the Rust Pulseq writer (hasty-seq pulseq.rs) through pypulseq 1.5 (rustlib/.venv).

For every part of a girfqa output directory:
  - <part>_check.seq (one instance of every segment type, rotations baked in): pypulseq reads it; its gradient
    waveforms at the raster points are compared with the generator's own raster (<part>_check_grad.npy, T/m), its ADC
    sample times with <part>_check_adc.npy, the md5 signature, block count and total duration;
  - <part>.seq (the real file): if it has no ROTATIONS extension, pypulseq reads it as well (blocks, duration,
    signature, ADC count); otherwise a minimal independent parser checks the signature, block / duration sums and
    that every rotation is a unit quaternion and every block's extension chain resolves.
Writes <dir>/roundtrip.json.

    pulseq_roundtrip.py data/hasty/seq/girf_qa/protocol [--full]
"""
import hashlib, json, os, sys, time
import numpy as np
import pypulseq as pp

GAMMA = 42.576e6


def md5_ok(path):
    txt = open(path).read()
    i = txt.find('\n[SIGNATURE]')
    h = [l.split()[1] for l in txt[i:].splitlines() if l.startswith('Hash')][0]
    return hashlib.md5(txt[:i].encode()).hexdigest() == h


def mini_parse(path):
    """independent check of a .seq with extensions pypulseq 1.5.0 cannot read"""
    sect, ext, rot, blocks, ext_types = None, {}, {}, [], {}
    defs = {}
    for line in open(path):
        l = line.strip()
        if not l or l.startswith('#'):
            continue
        if l.startswith('['):
            sect = l
            continue
        if l.startswith('extension '):
            _, name, tid = l.split()
            ext_types[int(tid)] = name
            sect = name
            continue
        t = l.split()
        if sect == '[DEFINITIONS]':
            defs[t[0]] = t[1:]
        elif sect == '[BLOCKS]':
            blocks.append([int(x) for x in t])
        elif sect == '[EXTENSIONS]':
            ext[int(t[0])] = tuple(int(x) for x in t[1:])
        elif sect == 'ROTATIONS':
            rot[int(t[0])] = np.array([float(x) for x in t[1:]])
    br = float(defs['BlockDurationRaster'][0])
    dur = sum(b[1] for b in blocks) * br
    bad_chain = 0
    for b in blocks:
        e = b[7]
        n = 0
        while e:
            if e not in ext:
                bad_chain += 1
                break
            e = ext[e][2]
            n += 1
            if n > 20:
                bad_chain += 1
                break
    qn = np.array([np.linalg.norm(q) for q in rot.values()]) if rot else np.ones(1)
    return dict(n_blocks=len(blocks), duration=dur, total_def=float(defs.get('TotalDuration', [np.nan])[0]), n_rotations=len(rot),
                max_quat_norm_err=float(np.abs(qn - 1).max()), bad_ext_chains=bad_chain, ext_types=ext_types)


def ge_system():
    """pypulseq system with the file's GE / pge2 rasters (check_timing uses the Sequence's system, not the file)"""
    return pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s', grad_raster_time=4e-6, rf_raster_time=2e-6,
                   adc_raster_time=2e-6, block_duration_raster=4e-6, rf_dead_time=100e-6, rf_ringdown_time=60e-6, adc_dead_time=40e-6, B0=3.0)


def raster_from_pypulseq(seq, n, dt):
    w = seq.waveforms()
    t = np.arange(n) * dt
    out = np.zeros((n, 3))
    for a in range(3):
        if w[a].shape[1]:
            out[:, a] = np.interp(t, w[a][0], w[a][1], left=0.0, right=0.0) / GAMMA
    return out


def main():
    d = sys.argv[1]
    full = '--full' in sys.argv
    prot = json.load(open(os.path.join(d, 'protocol.json')))
    res = {}
    for p in prot['parts']:
        name = p['name']
        r = {}
        t0 = time.time()
        cs = os.path.join(d, f'{name}_check.seq')
        if os.path.exists(cs):
            seq = pp.Sequence(system=ge_system())
            seq.read(cs)
            g_ref = np.load(os.path.join(d, f'{name}_check_grad.npy'))
            dt = seq.grad_raster_time
            g = raster_from_pypulseq(seq, len(g_ref), dt)
            gm = np.abs(g_ref).max()
            r['check_blocks'] = len(seq.block_events)
            r['check_md5'] = md5_ok(cs)
            r['check_max_dg_rel'] = float(np.abs(g - g_ref).max() / gm)
            r['check_max_dg_Tm'] = float(np.abs(g - g_ref).max())
            ta, _ = seq.adc_times()
            a_ref = np.load(os.path.join(d, f'{name}_check_adc.npy'))
            r['check_adc_n'] = [int(len(ta)), int(len(a_ref))]
            r['check_adc_max_dt_ns'] = float(np.abs(np.asarray(ta) - a_ref).max() * 1e9) if len(ta) == len(a_ref) and len(ta) else None
            r['check_duration'] = [float(sum(seq.block_durations.values())), float(sum(seq.block_durations.values()))]
            ok, errs = seq.check_timing()
            r['check_timing_ok'] = bool(ok)
            r['check_timing_errors'] = [str(e)[:200] for e in errs[:3]]
        fs = os.path.join(d, f'{name}.seq')
        mp = mini_parse(fs)
        r['file_md5'] = md5_ok(fs)
        r['file'] = {k: v for k, v in mp.items() if k != 'ext_types'}
        r['file_ext_types'] = list(mp['ext_types'].values())
        if (full or mp['n_blocks'] < 60000) and 'ROTATIONS' not in mp['ext_types'].values():
            seq = pp.Sequence(system=ge_system())
            seq.read(fs)
            r['pypulseq_full'] = dict(n_blocks=len(seq.block_events), duration=float(sum(seq.block_durations.values())), n_adc=int(len(seq.adc_times()[0])))
        r['sec'] = time.time() - t0
        res[name] = r
        print(name, json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(d, 'roundtrip.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
