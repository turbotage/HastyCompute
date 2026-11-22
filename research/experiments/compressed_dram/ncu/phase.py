import csv, sys, collections
# per-phase DRAM bytes of one gradient: label each kernel by its role from the sequence
def load(f):
    txt = open(f).read().splitlines()
    i = next(k for k,l in enumerate(txt) if l.startswith('"ID"'))
    rows = collections.OrderedDict()
    for r in csv.DictReader(txt[i:]):
        rows.setdefault(r['ID'], {'k': r['Kernel Name'], 'g': r['Grid Size']})[r['Metric Name']] = float(r['Metric Value'].replace(',', ''))
    return list(rows.values())
for f in sys.argv[1:]:
    L = load(f)
    # z passes have the largest grid y dim / are single launches between slab loops; identify by sequence
    adj = False; phases = collections.OrderedDict(); last_fft = None
    zcount = 0; xy_seen = 0
    ffts = [m for m in L if m['k'] == 'fft_axis']
    # the z pass grid: the fft grid that occurs exactly 2*batches times (forward + inverse) and is launched alone
    cnt = collections.Counter(m['g'] for m in ffts)
    zg = min(cnt, key=lambda g: cnt[g])
    xgs = [g for g in cnt if g != zg]
    state = 'fwd'
    for idx, m in enumerate(L):
        k, g = m['k'], m['g']
        if k.startswith('seg_adjoint') or k.startswith('seg_zero'): state = 'adj'
        if k.startswith('seg_forward') or k == 'seg_pad': state = 'fwd' if k == 'seg_pad' else state
        if k == 'fft_axis':
            if g == zg: role = 'z' if state == 'fwd' else 'z^-1'
            else:
                # first of the two non-z grids launched in a forward slab loop = x; in the inverse loop order is y then x
                role = None
                # x pass grid = the one with many (y,z) groups: grid y > 1 typically
                gy = int(g.strip('()').split(',')[1])
                isx = gy > 1
                role = ('x (pad)' if isx else 'y') if state == 'fwd' else ('x^-1 (crop)' if isx else 'y^-1')
        else:
            role = k
            if k.startswith('seg_forward'): state = 'fwd_done'
        if state == 'fwd_done' and k == 'fft_axis': pass
        key = (('F ' if state in ('fwd','fwd_done') and not k.startswith('seg_adj') else 'A ') + role)
        if k.startswith('seg_forward'): key = 'F gather ' + k
        if k.startswith('seg_adjoint') or k.startswith('seg_zero') or k == 'l2_discard': key = 'A ' + k
        if state == 'fwd_done' and k == 'fft_axis': key = ('F ' + role)
        a = phases.setdefault(key, [0, 0., 0., 0.]); a[0] += 1; a[1] += m['gpu__time_duration.sum']; a[2] += m['dram__bytes_read.sum']; a[3] += m['dram__bytes_write.sum']
        if k.startswith('seg_forward'): state = 'fwd'  # next batch's load begins
        if k == 'fft_axis' and role in ('x^-1 (crop)',): pass
    print('==', f)
    T = [0,0,0]
    for key, a in phases.items():
        print(f"  {key:34s} x{a[0]:4d} {a[1]/1e6:8.2f} ms  R {a[2]/1e9:7.2f}  W {a[3]/1e9:7.2f} GB")
        T[0]+=a[1]; T[1]+=a[2]; T[2]+=a[3]
    print(f"  {'total':34s}       {T[0]/1e6:8.2f} ms  R {T[1]/1e9:7.2f}  W {T[2]/1e9:7.2f} GB")
