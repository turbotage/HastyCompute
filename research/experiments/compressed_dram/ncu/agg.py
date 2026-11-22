import csv, sys, collections
for f in sys.argv[1:]:
    txt = open(f).read().splitlines()
    i = next(k for k,l in enumerate(txt) if l.startswith('"ID"'))
    rows = collections.OrderedDict()
    for r in csv.DictReader(txt[i:]):
        rows.setdefault(r['ID'], {'k': r['Kernel Name'], 'g': r['Grid Size'], 'b': r['Block Size']})[r['Metric Name']] = float(r['Metric Value'].replace(',', ''))
    # group consecutive kernels by (name, grid) into phases
    agg = collections.OrderedDict()
    seq = []
    for k, m in rows.items():
        key = (m['k'], m['g'])
        a = agg.setdefault(key, [0, 0.0, 0.0, 0.0]); a[0] += 1; a[1] += m['gpu__time_duration.sum']; a[2] += m['dram__bytes_read.sum']; a[3] += m['dram__bytes_write.sum']
    print('==', f, len(rows), 'kernels')
    tt = [0,0,0]
    for (k, g), a in agg.items():
        print(f"  {k[:28]:28s} grid {g:18s} x{a[0]:4d}  {a[1]/1e6:8.2f} ms  dram R {a[2]/1e9:7.3f} W {a[3]/1e9:7.3f} GB")
        tt[0]+=a[1]; tt[1]+=a[2]; tt[2]+=a[3]
    print(f"  total {tt[0]/1e6:.2f} ms R {tt[1]/1e9:.2f} W {tt[2]/1e9:.2f} GB")
