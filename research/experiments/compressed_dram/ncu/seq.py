import csv, sys, collections
# sequence of phases: runs of identical (kernel, grid, parity-of-slab-alternation) collapsed; x/y alternate in slab mode
for f in sys.argv[1:]:
    txt = open(f).read().splitlines()
    i = next(k for k,l in enumerate(txt) if l.startswith('"ID"'))
    rows = collections.OrderedDict()
    for r in csv.DictReader(txt[i:]):
        rows.setdefault(r['ID'], {'k': r['Kernel Name'], 'g': r['Grid Size']})[r['Metric Name']] = float(r['Metric Value'].replace(',', ''))
    L = list(rows.values())
    out = []  # phases: consecutive kernels; slab loops: pattern A B A B... grouped as one phase per kernel key
    cur = None
    for m in L:
        key = (m['k'], m['g'])
        if cur and (key in cur['keys']) and len(cur['keys']) <= 2 and cur['k'][0][0]=='fft_axis' and key[0]=='fft_axis' and cur['n']>0 and key[1] != '(1000, 320, 1)' and list(cur['keys'])[0][1] != '(1000, 320, 1)':
            pass
        if cur is None or not (key in cur['keys'] or (len(cur['keys']) == 1 and key[0]=='fft_axis' and list(cur['keys'])[0][0]=='fft_axis' and '320' not in key[1] and '320' not in list(cur['keys'])[0][1])):
            cur = {'keys': collections.OrderedDict(), 'k': [key], 'n': 0}
            out.append(cur)
        a = cur['keys'].setdefault(key, [0,0.,0.,0.]); a[0]+=1; a[1]+=m['gpu__time_duration.sum']; a[2]+=m['dram__bytes_read.sum']; a[3]+=m['dram__bytes_write.sum']; cur['n']+=1
    for ph in out:
        for (k,g),a in ph['keys'].items():
            print(f"  {k[:22]:22s} {g:18s} x{a[0]:4d} {a[1]/1e6:7.2f} ms R {a[2]/1e9:7.3f} W {a[3]/1e9:7.3f} GB")
        print('  --')
