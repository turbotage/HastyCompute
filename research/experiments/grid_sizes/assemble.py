#!/usr/bin/env python3
"""Assemble §18 from the scratchpad drafts and the generated tables; writes s18_full.md next to the drafts."""
import os, subprocess, re, sys
D = os.path.dirname(os.path.abspath(__file__))
G = '/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gs'
rd = lambda f: open(os.path.join(G, f)).read()
run = lambda s: subprocess.run([sys.executable, os.path.join(D, s)], capture_output=True, text=True).stdout
tm = run('tables_md.py')
sec = {}
for part in tm.split('### ')[1:]:
    title, body = part.split('\n', 1)
    sec[title.strip()] = body.strip()
res = rd('s18_results.md')
i = res.index('### 18.6')
s = rd('s18_head.md').replace('added here (18.4)', 'added here (18.5)') + res[:i] + rd('s18_fft.md') + rd('s18_built.md') + res[i:] + rd('s18_tail_gs.md')
s = s.replace('TABLES_BA', sec['before / after (best channel batch of each)'])
s = s.replace('TABLES_CURVE', sec['gradient vs channel batch'])
s = s.replace('TABLES_BW', sec['bandwidth, memory, lower bound, variants (after)'])
s = s.replace('PLANNER_TABLE', sec.get('planner (SegSplit::planned)', '(not run)'))
s = s.replace('PROFILE_TABLE', run('prof.py').strip())
s = s.replace('FFT_TABLE', run('fftpass.py').strip())
reg = open(os.path.join(G, 'regression.md')).read().strip() if os.path.exists(os.path.join(G, 'regression.md')) else 'REGRESSION'
s = s.replace('REGRESSION', reg)
open(os.path.join(G, 's18_full.md'), 'w').write(s)
left = [m for m in ('TABLES_', 'PROFILE_TABLE', 'FFT_TABLE', 'PLANNER_TABLE', 'REGRESSION') if m in s]
print(len(s.splitlines()), 'lines; placeholders left:', left)
