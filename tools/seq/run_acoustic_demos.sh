#!/bin/bash
# Demo sequences -> 10 s sound files (toy model, uncalibrated), one shared absolute scale + normalised copies.
#   rustlib/target/acoustic/release/acousticsim must be built:
#   CARGO_TARGET_DIR=rustlib/target/acoustic cargo build --release -j 4 -p hasty-seq --bin acousticsim
set -e
cd "$(dirname "$0")/../.."
PY=data/hasty/venvs/pypulseq/bin/python
B=rustlib/target/acoustic/release/acousticsim
D=data/hasty/seq/acoustic_demo
O=${1:-data/hasty/seq/studies/acoustic/demo}
C="--fs-db 126 --also-norm"
$PY tools/seq/make_acoustic_demos.py $D 2>&1 | grep -E "\.seq|EPI"
$PY tools/seq/make_music_seq.py $D/music 2>&1 | grep -v -i warn
run() { n=$1; shift; echo "== $n"; rm -f $O/$n/w*; $B "$@" $C --out $O/$n | grep -E "^\s+\[|^\s+[0-9.]+ Hz" | head -4; python3 tools/seq/plot_acoustic.py $O/$n 2>&1 | grep -E "figures|Error|Trace"; }
run se_2d   --seq $D/se_2d.seq  --rf-per-tr 2 --tr-start 20  --ntr 20
run gre_2d  --seq $D/gre_2d.seq --tr-start 300 --ntr 1250
run ute_3d  --seq $D/ute_3d.seq --tr-start 500 --ntr 2000
run epi_ms  --seq $D/epi_ms.seq --tr-start 16  --ntr 100
run bssfp_2d  --seq $D/bssfp_2d.seq  --tr-start 400 --ntr 2480
run fspgr_3d  --seq $D/fspgr_3d.seq  --tr-start 300 --ntr 1667
run ir_fspgr_3d --seq $D/ir_fspgr_3d.seq --tr-start 0 --ntr 530
run pcvipr_balanced --seq $D/pcvipr_balanced.seq --tr-start 400 --ntr 2212
run flow_4d --seq data/hasty/seq/girf_qa/protocol/h1_flow.seq --tr-ms 14.104 --at 300 --ntr 710
run music_x --seq $D/music/music_x.seq --tr-ms 100000 --tr-start 0 --ntr 1 --lead 0
python3 - "$O" <<'PYEOF'
import json, glob, shutil, os, sys
O = sys.argv[1]
names = {'se_2d': '1_spin_echo_2d', 'gre_2d': '2_gre_2d', 'ute_3d': '3_ute_3d_radial', 'epi_ms': '4_epi_multislice', 'bssfp_2d': '5_bssfp_2d', 'fspgr_3d': '6_fspgr_3d', 'ir_fspgr_3d': '7_ir_fspgr_3d', 'pcvipr_balanced': '8_pcvipr_balanced_pm1_flow', 'flow_4d': '9_flow_4d_yarnball', 'music_x': '10_music_ode_to_joy_x'}
os.makedirs(f'{O}/listen', exist_ok=True)
for f in glob.glob(f'{O}/listen/*.wav'): os.remove(f)
rows = []
for k, n in names.items():
    S = json.load(open(f'{O}/{k}/summary.json')); w = S['windows'][0]
    for f in glob.glob(f'{O}/{k}/w00_*.wav'):
        shutil.copy(f, f'{O}/listen/{n}' + ('_norm' if f.endswith('_norm.wav') else '_true_level') + '.wav')
    t = w['lines'][0]
    rows.append((n, w['t_end'] - w['t_start'], w['laeq'], w['lafmax'], w['lpeak_db'], t['f'], t['db_a'], w['limits']['iec_99_dba_exceeded']))
shutil.copy('data/hasty/seq/acoustic_demo/music/music_in.wav', f'{O}/listen/10_music_ode_to_joy_x_ORIGINAL_audio.wav')
with open(f'{O}/listen/README.md', 'w') as f:
    f.write("# Acoustic demo (toy model, literature-informed, UNCALIBRATED; reverb 0.4 s / -12 dB)\n\n`*_true_level.wav`: one shared scale for all files (full scale = 126 dB SPL sine), loudness comparable between files.\n`*_norm.wav`: each file normalised to 0.9 FS (timbre).\n`10_music..._ORIGINAL_audio.wav`: the melody that was played through the x gradient (no scanner colouring) for comparison.\n\n| file | s | LAeq dB(A) | LAFmax | Lpeak dB | strongest line Hz (dB(A)) | > 99 dB(A) |\n|---|---|---|---|---|---|---|\n")
    for r in rows: f.write(f"| {r[0]} | {r[1]:.1f} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.1f} | {r[5]:.0f} ({r[6]:.0f}) | {'yes' if r[7] else 'no'} |\n")
A = 'data/hasty/seq/studies/acoustic/audio2grad/ode'
if os.path.exists(A + '/achieved.wav'):
    for k in ('target', 'achieved', 'naive'): shutil.copy(f'{A}/{k}.wav', f'{O}/listen/11_audio2grad_ode_{k}.wav')
    open(f'{O}/listen/README.md', 'a').write("\n`11_audio2grad_ode_*`: tools/seq/audio2grad.py: target audio, the sound the inverse-designed 3-axis gradients give (achieved), and the naive direct drive (audio as x gradient).\n")
print(open(f'{O}/listen/README.md').read())
PYEOF
