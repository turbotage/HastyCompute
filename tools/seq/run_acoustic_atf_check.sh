#!/bin/bash
# Synthetic round trip of the ATF calibration chain (tools/seq/acoustic_atf.py): a perturbed "true" scanner model plays
# the sweep sequences through acousticsim (this stands in for the microphone recordings); the fit sees only the
# recordings and the sweep description, starts from peak picking, and is compared with the truth.
set -e
cd "$(dirname "$0")/../.."
PY=data/hasty/venvs/pypulseq/bin/python
SIM=rustlib/target/acoustic/release/acousticsim
A=${1:-data/hasty/seq/studies/acoustic/atf_check}
mkdir -p $A
$PY tools/seq/acoustic_atf.py make $A 2>&1 | grep -v -i warn
$SIM --out $A --reverb off --dump-model >/dev/null
$PY - "$A" <<'PYEOF'
import json, sys
a = sys.argv[1]
m = json.load(open(f"{a}/model_toy_3t.json"))
m["reverb"] = None
json.dump(m, open(f"{a}/model_start.json", "w"), indent=1)
# the "true" scanner: narrow modes shifted by up to 5 %, gains changed, one axis different
for ax, (df, g) in enumerate([(1.04, 0.8), (0.96, 1.2), (1.03, 0.9)]):
    for k in m["axes"][ax]:
        if k["q"] > 5:
            k["f0"] *= df
            k["peak"] *= g
m["name"] = "synthetic truth"
json.dump(m, open(f"{a}/model_true.json", "w"), indent=1)
PYEOF
for ax in x y z; do
  $SIM --seq $A/atf_$ax.seq --tr-ms 160 --tr-start 0 --ntr 148 --lead 0 --reverb off --model $A/model_true.json --out $A/rec_$ax >/dev/null
  P=$(ls $A/rec_$ax/w00_*_p.npy)
  $PY tools/seq/acoustic_atf.py fit $A $ax $P --model-in $( [ $ax = x ] && echo $A/model_start.json || echo $A/model_fit.json ) --out $A/model_fit.json 2>&1 | grep -v -i warn
done
echo "--- fitted vs truth"
$PY tools/seq/acoustic_atf.py eval $A/model_true.json $A/model_fit.json
echo "--- start (toy) vs truth"
$PY tools/seq/acoustic_atf.py eval $A/model_true.json $A/model_start.json
