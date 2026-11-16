#!/bin/bash
# Operator benchmark matrix: forward + adjoint time of one (resp bin, encode) unit of the 4D-flow gradient, per
# operator formulation, resolution and setting. Drives segnufft_bench (uniform-random k-space, random sample times;
# the static 167 k-sample case reproduces the real per-set time). Writes CSV to $OUT (default results.csv).
#   unit = all samples of one encode in one respiratory bin (3.34 M at 1 mm; ∝ 1/res: readout samples ∝ matrix)
#   per-frame:      F = 20 static images of unit/20 samples each (today's operator) → time × 20
#   knot / interval: temporal B-spline operator over M knots (deg 1 linear, 3 cubic), unit/M samples per interval
# Env: QUICK=1 (1 mm only), ACBFSB=1 (let the autotune also try register blocking; default: off, it never won).
set -u
B=/home/turbotage/Documents/GitHub/HastyCompute/rustlib/target/release/segnufft_bench
OUT=${OUT:-results.csv}
export HASTY_BENCH_PLAIN=0 HASTY_BENCH_GRID_GB=${GRID_GB:-6}
[ "${ACBFSB:-0}" = 1 ] || export HASTY_SEG_ACB=1 HASTY_SEG_FSB=1
echo "res_mm,variant,deg,knots,coils,nseg,order,samples_per_image,fwd_ms,adj_ms,unit_s,coil_batch,tile_fwd,tile_adj" > "$OUT"
run() { # res dims... variant deg knots coils nseg order unit
  local res=$1 d0=$2 d1=$3 d2=$4 var=$5 deg=$6 kn=$7 nc=$8 ns=$9 ord=${10} unit=${11}
  local per mult frames knot
  case $var in
    perframe) per=$((unit / 20)); mult=20; frames=""; knot=0 ;;
    knot)     per=$((unit / kn)); mult=1; frames="$kn,$deg"; knot=1 ;;
    interval) per=$((unit / kn)); mult=1; frames="$kn,$deg"; knot=0 ;;
  esac
  local out
  out=$(HASTY_BENCH_CFG="$ns,$ord" HASTY_BENCH_FRAMES="${frames:-1,1}" HASTY_BENCH_KNOT=$knot timeout 1800 $B $d0 $d1 $d2 $per $nc 2>&1)
  local line=$(echo "$out" | grep -E "SegNufft L=" | tail -1)
  local tune=$(echo "$out" | grep -E "autotune" | tail -1)
  local f=$(echo "$line" | sed -E 's/.*forward ([0-9.]+) ms.*/\1/') a=$(echo "$line" | sed -E 's/.*adjoint ([0-9.]+) ms.*/\1/') cb=$(echo "$line" | sed -E 's/.*coil batch ([0-9]+).*/\1/')
  local tf=$(echo "$tune" | sed -E 's/.*fwd ([0-9.]+) ms.*/\1/') ta=$(echo "$tune" | sed -E 's/.*adj ([0-9.]+) ms.*/\1/')
  if [ -z "$line" ]; then echo "FAILED $res $var $deg $kn: $(echo "$out" | tail -2)" >&2; return; fi
  local u=$(python3 -c "print(f'{($f + $a) * $mult / 1000:.3f}')")
  echo "$res,$var,$deg,$kn,$nc,$ns,$ord,$per,$f,$a,$u,$cb,$tf,$ta" | tee -a "$OUT"
}
grid() { # res d0 d1 d2 unit
  local r=$1 a=$2 b=$3 c=$4 u=$5
  run $r $a $b $c perframe 0 20 22 11 2 $u
  for dk in "1 10" "1 20" "3 8" "3 10" "3 20"; do set -- $dk; run $r $a $b $c knot $1 $2 22 11 2 $u; done
  for dk in "1 10" "3 10"; do set -- $dk; run $r $a $b $c interval $1 $2 22 11 2 $u; done
}
grid 1.0 198 189 228 3340000
# settings at 1 mm: fewer segments, fewer coils, no ΔC correction
for s in "22 6 2" "12 11 2" "22 11 0"; do set -- $s
  run 1.0 198 189 228 perframe 0 20 $1 $2 $3 3340000
  run 1.0 198 189 228 knot 3 10 $1 $2 $3 3340000
  run 1.0 198 189 228 knot 1 10 $1 $2 $3 3340000
done
if [ "${QUICK:-0}" != 1 ]; then
  grid 1.5 132 126 152 2230000
  grid 2.0 99 95 114 1670000
fi
