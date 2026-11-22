# kernaudit summary (NVRTC 13.0)

Theoretical occupancy from ptxas registers, static + dynamic shared memory (+1 KB reserved per block) and the per-SM limits of each architecture. No GPU was used.

Architectures:

| arch | GPUs | threads / SM | warps / SM | blocks / SM | regs / SM | smem / SM | smem opt-in / block |
|---|---|---|---|---|---|---|---|
| sm_80 | A100 40/80 GB | 2048 | 64 | 32 | 65536 | 164 KB | 163 KB |
| sm_89 | RTX 4090, L40S | 1536 | 48 | 24 | 65536 | 100 KB | 99 KB |
| sm_90 | H100 SXM/PCIe, H200, GH200 | 2048 | 64 | 32 | 65536 | 228 KB | 227 KB |
| sm_100 | B200 (not a target; reference) | 2048 | 64 | 32 | 65536 | 228 KB | 227 KB |
| sm_120 | RTX 5090 | 1536 | 48 | 24 | 65536 | 100 KB | 99 KB |

Cases:

- `vel2d_o3_cb5`: velocity recon: 2-D knots (resp linear x cardiac cubic), order 3, L 10, coil batch 5 (50 ch)
- `vel2d_o3_cb10`: as vel2d_o3_cb5 with coil batch 10 (100 ch: the big-VRAM batch)
- `knot1d_o2_cb5`: 1-D cubic knots, order 2, L 10, coil batch 5
- `static_o2_cb5`: static image, order 2, L 10, coil batch 5
- `static_o0_10x10`: static image, order 0, 10 coils x L 10 in one batch (prototype 10 x 10)
- `static_o0_48x1`: static image, order 0, 48 coils x L 1 in one batch (ra nxg 2 cb 3 spill case)

## Per configuration

Cell = registers / warps per SM (limiter); `S` = spills, `K` = stack frame, `X` = does not fit, `-` = not a candidate on that arch (shared memory).

### knot1d_o2_cb5 / kl2_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg2 cb4 chk12 stages1 lbmin0 | 128 | 18000 | 128/16 (regs) K48 | 128/16 (regs) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 127/16 (regs) K48 |
| nxg2 cb4 chk12 stages1 lbmin4 | 128 | 18000 | 128/16 (regs) K48 | 128/16 (regs) K48 | 128/16 (regs) K48 | 127/16 (regs) K48 | 126/16 (regs) K48 |
| nxg2 cb4 chk12 stages2 lbmin0 | 128 | 19104 | 128/16 (regs) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 128/16 (regs) K48 | 128/16 (regs) K48 |
| nxg2 cb4 chk12 stages2 lbmin3 | 128 | 19104 | 146/12 (regs) K48 | 147/12 (regs) K48 | 147/12 (regs) K48 | 156/12 (regs) K48 | 150/12 (regs) K48 |
| nxg2 cb4 chk12 stages2 lbmin4 | 128 | 19104 | 128/16 (regs) K48 | 128/16 (regs) K48 | 128/16 (regs) K48 | 126/16 (regs) K48 | 126/16 (regs) K48 |
| nxg2 cb4 chk16 stages1 lbmin0 | 128 | 24000 | 128/16 (regs) K48 | 128/16 (regs+smem) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 127/16 (regs+smem) K48 |
| nxg2 cb4 chk16 stages1 lbmin3 | 128 | 24000 | 150/12 (regs) K48 | 149/12 (regs) K48 | 151/12 (regs) K48 | 156/12 (regs) K48 | 151/12 (regs) K48 |
| nxg2 cb4 chk16 stages1 lbmin4 | 128 | 24000 | 128/16 (regs) K48 | 128/16 (regs+smem) K48 | 128/16 (regs) K48 | 127/16 (regs) K48 | 126/16 (regs+smem) K48 |
| nxg2 cb4 chk16 stages2 lbmin0 | 128 | 25472 | 128/16 (regs) K48 | 122/12 (smem) K48 | 125/16 (regs) K48 | 128/16 (regs) K48 | 128/12 (smem) K48 |
| nxg2 cb4 chk16 stages2 lbmin4 | 128 | 25472 | 128/16 (regs) K48 | 128/12 (smem) K48 | 128/16 (regs) K48 | 126/16 (regs) K48 | 126/12 (smem) K48 |
| nxg2 cb4 chk24 stages1 lbmin0 | 128 | 36000 | 128/16 (regs+smem) K48 | 128/8 (smem) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 127/8 (smem) K48 |
| nxg2 cb4 chk24 stages1 lbmin4 | 128 | 36000 | 128/16 (regs+smem) K48 | 128/8 (smem) K48 | 128/16 (regs) K48 | 127/16 (regs) K48 | 126/8 (smem) K48 |
| nxg2 cb4 chk24 stages2 lbmin0 | 128 | 38208 | 128/16 (regs+smem) K48 | 128/8 (smem) K48 | 124/16 (regs) K48 | 128/16 (regs) K48 | 128/8 (smem) K48 |
| nxg2 cb4 chk24 stages2 lbmin4 | 128 | 38208 | 128/16 (regs+smem) K48 | 128/8 (smem) K48 | 128/16 (regs) K48 | 126/16 (regs) K48 | 126/8 (smem) K48 |
| nxg2 cb4 chk8 stages1 lbmin0 | 128 | 12000 | 128/16 (regs) K48 | 128/16 (regs) K48 | 128/16 (regs) K48 | 128/16 (regs) K48 | 125/16 (regs) K48 |
| nxg2 cb4 chk8 stages1 lbmin4 | 128 | 12000 | 128/16 (regs) K48 | 128/16 (regs) K48 | 128/16 (regs) K48 | 123/16 (regs) K48 | 124/16 (regs) K48 |
| nxg2 cb4 chk8 stages2 lbmin0 | 128 | 12736 | 128/16 (regs) K48 | 122/16 (regs) K48 | 125/16 (regs) K48 | 128/16 (regs) K48 | 127/16 (regs) K48 |
| nxg2 cb4 chk8 stages2 lbmin4 | 128 | 12736 | 128/16 (regs) K48 | 128/16 (regs) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 124/16 (regs) K48 |
| nxg4 cb4 chk12 stages1 lbmin0 | 128 | 14928 | 96/20 (regs) K48 | 80/24 (regs+smem) K48 | 80/24 (regs) K48 | 80/24 (regs) K48 | 80/24 (regs+smem) K48 |
| nxg4 cb4 chk12 stages1 lbmin4 | 128 | 14928 | 117/16 (regs) K48 | 119/16 (regs) K48 | 121/16 (regs) K48 | 117/16 (regs) K48 | 112/16 (regs) K48 |
| nxg4 cb4 chk12 stages2 lbmin0 | 128 | 16032 | 96/20 (regs) K48 | 72/20 (smem) K48 | 80/24 (regs) K48 | 80/24 (regs) K48 | 72/20 (smem) K48 |
| nxg4 cb4 chk12 stages2 lbmin3 | 128 | 16032 | 114/16 (regs) K48 | 115/16 (regs) K48 | 117/16 (regs) K48 | 133/12 (regs) K48 | 127/16 (regs) K48 |
| nxg4 cb4 chk12 stages2 lbmin4 | 128 | 16032 | 114/16 (regs) K48 | 115/16 (regs) K48 | 117/16 (regs) K48 | 117/16 (regs) K48 | 112/16 (regs) K48 |
| nxg4 cb4 chk16 stages1 lbmin0 | 128 | 19904 | 96/20 (regs) K48 | 80/16 (smem) K48 | 80/24 (regs) K48 | 72/28 (regs) S20 K64 | 72/16 (smem) K48 |
| nxg4 cb4 chk16 stages1 lbmin3 | 128 | 19904 | 117/16 (regs) K48 | 119/16 (regs+smem) K48 | 121/16 (regs) K48 | 133/12 (regs) K48 | 127/16 (regs+smem) K48 |
| nxg4 cb4 chk16 stages1 lbmin4 | 128 | 19904 | 117/16 (regs) K48 | 119/16 (regs+smem) K48 | 121/16 (regs) K48 | 117/16 (regs) K48 | 112/16 (regs+smem) K48 |
| nxg4 cb4 chk16 stages2 lbmin0 | 128 | 21376 | 96/20 (regs) K48 | 80/16 (smem) K48 | 80/24 (regs) K48 | 72/28 (regs) K48 | 72/16 (smem) K48 |
| nxg4 cb4 chk16 stages2 lbmin4 | 128 | 21376 | 114/16 (regs) K48 | 113/16 (regs+smem) K48 | 115/16 (regs) K48 | 115/16 (regs) K48 | 113/16 (regs+smem) K48 |
| nxg4 cb4 chk24 stages1 lbmin0 | 128 | 29856 | 96/20 (regs+smem) K48 | 80/12 (smem) K48 | 80/24 (regs) K48 | 72/28 (regs+smem) S20 K64 | 72/12 (smem) K48 |
| nxg4 cb4 chk24 stages1 lbmin4 | 128 | 29856 | 117/16 (regs) K48 | 119/12 (smem) K48 | 121/16 (regs) K48 | 117/16 (regs) K48 | 112/12 (smem) K48 |
| nxg4 cb4 chk24 stages2 lbmin0 | 128 | 32064 | 96/20 (regs+smem) K48 | 72/12 (smem) K48 | 80/24 (regs) K48 | 80/24 (regs) K48 | 72/12 (smem) K48 |
| nxg4 cb4 chk24 stages2 lbmin4 | 128 | 32064 | 114/16 (regs) K48 | 115/12 (smem) K48 | 117/16 (regs) K48 | 115/16 (regs) K48 | 113/12 (smem) K48 |
| nxg4 cb4 chk8 stages1 lbmin0 | 128 | 9952 | 96/20 (regs) K48 | 80/24 (regs) K48 | 80/24 (regs) K48 | 80/24 (regs) K48 | 72/28 (regs) S4 K64 |
| nxg4 cb4 chk8 stages1 lbmin4 | 128 | 9952 | 117/16 (regs) K48 | 119/16 (regs) K48 | 121/16 (regs) K48 | 115/16 (regs) K48 | 113/16 (regs) K48 |
| nxg4 cb4 chk8 stages2 lbmin0 | 128 | 10688 | 96/20 (regs) K48 | 80/24 (regs) K48 | 80/24 (regs) K48 | 80/24 (regs) K48 | 72/28 (regs) K48 |
| nxg4 cb4 chk8 stages2 lbmin4 | 128 | 10688 | 114/16 (regs) K48 | 113/16 (regs) K48 | 115/16 (regs) K48 | 113/16 (regs) K48 | 113/16 (regs) K48 |
| nxg4 cb7 chk12 stages1 lbmin0 | 128 | 18000 | 128/16 (regs) K48 | 127/16 (regs) K48 | 123/16 (regs) K48 | 125/16 (regs) K48 | 127/16 (regs) K48 |
| nxg4 cb7 chk12 stages1 lbmin4 | 128 | 18000 | 128/16 (regs) K48 | 128/16 (regs) K48 | 128/16 (regs) K48 | 128/16 (regs) K48 | 125/16 (regs) K48 |
| nxg4 cb7 chk12 stages2 lbmin0 | 128 | 19104 | 128/16 (regs) K48 | 120/16 (regs) K48 | 120/16 (regs) K48 | 127/16 (regs) K48 | 125/16 (regs) K48 |
| nxg4 cb7 chk12 stages2 lbmin3 | 128 | 19104 | 145/12 (regs) K48 | 145/12 (regs) K48 | 147/12 (regs) K48 | 151/12 (regs) K48 | 147/12 (regs) K48 |
| nxg4 cb7 chk12 stages2 lbmin4 | 128 | 19104 | 128/16 (regs) K48 | 128/16 (regs) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 125/16 (regs) K48 |
| nxg4 cb7 chk16 stages1 lbmin0 | 128 | 24000 | 128/16 (regs) K48 | 127/16 (regs+smem) K48 | 123/16 (regs) K48 | 125/16 (regs) K48 | 127/16 (regs+smem) K48 |
| nxg4 cb7 chk16 stages1 lbmin3 | 128 | 24000 | 145/12 (regs) K48 | 145/12 (regs) K48 | 149/12 (regs) K48 | 154/12 (regs) K48 | 148/12 (regs) K48 |
| nxg4 cb7 chk16 stages1 lbmin4 | 128 | 24000 | 128/16 (regs) K48 | 128/16 (regs+smem) K48 | 128/16 (regs) K48 | 126/16 (regs) K48 | 125/16 (regs+smem) K48 |
| nxg4 cb7 chk16 stages2 lbmin0 | 128 | 25472 | 128/16 (regs) K48 | 121/12 (smem) K48 | 124/16 (regs) K48 | 127/16 (regs) K48 | 125/12 (smem) K48 |
| nxg4 cb7 chk16 stages2 lbmin4 | 128 | 25472 | 128/16 (regs) K48 | 128/12 (smem) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 125/12 (smem) K48 |
| nxg4 cb7 chk24 stages1 lbmin0 | 128 | 36000 | 128/16 (regs+smem) K48 | 127/8 (smem) K48 | 123/16 (regs) K48 | 125/16 (regs) K48 | 127/8 (smem) K48 |
| nxg4 cb7 chk24 stages1 lbmin4 | 128 | 36000 | 128/16 (regs+smem) K48 | 128/8 (smem) K48 | 128/16 (regs) K48 | 126/16 (regs) K48 | 125/8 (smem) K48 |
| nxg4 cb7 chk24 stages2 lbmin0 | 128 | 38208 | 128/16 (regs+smem) K48 | 120/8 (smem) K48 | 120/16 (regs) K48 | 128/16 (regs) K48 | 128/8 (smem) K48 |
| nxg4 cb7 chk24 stages2 lbmin4 | 128 | 38208 | 128/16 (regs+smem) K48 | 128/8 (smem) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 125/8 (smem) K48 |
| nxg4 cb7 chk8 stages1 lbmin0 | 128 | 12000 | 128/16 (regs) K48 | 127/16 (regs) K48 | 123/16 (regs) K48 | 125/16 (regs) K48 | 125/16 (regs) K48 |
| nxg4 cb7 chk8 stages1 lbmin4 | 128 | 12000 | 128/16 (regs) K48 | 128/16 (regs) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 123/16 (regs) K48 |
| nxg4 cb7 chk8 stages2 lbmin0 | 128 | 12736 | 128/16 (regs) K48 | 121/16 (regs) K48 | 124/16 (regs) K48 | 126/16 (regs) K48 | 124/16 (regs) K48 |
| nxg4 cb7 chk8 stages2 lbmin4 | 128 | 12736 | 128/16 (regs) K48 | 128/16 (regs) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 123/16 (regs) K48 |

### knot1d_o2_cb5 / ra2_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 128 | 18000 | 154/12 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 127/16 (regs) K40 | 127/16 (regs) K40 |
| nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 128 | 18000 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 127/16 (regs) K40 | 126/16 (regs) K40 |
| nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 128 | 19104 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K40 | 128/16 (regs) K40 |
| nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 128 | 19104 | 150/12 (regs) K32 | 150/12 (regs) K32 | 153/12 (regs) K32 | 156/12 (regs) K40 | 150/12 (regs) K40 |
| nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 128 | 19104 | 127/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 126/16 (regs) K40 | 126/16 (regs) K40 |
| nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 128 | 28032 | 146/12 (regs) K32 | 147/12 (regs+smem) K32 | 148/12 (regs) K32 | 159/12 (regs) K32 | 158/12 (regs+smem) K32 |
| nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 128 | 28032 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 127/16 (regs) K32 | 127/12 (smem) K32 |
| nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 128 | 24000 | 154/12 (regs) K32 | 128/16 (regs+smem) K32 | 128/16 (regs) K32 | 127/16 (regs) K40 | 125/16 (regs+smem) K40 |
| nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 128 | 24000 | 154/12 (regs) K32 | 154/12 (regs) K32 | 157/12 (regs) K32 | 156/12 (regs) K40 | 153/12 (regs) K40 |
| nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 128 | 24000 | 128/16 (regs) K32 | 128/16 (regs+smem) K32 | 128/16 (regs) K32 | 127/16 (regs) K40 | 126/16 (regs+smem) K40 |
| nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 128 | 25472 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 128/16 (regs) K40 | 128/12 (smem) K40 |
| nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 128 | 25472 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 127/16 (regs) K40 | 127/12 (smem) K40 |
| nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 128 | 36000 | 154/12 (regs) K32 | 128/8 (smem) K32 | 128/16 (regs) K32 | 127/16 (regs) K40 | 125/8 (smem) K40 |
| nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 128 | 36000 | 128/16 (regs+smem) K32 | 128/8 (smem) K32 | 128/16 (regs) K32 | 127/16 (regs) K40 | 126/8 (smem) K40 |
| nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 128 | 38208 | 128/16 (regs+smem) K32 | 128/8 (smem) K32 | 128/16 (regs) K32 | 128/16 (regs) K40 | 128/8 (smem) K40 |
| nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 128 | 38208 | 127/16 (regs+smem) K32 | 128/8 (smem) K32 | 128/16 (regs) K32 | 126/16 (regs) K40 | 126/8 (smem) K40 |
| nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 128 | 12000 | 154/12 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K40 | 126/16 (regs) K40 |
| nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 128 | 12000 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 123/16 (regs) K40 | 124/16 (regs) K40 |
| nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 128 | 12736 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K40 | 128/16 (regs) K40 |
| nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 128 | 12736 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 124/16 (regs) K40 | 124/16 (regs) K40 |
| nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 128 | 14928 | 123/16 (regs) K32 | 93/20 (regs) K32 | 80/24 (regs) K32 | 80/24 (regs) K40 | 80/24 (regs+smem) K40 |
| nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 128 | 14928 | 123/16 (regs) K32 | 123/16 (regs) K32 | 123/16 (regs) K32 | 117/16 (regs) K40 | 117/16 (regs) K40 |
| nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 128 | 16032 | 120/16 (regs) K32 | 94/20 (regs+smem) K32 | 96/20 (regs) K32 | 72/28 (regs) K40 | 72/20 (smem) K40 |
| nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 128 | 16032 | 120/16 (regs) K32 | 119/16 (regs) K32 | 123/16 (regs) K32 | 133/12 (regs) K40 | 128/16 (regs) K40 |
| nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 128 | 16032 | 120/16 (regs) K32 | 119/16 (regs) K32 | 123/16 (regs) K32 | 120/16 (regs) K40 | 117/16 (regs) K40 |
| nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 128 | 23936 | 114/16 (regs) K32 | 115/16 (regs+smem) K32 | 117/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs+smem) K32 |
| nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 128 | 23936 | 114/16 (regs) K32 | 115/16 (regs+smem) K32 | 117/16 (regs) K32 | 119/16 (regs) K32 | 116/16 (regs+smem) K32 |
| nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 128 | 19904 | 123/16 (regs) K32 | 93/16 (smem) K32 | 80/24 (regs) K32 | 72/28 (regs) K40 | 72/16 (smem) K40 |
| nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 128 | 19904 | 121/16 (regs) K32 | 122/16 (regs+smem) K32 | 125/16 (regs) K32 | 133/12 (regs) K40 | 128/16 (regs+smem) K40 |
| nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 128 | 19904 | 123/16 (regs) K32 | 123/16 (regs+smem) K32 | 123/16 (regs) K32 | 117/16 (regs) K40 | 117/16 (regs+smem) K40 |
| nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 128 | 21376 | 96/20 (regs) K32 | 95/16 (smem) K32 | 80/24 (regs) K32 | 72/28 (regs) K40 | 72/16 (smem) K40 |
| nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 128 | 21376 | 118/16 (regs) K32 | 121/16 (regs+smem) K32 | 121/16 (regs) K32 | 116/16 (regs) K40 | 113/16 (regs+smem) K40 |
| nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 128 | 29856 | 123/16 (regs) K32 | 93/12 (smem) K32 | 80/24 (regs) K32 | 72/28 (regs+smem) K40 | 72/12 (smem) K40 |
| nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 128 | 29856 | 123/16 (regs) K32 | 123/12 (smem) K32 | 123/16 (regs) K32 | 117/16 (regs) K40 | 117/12 (smem) K40 |
| nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 128 | 32064 | 120/16 (regs) K32 | 94/12 (smem) K32 | 96/20 (regs) K32 | 72/28 (regs+smem) K40 | 72/12 (smem) K40 |
| nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 128 | 32064 | 120/16 (regs) K32 | 119/12 (smem) K32 | 123/16 (regs) K32 | 117/16 (regs) K40 | 112/12 (smem) K40 |
| nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 128 | 9952 | 123/16 (regs) K32 | 93/20 (regs) K32 | 80/24 (regs) K32 | 80/24 (regs) K40 | 72/28 (regs) S4 K48 |
| nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 128 | 9952 | 123/16 (regs) K32 | 123/16 (regs) K32 | 123/16 (regs) K32 | 117/16 (regs) K40 | 113/16 (regs) K40 |
| nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 128 | 10688 | 96/20 (regs) K32 | 95/20 (regs) K32 | 80/24 (regs) K32 | 80/24 (regs) K40 | 72/28 (regs) K40 |
| nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 128 | 10688 | 118/16 (regs) K32 | 121/16 (regs) K32 | 121/16 (regs) K32 | 115/16 (regs) K40 | 112/16 (regs) K40 |
| nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 128 | 18000 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 127/16 (regs) K40 | 125/16 (regs) K40 |
| nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | 18000 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K40 | 125/16 (regs) K40 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 128 | 19104 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 127/16 (regs) K40 | 125/16 (regs) K40 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 128 | 19104 | 147/12 (regs) K32 | 146/12 (regs) K32 | 151/12 (regs) K32 | 150/12 (regs) K40 | 145/12 (regs) K40 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 128 | 19104 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 125/16 (regs) K40 | 122/16 (regs) K40 |
| nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 128 | 28032 | 143/12 (regs) K32 | 144/12 (regs+smem) K32 | 145/12 (regs) K32 | 158/12 (regs) K32 | 154/12 (regs+smem) K32 |
| nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | 28032 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 124/16 (regs) K32 | 124/12 (smem) K32 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 128 | 24000 | 128/16 (regs) K32 | 128/16 (regs+smem) K32 | 128/16 (regs) K32 | 127/16 (regs) K40 | 125/16 (regs+smem) K40 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 128 | 24000 | 150/12 (regs) K32 | 150/12 (regs) K32 | 153/12 (regs) K32 | 152/12 (regs) K40 | 148/12 (regs) K40 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 128 | 24000 | 128/16 (regs) K32 | 128/16 (regs+smem) K32 | 128/16 (regs) K32 | 126/16 (regs) K40 | 125/16 (regs+smem) K40 |
| nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 128 | 25472 | 128/16 (regs) K32 | 128/12 (smem) K32 | 124/16 (regs) K32 | 128/16 (regs) K40 | 127/12 (smem) K40 |
| nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 128 | 25472 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 124/16 (regs) K40 | 125/12 (smem) K40 |
| nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 128 | 36000 | 128/16 (regs+smem) K32 | 128/8 (smem) K32 | 128/16 (regs) K32 | 127/16 (regs) K40 | 125/8 (smem) K40 |
| nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | 36000 | 128/16 (regs+smem) K32 | 128/8 (smem) K32 | 128/16 (regs) K32 | 126/16 (regs) K40 | 125/8 (smem) K40 |
| nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 128 | 38208 | 128/16 (regs+smem) K32 | 128/8 (smem) K32 | 128/16 (regs) K32 | 127/16 (regs) K40 | 125/8 (smem) K40 |
| nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 128 | 38208 | 128/16 (regs+smem) K32 | 128/8 (smem) K32 | 128/16 (regs) K32 | 125/16 (regs) K40 | 126/8 (smem) K40 |
| nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 128 | 12000 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 125/16 (regs) K40 | 123/16 (regs) K40 |
| nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 128 | 12000 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 124/16 (regs) K40 | 123/16 (regs) K40 |
| nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 128 | 12736 | 128/16 (regs) K32 | 128/16 (regs) K32 | 124/16 (regs) K32 | 127/16 (regs) K40 | 122/16 (regs) K40 |
| nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 128 | 12736 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K40 | 123/16 (regs) K40 |

### knot1d_o2_cb5 / vt2_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 128 | 24064 | 254/8 (regs) K32 | 254/8 (regs) K32 | 244/8 (regs) K32 | 254/8 (regs) K32 | 255/8 (regs) S4 K40 |
| 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 128 | 29248 | 254/8 (regs) K32 | 254/8 (regs) K32 | 244/8 (regs) K32 | 254/8 (regs) K32 | 255/8 (regs) S4 K40 |
| 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 128 | 39616 | 254/8 (regs) K32 | 254/8 (regs+smem) K32 | 244/8 (regs) K32 | 254/8 (regs) K32 | 255/8 (regs+smem) S4 K40 |
| 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 128 | 49984 | 254/8 (regs) K32 | 254/8 (regs+smem) K32 | 244/8 (regs) K32 | 254/8 (regs) K32 | 255/8 (regs+smem) S4 K40 |
| 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 256 | 30464 | 240/8 (regs) K32 | 242/8 (regs) K32 | 240/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 |
| 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 256 | 35648 | 240/8 (regs) K32 | 242/8 (regs) K32 | 240/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 |
| 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 256 | 46016 | 240/8 (regs) K32 | 242/8 (regs) K32 | 240/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 |
| 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 256 | 56384 | 240/8 (regs) K32 | 242/8 (regs+smem) K32 | 240/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs+smem) K32 |

### static_o0_10x10 / ra2_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg2 cb7 chk12 inplace1 stages1 lbmin0 | 128 | 15456 | 168/12 (regs) S40 K72 | 168/12 (regs) S40 K72 | 195/8 (regs) K32 | 159/12 (regs) K32 | 160/12 (regs) K32 |
| nxg2 cb7 chk12 inplace1 stages1 lbmin4 | 128 | 15456 | 128/16 (regs) S220 K168 | 128/16 (regs) S220 K168 | 128/16 (regs) S220 K168 | 128/16 (regs) S216 K152 | 128/16 (regs) S232 K168 |
| nxg2 cb7 chk12 inplace1 stages2 lbmin0 | 128 | 17040 | 168/12 (regs) S40 K72 | 168/12 (regs) S40 K72 | 193/8 (regs) K32 | 157/12 (regs) K32 | 157/12 (regs) K32 |
| nxg2 cb7 chk12 inplace1 stages2 lbmin3 | 128 | 17040 | 168/12 (regs) K32 | 168/12 (regs) K32 | 168/12 (regs) K32 | 158/12 (regs) K32 | 158/12 (regs) K32 |
| nxg2 cb7 chk12 inplace1 stages2 lbmin4 | 128 | 17040 | 128/16 (regs) S212 K152 | 128/16 (regs) S212 K152 | 128/16 (regs) S212 K152 | 128/16 (regs) S232 K144 | 128/16 (regs) S252 K160 |
| nxg2 cb7 chk16 inplace0 stages1 lbmin3 | 128 | 24640 | 168/12 (regs) S16 K48 | 168/12 (regs+smem) S16 K48 | 168/12 (regs) S20 K56 | 164/12 (regs) K32 | 163/12 (regs+smem) K32 |
| nxg2 cb7 chk16 inplace0 stages1 lbmin4 | 128 | 24640 | 128/16 (regs) S212 K176 | 128/12 (smem) S212 K176 | 128/16 (regs) S220 K176 | 128/16 (regs) S232 K176 | 128/12 (smem) S228 K176 |
| nxg2 cb7 chk16 inplace1 stages1 lbmin0 | 128 | 20608 | 199/8 (regs) K32 | 196/8 (regs) K32 | 200/8 (regs) K32 | 158/12 (regs) K32 | 156/12 (regs) K32 |
| nxg2 cb7 chk16 inplace1 stages1 lbmin3 | 128 | 20608 | 168/12 (regs) S20 K56 | 168/12 (regs) S20 K56 | 168/12 (regs) S20 K56 | 159/12 (regs) K32 | 157/12 (regs) K32 |
| nxg2 cb7 chk16 inplace1 stages1 lbmin4 | 128 | 20608 | 128/16 (regs) S264 K184 | 128/16 (regs+smem) S264 K184 | 128/16 (regs) S264 K184 | 128/16 (regs) S264 K168 | 128/16 (regs+smem) S288 K192 |
| nxg2 cb7 chk16 inplace1 stages2 lbmin0 | 128 | 22720 | 196/8 (regs) K32 | 193/8 (regs) K32 | 195/8 (regs) K32 | 158/12 (regs) K32 | 158/12 (regs) K32 |
| nxg2 cb7 chk16 inplace1 stages2 lbmin4 | 128 | 22720 | 128/16 (regs) S248 K160 | 128/16 (regs+smem) S248 K160 | 128/16 (regs) S248 K160 | 128/16 (regs) S268 K152 | 128/16 (regs+smem) S288 K168 |
| nxg2 cb7 chk24 inplace1 stages1 lbmin0 | 128 | 30912 | 199/8 (regs) K32 | 196/8 (regs) K32 | 200/8 (regs) K32 | 158/12 (regs) K32 | 156/12 (regs+smem) K32 |
| nxg2 cb7 chk24 inplace1 stages1 lbmin4 | 128 | 30912 | 128/16 (regs) S264 K184 | 128/12 (smem) S264 K184 | 128/16 (regs) S264 K184 | 128/16 (regs) S264 K168 | 128/12 (smem) S288 K192 |
| nxg2 cb7 chk24 inplace1 stages2 lbmin0 | 128 | 34080 | 197/8 (regs) K32 | 194/8 (regs+smem) K32 | 196/8 (regs) K32 | 158/12 (regs) K32 | 157/8 (smem) K32 |
| nxg2 cb7 chk24 inplace1 stages2 lbmin4 | 128 | 34080 | 128/16 (regs+smem) S248 K160 | 128/8 (smem) S248 K160 | 128/16 (regs) S248 K160 | 128/16 (regs) S268 K152 | 128/8 (smem) S288 K168 |
| nxg2 cb7 chk8 inplace1 stages1 lbmin0 | 128 | 14464 | 168/12 (regs) S40 K72 | 168/12 (regs) S40 K72 | 195/8 (regs) K32 | 159/12 (regs) K32 | 159/12 (regs) K32 |
| nxg2 cb7 chk8 inplace1 stages1 lbmin4 | 128 | 14464 | 128/16 (regs) S220 K168 | 128/16 (regs) S220 K168 | 128/16 (regs) S220 K168 | 128/16 (regs) S216 K152 | 128/16 (regs) S232 K168 |
| nxg2 cb7 chk8 inplace1 stages2 lbmin0 | 128 | 14464 | 168/12 (regs) S40 K72 | 168/12 (regs) S40 K72 | 192/8 (regs) K32 | 158/12 (regs) K32 | 156/12 (regs) K32 |
| nxg2 cb7 chk8 inplace1 stages2 lbmin4 | 128 | 14464 | 128/16 (regs) S212 K152 | 128/16 (regs) S212 K152 | 128/16 (regs) S212 K152 | 128/16 (regs) S232 K144 | 128/16 (regs) S252 K160 |
| nxg4 cb13 chk12 inplace1 stages1 lbmin0 | 128 | 13920 | 196/8 (regs) K32 | 193/8 (regs) K32 | 197/8 (regs) K32 | 156/12 (regs) K32 | 156/12 (regs) K32 |
| nxg4 cb13 chk12 inplace1 stages1 lbmin4 | 128 | 13920 | 128/16 (regs) S268 K200 | 128/16 (regs) S268 K200 | 128/16 (regs) S268 K208 | 128/16 (regs) S260 K176 | 128/16 (regs) S276 K200 |
| nxg4 cb13 chk12 inplace1 stages2 lbmin0 | 128 | 15504 | 196/8 (regs) K32 | 193/8 (regs) K32 | 195/8 (regs) K32 | 155/12 (regs) K32 | 155/12 (regs) K32 |
| nxg4 cb13 chk12 inplace1 stages2 lbmin3 | 128 | 15504 | 168/12 (regs) K32 | 164/12 (regs) K32 | 168/12 (regs) K32 | 158/12 (regs) K32 | 158/12 (regs) K32 |
| nxg4 cb13 chk12 inplace1 stages2 lbmin4 | 128 | 15504 | 128/16 (regs) S260 K184 | 128/16 (regs) S260 K184 | 128/16 (regs) S260 K184 | 128/16 (regs) S284 K176 | 128/16 (regs) S304 K192 |
| nxg4 cb13 chk16 inplace0 stages1 lbmin3 | 128 | 22592 | 168/12 (regs) S20 K56 | 168/12 (regs) S12 K48 | 168/12 (regs) S16 K48 | 160/12 (regs) K32 | 163/12 (regs) K32 |
| nxg4 cb13 chk16 inplace0 stages1 lbmin4 | 128 | 22592 | 128/16 (regs) S260 K208 | 128/16 (regs+smem) S260 K208 | 128/16 (regs) S316 K240 | 128/16 (regs) S276 K208 | 128/16 (regs+smem) S272 K208 |
| nxg4 cb13 chk16 inplace1 stages1 lbmin0 | 128 | 18560 | 201/8 (regs) K32 | 198/8 (regs) K32 | 202/8 (regs) K32 | 159/12 (regs) K32 | 159/12 (regs) K32 |
| nxg4 cb13 chk16 inplace1 stages1 lbmin3 | 128 | 18560 | 168/12 (regs) S20 K56 | 168/12 (regs) S16 K48 | 168/12 (regs) S24 K56 | 160/12 (regs) K32 | 159/12 (regs) K32 |
| nxg4 cb13 chk16 inplace1 stages1 lbmin4 | 128 | 18560 | 128/16 (regs) S312 K224 | 128/16 (regs) S312 K224 | 128/16 (regs) S360 K248 | 128/16 (regs) S308 K200 | 128/16 (regs) S332 K224 |
| nxg4 cb13 chk16 inplace1 stages2 lbmin0 | 128 | 20672 | 198/8 (regs) K32 | 195/8 (regs) K32 | 197/8 (regs) K32 | 155/12 (regs) K32 | 156/12 (regs) K32 |
| nxg4 cb13 chk16 inplace1 stages2 lbmin4 | 128 | 20672 | 128/16 (regs) S296 K200 | 128/16 (regs+smem) S296 K200 | 128/16 (regs) S296 K200 | 128/16 (regs) S320 K192 | 128/16 (regs+smem) S340 K200 |
| nxg4 cb13 chk24 inplace1 stages1 lbmin0 | 128 | 27840 | 201/8 (regs) K32 | 198/8 (regs) K32 | 202/8 (regs) K32 | 159/12 (regs) K32 | 159/12 (regs+smem) K32 |
| nxg4 cb13 chk24 inplace1 stages1 lbmin4 | 128 | 27840 | 128/16 (regs) S312 K224 | 128/12 (smem) S312 K224 | 128/16 (regs) S360 K248 | 128/16 (regs) S308 K200 | 128/12 (smem) S332 K224 |
| nxg4 cb13 chk24 inplace1 stages2 lbmin0 | 128 | 31008 | 199/8 (regs) K32 | 196/8 (regs) K32 | 198/8 (regs) K32 | 156/12 (regs) K32 | 157/12 (regs+smem) K32 |
| nxg4 cb13 chk24 inplace1 stages2 lbmin4 | 128 | 31008 | 128/16 (regs) S296 K200 | 128/12 (smem) S296 K200 | 128/16 (regs) S296 K200 | 128/16 (regs) S320 K192 | 128/12 (smem) S340 K200 |
| nxg4 cb13 chk8 inplace1 stages1 lbmin0 | 128 | 13440 | 196/8 (regs) K32 | 193/8 (regs) K32 | 197/8 (regs) K32 | 156/12 (regs) K32 | 157/12 (regs) K32 |
| nxg4 cb13 chk8 inplace1 stages1 lbmin4 | 128 | 13440 | 128/16 (regs) S268 K200 | 128/16 (regs) S268 K200 | 128/16 (regs) S268 K208 | 128/16 (regs) S260 K176 | 128/16 (regs) S276 K200 |
| nxg4 cb13 chk8 inplace1 stages2 lbmin0 | 128 | 13440 | 195/8 (regs) K32 | 192/8 (regs) K32 | 166/12 (regs) K32 | 162/12 (regs) K32 | 156/12 (regs) K32 |
| nxg4 cb13 chk8 inplace1 stages2 lbmin4 | 128 | 13440 | 128/16 (regs) S260 K184 | 128/16 (regs) S260 K184 | 128/16 (regs) S260 K184 | 128/16 (regs) S284 K176 | 128/16 (regs) S304 K192 |
| nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 128 | 9024 | 128/16 (regs) S8 K40 | 120/16 (regs) K32 | 120/16 (regs) K32 | 109/16 (regs) K32 | 110/16 (regs) K32 |
| nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | 9024 | 128/16 (regs) S4 K40 | 121/16 (regs) K32 | 121/16 (regs) K32 | 110/16 (regs) K32 | 110/16 (regs) K32 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 128 | 10320 | 125/16 (regs) K32 | 122/16 (regs) K32 | 122/16 (regs) K32 | 107/16 (regs) K32 | 109/16 (regs) K32 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 128 | 10320 | 138/12 (regs) K32 | 138/12 (regs) K32 | 139/12 (regs) K32 | 141/12 (regs) K32 | 136/12 (regs) K32 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 128 | 10320 | 124/16 (regs) K32 | 119/16 (regs) K32 | 119/16 (regs) K32 | 107/16 (regs) K32 | 109/16 (regs) K32 |
| nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 128 | 16064 | 136/12 (regs) K32 | 136/12 (regs) K32 | 137/12 (regs) K32 | 138/12 (regs) K32 | 132/12 (regs) K32 |
| nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | 16064 | 128/16 (regs) S8 K40 | 125/16 (regs) K32 | 122/16 (regs) K32 | 119/16 (regs) K32 | 116/16 (regs) K32 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 128 | 12032 | 128/16 (regs) S8 K40 | 120/16 (regs) K32 | 120/16 (regs) K32 | 109/16 (regs) K32 | 110/16 (regs) K32 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 128 | 12032 | 136/12 (regs) K32 | 136/12 (regs) K32 | 139/12 (regs) K32 | 142/12 (regs) K32 | 136/12 (regs) K32 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 128 | 12032 | 128/16 (regs) S4 K40 | 121/16 (regs) K32 | 121/16 (regs) K32 | 110/16 (regs) K32 | 110/16 (regs) K32 |
| nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 128 | 13760 | 128/16 (regs) K32 | 120/16 (regs) K32 | 120/16 (regs) K32 | 108/16 (regs) K32 | 109/16 (regs) K32 |
| nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 128 | 13760 | 128/16 (regs) K32 | 121/16 (regs) K32 | 122/16 (regs) K32 | 108/16 (regs) K32 | 109/16 (regs) K32 |
| nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 128 | 18048 | 128/16 (regs) S40 K72 | 122/16 (regs) K32 | 120/16 (regs) K32 | 119/16 (regs) K32 | 118/16 (regs) K32 |
| nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | 18048 | 128/16 (regs) S4 K40 | 121/16 (regs) K32 | 121/16 (regs) K32 | 115/16 (regs) K32 | 114/16 (regs) K32 |
| nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 128 | 20640 | 128/16 (regs) K32 | 118/16 (regs+smem) K32 | 119/16 (regs) K32 | 116/16 (regs) K32 | 113/16 (regs+smem) K32 |
| nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 128 | 20640 | 124/16 (regs) K32 | 119/16 (regs+smem) K32 | 119/16 (regs) K32 | 108/16 (regs) K32 | 110/16 (regs+smem) K32 |
| nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 128 | 7296 | 128/16 (regs) S8 K40 | 120/16 (regs) K32 | 120/16 (regs) K32 | 109/16 (regs) K32 | 110/16 (regs) K32 |
| nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 128 | 7296 | 128/16 (regs) S4 K40 | 121/16 (regs) K32 | 121/16 (regs) K32 | 110/16 (regs) K32 | 110/16 (regs) K32 |
| nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 128 | 7296 | 128/16 (regs) K32 | 120/16 (regs) K32 | 120/16 (regs) K32 | 108/16 (regs) K32 | 109/16 (regs) K32 |
| nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 128 | 7296 | 128/16 (regs) K32 | 121/16 (regs) K32 | 122/16 (regs) K32 | 108/16 (regs) K32 | 111/16 (regs) K32 |

### static_o0_10x10 / vs2_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 4x4x16 smax64 nwc7 nws1 vxpad0 sch12 | 224 | 15936 | 254/7 (regs) K32 | 254/7 (regs) K32 | 238/7 (regs) K32 | 180/7 (regs) K32 | 242/7 (regs) K32 |
| 4x4x16 smax64 nwc7 nws1 vxpad0 sch16 | 224 | 20416 | 254/7 (regs) K32 | 254/7 (regs) K32 | 238/7 (regs) K32 | 180/7 (regs) K32 | 242/7 (regs) K32 |
| 4x4x16 smax64 nwc7 nws1 vxpad0 sch24 | 224 | 29376 | 254/7 (regs) K32 | 254/7 (regs) K32 | 238/7 (regs) K32 | 180/7 (regs) K32 | 242/7 (regs) K32 |
| 4x4x16 smax64 nwc7 nws1 vxpad0 sch32 | 224 | 38336 | 254/7 (regs) K32 | 254/7 (regs) K32 | 238/7 (regs) K32 | 180/7 (regs) K32 | 242/7 (regs) K32 |
| 4x4x16 smax64 nwc7 nws1 vxpad1 sch12 | 224 | 16320 | 254/7 (regs) K32 | 254/7 (regs) K32 | 238/7 (regs) K32 | 254/7 (regs) K32 | 242/7 (regs) K32 |
| 4x4x16 smax64 nwc7 nws1 vxpad1 sch16 | 224 | 20928 | 254/7 (regs) K32 | 254/7 (regs) K32 | 238/7 (regs) K32 | 254/7 (regs) K32 | 242/7 (regs) K32 |
| 4x4x16 smax64 nwc7 nws1 vxpad1 sch24 | 224 | 30144 | 254/7 (regs) K32 | 254/7 (regs) K32 | 238/7 (regs) K32 | 254/7 (regs) K32 | 242/7 (regs) K32 |
| 4x4x16 smax64 nwc7 nws1 vxpad1 sch32 | 224 | 39360 | 254/7 (regs) K32 | 254/7 (regs) K32 | 238/7 (regs) K32 | 254/7 (regs) K32 | 242/7 (regs) K32 |

### static_o0_48x1 / ra2_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg2 cb3 chk12 inplace1 stages1 lbmin0 | 128 | 12096 | 128/16 (regs) S16 K48 | 128/16 (regs) S16 K48 | 128/16 (regs) K32 | 119/16 (regs) K32 | 119/16 (regs) K32 |
| nxg2 cb3 chk12 inplace1 stages1 lbmin4 | 128 | 12096 | 128/16 (regs) K32 | 128/16 (regs) K32 | 126/16 (regs) K32 | 114/16 (regs) K32 | 117/16 (regs) K32 |
| nxg2 cb3 chk12 inplace1 stages2 lbmin0 | 128 | 17328 | 128/16 (regs) K32 | 121/16 (regs) K32 | 119/16 (regs) K32 | 119/16 (regs) K32 | 116/16 (regs) K32 |
| nxg2 cb3 chk12 inplace1 stages2 lbmin3 | 128 | 17328 | 140/12 (regs) K32 | 140/12 (regs) K32 | 140/12 (regs) K32 | 134/12 (regs) K32 | 132/12 (regs) K32 |
| nxg2 cb3 chk12 inplace1 stages2 lbmin4 | 128 | 17328 | 124/16 (regs) K32 | 123/16 (regs) K32 | 121/16 (regs) K32 | 116/16 (regs) K32 | 117/16 (regs) K32 |
| nxg2 cb3 chk16 inplace0 stages1 lbmin3 | 128 | 20160 | 147/12 (regs) K32 | 147/12 (regs) K32 | 149/12 (regs) K32 | 143/12 (regs) K32 | 140/12 (regs) K32 |
| nxg2 cb3 chk16 inplace0 stages1 lbmin4 | 128 | 20160 | 128/16 (regs) K32 | 126/16 (regs+smem) K32 | 127/16 (regs) K32 | 118/16 (regs) K32 | 116/16 (regs+smem) K32 |
| nxg2 cb3 chk16 inplace1 stages1 lbmin0 | 128 | 16128 | 149/12 (regs) K32 | 128/16 (regs) S24 K56 | 149/12 (regs) K32 | 121/16 (regs) K32 | 118/16 (regs) K32 |
| nxg2 cb3 chk16 inplace1 stages1 lbmin3 | 128 | 16128 | 149/12 (regs) K32 | 149/12 (regs) K32 | 151/12 (regs) K32 | 141/12 (regs) K32 | 138/12 (regs) K32 |
| nxg2 cb3 chk16 inplace1 stages1 lbmin4 | 128 | 16128 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 118/16 (regs) K32 | 119/16 (regs) K32 |
| nxg2 cb3 chk16 inplace1 stages2 lbmin0 | 128 | 23104 | 128/16 (regs) K32 | 128/16 (regs+smem) K32 | 128/16 (regs) K32 | 119/16 (regs) K32 | 120/16 (regs+smem) K32 |
| nxg2 cb3 chk16 inplace1 stages2 lbmin4 | 128 | 23104 | 126/16 (regs) K32 | 123/16 (regs+smem) K32 | 121/16 (regs) K32 | 113/16 (regs) K32 | 118/16 (regs+smem) K32 |
| nxg2 cb3 chk24 inplace1 stages1 lbmin0 | 128 | 24192 | 158/12 (regs) K32 | 158/12 (regs) K32 | 155/12 (regs) K32 | 146/12 (regs) K32 | 147/12 (regs) K32 |
| nxg2 cb3 chk24 inplace1 stages1 lbmin4 | 128 | 24192 | 128/16 (regs) K32 | 128/16 (regs+smem) K32 | 128/16 (regs) S4 K40 | 123/16 (regs) K32 | 122/16 (regs+smem) K32 |
| nxg2 cb3 chk24 inplace1 stages2 lbmin0 | 128 | 34656 | 155/12 (regs) K32 | 153/8 (smem) K32 | 154/12 (regs) K32 | 128/16 (regs) K32 | 135/8 (smem) K32 |
| nxg2 cb3 chk24 inplace1 stages2 lbmin4 | 128 | 34656 | 128/16 (regs+smem) K32 | 128/8 (smem) K32 | 128/16 (regs) K32 | 115/16 (regs) K32 | 116/8 (smem) K32 |
| nxg2 cb3 chk8 inplace1 stages1 lbmin0 | 128 | 8064 | 128/16 (regs) K32 | 122/16 (regs) K32 | 119/16 (regs) K32 | 120/16 (regs) K32 | 113/16 (regs) K32 |
| nxg2 cb3 chk8 inplace1 stages1 lbmin4 | 128 | 8064 | 128/16 (regs) K32 | 124/16 (regs) K32 | 121/16 (regs) K32 | 114/16 (regs) K32 | 116/16 (regs) K32 |
| nxg2 cb3 chk8 inplace1 stages2 lbmin0 | 128 | 11552 | 121/16 (regs) K32 | 120/16 (regs) K32 | 122/16 (regs) K32 | 113/16 (regs) K32 | 102/16 (regs) K32 |
| nxg2 cb3 chk8 inplace1 stages2 lbmin4 | 128 | 11552 | 121/16 (regs) K32 | 122/16 (regs) K32 | 123/16 (regs) K32 | 112/16 (regs) K32 | 115/16 (regs) K32 |
| nxg4 cb3 chk12 inplace1 stages1 lbmin0 | 128 | 6816 | 110/16 (regs) K32 | 96/20 (regs) K32 | 96/20 (regs) S8 K40 | 72/28 (regs) K32 | 72/28 (regs) K32 |
| nxg4 cb3 chk12 inplace1 stages1 lbmin4 | 128 | 6816 | 110/16 (regs) K32 | 109/16 (regs) K32 | 113/16 (regs) K32 | 102/16 (regs) K32 | 99/16 (regs) K32 |
| nxg4 cb3 chk12 inplace1 stages2 lbmin0 | 128 | 9840 | 96/20 (regs) K32 | 80/24 (regs) K32 | 80/24 (regs) S8 K40 | 72/28 (regs) K32 | 79/24 (regs) K32 |
| nxg4 cb3 chk12 inplace1 stages2 lbmin3 | 128 | 9840 | 107/16 (regs) K32 | 106/16 (regs) K32 | 108/16 (regs) K32 | 112/16 (regs) K32 | 107/16 (regs) K32 |
| nxg4 cb3 chk12 inplace1 stages2 lbmin4 | 128 | 9840 | 107/16 (regs) K32 | 106/16 (regs) K32 | 108/16 (regs) K32 | 102/16 (regs) K32 | 98/16 (regs) K32 |
| nxg4 cb3 chk16 inplace0 stages1 lbmin3 | 128 | 13120 | 114/16 (regs) K32 | 114/16 (regs) K32 | 116/16 (regs) K32 | 116/16 (regs) K32 | 114/16 (regs) K32 |
| nxg4 cb3 chk16 inplace0 stages1 lbmin4 | 128 | 13120 | 114/16 (regs) K32 | 114/16 (regs) K32 | 116/16 (regs) K32 | 106/16 (regs) K32 | 103/16 (regs) K32 |
| nxg4 cb3 chk16 inplace1 stages1 lbmin0 | 128 | 9088 | 115/16 (regs) K32 | 112/16 (regs) K32 | 116/16 (regs) K32 | 72/28 (regs) K32 | 72/28 (regs) S4 K40 |
| nxg4 cb3 chk16 inplace1 stages1 lbmin3 | 128 | 9088 | 115/16 (regs) K32 | 115/16 (regs) K32 | 118/16 (regs) K32 | 119/16 (regs) K32 | 115/16 (regs) K32 |
| nxg4 cb3 chk16 inplace1 stages1 lbmin4 | 128 | 9088 | 115/16 (regs) K32 | 115/16 (regs) K32 | 116/16 (regs) K32 | 104/16 (regs) K32 | 101/16 (regs) K32 |
| nxg4 cb3 chk16 inplace1 stages2 lbmin0 | 128 | 13120 | 111/16 (regs) K32 | 108/16 (regs) K32 | 96/20 (regs) K32 | 72/28 (regs) K32 | 80/24 (regs) K32 |
| nxg4 cb3 chk16 inplace1 stages2 lbmin4 | 128 | 13120 | 111/16 (regs) K32 | 111/16 (regs) K32 | 112/16 (regs) K32 | 102/16 (regs) K32 | 99/16 (regs) K32 |
| nxg4 cb3 chk24 inplace1 stages1 lbmin0 | 128 | 13632 | 115/16 (regs) K32 | 113/16 (regs) K32 | 112/16 (regs) K32 | 80/24 (regs) K32 | 80/24 (regs+smem) K32 |
| nxg4 cb3 chk24 inplace1 stages1 lbmin4 | 128 | 13632 | 115/16 (regs) K32 | 115/16 (regs) K32 | 114/16 (regs) K32 | 108/16 (regs) K32 | 106/16 (regs) K32 |
| nxg4 cb3 chk24 inplace1 stages2 lbmin0 | 128 | 19680 | 115/16 (regs) K32 | 112/16 (regs+smem) K32 | 114/16 (regs) K32 | 80/24 (regs) K32 | 80/16 (smem) K32 |
| nxg4 cb3 chk24 inplace1 stages2 lbmin4 | 128 | 19680 | 115/16 (regs) K32 | 115/16 (regs+smem) K32 | 116/16 (regs) K32 | 106/16 (regs) K32 | 105/16 (regs+smem) K32 |
| nxg4 cb3 chk8 inplace1 stages1 lbmin0 | 128 | 4544 | 96/20 (regs) K32 | 80/24 (regs) K32 | 80/24 (regs) S4 K40 | 80/24 (regs) K32 | 72/28 (regs) K32 |
| nxg4 cb3 chk8 inplace1 stages1 lbmin4 | 128 | 4544 | 106/16 (regs) K32 | 106/16 (regs) K32 | 109/16 (regs) K32 | 104/16 (regs) K32 | 102/16 (regs) K32 |
| nxg4 cb3 chk8 inplace1 stages2 lbmin0 | 128 | 6560 | 96/20 (regs) K32 | 80/24 (regs) K32 | 80/24 (regs) S16 K48 | 64/32 (regs) K32 | 72/28 (regs) K32 |
| nxg4 cb3 chk8 inplace1 stages2 lbmin4 | 128 | 6560 | 103/16 (regs) K32 | 103/16 (regs) K32 | 105/16 (regs) K32 | 100/16 (regs) K32 | 98/16 (regs) K32 |
| nxg4 cb6 chk12 inplace1 stages1 lbmin0 | 128 | 10560 | 128/16 (regs) S4 K40 | 119/16 (regs) K32 | 119/16 (regs) K32 | 118/16 (regs) K32 | 117/16 (regs) K32 |
| nxg4 cb6 chk12 inplace1 stages1 lbmin4 | 128 | 10560 | 128/16 (regs) K32 | 120/16 (regs) K32 | 119/16 (regs) K32 | 121/16 (regs) K32 | 116/16 (regs) K32 |
| nxg4 cb6 chk12 inplace1 stages2 lbmin0 | 128 | 15792 | 128/16 (regs) K32 | 119/16 (regs) K32 | 115/16 (regs) K32 | 118/16 (regs) K32 | 118/16 (regs) K32 |
| nxg4 cb6 chk12 inplace1 stages2 lbmin3 | 128 | 15792 | 137/12 (regs) K32 | 137/12 (regs) K32 | 138/12 (regs) K32 | 135/12 (regs) K32 | 132/12 (regs) K32 |
| nxg4 cb6 chk12 inplace1 stages2 lbmin4 | 128 | 15792 | 124/16 (regs) K32 | 120/16 (regs) K32 | 117/16 (regs) K32 | 111/16 (regs) K32 | 113/16 (regs) K32 |
| nxg4 cb6 chk16 inplace0 stages1 lbmin3 | 128 | 18112 | 146/12 (regs) K32 | 146/12 (regs) K32 | 148/12 (regs) K32 | 145/12 (regs) K32 | 142/12 (regs) K32 |
| nxg4 cb6 chk16 inplace0 stages1 lbmin4 | 128 | 18112 | 128/16 (regs) K32 | 120/16 (regs) K32 | 120/16 (regs) K32 | 119/16 (regs) K32 | 124/16 (regs) K32 |
| nxg4 cb6 chk16 inplace1 stages1 lbmin0 | 128 | 14080 | 148/12 (regs) K32 | 145/12 (regs) K32 | 148/12 (regs) K32 | 119/16 (regs) K32 | 117/16 (regs) K32 |
| nxg4 cb6 chk16 inplace1 stages1 lbmin3 | 128 | 14080 | 148/12 (regs) K32 | 148/12 (regs) K32 | 150/12 (regs) K32 | 144/12 (regs) K32 | 139/12 (regs) K32 |
| nxg4 cb6 chk16 inplace1 stages1 lbmin4 | 128 | 14080 | 128/16 (regs) K32 | 125/16 (regs) K32 | 128/16 (regs) K32 | 119/16 (regs) K32 | 123/16 (regs) K32 |
| nxg4 cb6 chk16 inplace1 stages2 lbmin0 | 128 | 21056 | 128/16 (regs) K32 | 119/16 (regs+smem) K32 | 118/16 (regs) K32 | 119/16 (regs) K32 | 119/16 (regs+smem) K32 |
| nxg4 cb6 chk16 inplace1 stages2 lbmin4 | 128 | 21056 | 128/16 (regs) K32 | 120/16 (regs+smem) K32 | 119/16 (regs) K32 | 112/16 (regs) K32 | 114/16 (regs+smem) K32 |
| nxg4 cb6 chk24 inplace1 stages1 lbmin0 | 128 | 21120 | 155/12 (regs) K32 | 155/12 (regs) K32 | 154/12 (regs) K32 | 118/16 (regs) K32 | 118/16 (regs+smem) K32 |
| nxg4 cb6 chk24 inplace1 stages1 lbmin4 | 128 | 21120 | 128/16 (regs) K32 | 128/16 (regs+smem) K32 | 128/16 (regs) K32 | 122/16 (regs) K32 | 120/16 (regs+smem) K32 |
| nxg4 cb6 chk24 inplace1 stages2 lbmin0 | 128 | 31584 | 146/12 (regs) K32 | 150/12 (regs+smem) K32 | 152/12 (regs) K32 | 121/16 (regs) K32 | 121/12 (smem) K32 |
| nxg4 cb6 chk24 inplace1 stages2 lbmin4 | 128 | 31584 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 118/16 (regs) K32 | 119/12 (smem) K32 |
| nxg4 cb6 chk8 inplace1 stages1 lbmin0 | 128 | 7040 | 126/16 (regs) K32 | 122/16 (regs) K32 | 118/16 (regs) K32 | 120/16 (regs) K32 | 118/16 (regs) K32 |
| nxg4 cb6 chk8 inplace1 stages1 lbmin4 | 128 | 7040 | 124/16 (regs) K32 | 120/16 (regs) K32 | 120/16 (regs) K32 | 116/16 (regs) K32 | 108/16 (regs) K32 |
| nxg4 cb6 chk8 inplace1 stages2 lbmin0 | 128 | 10528 | 121/16 (regs) K32 | 120/16 (regs) K32 | 119/16 (regs) K32 | 113/16 (regs) K32 | 117/16 (regs) K32 |
| nxg4 cb6 chk8 inplace1 stages2 lbmin4 | 128 | 10528 | 121/16 (regs) K32 | 120/16 (regs) K32 | 121/16 (regs) K32 | 113/16 (regs) K32 | 117/16 (regs) K32 |

### static_o0_48x1 / vs2_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 4x4x16 smax32 nwc1 nws1 vxpad0 sch12 | 32 | 5952 | 248/8 (regs) K32 | 248/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 | 244/8 (regs) K32 |
| 4x4x16 smax32 nwc1 nws1 vxpad0 sch16 | 32 | 7360 | 248/8 (regs) K32 | 248/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 | 244/8 (regs) K32 |
| 4x4x16 smax32 nwc1 nws1 vxpad0 sch24 | 32 | 10176 | 248/8 (regs) K32 | 248/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 | 244/8 (regs) K32 |
| 4x4x16 smax32 nwc1 nws1 vxpad1 sch12 | 32 | 6336 | 253/8 (regs) K32 | 255/8 (regs) K32 | 168/12 (regs) S4 K40 | 246/8 (regs) K32 | 255/8 (regs) S44 K80 |
| 4x4x16 smax32 nwc1 nws1 vxpad1 sch16 | 32 | 7872 | 253/8 (regs) K32 | 255/8 (regs) K32 | 168/12 (regs) S4 K40 | 246/8 (regs) K32 | 255/8 (regs) S44 K80 |
| 4x4x16 smax32 nwc1 nws1 vxpad1 sch24 | 32 | 10944 | 253/8 (regs) K32 | 255/8 (regs+smem) K32 | 168/12 (regs) S4 K40 | 246/8 (regs) K32 | 255/8 (regs+smem) S44 K80 |
| 4x4x16 smax32 nwc1 nws2 vxpad0 sch12 | 64 | 5952 | 234/8 (regs) K32 | 236/8 (regs) K32 | 239/8 (regs) K32 | 242/8 (regs) K32 | 244/8 (regs) K32 |
| 4x4x16 smax32 nwc1 nws2 vxpad0 sch16 | 64 | 7360 | 234/8 (regs) K32 | 236/8 (regs) K32 | 239/8 (regs) K32 | 242/8 (regs) K32 | 246/8 (regs) K32 |
| 4x4x16 smax32 nwc1 nws2 vxpad0 sch24 | 64 | 10176 | 234/8 (regs) K32 | 236/8 (regs) K32 | 239/8 (regs) K32 | 242/8 (regs) K32 | 246/8 (regs) K32 |
| 4x4x16 smax32 nwc1 nws2 vxpad1 sch12 | 64 | 6336 | 240/8 (regs) K32 | 242/8 (regs) K32 | 240/8 (regs) K32 | 242/8 (regs) K32 | 242/8 (regs) K32 |
| 4x4x16 smax32 nwc1 nws2 vxpad1 sch16 | 64 | 7872 | 240/8 (regs) K32 | 242/8 (regs) K32 | 240/8 (regs) K32 | 242/8 (regs) K32 | 244/8 (regs) K32 |
| 4x4x16 smax32 nwc1 nws2 vxpad1 sch24 | 64 | 10944 | 240/8 (regs) K32 | 242/8 (regs) K32 | 240/8 (regs) K32 | 242/8 (regs) K32 | 244/8 (regs) K32 |
| 4x4x16 smax32 nwc2 nws1 vxpad0 sch12 | 64 | 7616 | 240/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 | 242/8 (regs) K32 | 234/8 (regs) K32 |
| 4x4x16 smax32 nwc2 nws1 vxpad0 sch16 | 64 | 9536 | 240/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 | 242/8 (regs) K32 | 234/8 (regs) K32 |
| 4x4x16 smax32 nwc2 nws1 vxpad0 sch24 | 64 | 13376 | 240/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 | 242/8 (regs) K32 | 234/8 (regs) K32 |
| 4x4x16 smax32 nwc2 nws1 vxpad1 sch12 | 64 | 8000 | 254/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 | 242/8 (regs) K32 | 234/8 (regs) K32 |
| 4x4x16 smax32 nwc2 nws1 vxpad1 sch16 | 64 | 10048 | 254/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 | 242/8 (regs) K32 | 234/8 (regs) K32 |
| 4x4x16 smax32 nwc2 nws1 vxpad1 sch24 | 64 | 14144 | 254/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 | 242/8 (regs) K32 | 234/8 (regs) K32 |
| 4x4x16 smax32 nwc2 nws2 vxpad0 sch12 | 128 | 7616 | 224/8 (regs) K32 | 234/8 (regs) K32 | 238/8 (regs) K32 | 180/8 (regs) K32 | 208/8 (regs) K32 |
| 4x4x16 smax32 nwc2 nws2 vxpad0 sch16 | 128 | 9536 | 224/8 (regs) K32 | 234/8 (regs) K32 | 238/8 (regs) K32 | 180/8 (regs) K32 | 208/8 (regs) K32 |
| 4x4x16 smax32 nwc2 nws2 vxpad0 sch24 | 128 | 13376 | 224/8 (regs) K32 | 234/8 (regs) K32 | 238/8 (regs) K32 | 180/8 (regs) K32 | 208/8 (regs) K32 |
| 4x4x16 smax32 nwc2 nws2 vxpad1 sch12 | 128 | 8000 | 234/8 (regs) K32 | 242/8 (regs) K32 | 238/8 (regs) K32 | 182/8 (regs) K32 | 206/8 (regs) K32 |
| 4x4x16 smax32 nwc2 nws2 vxpad1 sch16 | 128 | 10048 | 234/8 (regs) K32 | 242/8 (regs) K32 | 238/8 (regs) K32 | 182/8 (regs) K32 | 206/8 (regs) K32 |
| 4x4x16 smax32 nwc2 nws2 vxpad1 sch24 | 128 | 14144 | 234/8 (regs) K32 | 242/8 (regs) K32 | 238/8 (regs) K32 | 182/8 (regs) K32 | 206/8 (regs) K32 |
| 4x4x16 smax32 nwc3 nws1 vxpad0 sch12 | 96 | 9280 | 254/6 (regs) K32 | 254/6 (regs) K32 | 254/6 (regs) K32 | 173/6 (regs) K32 | 216/6 (regs) K32 |
| 4x4x16 smax32 nwc3 nws1 vxpad0 sch16 | 96 | 11712 | 254/6 (regs) K32 | 254/6 (regs) K32 | 254/6 (regs) K32 | 173/6 (regs) K32 | 216/6 (regs) K32 |
| 4x4x16 smax32 nwc3 nws1 vxpad0 sch24 | 96 | 16576 | 254/6 (regs) K32 | 254/6 (regs) K32 | 254/6 (regs) K32 | 173/6 (regs) K32 | 216/6 (regs) K32 |
| 4x4x16 smax32 nwc3 nws1 vxpad1 sch12 | 96 | 9664 | 254/6 (regs) K32 | 254/6 (regs) K32 | 239/6 (regs) K32 | 174/6 (regs) K32 | 168/12 (regs) S12 K48 |
| 4x4x16 smax32 nwc3 nws1 vxpad1 sch16 | 96 | 12224 | 254/6 (regs) K32 | 254/6 (regs) K32 | 239/6 (regs) K32 | 174/6 (regs) K32 | 168/12 (regs) S12 K48 |
| 4x4x16 smax32 nwc3 nws1 vxpad1 sch24 | 96 | 17344 | 254/6 (regs) K32 | 254/6 (regs) K32 | 239/6 (regs) K32 | 174/6 (regs) K32 | 168/12 (regs) S12 K48 |
| 4x4x16 smax32 nwc3 nws2 vxpad0 sch12 | 192 | 9280 | 240/6 (regs) K32 | 238/6 (regs) K32 | 238/6 (regs) K32 | 180/6 (regs) K32 | 206/6 (regs) K32 |
| 4x4x16 smax32 nwc3 nws2 vxpad0 sch16 | 192 | 11712 | 240/6 (regs) K32 | 238/6 (regs) K32 | 238/6 (regs) K32 | 180/6 (regs) K32 | 206/6 (regs) K32 |
| 4x4x16 smax32 nwc3 nws2 vxpad0 sch24 | 192 | 16576 | 240/6 (regs) K32 | 238/6 (regs) K32 | 238/6 (regs) K32 | 180/6 (regs) K32 | 206/6 (regs) K32 |
| 4x4x16 smax32 nwc3 nws2 vxpad1 sch12 | 192 | 9664 | 240/6 (regs) K32 | 242/6 (regs) K32 | 239/6 (regs) K32 | 176/6 (regs) K32 | 208/6 (regs) K32 |
| 4x4x16 smax32 nwc3 nws2 vxpad1 sch16 | 192 | 12224 | 240/6 (regs) K32 | 242/6 (regs) K32 | 239/6 (regs) K32 | 176/6 (regs) K32 | 208/6 (regs) K32 |
| 4x4x16 smax32 nwc3 nws2 vxpad1 sch24 | 192 | 17344 | 240/6 (regs) K32 | 242/6 (regs) K32 | 239/6 (regs) K32 | 176/6 (regs) K32 | 208/6 (regs) K32 |

### static_o2_cb5 / ra2_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 128 | 18000 | 128/16 (regs) K32 | 128/16 (regs) K32 | 127/16 (regs) K32 | 127/16 (regs) K32 | 127/16 (regs) K32 |
| nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 128 | 18000 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 124/16 (regs) K32 | 124/16 (regs) K32 |
| nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 128 | 19104 | 127/16 (regs) K32 | 127/16 (regs) K32 | 128/16 (regs) K32 | 126/16 (regs) K32 | 124/16 (regs) K32 |
| nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 128 | 19104 | 149/12 (regs) K32 | 146/12 (regs) K32 | 149/12 (regs) K32 | 159/12 (regs) K32 | 153/12 (regs) K32 |
| nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 128 | 19104 | 127/16 (regs) K32 | 127/16 (regs) K32 | 128/16 (regs) K32 | 121/16 (regs) K32 | 121/16 (regs) K32 |
| nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 128 | 28032 | 146/12 (regs) K32 | 147/12 (regs+smem) K32 | 151/12 (regs) K32 | 160/12 (regs) K32 | 156/12 (regs+smem) K32 |
| nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 128 | 28032 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 123/16 (regs) K32 | 123/12 (smem) K32 |
| nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 128 | 24000 | 128/16 (regs) K32 | 128/16 (regs+smem) K32 | 127/16 (regs) K32 | 127/16 (regs) K32 | 127/16 (regs+smem) K32 |
| nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 128 | 24000 | 151/12 (regs) K32 | 150/12 (regs) K32 | 152/12 (regs) K32 | 158/12 (regs) K32 | 152/12 (regs) K32 |
| nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 128 | 24000 | 128/16 (regs) K32 | 128/16 (regs+smem) K32 | 128/16 (regs) K32 | 124/16 (regs) K32 | 124/16 (regs+smem) K32 |
| nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 128 | 25472 | 128/16 (regs) K32 | 127/12 (smem) K32 | 125/16 (regs) K32 | 126/16 (regs) K32 | 124/12 (smem) K32 |
| nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 128 | 25472 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 122/16 (regs) K32 | 122/12 (smem) K32 |
| nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 128 | 36000 | 128/16 (regs+smem) K32 | 128/8 (smem) K32 | 127/16 (regs) K32 | 127/16 (regs) K32 | 127/8 (smem) K32 |
| nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 128 | 36000 | 128/16 (regs+smem) K32 | 128/8 (smem) K32 | 128/16 (regs) K32 | 124/16 (regs) K32 | 124/8 (smem) K32 |
| nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 128 | 38208 | 127/16 (regs+smem) K32 | 127/8 (smem) K32 | 128/16 (regs) K32 | 126/16 (regs) K32 | 124/8 (smem) K32 |
| nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 128 | 38208 | 127/16 (regs+smem) K32 | 127/8 (smem) K32 | 128/16 (regs) K32 | 121/16 (regs) K32 | 121/8 (smem) K32 |
| nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 128 | 12000 | 128/16 (regs) K32 | 128/16 (regs) K32 | 127/16 (regs) K32 | 128/16 (regs) K32 | 125/16 (regs) K32 |
| nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 128 | 12000 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 124/16 (regs) K32 | 123/16 (regs) K32 |
| nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 128 | 12736 | 128/16 (regs) K32 | 127/16 (regs) K32 | 125/16 (regs) K32 | 126/16 (regs) K32 | 124/16 (regs) K32 |
| nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 128 | 12736 | 128/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs) K32 | 123/16 (regs) K32 | 123/16 (regs) K32 |
| nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 128 | 14928 | 96/20 (regs) K32 | 72/24 (smem) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 | 80/24 (regs+smem) K32 |
| nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 128 | 14928 | 117/16 (regs) K32 | 118/16 (regs) K32 | 121/16 (regs) K32 | 119/16 (regs) K32 | 114/16 (regs) K32 |
| nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 128 | 16032 | 96/20 (regs) K32 | 72/24 (smem) K32 | 72/28 (regs) K32 | 80/24 (regs) K32 | 80/24 (regs+smem) K32 |
| nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 128 | 16032 | 114/16 (regs) K32 | 115/16 (regs) K32 | 116/16 (regs) K32 | 134/12 (regs) K32 | 128/16 (regs) K32 |
| nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 128 | 16032 | 114/16 (regs) K32 | 115/16 (regs) K32 | 116/16 (regs) K32 | 118/16 (regs) K32 | 114/16 (regs) K32 |
| nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 128 | 23936 | 114/16 (regs) K32 | 115/16 (regs+smem) K32 | 119/16 (regs) K32 | 128/16 (regs) K32 | 125/16 (regs+smem) K32 |
| nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 128 | 23936 | 114/16 (regs) K32 | 115/16 (regs+smem) K32 | 119/16 (regs) K32 | 120/16 (regs) K32 | 116/16 (regs+smem) K32 |
| nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 128 | 19904 | 96/20 (regs) K32 | 72/16 (smem) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 | 80/16 (smem) K32 |
| nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 128 | 19904 | 117/16 (regs) K32 | 118/16 (regs+smem) K32 | 121/16 (regs) K32 | 134/12 (regs) K32 | 128/16 (regs+smem) K32 |
| nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 128 | 19904 | 117/16 (regs) K32 | 118/16 (regs+smem) K32 | 121/16 (regs) K32 | 119/16 (regs) K32 | 114/16 (regs+smem) K32 |
| nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 128 | 21376 | 96/20 (regs) K32 | 72/16 (smem) K32 | 80/24 (regs) K32 | 72/28 (regs) K32 | 72/16 (smem) S8 K40 |
| nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 128 | 21376 | 114/16 (regs) K32 | 114/16 (regs+smem) K32 | 115/16 (regs) K32 | 121/16 (regs) K32 | 114/16 (regs+smem) K32 |
| nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 128 | 29856 | 96/20 (regs+smem) K32 | 72/12 (smem) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 | 80/12 (smem) K32 |
| nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 128 | 29856 | 117/16 (regs) K32 | 118/12 (smem) K32 | 121/16 (regs) K32 | 119/16 (regs) K32 | 114/12 (smem) K32 |
| nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 128 | 32064 | 96/20 (regs+smem) K32 | 72/12 (smem) K32 | 72/28 (regs+smem) K32 | 72/28 (regs+smem) S12 K48 | 80/12 (smem) K32 |
| nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 128 | 32064 | 114/16 (regs) K32 | 115/12 (smem) K32 | 116/16 (regs) K32 | 121/16 (regs) K32 | 114/12 (smem) K32 |
| nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 128 | 9952 | 96/20 (regs) K32 | 72/28 (regs) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 |
| nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 128 | 9952 | 117/16 (regs) K32 | 118/16 (regs) K32 | 121/16 (regs) K32 | 116/16 (regs) K32 | 114/16 (regs) K32 |
| nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 128 | 10688 | 96/20 (regs) K32 | 72/28 (regs) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 | 72/28 (regs) S8 K40 |
| nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 128 | 10688 | 114/16 (regs) K32 | 114/16 (regs) K32 | 115/16 (regs) K32 | 115/16 (regs) K32 | 112/16 (regs) K32 |
| nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 128 | 18000 | 128/16 (regs) K32 | 126/16 (regs) K32 | 123/16 (regs) K32 | 127/16 (regs) K32 | 126/16 (regs) K32 |
| nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | 18000 | 128/16 (regs) K32 | 127/16 (regs) K32 | 127/16 (regs) K32 | 124/16 (regs) K32 | 128/16 (regs) K32 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 128 | 19104 | 126/16 (regs) K32 | 121/16 (regs) K32 | 123/16 (regs) K32 | 122/16 (regs) K32 | 123/16 (regs) K32 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 128 | 19104 | 145/12 (regs) K32 | 147/12 (regs) K32 | 146/12 (regs) K32 | 153/12 (regs) K32 | 149/12 (regs) K32 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 128 | 19104 | 128/16 (regs) K32 | 125/16 (regs) K32 | 126/16 (regs) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 |
| nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 128 | 28032 | 145/12 (regs) K32 | 145/12 (regs+smem) K32 | 144/12 (regs) K32 | 157/12 (regs) K32 | 151/12 (regs+smem) K32 |
| nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | 28032 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 123/16 (regs) K32 | 123/12 (smem) K32 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 128 | 24000 | 128/16 (regs) K32 | 126/16 (regs+smem) K32 | 123/16 (regs) K32 | 127/16 (regs) K32 | 126/16 (regs+smem) K32 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 128 | 24000 | 145/12 (regs) K32 | 144/12 (regs) K32 | 148/12 (regs) K32 | 156/12 (regs) K32 | 151/12 (regs) K32 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 128 | 24000 | 128/16 (regs) K32 | 127/16 (regs+smem) K32 | 127/16 (regs) K32 | 123/16 (regs) K32 | 126/16 (regs+smem) K32 |
| nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 128 | 25472 | 127/16 (regs) K32 | 120/12 (smem) K32 | 122/16 (regs) K32 | 122/16 (regs) K32 | 123/12 (smem) K32 |
| nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 128 | 25472 | 127/16 (regs) K32 | 127/12 (smem) K32 | 128/16 (regs) K32 | 123/16 (regs) K32 | 122/12 (smem) K32 |
| nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 128 | 36000 | 128/16 (regs+smem) K32 | 126/8 (smem) K32 | 123/16 (regs) K32 | 127/16 (regs) K32 | 126/8 (smem) K32 |
| nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | 36000 | 128/16 (regs+smem) K32 | 127/8 (smem) K32 | 127/16 (regs) K32 | 123/16 (regs) K32 | 126/8 (smem) K32 |
| nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 128 | 38208 | 126/16 (regs+smem) K32 | 121/8 (smem) K32 | 123/16 (regs) K32 | 122/16 (regs) K32 | 123/8 (smem) K32 |
| nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 128 | 38208 | 128/16 (regs+smem) K32 | 125/8 (smem) K32 | 126/16 (regs) K32 | 123/16 (regs) K32 | 122/8 (smem) K32 |
| nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 128 | 12000 | 128/16 (regs) K32 | 126/16 (regs) K32 | 123/16 (regs) K32 | 126/16 (regs) K32 | 126/16 (regs) K32 |
| nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 128 | 12000 | 128/16 (regs) K32 | 127/16 (regs) K32 | 127/16 (regs) K32 | 123/16 (regs) K32 | 124/16 (regs) K32 |
| nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 128 | 12736 | 127/16 (regs) K32 | 120/16 (regs) K32 | 122/16 (regs) K32 | 124/16 (regs) K32 | 124/16 (regs) K32 |
| nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 128 | 12736 | 127/16 (regs) K32 | 127/16 (regs) K32 | 128/16 (regs) K32 | 124/16 (regs) K32 | 121/16 (regs) K32 |

### static_o2_cb5 / vt2_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 128 | 24064 | 254/8 (regs) K32 | 254/8 (regs) K32 | 244/8 (regs) K32 | 254/8 (regs) K32 | 255/8 (regs) S4 K40 |
| 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 128 | 29248 | 254/8 (regs) K32 | 254/8 (regs) K32 | 244/8 (regs) K32 | 254/8 (regs) K32 | 255/8 (regs) S4 K40 |
| 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 128 | 39616 | 254/8 (regs) K32 | 254/8 (regs+smem) K32 | 244/8 (regs) K32 | 254/8 (regs) K32 | 255/8 (regs+smem) S4 K40 |
| 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 128 | 49984 | 254/8 (regs) K32 | 254/8 (regs+smem) K32 | 244/8 (regs) K32 | 254/8 (regs) K32 | 255/8 (regs+smem) S4 K40 |
| 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 256 | 30464 | 240/8 (regs) K32 | 242/8 (regs) K32 | 240/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 |
| 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 256 | 35648 | 240/8 (regs) K32 | 242/8 (regs) K32 | 240/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 |
| 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 256 | 46016 | 240/8 (regs) K32 | 242/8 (regs) K32 | 240/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs) K32 |
| 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 256 | 56384 | 240/8 (regs) K32 | 242/8 (regs+smem) K32 | 240/8 (regs) K32 | 254/8 (regs) K32 | 254/8 (regs+smem) K32 |

### vel2d_o3_cb10 / kl2_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg2 cb7 chk12 stages1 lbmin0 | 128 | 26416 | 222/8 (regs) K48 | 221/8 (regs) K48 | 222/8 (regs) K48 | 201/8 (regs) K48 | 200/8 (regs) K48 |
| nxg2 cb7 chk12 stages1 lbmin4 | 128 | 26416 | 128/16 (regs) S412 K224 | 128/12 (smem) S412 K224 | 128/16 (regs) S408 K224 | 128/16 (regs) S308 K192 | 128/12 (smem) S308 K192 |
| nxg2 cb7 chk12 stages2 lbmin0 | 128 | 28192 | 215/8 (regs) K48 | 214/8 (regs) K48 | 215/8 (regs) K48 | 168/12 (regs) K48 | 194/8 (regs) K48 |
| nxg2 cb7 chk12 stages2 lbmin3 | 128 | 28192 | 168/12 (regs) S20 K80 | 168/12 (regs+smem) S20 K80 | 168/12 (regs) S20 K80 | 165/12 (regs) K48 | 168/12 (regs+smem) K48 |
| nxg2 cb7 chk12 stages2 lbmin4 | 128 | 28192 | 128/16 (regs) S340 K192 | 128/12 (smem) S340 K192 | 128/16 (regs) S336 K192 | 128/16 (regs) S240 K160 | 128/12 (smem) S240 K160 |
| nxg2 cb7 chk16 stages1 lbmin0 | 128 | 35088 | 227/8 (regs) K48 | 226/8 (regs+smem) K48 | 227/8 (regs) K48 | 204/8 (regs) K48 | 204/8 (regs+smem) K48 |
| nxg2 cb7 chk16 stages1 lbmin3 | 128 | 35088 | 168/12 (regs) S68 K96 | 168/8 (smem) S68 K96 | 168/12 (regs) S68 K96 | 166/12 (regs) K48 | 166/8 (smem) K48 |
| nxg2 cb7 chk16 stages1 lbmin4 | 128 | 35088 | 128/16 (regs+smem) S456 K240 | 128/8 (smem) S456 K240 | 128/16 (regs) S452 K256 | 128/16 (regs) S348 K208 | 128/8 (smem) S348 K208 |
| nxg2 cb7 chk16 stages2 lbmin0 | 128 | 37456 | 220/8 (regs) K48 | 216/8 (regs+smem) K48 | 219/8 (regs) K48 | 167/12 (regs) K48 | 195/8 (regs+smem) K48 |
| nxg2 cb7 chk16 stages2 lbmin4 | 128 | 37456 | 128/16 (regs+smem) S376 K208 | 128/8 (smem) S376 K208 | 128/16 (regs) S372 K192 | 128/16 (regs) S268 K176 | 128/8 (smem) S272 K176 |
| nxg2 cb7 chk24 stages1 lbmin0 | 128 | 52432 | 227/8 (regs) K48 | 226/4 (smem) K48 | 227/8 (regs) K48 | 204/8 (regs) K48 | 204/4 (smem) K48 |
| nxg2 cb7 chk24 stages1 lbmin4 | 128 | 52432 | 128/12 (smem) S456 K240 | 128/4 (smem) S456 K240 | 128/16 (regs+smem) S452 K256 | 128/16 (regs+smem) S348 K208 | 128/4 (smem) S348 K208 |
| nxg2 cb7 chk24 stages2 lbmin0 | 128 | 55984 | 220/8 (regs+smem) K48 | 216/4 (smem) K48 | 219/8 (regs) K48 | 197/8 (regs) K48 | 196/4 (smem) K48 |
| nxg2 cb7 chk24 stages2 lbmin4 | 128 | 55984 | 128/8 (smem) S376 K208 | 128/4 (smem) S376 K208 | 128/16 (regs+smem) S372 K192 | 128/16 (regs+smem) S268 K176 | 128/4 (smem) S272 K176 |
| nxg2 cb7 chk8 stages1 lbmin0 | 128 | 17744 | 222/8 (regs) K48 | 221/8 (regs) K48 | 222/8 (regs) K48 | 195/8 (regs) K48 | 198/8 (regs) K48 |
| nxg2 cb7 chk8 stages1 lbmin4 | 128 | 17744 | 128/16 (regs) S412 K224 | 128/16 (regs) S412 K224 | 128/16 (regs) S408 K224 | 128/16 (regs) S272 K192 | 128/16 (regs) S272 K192 |
| nxg2 cb7 chk8 stages2 lbmin0 | 128 | 18928 | 215/8 (regs) K48 | 214/8 (regs) K48 | 215/8 (regs) K48 | 161/12 (regs) K48 | 168/12 (regs) K48 |
| nxg2 cb7 chk8 stages2 lbmin4 | 128 | 18928 | 128/16 (regs) S340 K192 | 128/16 (regs) S340 K192 | 128/16 (regs) S336 K192 | 128/16 (regs) S196 K144 | 128/16 (regs) S200 K144 |
| nxg4 cb13 chk12 stages1 lbmin0 | 128 | 24880 | 214/8 (regs) K48 | 211/8 (regs) K48 | 213/8 (regs) K48 | 167/12 (regs) K48 | 167/12 (regs+smem) K48 |
| nxg4 cb13 chk12 stages1 lbmin4 | 128 | 24880 | 128/16 (regs) S348 K224 | 128/12 (smem) S348 K224 | 128/16 (regs) S356 K224 | 128/16 (regs) S292 K208 | 128/12 (smem) S292 K208 |
| nxg4 cb13 chk12 stages2 lbmin0 | 128 | 26656 | 207/8 (regs) K48 | 166/12 (regs+smem) K48 | 208/8 (regs) K48 | 160/12 (regs) K48 | 164/12 (regs+smem) K48 |
| nxg4 cb13 chk12 stages2 lbmin3 | 128 | 26656 | 168/12 (regs) K48 | 167/12 (regs+smem) K48 | 166/12 (regs) K48 | 164/12 (regs) K48 | 164/12 (regs+smem) K48 |
| nxg4 cb13 chk12 stages2 lbmin4 | 128 | 26656 | 128/16 (regs) S280 K192 | 128/12 (smem) S280 K192 | 128/16 (regs) S276 K192 | 128/16 (regs) S228 K176 | 128/12 (smem) S228 K176 |
| nxg4 cb13 chk16 stages1 lbmin0 | 128 | 33040 | 219/8 (regs) K48 | 218/8 (regs) K48 | 222/8 (regs) K48 | 187/8 (regs) K48 | 196/8 (regs) K48 |
| nxg4 cb13 chk16 stages1 lbmin3 | 128 | 33040 | 168/12 (regs) S24 K80 | 168/12 (regs+smem) S24 K80 | 168/12 (regs) S28 K80 | 165/12 (regs) K48 | 167/12 (regs+smem) K48 |
| nxg4 cb13 chk16 stages1 lbmin4 | 128 | 33040 | 128/16 (regs+smem) S392 K240 | 128/12 (smem) S392 K240 | 128/16 (regs) S400 K256 | 128/16 (regs) S332 K224 | 128/12 (smem) S332 K224 |
| nxg4 cb13 chk16 stages2 lbmin0 | 128 | 35408 | 210/8 (regs) K48 | 209/8 (regs+smem) K48 | 211/8 (regs) K48 | 158/12 (regs) K48 | 158/8 (smem) K48 |
| nxg4 cb13 chk16 stages2 lbmin4 | 128 | 35408 | 128/16 (regs+smem) S316 K208 | 128/8 (smem) S316 K208 | 128/16 (regs) S316 K208 | 128/16 (regs) S240 K176 | 128/8 (smem) S240 K176 |
| nxg4 cb13 chk24 stages1 lbmin0 | 128 | 49360 | 219/8 (regs) K48 | 218/8 (regs+smem) K48 | 222/8 (regs) K48 | 187/8 (regs) K48 | 196/8 (regs+smem) K48 |
| nxg4 cb13 chk24 stages1 lbmin4 | 128 | 49360 | 128/12 (smem) S392 K240 | 128/8 (smem) S392 K240 | 128/16 (regs+smem) S400 K256 | 128/16 (regs+smem) S332 K224 | 128/8 (smem) S332 K224 |
| nxg4 cb13 chk24 stages2 lbmin0 | 128 | 52912 | 210/8 (regs) K48 | 209/4 (smem) K48 | 211/8 (regs) K48 | 160/12 (regs) K48 | 160/4 (smem) K48 |
| nxg4 cb13 chk24 stages2 lbmin4 | 128 | 52912 | 128/12 (smem) S316 K208 | 128/4 (smem) S316 K208 | 128/16 (regs+smem) S312 K208 | 128/16 (regs+smem) S240 K176 | 128/4 (smem) S240 K176 |
| nxg4 cb13 chk8 stages1 lbmin0 | 128 | 16720 | 214/8 (regs) K48 | 211/8 (regs) K48 | 213/8 (regs) K48 | 167/12 (regs) K48 | 167/12 (regs) K48 |
| nxg4 cb13 chk8 stages1 lbmin4 | 128 | 16720 | 128/16 (regs) S348 K224 | 128/16 (regs) S348 K224 | 128/16 (regs) S356 K224 | 128/16 (regs) S296 K208 | 128/16 (regs) S296 K208 |
| nxg4 cb13 chk8 stages2 lbmin0 | 128 | 17904 | 207/8 (regs) K48 | 167/12 (regs) K48 | 208/8 (regs) K48 | 162/12 (regs) K48 | 159/12 (regs) K48 |
| nxg4 cb13 chk8 stages2 lbmin4 | 128 | 17904 | 128/16 (regs) S280 K192 | 128/16 (regs) S280 K192 | 128/16 (regs) S280 K192 | 128/16 (regs) S228 K176 | 128/16 (regs) S228 K176 |
| nxg4 cb7 chk12 stages1 lbmin0 | 128 | 19984 | 159/12 (regs) K48 | 158/12 (regs) K48 | 157/12 (regs) K48 | 123/16 (regs) K48 | 125/16 (regs+smem) K48 |
| nxg4 cb7 chk12 stages1 lbmin4 | 128 | 19984 | 128/16 (regs) S8 K64 | 125/16 (regs+smem) K48 | 127/16 (regs) K48 | 123/16 (regs) K48 | 124/16 (regs+smem) K48 |
| nxg4 cb7 chk12 stages2 lbmin0 | 128 | 21472 | 156/12 (regs) K48 | 123/16 (regs+smem) K48 | 125/16 (regs) K48 | 123/16 (regs) K48 | 127/16 (regs+smem) K48 |
| nxg4 cb7 chk12 stages2 lbmin3 | 128 | 21472 | 158/12 (regs) K48 | 158/12 (regs) K48 | 161/12 (regs) K48 | 156/12 (regs) K48 | 153/12 (regs) K48 |
| nxg4 cb7 chk12 stages2 lbmin4 | 128 | 21472 | 126/16 (regs) K48 | 126/16 (regs+smem) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 123/16 (regs+smem) K48 |
| nxg4 cb7 chk16 stages1 lbmin0 | 128 | 26512 | 159/12 (regs) K48 | 158/12 (regs+smem) K48 | 157/12 (regs) K48 | 123/16 (regs) K48 | 125/12 (smem) K48 |
| nxg4 cb7 chk16 stages1 lbmin3 | 128 | 26512 | 158/12 (regs) K48 | 160/12 (regs+smem) K48 | 159/12 (regs) K48 | 159/12 (regs) K48 | 156/12 (regs+smem) K48 |
| nxg4 cb7 chk16 stages1 lbmin4 | 128 | 26512 | 128/16 (regs) S8 K64 | 125/12 (smem) K48 | 127/16 (regs) K48 | 123/16 (regs) K48 | 124/12 (smem) K48 |
| nxg4 cb7 chk16 stages2 lbmin0 | 128 | 28496 | 156/12 (regs) K48 | 123/12 (smem) K48 | 124/16 (regs) K48 | 123/16 (regs) K48 | 123/12 (smem) K48 |
| nxg4 cb7 chk16 stages2 lbmin4 | 128 | 28496 | 126/16 (regs) K48 | 126/12 (smem) K48 | 126/16 (regs) K48 | 123/16 (regs) K48 | 123/12 (smem) K48 |
| nxg4 cb7 chk24 stages1 lbmin0 | 128 | 39568 | 160/12 (regs) K48 | 158/8 (smem) K48 | 158/12 (regs) K48 | 126/16 (regs) K48 | 125/8 (smem) K48 |
| nxg4 cb7 chk24 stages1 lbmin4 | 128 | 39568 | 128/16 (regs+smem) S24 K80 | 128/8 (smem) K48 | 126/16 (regs) K48 | 122/16 (regs) K48 | 124/8 (smem) K48 |
| nxg4 cb7 chk24 stages2 lbmin0 | 128 | 42544 | 154/12 (regs+smem) K48 | 124/8 (smem) K48 | 126/16 (regs) K48 | 122/16 (regs) K48 | 125/8 (smem) K48 |
| nxg4 cb7 chk24 stages2 lbmin4 | 128 | 42544 | 128/12 (smem) K48 | 126/8 (smem) K48 | 128/16 (regs) K48 | 122/16 (regs) K48 | 126/8 (smem) K48 |
| nxg4 cb7 chk8 stages1 lbmin0 | 128 | 13456 | 159/12 (regs) K48 | 158/12 (regs) K48 | 157/12 (regs) K48 | 123/16 (regs) K48 | 122/16 (regs) K48 |
| nxg4 cb7 chk8 stages1 lbmin4 | 128 | 13456 | 128/16 (regs) S8 K64 | 125/16 (regs) K48 | 127/16 (regs) K48 | 121/16 (regs) K48 | 123/16 (regs) K48 |
| nxg4 cb7 chk8 stages2 lbmin0 | 128 | 14448 | 156/12 (regs) K48 | 123/16 (regs) K48 | 124/16 (regs) K48 | 125/16 (regs) K48 | 127/16 (regs) K48 |
| nxg4 cb7 chk8 stages2 lbmin4 | 128 | 14448 | 126/16 (regs) K48 | 126/16 (regs) K48 | 126/16 (regs) K48 | 125/16 (regs) K48 | 124/16 (regs) K48 |

### vel2d_o3_cb10 / ra2_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg2 cb7 chk12 inplace1 stages1 lbmin0 | 128 | 26416 | 228/8 (regs) K32 | 226/8 (regs) K32 | 229/8 (regs) K32 | 202/8 (regs) K48 | 202/8 (regs) K48 |
| nxg2 cb7 chk12 inplace1 stages1 lbmin4 | 128 | 26416 | 128/16 (regs) S412 K200 | 128/12 (smem) S412 K200 | 128/16 (regs) S408 K208 | 128/16 (regs) S308 K192 | 128/12 (smem) S308 K192 |
| nxg2 cb7 chk12 inplace1 stages2 lbmin0 | 128 | 28192 | 223/8 (regs) K32 | 221/8 (regs) K32 | 223/8 (regs) K32 | 196/8 (regs) K48 | 196/8 (regs) K48 |
| nxg2 cb7 chk12 inplace1 stages2 lbmin3 | 128 | 28192 | 168/12 (regs) S20 K56 | 168/12 (regs+smem) S20 K56 | 168/12 (regs) S20 K56 | 167/12 (regs) K48 | 167/12 (regs+smem) K48 |
| nxg2 cb7 chk12 inplace1 stages2 lbmin4 | 128 | 28192 | 128/16 (regs) S340 K168 | 128/12 (smem) S340 K168 | 128/16 (regs) S336 K168 | 128/16 (regs) S244 K160 | 128/12 (smem) S244 K160 |
| nxg2 cb7 chk16 inplace0 stages1 lbmin3 | 128 | 38464 | 168/12 (regs) S48 K72 | 168/8 (smem) S48 K72 | 168/12 (regs) S48 K72 | 165/12 (regs) K32 | 165/8 (smem) K32 |
| nxg2 cb7 chk16 inplace0 stages1 lbmin4 | 128 | 38464 | 128/16 (regs+smem) S364 K224 | 128/8 (smem) S364 K224 | 128/16 (regs) S360 K224 | 128/16 (regs) S248 K176 | 128/8 (smem) S248 K176 |
| nxg2 cb7 chk16 inplace1 stages1 lbmin0 | 128 | 35088 | 234/8 (regs) K32 | 229/8 (regs+smem) K32 | 234/8 (regs) K32 | 238/8 (regs) K48 | 232/8 (regs+smem) K48 |
| nxg2 cb7 chk16 inplace1 stages1 lbmin3 | 128 | 35088 | 168/12 (regs) S68 K80 | 168/8 (smem) S68 K80 | 168/12 (regs) S68 K80 | 167/12 (regs) K48 | 167/8 (smem) K48 |
| nxg2 cb7 chk16 inplace1 stages1 lbmin4 | 128 | 35088 | 128/16 (regs+smem) S456 K216 | 128/8 (smem) S456 K216 | 128/16 (regs) S452 K224 | 128/16 (regs) S352 K208 | 128/8 (smem) S352 K208 |
| nxg2 cb7 chk16 inplace1 stages2 lbmin0 | 128 | 37456 | 225/8 (regs) K32 | 225/8 (regs+smem) K32 | 226/8 (regs) K32 | 198/8 (regs) K48 | 197/8 (regs+smem) K48 |
| nxg2 cb7 chk16 inplace1 stages2 lbmin4 | 128 | 37456 | 128/16 (regs+smem) S376 K184 | 128/8 (smem) S376 K184 | 128/16 (regs) S372 K176 | 128/16 (regs) S272 K176 | 128/8 (smem) S272 K176 |
| nxg2 cb7 chk24 inplace1 stages1 lbmin0 | 128 | 52432 | 234/8 (regs) K32 | 229/4 (smem) K32 | 234/8 (regs) K32 | 238/8 (regs) K48 | 232/4 (smem) K48 |
| nxg2 cb7 chk24 inplace1 stages1 lbmin4 | 128 | 52432 | 128/12 (smem) S456 K216 | 128/4 (smem) S456 K216 | 128/16 (regs+smem) S452 K224 | 128/16 (regs+smem) S352 K208 | 128/4 (smem) S352 K208 |
| nxg2 cb7 chk24 inplace1 stages2 lbmin0 | 128 | 55984 | 225/8 (regs+smem) K32 | 225/4 (smem) K32 | 226/8 (regs) K32 | 198/8 (regs) K48 | 198/4 (smem) K48 |
| nxg2 cb7 chk24 inplace1 stages2 lbmin4 | 128 | 55984 | 128/8 (smem) S376 K184 | 128/4 (smem) S376 K184 | 128/16 (regs+smem) S372 K176 | 128/16 (regs+smem) S272 K176 | 128/4 (smem) S272 K176 |
| nxg2 cb7 chk8 inplace1 stages1 lbmin0 | 128 | 17744 | 228/8 (regs) K32 | 226/8 (regs) K32 | 229/8 (regs) K32 | 168/12 (regs) K48 | 168/12 (regs) K48 |
| nxg2 cb7 chk8 inplace1 stages1 lbmin4 | 128 | 17744 | 128/16 (regs) S412 K200 | 128/16 (regs) S412 K200 | 128/16 (regs) S408 K208 | 128/16 (regs) S272 K192 | 128/16 (regs) S272 K192 |
| nxg2 cb7 chk8 inplace1 stages2 lbmin0 | 128 | 18928 | 223/8 (regs) K32 | 219/8 (regs) K32 | 223/8 (regs) K32 | 166/12 (regs) K48 | 166/12 (regs) K48 |
| nxg2 cb7 chk8 inplace1 stages2 lbmin4 | 128 | 18928 | 128/16 (regs) S340 K168 | 128/16 (regs) S340 K168 | 128/16 (regs) S336 K168 | 128/16 (regs) S196 K144 | 128/16 (regs) S224 K144 |
| nxg4 cb13 chk12 inplace1 stages1 lbmin0 | 128 | 24880 | 221/8 (regs) K32 | 217/8 (regs) K32 | 219/8 (regs) K32 | 165/12 (regs) K48 | 167/12 (regs+smem) K48 |
| nxg4 cb13 chk12 inplace1 stages1 lbmin4 | 128 | 24880 | 128/16 (regs) S376 K208 | 128/12 (smem) S376 K208 | 128/16 (regs) S356 K216 | 128/16 (regs) S292 K208 | 128/12 (smem) S292 K208 |
| nxg4 cb13 chk12 inplace1 stages2 lbmin0 | 128 | 26656 | 213/8 (regs) K32 | 210/8 (regs) K32 | 213/8 (regs) K32 | 161/12 (regs) K48 | 161/12 (regs+smem) K48 |
| nxg4 cb13 chk12 inplace1 stages2 lbmin3 | 128 | 26656 | 168/12 (regs) K32 | 168/12 (regs+smem) K32 | 168/12 (regs) K32 | 161/12 (regs) K48 | 162/12 (regs+smem) K48 |
| nxg4 cb13 chk12 inplace1 stages2 lbmin4 | 128 | 26656 | 128/16 (regs) S304 K176 | 128/12 (smem) S304 K176 | 128/16 (regs) S280 K168 | 128/16 (regs) S228 K176 | 128/12 (smem) S228 K176 |
| nxg4 cb13 chk16 inplace0 stages1 lbmin3 | 128 | 36416 | 168/12 (regs) S24 K56 | 168/8 (smem) S16 K48 | 168/12 (regs) K32 | 163/12 (regs) K32 | 162/8 (smem) K32 |
| nxg4 cb13 chk16 inplace0 stages1 lbmin4 | 128 | 36416 | 128/16 (regs+smem) S316 K216 | 128/8 (smem) S316 K216 | 128/16 (regs) S296 K216 | 128/16 (regs) S224 K184 | 128/8 (smem) S224 K184 |
| nxg4 cb13 chk16 inplace1 stages1 lbmin0 | 128 | 33040 | 225/8 (regs) K32 | 222/8 (regs) K32 | 225/8 (regs) K32 | 168/12 (regs) K48 | 167/12 (regs+smem) K48 |
| nxg4 cb13 chk16 inplace1 stages1 lbmin3 | 128 | 33040 | 168/12 (regs) S24 K56 | 168/12 (regs+smem) S24 K56 | 168/12 (regs) S24 K56 | 162/12 (regs) K48 | 162/12 (regs+smem) K48 |
| nxg4 cb13 chk16 inplace1 stages1 lbmin4 | 128 | 33040 | 128/16 (regs+smem) S420 K224 | 128/12 (smem) S420 K224 | 128/16 (regs) S400 K232 | 128/16 (regs) S332 K224 | 128/12 (smem) S332 K224 |
| nxg4 cb13 chk16 inplace1 stages2 lbmin0 | 128 | 35408 | 217/8 (regs) K32 | 213/8 (regs+smem) K32 | 217/8 (regs) K32 | 160/12 (regs) K48 | 158/8 (smem) K48 |
| nxg4 cb13 chk16 inplace1 stages2 lbmin4 | 128 | 35408 | 128/16 (regs+smem) S340 K192 | 128/8 (smem) S340 K192 | 128/16 (regs) S316 K184 | 128/16 (regs) S240 K176 | 128/8 (smem) S240 K176 |
| nxg4 cb13 chk24 inplace1 stages1 lbmin0 | 128 | 49360 | 225/8 (regs) K32 | 222/8 (regs+smem) K32 | 225/8 (regs) K32 | 168/12 (regs) K48 | 167/8 (smem) K48 |
| nxg4 cb13 chk24 inplace1 stages1 lbmin4 | 128 | 49360 | 128/12 (smem) S420 K224 | 128/8 (smem) S420 K224 | 128/16 (regs+smem) S400 K232 | 128/16 (regs+smem) S332 K224 | 128/8 (smem) S332 K224 |
| nxg4 cb13 chk24 inplace1 stages2 lbmin0 | 128 | 52912 | 217/8 (regs) K32 | 213/4 (smem) K32 | 217/8 (regs) K32 | 160/12 (regs) K48 | 160/4 (smem) K48 |
| nxg4 cb13 chk24 inplace1 stages2 lbmin4 | 128 | 52912 | 128/12 (smem) S340 K192 | 128/4 (smem) S340 K192 | 128/16 (regs+smem) S316 K184 | 128/16 (regs+smem) S240 K176 | 128/4 (smem) S240 K176 |
| nxg4 cb13 chk8 inplace1 stages1 lbmin0 | 128 | 16720 | 221/8 (regs) K32 | 217/8 (regs) K32 | 219/8 (regs) K32 | 162/12 (regs) K48 | 166/12 (regs) K48 |
| nxg4 cb13 chk8 inplace1 stages1 lbmin4 | 128 | 16720 | 128/16 (regs) S376 K208 | 128/16 (regs) S376 K208 | 128/16 (regs) S356 K216 | 128/16 (regs) S296 K208 | 128/16 (regs) S296 K208 |
| nxg4 cb13 chk8 inplace1 stages2 lbmin0 | 128 | 17904 | 213/8 (regs) K32 | 210/8 (regs) K32 | 213/8 (regs) K32 | 160/12 (regs) K48 | 160/12 (regs) K48 |
| nxg4 cb13 chk8 inplace1 stages2 lbmin4 | 128 | 17904 | 128/16 (regs) S304 K176 | 128/16 (regs) S304 K176 | 128/16 (regs) S280 K168 | 128/16 (regs) S228 K176 | 128/16 (regs) S228 K176 |
| nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 128 | 19984 | 160/12 (regs) K32 | 161/12 (regs) K32 | 158/12 (regs) K32 | 119/16 (regs) K48 | 120/16 (regs+smem) K48 |
| nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | 19984 | 128/16 (regs) S8 K40 | 128/16 (regs+smem) K32 | 127/16 (regs) K32 | 122/16 (regs) K48 | 122/16 (regs+smem) K48 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 128 | 21472 | 156/12 (regs) K32 | 125/16 (regs+smem) K32 | 125/16 (regs) K32 | 121/16 (regs) K48 | 119/16 (regs+smem) K48 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 128 | 21472 | 156/12 (regs) K32 | 157/12 (regs) K32 | 158/12 (regs) K32 | 160/12 (regs) K48 | 157/12 (regs) K48 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 128 | 21472 | 126/16 (regs) K32 | 125/16 (regs+smem) K32 | 126/16 (regs) K32 | 124/16 (regs) K48 | 124/16 (regs+smem) K48 |
| nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 128 | 29888 | 157/12 (regs) K32 | 159/12 (regs+smem) K32 | 162/12 (regs) K32 | 158/12 (regs) K32 | 152/12 (regs+smem) K32 |
| nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | 29888 | 128/16 (regs) S4 K40 | 128/12 (smem) K32 | 126/16 (regs) K32 | 122/16 (regs) K32 | 123/12 (smem) K32 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 128 | 26512 | 160/12 (regs) K32 | 161/12 (regs+smem) K32 | 158/12 (regs) K32 | 119/16 (regs) K48 | 120/12 (smem) K48 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 128 | 26512 | 161/12 (regs) K32 | 162/12 (regs+smem) K32 | 162/12 (regs) K32 | 158/12 (regs) K48 | 157/12 (regs+smem) K48 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 128 | 26512 | 128/16 (regs) S8 K40 | 128/12 (smem) K32 | 127/16 (regs) K32 | 122/16 (regs) K48 | 122/12 (smem) K48 |
| nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 128 | 28496 | 156/12 (regs) K32 | 125/12 (smem) K32 | 126/16 (regs) K32 | 126/16 (regs) K48 | 124/12 (smem) K48 |
| nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 128 | 28496 | 126/16 (regs) K32 | 125/12 (smem) K32 | 124/16 (regs) K32 | 125/16 (regs) K48 | 124/12 (smem) K48 |
| nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 128 | 39568 | 161/12 (regs) K32 | 162/8 (smem) K32 | 159/12 (regs) K32 | 123/16 (regs) K48 | 123/8 (smem) K48 |
| nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | 39568 | 128/16 (regs+smem) S24 K56 | 128/8 (smem) S8 K40 | 128/16 (regs) S12 K48 | 124/16 (regs) K48 | 124/8 (smem) K48 |
| nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 128 | 42544 | 158/12 (regs+smem) K32 | 158/8 (smem) K32 | 158/12 (regs) K32 | 123/16 (regs) K48 | 121/8 (smem) K48 |
| nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 128 | 42544 | 126/12 (smem) K32 | 128/8 (smem) K32 | 127/16 (regs) K32 | 124/16 (regs) K48 | 125/8 (smem) K48 |
| nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 128 | 13456 | 160/12 (regs) K32 | 161/12 (regs) K32 | 158/12 (regs) K32 | 121/16 (regs) K48 | 122/16 (regs) K48 |
| nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 128 | 13456 | 128/16 (regs) S8 K40 | 128/16 (regs) K32 | 127/16 (regs) K32 | 122/16 (regs) K48 | 121/16 (regs) K48 |
| nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 128 | 14448 | 156/12 (regs) K32 | 125/16 (regs) K32 | 126/16 (regs) K32 | 124/16 (regs) K48 | 121/16 (regs) K48 |
| nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 128 | 14448 | 126/16 (regs) K32 | 125/16 (regs) K32 | 124/16 (regs) K32 | 121/16 (regs) K48 | 122/16 (regs) K48 |

### vel2d_o3_cb10 / vt2_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 4x4x16 smax48 nwc7 nws1 vxpad0 sch12 | 224 | 35200 | 254/7 (regs) K32 | 254/7 (regs) K32 | 254/7 (regs) K32 | 248/7 (regs) K32 | 252/7 (regs) K32 |
| 4x4x16 smax48 nwc7 nws1 vxpad0 sch16 | 224 | 42368 | 254/7 (regs) K32 | 254/7 (regs) K32 | 254/7 (regs) K32 | 248/7 (regs) K32 | 252/7 (regs) K32 |
| 4x4x16 smax48 nwc7 nws1 vxpad0 sch24 | 224 | 56704 | 254/7 (regs) K32 | 254/7 (regs+smem) K32 | 254/7 (regs) K32 | 248/7 (regs) K32 | 252/7 (regs+smem) K32 |
| 4x4x16 smax48 nwc7 nws1 vxpad0 sch32 | 224 | 71040 | 254/7 (regs) K32 | 254/7 (regs+smem) K32 | 254/7 (regs) K32 | 248/7 (regs) K32 | 252/7 (regs+smem) K32 |

### vel2d_o3_cb5 / kl2_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg2 cb4 chk12 stages1 lbmin0 | 128 | 19792 | 157/12 (regs) K48 | 158/12 (regs) K48 | 158/12 (regs) K48 | 156/12 (regs) K48 | 156/12 (regs) K48 |
| nxg2 cb4 chk12 stages1 lbmin4 | 128 | 19792 | 128/16 (regs) S8 K64 | 128/16 (regs+smem) K48 | 128/16 (regs) K48 | 126/16 (regs) K48 | 125/16 (regs+smem) K48 |
| nxg2 cb4 chk12 stages2 lbmin0 | 128 | 21088 | 160/12 (regs) K48 | 123/16 (regs+smem) K48 | 127/16 (regs) K48 | 126/16 (regs) K48 | 127/16 (regs+smem) K48 |
| nxg2 cb4 chk12 stages2 lbmin3 | 128 | 21088 | 158/12 (regs) K48 | 158/12 (regs) K48 | 160/12 (regs) K48 | 161/12 (regs) K48 | 156/12 (regs) K48 |
| nxg2 cb4 chk12 stages2 lbmin4 | 128 | 21088 | 128/16 (regs) K48 | 126/16 (regs+smem) K48 | 127/16 (regs) K48 | 125/16 (regs) K48 | 125/16 (regs+smem) K48 |
| nxg2 cb4 chk16 stages1 lbmin0 | 128 | 26256 | 157/12 (regs) K48 | 158/12 (regs+smem) K48 | 158/12 (regs) K48 | 156/12 (regs) K48 | 158/12 (regs+smem) K48 |
| nxg2 cb4 chk16 stages1 lbmin3 | 128 | 26256 | 164/12 (regs) K48 | 162/12 (regs+smem) K48 | 162/12 (regs) K48 | 157/12 (regs) K48 | 162/12 (regs+smem) K48 |
| nxg2 cb4 chk16 stages1 lbmin4 | 128 | 26256 | 128/16 (regs) S8 K64 | 128/12 (smem) K48 | 128/16 (regs) K48 | 126/16 (regs) K48 | 125/12 (smem) K48 |
| nxg2 cb4 chk16 stages2 lbmin0 | 128 | 27984 | 160/12 (regs) K48 | 123/12 (smem) K48 | 127/16 (regs) K48 | 124/16 (regs) K48 | 124/12 (smem) K48 |
| nxg2 cb4 chk16 stages2 lbmin4 | 128 | 27984 | 128/16 (regs) K48 | 126/12 (smem) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 124/12 (smem) K48 |
| nxg2 cb4 chk24 stages1 lbmin0 | 128 | 39184 | 157/12 (regs) K48 | 158/8 (smem) K48 | 158/12 (regs) K48 | 156/12 (regs) K48 | 158/8 (smem) K48 |
| nxg2 cb4 chk24 stages1 lbmin4 | 128 | 39184 | 128/16 (regs+smem) S8 K64 | 128/8 (smem) K48 | 126/16 (regs) K48 | 126/16 (regs) K48 | 125/8 (smem) K48 |
| nxg2 cb4 chk24 stages2 lbmin0 | 128 | 41776 | 160/12 (regs+smem) K48 | 123/8 (smem) K48 | 127/16 (regs) K48 | 124/16 (regs) K48 | 125/8 (smem) K48 |
| nxg2 cb4 chk24 stages2 lbmin4 | 128 | 41776 | 128/12 (smem) K48 | 126/8 (smem) K48 | 127/16 (regs) K48 | 125/16 (regs) K48 | 125/8 (smem) K48 |
| nxg2 cb4 chk8 stages1 lbmin0 | 128 | 13328 | 157/12 (regs) K48 | 158/12 (regs) K48 | 158/12 (regs) K48 | 159/12 (regs) K48 | 157/12 (regs) K48 |
| nxg2 cb4 chk8 stages1 lbmin4 | 128 | 13328 | 128/16 (regs) S8 K64 | 128/16 (regs) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 126/16 (regs) K48 |
| nxg2 cb4 chk8 stages2 lbmin0 | 128 | 14192 | 160/12 (regs) K48 | 123/16 (regs) K48 | 127/16 (regs) K48 | 124/16 (regs) K48 | 125/16 (regs) K48 |
| nxg2 cb4 chk8 stages2 lbmin4 | 128 | 14192 | 128/16 (regs) K48 | 126/16 (regs) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 127/16 (regs) K48 |
| nxg4 cb4 chk12 stages1 lbmin0 | 128 | 16720 | 124/16 (regs) K48 | 121/16 (regs) K48 | 119/16 (regs) K48 | 121/16 (regs) K48 | 122/16 (regs) K48 |
| nxg4 cb4 chk12 stages1 lbmin4 | 128 | 16720 | 124/16 (regs) K48 | 124/16 (regs) K48 | 124/16 (regs) K48 | 121/16 (regs) K48 | 118/16 (regs) K48 |
| nxg4 cb4 chk12 stages2 lbmin0 | 128 | 18016 | 124/16 (regs) K48 | 94/20 (regs+smem) K48 | 96/20 (regs) K48 | 79/24 (regs) K48 | 93/20 (regs+smem) K48 |
| nxg4 cb4 chk12 stages2 lbmin3 | 128 | 18016 | 135/12 (regs) K48 | 133/12 (regs) K48 | 135/12 (regs) K48 | 148/12 (regs) K48 | 143/12 (regs) K48 |
| nxg4 cb4 chk12 stages2 lbmin4 | 128 | 18016 | 121/16 (regs) K48 | 123/16 (regs) K48 | 126/16 (regs) K48 | 123/16 (regs) K48 | 122/16 (regs) K48 |
| nxg4 cb4 chk16 stages1 lbmin0 | 128 | 22160 | 124/16 (regs) K48 | 121/16 (regs+smem) K48 | 119/16 (regs) K48 | 121/16 (regs) K48 | 122/16 (regs+smem) K48 |
| nxg4 cb4 chk16 stages1 lbmin3 | 128 | 22160 | 138/12 (regs) K48 | 137/12 (regs) K48 | 141/12 (regs) K48 | 153/12 (regs) K48 | 148/12 (regs) K48 |
| nxg4 cb4 chk16 stages1 lbmin4 | 128 | 22160 | 124/16 (regs) K48 | 124/16 (regs+smem) K48 | 124/16 (regs) K48 | 121/16 (regs) K48 | 118/16 (regs+smem) K48 |
| nxg4 cb4 chk16 stages2 lbmin0 | 128 | 23888 | 124/16 (regs) K48 | 93/16 (smem) K48 | 96/20 (regs) K48 | 80/24 (regs) K48 | 72/16 (smem) K48 |
| nxg4 cb4 chk16 stages2 lbmin4 | 128 | 23888 | 121/16 (regs) K48 | 123/16 (regs+smem) K48 | 126/16 (regs) K48 | 123/16 (regs) K48 | 122/16 (regs+smem) K48 |
| nxg4 cb4 chk24 stages1 lbmin0 | 128 | 33040 | 124/16 (regs+smem) K48 | 121/12 (smem) K48 | 119/16 (regs) K48 | 121/16 (regs) K48 | 122/12 (smem) K48 |
| nxg4 cb4 chk24 stages1 lbmin4 | 128 | 33040 | 124/16 (regs+smem) K48 | 124/12 (smem) K48 | 124/16 (regs) K48 | 121/16 (regs) K48 | 118/12 (smem) K48 |
| nxg4 cb4 chk24 stages2 lbmin0 | 128 | 35632 | 124/16 (regs+smem) K48 | 94/8 (smem) K48 | 96/20 (regs) K48 | 72/24 (smem) K48 | 72/8 (smem) K48 |
| nxg4 cb4 chk24 stages2 lbmin4 | 128 | 35632 | 121/16 (regs+smem) K48 | 123/8 (smem) K48 | 126/16 (regs) K48 | 123/16 (regs) K48 | 122/8 (smem) K48 |
| nxg4 cb4 chk8 stages1 lbmin0 | 128 | 11280 | 124/16 (regs) K48 | 121/16 (regs) K48 | 119/16 (regs) K48 | 96/20 (regs) K48 | 96/20 (regs) K48 |
| nxg4 cb4 chk8 stages1 lbmin4 | 128 | 11280 | 124/16 (regs) K48 | 124/16 (regs) K48 | 124/16 (regs) K48 | 120/16 (regs) K48 | 124/16 (regs) K48 |
| nxg4 cb4 chk8 stages2 lbmin0 | 128 | 12144 | 124/16 (regs) K48 | 93/20 (regs) K48 | 96/20 (regs) K48 | 80/24 (regs) K48 | 122/16 (regs) K48 |
| nxg4 cb4 chk8 stages2 lbmin4 | 128 | 12144 | 121/16 (regs) K48 | 123/16 (regs) K48 | 126/16 (regs) K48 | 121/16 (regs) K48 | 120/16 (regs) K48 |
| nxg4 cb7 chk12 stages1 lbmin0 | 128 | 19792 | 157/12 (regs) K48 | 160/12 (regs) K48 | 157/12 (regs) K48 | 123/16 (regs) K48 | 125/16 (regs+smem) K48 |
| nxg4 cb7 chk12 stages1 lbmin4 | 128 | 19792 | 128/16 (regs) K48 | 125/16 (regs+smem) K48 | 127/16 (regs) K48 | 123/16 (regs) K48 | 124/16 (regs+smem) K48 |
| nxg4 cb7 chk12 stages2 lbmin0 | 128 | 21088 | 156/12 (regs) K48 | 123/16 (regs+smem) K48 | 125/16 (regs) K48 | 123/16 (regs) K48 | 127/16 (regs+smem) K48 |
| nxg4 cb7 chk12 stages2 lbmin3 | 128 | 21088 | 158/12 (regs) K48 | 158/12 (regs) K48 | 161/12 (regs) K48 | 156/12 (regs) K48 | 153/12 (regs) K48 |
| nxg4 cb7 chk12 stages2 lbmin4 | 128 | 21088 | 126/16 (regs) K48 | 126/16 (regs+smem) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 123/16 (regs+smem) K48 |
| nxg4 cb7 chk16 stages1 lbmin0 | 128 | 26256 | 157/12 (regs) K48 | 160/12 (regs+smem) K48 | 157/12 (regs) K48 | 123/16 (regs) K48 | 125/12 (smem) K48 |
| nxg4 cb7 chk16 stages1 lbmin3 | 128 | 26256 | 160/12 (regs) K48 | 160/12 (regs+smem) K48 | 161/12 (regs) K48 | 159/12 (regs) K48 | 156/12 (regs+smem) K48 |
| nxg4 cb7 chk16 stages1 lbmin4 | 128 | 26256 | 128/16 (regs) K48 | 125/12 (smem) K48 | 127/16 (regs) K48 | 123/16 (regs) K48 | 124/12 (smem) K48 |
| nxg4 cb7 chk16 stages2 lbmin0 | 128 | 27984 | 156/12 (regs) K48 | 123/12 (smem) K48 | 123/16 (regs) K48 | 123/16 (regs) K48 | 123/12 (smem) K48 |
| nxg4 cb7 chk16 stages2 lbmin4 | 128 | 27984 | 126/16 (regs) K48 | 126/12 (smem) K48 | 125/16 (regs) K48 | 123/16 (regs) K48 | 123/12 (smem) K48 |
| nxg4 cb7 chk24 stages1 lbmin0 | 128 | 39184 | 157/12 (regs) K48 | 160/8 (smem) K48 | 157/12 (regs) K48 | 123/16 (regs) K48 | 125/8 (smem) K48 |
| nxg4 cb7 chk24 stages1 lbmin4 | 128 | 39184 | 128/16 (regs+smem) K48 | 125/8 (smem) K48 | 127/16 (regs) K48 | 123/16 (regs) K48 | 124/8 (smem) K48 |
| nxg4 cb7 chk24 stages2 lbmin0 | 128 | 41776 | 156/12 (regs+smem) K48 | 123/8 (smem) K48 | 125/16 (regs) K48 | 123/16 (regs) K48 | 125/8 (smem) K48 |
| nxg4 cb7 chk24 stages2 lbmin4 | 128 | 41776 | 126/12 (smem) K48 | 126/8 (smem) K48 | 128/16 (regs) K48 | 122/16 (regs) K48 | 121/8 (smem) K48 |
| nxg4 cb7 chk8 stages1 lbmin0 | 128 | 13328 | 157/12 (regs) K48 | 160/12 (regs) K48 | 157/12 (regs) K48 | 123/16 (regs) K48 | 122/16 (regs) K48 |
| nxg4 cb7 chk8 stages1 lbmin4 | 128 | 13328 | 128/16 (regs) K48 | 125/16 (regs) K48 | 127/16 (regs) K48 | 121/16 (regs) K48 | 123/16 (regs) K48 |
| nxg4 cb7 chk8 stages2 lbmin0 | 128 | 14192 | 156/12 (regs) K48 | 123/16 (regs) K48 | 123/16 (regs) K48 | 125/16 (regs) K48 | 127/16 (regs) K48 |
| nxg4 cb7 chk8 stages2 lbmin4 | 128 | 14192 | 126/16 (regs) K48 | 126/16 (regs) K48 | 125/16 (regs) K48 | 125/16 (regs) K48 | 124/16 (regs) K48 |

### vel2d_o3_cb5 / ra2_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 128 | 19792 | 161/12 (regs) K32 | 162/12 (regs) K32 | 161/12 (regs) K32 | 128/16 (regs) K48 | 159/12 (regs) K48 |
| nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 128 | 19792 | 128/16 (regs) S8 K40 | 128/16 (regs+smem) S8 K40 | 128/16 (regs) S8 K40 | 125/16 (regs) K48 | 125/16 (regs+smem) K48 |
| nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 128 | 21088 | 158/12 (regs) K32 | 158/12 (regs) K32 | 157/12 (regs) K32 | 123/16 (regs) K48 | 125/16 (regs+smem) K48 |
| nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 128 | 21088 | 157/12 (regs) K32 | 159/12 (regs) K32 | 158/12 (regs) K32 | 160/12 (regs) K48 | 157/12 (regs) K48 |
| nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 128 | 21088 | 128/16 (regs) K32 | 124/16 (regs+smem) K32 | 126/16 (regs) K32 | 126/16 (regs) K48 | 125/16 (regs+smem) K48 |
| nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 128 | 29632 | 157/12 (regs) K32 | 158/12 (regs+smem) K32 | 160/12 (regs) K32 | 160/12 (regs) K32 | 156/12 (regs+smem) K32 |
| nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 128 | 29632 | 128/16 (regs) K32 | 127/12 (smem) K32 | 125/16 (regs) K32 | 122/16 (regs) K32 | 125/12 (smem) K32 |
| nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 128 | 26256 | 161/12 (regs) K32 | 162/12 (regs+smem) K32 | 161/12 (regs) K32 | 128/16 (regs) K48 | 128/12 (smem) K48 |
| nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 128 | 26256 | 162/12 (regs) K32 | 162/12 (regs+smem) K32 | 161/12 (regs) K32 | 159/12 (regs) K48 | 162/12 (regs+smem) K48 |
| nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 128 | 26256 | 128/16 (regs) S8 K40 | 128/12 (smem) S8 K40 | 128/16 (regs) S8 K40 | 125/16 (regs) K48 | 125/12 (smem) K48 |
| nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 128 | 27984 | 158/12 (regs) K32 | 158/12 (regs+smem) K32 | 158/12 (regs) K32 | 123/16 (regs) K48 | 125/12 (smem) K48 |
| nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 128 | 27984 | 128/16 (regs) K32 | 127/12 (smem) K32 | 125/16 (regs) K32 | 125/16 (regs) K48 | 125/12 (smem) K48 |
| nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 128 | 39184 | 161/12 (regs) K32 | 162/8 (smem) K32 | 161/12 (regs) K32 | 128/16 (regs) K48 | 128/8 (smem) K48 |
| nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 128 | 39184 | 128/16 (regs+smem) S8 K40 | 128/8 (smem) S8 K40 | 128/16 (regs) S8 K40 | 125/16 (regs) K48 | 125/8 (smem) K48 |
| nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 128 | 41776 | 158/12 (regs+smem) K32 | 158/8 (smem) K32 | 157/12 (regs) K32 | 124/16 (regs) K48 | 121/8 (smem) K48 |
| nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 128 | 41776 | 128/12 (smem) K32 | 124/8 (smem) K32 | 126/16 (regs) K32 | 126/16 (regs) K48 | 125/8 (smem) K48 |
| nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 128 | 13328 | 161/12 (regs) K32 | 162/12 (regs) K32 | 161/12 (regs) K32 | 128/16 (regs) K48 | 128/16 (regs) K48 |
| nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 128 | 13328 | 128/16 (regs) S8 K40 | 128/16 (regs) S8 K40 | 128/16 (regs) S8 K40 | 122/16 (regs) K48 | 124/16 (regs) K48 |
| nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 128 | 14192 | 158/12 (regs) K32 | 158/12 (regs) K32 | 158/12 (regs) K32 | 127/16 (regs) K48 | 128/16 (regs) K48 |
| nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 128 | 14192 | 128/16 (regs) K32 | 127/16 (regs) K32 | 125/16 (regs) K32 | 125/16 (regs) K48 | 124/16 (regs) K48 |
| nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 128 | 16720 | 128/16 (regs) K32 | 120/16 (regs) K32 | 119/16 (regs) K32 | 121/16 (regs) K48 | 121/16 (regs) K48 |
| nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 128 | 16720 | 128/16 (regs) K32 | 128/16 (regs) K32 | 124/16 (regs) K32 | 121/16 (regs) K48 | 124/16 (regs) K48 |
| nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 128 | 18016 | 122/16 (regs) K32 | 123/16 (regs) K32 | 119/16 (regs) K32 | 72/28 (regs) S48 K64 | 93/20 (regs+smem) K48 |
| nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 128 | 18016 | 138/12 (regs) K32 | 137/12 (regs) K32 | 140/12 (regs) K32 | 151/12 (regs) K48 | 144/12 (regs) K48 |
| nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 128 | 18016 | 124/16 (regs) K32 | 124/16 (regs) K32 | 124/16 (regs) K32 | 123/16 (regs) K48 | 122/16 (regs) K48 |
| nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 128 | 25536 | 134/12 (regs) K32 | 134/12 (regs+smem) K32 | 138/12 (regs) K32 | 142/12 (regs) K32 | 136/12 (regs+smem) K32 |
| nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 128 | 25536 | 121/16 (regs) K32 | 124/12 (smem) K32 | 124/16 (regs) K32 | 121/16 (regs) K32 | 128/12 (smem) K32 |
| nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 128 | 22160 | 128/16 (regs) K32 | 120/16 (regs+smem) K32 | 119/16 (regs) K32 | 121/16 (regs) K48 | 121/16 (regs+smem) K48 |
| nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 128 | 22160 | 146/12 (regs) K32 | 145/12 (regs) K32 | 148/12 (regs) K32 | 152/12 (regs) K48 | 148/12 (regs) K48 |
| nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 128 | 22160 | 128/16 (regs) K32 | 128/16 (regs+smem) K32 | 124/16 (regs) K32 | 121/16 (regs) K48 | 124/16 (regs+smem) K48 |
| nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 128 | 23888 | 122/16 (regs) K32 | 123/16 (regs+smem) K32 | 119/16 (regs) K32 | 80/24 (regs) K48 | 80/16 (smem) K48 |
| nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 128 | 23888 | 124/16 (regs) K32 | 124/16 (regs+smem) K32 | 125/16 (regs) K32 | 124/16 (regs) K48 | 122/16 (regs+smem) K48 |
| nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 128 | 33040 | 128/16 (regs+smem) K32 | 120/12 (smem) K32 | 119/16 (regs) K32 | 121/16 (regs) K48 | 121/12 (smem) K48 |
| nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 128 | 33040 | 128/16 (regs+smem) K32 | 128/12 (smem) K32 | 124/16 (regs) K32 | 121/16 (regs) K48 | 124/12 (smem) K48 |
| nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 128 | 35632 | 122/16 (regs+smem) K32 | 123/8 (smem) K32 | 119/16 (regs) K32 | 72/24 (smem) K48 | 72/8 (smem) K48 |
| nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 128 | 35632 | 124/16 (regs+smem) K32 | 124/8 (smem) K32 | 124/16 (regs) K32 | 124/16 (regs) K48 | 122/8 (smem) K48 |
| nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 128 | 11280 | 128/16 (regs) K32 | 120/16 (regs) K32 | 119/16 (regs) K32 | 95/20 (regs) K48 | 95/20 (regs) K48 |
| nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 128 | 11280 | 128/16 (regs) K32 | 128/16 (regs) K32 | 124/16 (regs) K32 | 120/16 (regs) K48 | 124/16 (regs) K48 |
| nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 128 | 12144 | 122/16 (regs) K32 | 123/16 (regs) K32 | 119/16 (regs) K32 | 80/24 (regs) K48 | 96/20 (regs) K48 |
| nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 128 | 12144 | 124/16 (regs) K32 | 124/16 (regs) K32 | 125/16 (regs) K32 | 119/16 (regs) K48 | 121/16 (regs) K48 |
| nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 128 | 19792 | 159/12 (regs) K32 | 159/12 (regs) K32 | 158/12 (regs) K32 | 119/16 (regs) K48 | 120/16 (regs+smem) K48 |
| nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | 19792 | 128/16 (regs) K32 | 126/16 (regs+smem) K32 | 126/16 (regs) K32 | 122/16 (regs) K48 | 122/16 (regs+smem) K48 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 128 | 21088 | 156/12 (regs) K32 | 125/16 (regs+smem) K32 | 126/16 (regs) K32 | 121/16 (regs) K48 | 119/16 (regs+smem) K48 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 128 | 21088 | 156/12 (regs) K32 | 157/12 (regs) K32 | 158/12 (regs) K32 | 160/12 (regs) K48 | 157/12 (regs) K48 |
| nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 128 | 21088 | 126/16 (regs) K32 | 125/16 (regs+smem) K32 | 126/16 (regs) K32 | 124/16 (regs) K48 | 124/16 (regs+smem) K48 |
| nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 128 | 29632 | 156/12 (regs) K32 | 157/12 (regs+smem) K32 | 158/12 (regs) K32 | 158/12 (regs) K32 | 152/12 (regs+smem) K32 |
| nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | 29632 | 128/16 (regs) K32 | 127/12 (smem) K32 | 125/16 (regs) K32 | 122/16 (regs) K32 | 123/12 (smem) K32 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 128 | 26256 | 159/12 (regs) K32 | 159/12 (regs+smem) K32 | 158/12 (regs) K32 | 119/16 (regs) K48 | 120/12 (smem) K48 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 128 | 26256 | 160/12 (regs) K32 | 162/12 (regs+smem) K32 | 161/12 (regs) K32 | 158/12 (regs) K48 | 157/12 (regs+smem) K48 |
| nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 128 | 26256 | 128/16 (regs) K32 | 126/12 (smem) K32 | 126/16 (regs) K32 | 122/16 (regs) K48 | 122/12 (smem) K48 |
| nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 128 | 27984 | 156/12 (regs) K32 | 125/12 (smem) K32 | 124/16 (regs) K32 | 126/16 (regs) K48 | 124/12 (smem) K48 |
| nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 128 | 27984 | 126/16 (regs) K32 | 125/12 (smem) K32 | 124/16 (regs) K32 | 125/16 (regs) K48 | 124/12 (smem) K48 |
| nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 128 | 39184 | 159/12 (regs) K32 | 159/8 (smem) K32 | 158/12 (regs) K32 | 119/16 (regs) K48 | 120/8 (smem) K48 |
| nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | 39184 | 128/16 (regs+smem) K32 | 126/8 (smem) K32 | 126/16 (regs) K32 | 122/16 (regs) K48 | 122/8 (smem) K48 |
| nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 128 | 41776 | 156/12 (regs+smem) K32 | 125/8 (smem) K32 | 126/16 (regs) K32 | 121/16 (regs) K48 | 119/8 (smem) K48 |
| nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 128 | 41776 | 126/12 (smem) K32 | 125/8 (smem) K32 | 126/16 (regs) K32 | 123/16 (regs) K48 | 122/8 (smem) K48 |
| nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 128 | 13328 | 159/12 (regs) K32 | 159/12 (regs) K32 | 158/12 (regs) K32 | 121/16 (regs) K48 | 122/16 (regs) K48 |
| nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 128 | 13328 | 128/16 (regs) K32 | 126/16 (regs) K32 | 126/16 (regs) K32 | 122/16 (regs) K48 | 121/16 (regs) K48 |
| nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 128 | 14192 | 156/12 (regs) K32 | 125/16 (regs) K32 | 124/16 (regs) K32 | 124/16 (regs) K48 | 121/16 (regs) K48 |
| nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 128 | 14192 | 126/16 (regs) K32 | 125/16 (regs) K32 | 124/16 (regs) K32 | 121/16 (regs) K48 | 122/16 (regs) K48 |

### vel2d_o3_cb5 / vt2_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 128 | 25408 | 254/8 (regs) K32 | 254/8 (regs) K32 | 244/8 (regs) K32 | 244/8 (regs) K32 | 246/8 (regs) K32 |
| 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 128 | 31040 | 254/8 (regs) K32 | 254/8 (regs) K32 | 244/8 (regs) K32 | 244/8 (regs) K32 | 246/8 (regs) K32 |
| 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 128 | 42304 | 254/8 (regs) K32 | 254/8 (regs+smem) K32 | 244/8 (regs) K32 | 244/8 (regs) K32 | 246/8 (regs+smem) K32 |
| 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 128 | 53568 | 254/8 (regs) K32 | 254/4 (smem) K32 | 244/8 (regs) K32 | 244/8 (regs) K32 | 246/4 (smem) K32 |
| 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 256 | 31808 | 238/8 (regs) K32 | 242/8 (regs) K32 | 241/8 (regs) K32 | 248/8 (regs) K32 | 254/8 (regs) K32 |
| 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 256 | 37440 | 238/8 (regs) K32 | 242/8 (regs) K32 | 241/8 (regs) K32 | 250/8 (regs) K32 | 254/8 (regs) K32 |
| 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 256 | 48704 | 238/8 (regs) K32 | 242/8 (regs) K32 | 241/8 (regs) K32 | 250/8 (regs) K32 | 254/8 (regs) K32 |
| 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 256 | 59968 | 238/8 (regs) K32 | 242/8 (regs+smem) K32 | 241/8 (regs) K32 | 248/8 (regs) K32 | 254/8 (regs+smem) K32 |

## Flags (spills, stack, no fit, below 8 warps / SM, compile errors)

| arch | case | family | cfg | regs | flags |
|---|---|---|---|---|---|
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin0 | 124 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin0 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin0 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin0 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin4 | 127 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin4 | 126 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin0 | 128 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin0 | 128 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin0 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin0 | 124 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin3 | 156 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin3 | 150 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin3 | 146 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin3 | 147 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin3 | 147 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin4 | 126 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin4 | 126 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin0 | 124 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin0 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin0 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin0 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin3 | 156 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin3 | 151 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin3 | 150 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin3 | 149 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin3 | 151 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin4 | 127 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin4 | 126 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin0 | 128 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin0 | 128 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin0 | 122 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin0 | 125 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin4 | 126 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin4 | 126 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin0 | 124 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin0 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin0 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin0 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin4 | 127 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin4 | 126 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin0 | 128 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin0 | 128 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin0 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin0 | 124 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin4 | 126 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin4 | 126 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin0 | 128 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin0 | 125 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin0 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin0 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin4 | 123 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin0 | 128 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin0 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin0 | 122 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin0 | 125 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin4 | 124 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin0 | 80 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin0 | 80 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin0 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin0 | 80 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin0 | 80 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin4 | 117 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin4 | 112 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin4 | 117 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin4 | 119 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin4 | 121 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin0 | 80 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin0 | 72 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin0 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin0 | 72 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin0 | 80 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin3 | 133 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin3 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin3 | 114 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin3 | 115 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin3 | 117 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin4 | 117 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin4 | 112 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin4 | 114 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin4 | 115 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin4 | 117 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin0 | 72 | SPILL st20 ld8, STACK 64 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin0 | 72 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin0 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin0 | 80 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin0 | 80 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin3 | 133 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin3 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin3 | 117 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin3 | 119 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin3 | 121 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin4 | 117 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin4 | 112 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin4 | 117 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin4 | 119 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin4 | 121 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin0 | 72 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin0 | 72 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin0 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin0 | 80 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin0 | 80 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin4 | 115 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin4 | 113 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin4 | 114 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin4 | 113 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin4 | 115 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin0 | 72 | SPILL st20 ld8, STACK 64 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin0 | 72 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin0 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin0 | 80 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin0 | 80 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin4 | 117 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin4 | 112 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin4 | 117 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin4 | 119 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin4 | 121 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin0 | 80 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin0 | 72 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin0 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin0 | 72 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin0 | 80 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin4 | 115 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin4 | 113 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin4 | 114 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin4 | 115 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin4 | 117 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin0 | 80 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin0 | 72 | SPILL st4 ld4, STACK 64 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin0 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin0 | 80 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin0 | 80 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin4 | 115 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin4 | 113 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin4 | 117 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin4 | 119 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin4 | 121 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin0 | 80 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin0 | 72 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin0 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin0 | 80 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin0 | 80 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin4 | 113 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin4 | 113 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin4 | 114 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin4 | 113 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin4 | 115 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 125 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 127 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 123 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 128 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 125 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 127 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 125 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 120 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 120 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 151 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 147 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 145 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 145 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 147 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 125 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 125 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 127 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 123 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 154 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 148 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 145 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 145 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 149 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 126 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 125 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 127 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 125 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 121 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 124 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 125 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 125 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 127 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 123 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 126 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 125 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 128 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 128 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 120 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 120 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 125 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 125 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 125 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 127 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 123 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 124 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 123 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 126 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 124 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 121 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 124 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 123 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 127 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 127 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 154 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 127 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 126 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 128 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 128 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 156 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 150 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 150 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 150 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 153 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 126 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 126 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 127 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 159 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 158 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 146 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 147 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 148 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 127 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 127 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 127 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 125 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 154 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 156 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 153 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 154 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 154 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 157 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 127 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 126 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 128 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 128 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 127 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 127 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 127 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 125 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 154 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 127 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 126 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 128 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 128 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 126 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 126 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 127 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 128 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 126 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 154 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 123 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 124 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 128 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 128 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 124 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 124 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 80 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 80 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 123 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 93 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 117 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 117 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 72 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 72 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 120 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 94 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 133 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 128 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 120 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 119 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 123 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 120 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 117 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 120 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 119 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 123 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 128 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 114 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 115 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 117 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 119 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 116 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 114 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 115 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 117 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 72 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 72 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 123 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 93 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 133 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 128 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 121 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 122 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 125 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 117 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 117 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 72 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 72 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 95 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 116 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 113 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 118 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 72 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 72 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 123 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 93 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 117 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 117 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 72 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 72 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 120 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 94 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 117 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 112 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 120 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 119 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 123 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 80 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 72 | SPILL st4 ld4, STACK 48 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 123 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 93 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 117 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 113 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 80 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 72 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 95 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 115 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 112 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 118 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 127 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 125 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 125 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 127 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 125 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 150 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 145 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 147 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 146 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 151 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 125 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 122 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 158 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 154 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 143 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 144 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 145 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 124 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 124 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 127 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 125 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 152 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 148 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 150 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 150 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 153 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 126 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 125 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 128 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 127 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 124 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 124 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 125 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 127 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 125 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 126 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 125 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 127 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 125 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 125 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 126 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 125 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 123 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 124 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 123 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 127 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 122 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 124 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 128 | STACK 40 |
| sm_120 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 123 | STACK 40 |
| sm_80 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 254 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 255 | SPILL st4 ld4, STACK 40 |
| sm_80 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 254 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 254 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 244 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 254 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 255 | SPILL st4 ld4, STACK 40 |
| sm_80 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 254 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 254 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 244 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 254 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 255 | SPILL st4 ld4, STACK 40 |
| sm_80 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 254 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 254 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 244 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 254 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 255 | SPILL st4 ld4, STACK 40 |
| sm_80 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 254 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 254 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 244 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 254 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 254 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 240 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 242 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 240 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 254 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 254 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 240 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 242 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 240 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 254 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 254 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 240 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 242 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 240 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 254 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 254 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 240 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 242 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 240 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin0 | 160 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin0 | 168 | SPILL st40 ld52, STACK 72 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin0 | 168 | SPILL st40 ld52, STACK 72 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin0 | 195 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st216 ld176, STACK 152 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st232 ld196, STACK 168 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st220 ld172, STACK 168 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st220 ld172, STACK 168 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st220 ld172, STACK 168 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin0 | 157 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin0 | 157 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin0 | 168 | SPILL st40 ld52, STACK 72 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin0 | 168 | SPILL st40 ld52, STACK 72 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin0 | 193 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin3 | 158 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin3 | 158 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin3 | 168 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin3 | 168 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin3 | 168 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st232 ld184, STACK 144 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st252 ld200, STACK 160 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st212 ld168, STACK 152 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st212 ld168, STACK 152 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st212 ld168, STACK 152 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin3 | 164 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin3 | 163 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin3 | 168 | SPILL st16 ld16, STACK 48 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin3 | 168 | SPILL st16 ld16, STACK 48 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin3 | 168 | SPILL st20 ld20, STACK 56 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st232 ld224, STACK 176 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st228 ld224, STACK 176 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st212 ld200, STACK 176 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st212 ld184, STACK 176 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st220 ld208, STACK 176 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin0 | 158 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin0 | 156 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin0 | 199 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin0 | 196 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin0 | 200 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin3 | 159 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin3 | 157 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin3 | 168 | SPILL st20 ld20, STACK 56 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin3 | 168 | SPILL st20 ld20, STACK 56 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin3 | 168 | SPILL st20 ld20, STACK 56 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st264 ld204, STACK 168 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st288 ld236, STACK 192 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st264 ld192, STACK 184 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st264 ld192, STACK 184 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st264 ld192, STACK 184 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin0 | 196 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin0 | 193 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin0 | 195 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st268 ld200, STACK 152 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st288 ld212, STACK 168 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st248 ld184, STACK 160 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st248 ld184, STACK 160 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st248 ld184, STACK 160 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin0 | 158 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin0 | 156 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin0 | 199 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin0 | 196 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin0 | 200 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st264 ld204, STACK 168 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st288 ld236, STACK 192 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st264 ld192, STACK 184 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st264 ld192, STACK 184 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st264 ld192, STACK 184 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin0 | 157 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin0 | 197 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin0 | 194 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin0 | 196 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st268 ld200, STACK 152 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st288 ld212, STACK 168 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st248 ld184, STACK 160 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st248 ld184, STACK 160 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st248 ld184, STACK 160 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin0 | 168 | SPILL st40 ld52, STACK 72 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin0 | 168 | SPILL st40 ld52, STACK 72 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin0 | 195 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st216 ld176, STACK 152 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st232 ld196, STACK 168 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st220 ld172, STACK 168 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st220 ld172, STACK 168 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st220 ld172, STACK 168 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin0 | 156 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin0 | 168 | SPILL st40 ld52, STACK 72 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin0 | 168 | SPILL st40 ld52, STACK 72 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin0 | 192 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st232 ld184, STACK 144 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st252 ld200, STACK 160 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st212 ld168, STACK 152 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st212 ld168, STACK 152 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st212 ld168, STACK 152 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin0 | 156 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin0 | 156 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin0 | 196 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin0 | 193 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin0 | 197 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st260 ld232, STACK 176 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st276 ld252, STACK 200 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st268 ld232, STACK 200 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st268 ld232, STACK 200 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st268 ld232, STACK 208 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin0 | 155 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin0 | 155 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin0 | 196 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin0 | 193 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin0 | 195 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin3 | 158 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin3 | 158 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin3 | 168 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin3 | 164 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin3 | 168 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st284 ld244, STACK 176 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st304 ld260, STACK 192 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st260 ld228, STACK 184 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st260 ld228, STACK 184 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st260 ld228, STACK 184 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin3 | 160 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin3 | 163 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin3 | 168 | SPILL st20 ld20, STACK 56 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin3 | 168 | SPILL st12 ld12, STACK 48 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin3 | 168 | SPILL st16 ld16, STACK 48 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st276 ld280, STACK 208 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st272 ld280, STACK 208 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st260 ld260, STACK 208 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st260 ld244, STACK 208 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st316 ld312, STACK 240 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin0 | 201 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin0 | 198 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin0 | 202 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin3 | 160 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin3 | 159 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin3 | 168 | SPILL st20 ld20, STACK 56 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin3 | 168 | SPILL st16 ld16, STACK 48 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin3 | 168 | SPILL st24 ld24, STACK 56 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st308 ld260, STACK 200 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st332 ld292, STACK 224 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st312 ld252, STACK 224 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st312 ld252, STACK 224 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st360 ld296, STACK 248 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin0 | 155 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin0 | 156 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin0 | 198 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin0 | 195 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin0 | 197 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st320 ld260, STACK 192 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st340 ld272, STACK 200 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st296 ld244, STACK 200 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st296 ld244, STACK 200 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st296 ld244, STACK 200 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin0 | 201 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin0 | 198 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin0 | 202 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st308 ld260, STACK 200 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st332 ld292, STACK 224 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st312 ld252, STACK 224 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st312 ld252, STACK 224 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st360 ld296, STACK 248 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin0 | 156 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin0 | 157 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin0 | 199 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin0 | 196 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin0 | 198 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st320 ld260, STACK 192 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st340 ld272, STACK 200 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st296 ld244, STACK 200 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st296 ld244, STACK 200 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st296 ld244, STACK 200 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin0 | 156 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin0 | 157 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin0 | 196 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin0 | 193 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin0 | 197 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st260 ld232, STACK 176 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st276 ld252, STACK 200 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st268 ld232, STACK 200 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st268 ld232, STACK 200 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st268 ld232, STACK 208 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin0 | 162 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin0 | 156 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin0 | 195 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin0 | 192 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin0 | 166 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st284 ld244, STACK 176 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st304 ld260, STACK 192 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st260 ld228, STACK 184 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st260 ld228, STACK 184 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st260 ld228, STACK 184 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 109 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 120 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 120 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 110 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 107 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 109 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 125 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 122 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 122 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 141 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 136 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 138 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 138 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 139 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 107 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 109 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 119 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 119 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 138 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 132 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 136 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 136 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 137 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 119 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 116 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 125 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 122 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 109 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 120 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 120 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 142 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 136 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 136 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 136 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 139 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 110 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 108 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 109 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 120 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 120 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 108 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 109 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 122 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 119 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 118 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 128 | SPILL st40 ld52, STACK 72 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 122 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 120 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 115 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 114 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 116 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 113 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 118 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 119 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 108 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 119 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 119 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 109 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 120 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 120 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 110 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 108 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 109 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 120 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 120 | STACK 32 |
| sm_100 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 108 | STACK 32 |
| sm_120 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 111 | STACK 32 |
| sm_80 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_90 | static_o0_10x10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 122 | STACK 32 |
| sm_100 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch12 | 180 | STACK 32, LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch12 | 242 | STACK 32, LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch12 | 254 | STACK 32, LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch12 | 254 | STACK 32, LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch12 | 238 | STACK 32, LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch16 | 180 | STACK 32, LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch16 | 242 | STACK 32, LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch16 | 254 | STACK 32, LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch16 | 254 | STACK 32, LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch16 | 238 | STACK 32, LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch24 | 180 | STACK 32, LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch24 | 242 | STACK 32, LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch24 | 254 | STACK 32, LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch24 | 254 | STACK 32, LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch24 | 238 | STACK 32, LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch32 | 180 | STACK 32, LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch32 | 242 | STACK 32, LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch32 | 254 | STACK 32, LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch32 | 254 | STACK 32, LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 sch32 | 238 | STACK 32, LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch12 | 254 | STACK 32, LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch12 | 242 | STACK 32, LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch12 | 254 | STACK 32, LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch12 | 254 | STACK 32, LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch12 | 238 | STACK 32, LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch16 | 254 | STACK 32, LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch16 | 242 | STACK 32, LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch16 | 254 | STACK 32, LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch16 | 254 | STACK 32, LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch16 | 238 | STACK 32, LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch24 | 254 | STACK 32, LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch24 | 242 | STACK 32, LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch24 | 254 | STACK 32, LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch24 | 254 | STACK 32, LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch24 | 238 | STACK 32, LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch32 | 254 | STACK 32, LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch32 | 242 | STACK 32, LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch32 | 254 | STACK 32, LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch32 | 254 | STACK 32, LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs2_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 sch32 | 238 | STACK 32, LOW_OCC 7w |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages1 lbmin0 | 119 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages1 lbmin0 | 119 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages1 lbmin0 | 128 | SPILL st16 ld28, STACK 48 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages1 lbmin0 | 128 | SPILL st16 ld28, STACK 48 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages1 lbmin4 | 114 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages1 lbmin4 | 117 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages1 lbmin4 | 126 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin0 | 119 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin0 | 116 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin0 | 121 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin0 | 119 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin3 | 134 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin3 | 132 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin3 | 140 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin3 | 140 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin3 | 140 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin4 | 116 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin4 | 117 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin4 | 123 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk12 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace0 stages1 lbmin3 | 143 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace0 stages1 lbmin3 | 140 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace0 stages1 lbmin3 | 147 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace0 stages1 lbmin3 | 147 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace0 stages1 lbmin3 | 149 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace0 stages1 lbmin4 | 118 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace0 stages1 lbmin4 | 116 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace0 stages1 lbmin4 | 126 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace0 stages1 lbmin4 | 127 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin0 | 121 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin0 | 118 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin0 | 149 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin0 | 128 | SPILL st24 ld36, STACK 56 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin0 | 149 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin3 | 141 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin3 | 138 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin3 | 149 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin3 | 149 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin3 | 151 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin4 | 118 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin4 | 119 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages2 lbmin0 | 119 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages2 lbmin0 | 120 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages2 lbmin4 | 113 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages2 lbmin4 | 118 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages2 lbmin4 | 123 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk16 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages1 lbmin0 | 146 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages1 lbmin0 | 147 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages1 lbmin0 | 158 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages1 lbmin0 | 158 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages1 lbmin0 | 155 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages1 lbmin4 | 122 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages2 lbmin0 | 135 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages2 lbmin0 | 155 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages2 lbmin0 | 153 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages2 lbmin0 | 154 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages2 lbmin4 | 115 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages2 lbmin4 | 116 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages1 lbmin0 | 120 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages1 lbmin0 | 113 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages1 lbmin0 | 122 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages1 lbmin0 | 119 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages1 lbmin4 | 114 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages1 lbmin4 | 116 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages2 lbmin0 | 113 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages2 lbmin0 | 102 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages2 lbmin0 | 121 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages2 lbmin0 | 120 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages2 lbmin0 | 122 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages2 lbmin4 | 112 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages2 lbmin4 | 115 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages2 lbmin4 | 122 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg2 cb3 chk8 inplace1 stages2 lbmin4 | 123 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages1 lbmin0 | 72 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages1 lbmin0 | 72 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages1 lbmin0 | 110 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages1 lbmin0 | 96 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages1 lbmin0 | 96 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages1 lbmin4 | 102 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages1 lbmin4 | 99 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages1 lbmin4 | 110 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages1 lbmin4 | 109 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages1 lbmin4 | 113 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin0 | 72 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin0 | 79 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin0 | 96 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin0 | 80 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin0 | 80 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin3 | 112 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin3 | 107 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin3 | 107 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin3 | 106 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin3 | 108 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin4 | 102 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin4 | 98 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin4 | 107 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin4 | 106 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk12 inplace1 stages2 lbmin4 | 108 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace0 stages1 lbmin3 | 116 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace0 stages1 lbmin3 | 114 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace0 stages1 lbmin3 | 114 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace0 stages1 lbmin3 | 114 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace0 stages1 lbmin3 | 116 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace0 stages1 lbmin4 | 106 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace0 stages1 lbmin4 | 103 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace0 stages1 lbmin4 | 114 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace0 stages1 lbmin4 | 114 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace0 stages1 lbmin4 | 116 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin0 | 72 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin0 | 72 | SPILL st4 ld4, STACK 40 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin0 | 115 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin0 | 112 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin0 | 116 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin3 | 119 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin3 | 115 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin3 | 115 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin3 | 115 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin3 | 118 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin4 | 104 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin4 | 101 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin4 | 115 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin4 | 115 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages1 lbmin4 | 116 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages2 lbmin0 | 72 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages2 lbmin0 | 80 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages2 lbmin0 | 111 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages2 lbmin0 | 108 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages2 lbmin0 | 96 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages2 lbmin4 | 102 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages2 lbmin4 | 99 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages2 lbmin4 | 111 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages2 lbmin4 | 111 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk16 inplace1 stages2 lbmin4 | 112 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages1 lbmin0 | 115 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages1 lbmin0 | 113 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages1 lbmin0 | 112 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages1 lbmin4 | 108 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages1 lbmin4 | 106 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages1 lbmin4 | 115 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages1 lbmin4 | 115 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages1 lbmin4 | 114 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages2 lbmin0 | 80 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages2 lbmin0 | 80 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages2 lbmin0 | 115 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages2 lbmin0 | 112 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages2 lbmin0 | 114 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages2 lbmin4 | 106 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages2 lbmin4 | 105 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages2 lbmin4 | 115 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages2 lbmin4 | 115 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk24 inplace1 stages2 lbmin4 | 116 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages1 lbmin0 | 72 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages1 lbmin0 | 96 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages1 lbmin0 | 80 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages1 lbmin4 | 104 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages1 lbmin4 | 102 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages1 lbmin4 | 106 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages1 lbmin4 | 106 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages1 lbmin4 | 109 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages2 lbmin0 | 64 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages2 lbmin0 | 72 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages2 lbmin0 | 96 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages2 lbmin0 | 80 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages2 lbmin0 | 80 | SPILL st16 ld16, STACK 48 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages2 lbmin4 | 100 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages2 lbmin4 | 98 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages2 lbmin4 | 103 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages2 lbmin4 | 103 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb3 chk8 inplace1 stages2 lbmin4 | 105 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages1 lbmin0 | 118 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages1 lbmin0 | 117 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages1 lbmin0 | 128 | SPILL st4 ld16, STACK 40 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages1 lbmin0 | 119 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages1 lbmin0 | 119 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages1 lbmin4 | 116 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages1 lbmin4 | 120 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages1 lbmin4 | 119 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin0 | 118 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin0 | 118 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin0 | 119 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin0 | 115 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin3 | 135 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin3 | 132 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin3 | 137 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin3 | 137 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin3 | 138 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin4 | 111 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin4 | 113 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin4 | 120 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk12 inplace1 stages2 lbmin4 | 117 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace0 stages1 lbmin3 | 145 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace0 stages1 lbmin3 | 142 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace0 stages1 lbmin3 | 146 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace0 stages1 lbmin3 | 146 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace0 stages1 lbmin3 | 148 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace0 stages1 lbmin4 | 119 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace0 stages1 lbmin4 | 124 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace0 stages1 lbmin4 | 120 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace0 stages1 lbmin4 | 120 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin0 | 119 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin0 | 117 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin0 | 148 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin0 | 145 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin0 | 148 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin3 | 144 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin3 | 139 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin3 | 148 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin3 | 148 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin3 | 150 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin4 | 119 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin4 | 125 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages2 lbmin0 | 119 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages2 lbmin0 | 119 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages2 lbmin0 | 119 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages2 lbmin0 | 118 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages2 lbmin4 | 112 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages2 lbmin4 | 114 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages2 lbmin4 | 120 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk16 inplace1 stages2 lbmin4 | 119 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages1 lbmin0 | 118 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages1 lbmin0 | 118 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages1 lbmin0 | 155 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages1 lbmin0 | 155 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages1 lbmin0 | 154 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages1 lbmin4 | 122 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages1 lbmin4 | 120 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages2 lbmin0 | 121 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages2 lbmin0 | 121 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages2 lbmin0 | 146 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages2 lbmin0 | 150 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages2 lbmin0 | 152 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages2 lbmin4 | 118 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages2 lbmin4 | 119 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages1 lbmin0 | 120 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages1 lbmin0 | 118 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages1 lbmin0 | 126 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages1 lbmin0 | 122 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages1 lbmin0 | 118 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages1 lbmin4 | 116 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages1 lbmin4 | 108 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages1 lbmin4 | 120 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages1 lbmin4 | 120 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages2 lbmin0 | 113 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages2 lbmin0 | 117 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages2 lbmin0 | 121 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages2 lbmin0 | 120 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages2 lbmin0 | 119 | STACK 32 |
| sm_100 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages2 lbmin4 | 113 | STACK 32 |
| sm_120 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages2 lbmin4 | 117 | STACK 32 |
| sm_80 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_89 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages2 lbmin4 | 120 | STACK 32 |
| sm_90 | static_o0_48x1 | ra2_adj | nxg4 cb6 chk8 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch12 | 254 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch12 | 244 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch12 | 248 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch12 | 248 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch12 | 254 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch16 | 254 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch16 | 244 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch16 | 248 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch16 | 248 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch16 | 254 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch24 | 254 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch24 | 244 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch24 | 248 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch24 | 248 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 sch24 | 254 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch12 | 246 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch12 | 255 | SPILL st44 ld68, STACK 80 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch12 | 253 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch12 | 255 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch12 | 168 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch16 | 246 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch16 | 255 | SPILL st44 ld68, STACK 80 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch16 | 253 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch16 | 255 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch16 | 168 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch24 | 246 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch24 | 255 | SPILL st44 ld68, STACK 80 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch24 | 253 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch24 | 255 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 sch24 | 168 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch12 | 242 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch12 | 244 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch12 | 234 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch12 | 236 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch12 | 239 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch16 | 242 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch16 | 246 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch16 | 234 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch16 | 236 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch16 | 239 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch24 | 242 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch24 | 246 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch24 | 234 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch24 | 236 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad0 sch24 | 239 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch12 | 242 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch12 | 242 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch12 | 240 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch12 | 242 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch12 | 240 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch16 | 242 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch16 | 244 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch16 | 240 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch16 | 242 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch16 | 240 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch24 | 242 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch24 | 244 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch24 | 240 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch24 | 242 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc1 nws2 vxpad1 sch24 | 240 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch12 | 242 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch12 | 234 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch12 | 240 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch12 | 254 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch12 | 254 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch16 | 242 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch16 | 234 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch16 | 240 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch16 | 254 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch16 | 254 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch24 | 242 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch24 | 234 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch24 | 240 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch24 | 254 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad0 sch24 | 254 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch12 | 242 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch12 | 234 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch12 | 254 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch12 | 254 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch12 | 254 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch16 | 242 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch16 | 234 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch16 | 254 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch16 | 254 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch16 | 254 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch24 | 242 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch24 | 234 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch24 | 254 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch24 | 254 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws1 vxpad1 sch24 | 254 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch12 | 180 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch12 | 208 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch12 | 224 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch12 | 234 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch12 | 238 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch16 | 180 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch16 | 208 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch16 | 224 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch16 | 234 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch16 | 238 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch24 | 180 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch24 | 208 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch24 | 224 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch24 | 234 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad0 sch24 | 238 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch12 | 182 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch12 | 206 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch12 | 234 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch12 | 242 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch12 | 238 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch16 | 182 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch16 | 206 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch16 | 234 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch16 | 242 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch16 | 238 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch24 | 182 | STACK 32 |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch24 | 206 | STACK 32 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch24 | 234 | STACK 32 |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch24 | 242 | STACK 32 |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc2 nws2 vxpad1 sch24 | 238 | STACK 32 |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch12 | 173 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch12 | 216 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch12 | 254 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch12 | 254 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch12 | 254 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch16 | 173 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch16 | 216 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch16 | 254 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch16 | 254 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch16 | 254 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch24 | 173 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch24 | 216 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch24 | 254 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch24 | 254 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 sch24 | 254 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch12 | 174 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch12 | 168 | SPILL st12 ld44, STACK 48 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch12 | 254 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch12 | 254 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch12 | 239 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch16 | 174 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch16 | 168 | SPILL st12 ld44, STACK 48 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch16 | 254 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch16 | 254 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch16 | 239 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch24 | 174 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch24 | 168 | SPILL st12 ld44, STACK 48 |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch24 | 254 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch24 | 254 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 sch24 | 239 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch12 | 180 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch12 | 206 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch12 | 240 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch12 | 238 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch12 | 238 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch16 | 180 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch16 | 206 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch16 | 240 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch16 | 238 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch16 | 238 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch24 | 180 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch24 | 206 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch24 | 240 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch24 | 238 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 sch24 | 238 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch12 | 176 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch12 | 208 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch12 | 240 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch12 | 242 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch12 | 239 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch16 | 176 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch16 | 208 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch16 | 240 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch16 | 242 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch16 | 239 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch24 | 176 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch24 | 208 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch24 | 240 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch24 | 242 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs2_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 sch24 | 239 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 127 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 127 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 126 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 124 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 127 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 127 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 159 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 153 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 149 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 146 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 149 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 127 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 127 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 160 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 156 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 146 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 147 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 151 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 123 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 123 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 127 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 127 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 158 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 152 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 151 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 150 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 152 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 126 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 124 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 127 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 125 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 122 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 122 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 127 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 127 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 126 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 124 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 127 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 127 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 127 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 127 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 125 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 127 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 126 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 124 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 127 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 125 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 123 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 123 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 119 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 114 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 117 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 118 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 72 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 134 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 128 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 114 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 115 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 116 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 118 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 114 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 114 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 115 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 116 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 128 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 125 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 114 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 115 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 119 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 120 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 116 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 114 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 115 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 119 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 134 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 128 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 117 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 118 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 121 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 119 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 114 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 117 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 118 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 72 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 72 | SPILL st8 ld8, STACK 40 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 114 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 114 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 114 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 115 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 119 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 114 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 117 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 118 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 72 | SPILL st12 ld12, STACK 48 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 72 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 114 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 114 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 115 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 116 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 116 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 114 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 117 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 118 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 121 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 72 | SPILL st8 ld8, STACK 40 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 115 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 112 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 114 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 114 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 115 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 126 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 126 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 123 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 127 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 127 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 122 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 123 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 126 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 121 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 123 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 153 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 149 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 145 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 147 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 146 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 123 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 157 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 151 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 145 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 145 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 144 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 123 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 123 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 126 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 126 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 123 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 156 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 151 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 145 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 144 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 148 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 126 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 127 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 127 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 122 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 123 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 127 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 120 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 122 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 123 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 122 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 127 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 127 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 126 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 126 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 123 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 126 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 127 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 127 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 122 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 123 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 126 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 121 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 123 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 123 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 122 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 126 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 126 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 126 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 123 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 123 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 127 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 127 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 124 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 124 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 127 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 120 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 122 | STACK 32 |
| sm_100 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_120 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 121 | STACK 32 |
| sm_80 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 127 | STACK 32 |
| sm_89 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 127 | STACK 32 |
| sm_90 | static_o2_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 254 | STACK 32 |
| sm_120 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 255 | SPILL st4 ld4, STACK 40 |
| sm_80 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 254 | STACK 32 |
| sm_89 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 254 | STACK 32 |
| sm_90 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 244 | STACK 32 |
| sm_100 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 254 | STACK 32 |
| sm_120 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 255 | SPILL st4 ld4, STACK 40 |
| sm_80 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 254 | STACK 32 |
| sm_89 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 254 | STACK 32 |
| sm_90 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 244 | STACK 32 |
| sm_100 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 254 | STACK 32 |
| sm_120 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 255 | SPILL st4 ld4, STACK 40 |
| sm_80 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 254 | STACK 32 |
| sm_89 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 254 | STACK 32 |
| sm_90 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 244 | STACK 32 |
| sm_100 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 254 | STACK 32 |
| sm_120 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 255 | SPILL st4 ld4, STACK 40 |
| sm_80 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 254 | STACK 32 |
| sm_89 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 254 | STACK 32 |
| sm_90 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 244 | STACK 32 |
| sm_100 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 254 | STACK 32 |
| sm_120 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 254 | STACK 32 |
| sm_80 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 240 | STACK 32 |
| sm_89 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 242 | STACK 32 |
| sm_90 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 240 | STACK 32 |
| sm_100 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 254 | STACK 32 |
| sm_120 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 254 | STACK 32 |
| sm_80 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 240 | STACK 32 |
| sm_89 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 242 | STACK 32 |
| sm_90 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 240 | STACK 32 |
| sm_100 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 254 | STACK 32 |
| sm_120 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 254 | STACK 32 |
| sm_80 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 240 | STACK 32 |
| sm_89 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 242 | STACK 32 |
| sm_90 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 240 | STACK 32 |
| sm_100 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 254 | STACK 32 |
| sm_120 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 254 | STACK 32 |
| sm_80 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 240 | STACK 32 |
| sm_89 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 242 | STACK 32 |
| sm_90 | static_o2_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 240 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages1 lbmin0 | 201 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages1 lbmin0 | 200 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages1 lbmin0 | 222 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages1 lbmin0 | 221 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages1 lbmin0 | 222 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages1 lbmin4 | 128 | SPILL st308 ld220, STACK 192 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages1 lbmin4 | 128 | SPILL st308 ld220, STACK 192 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages1 lbmin4 | 128 | SPILL st412 ld240, STACK 224 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages1 lbmin4 | 128 | SPILL st412 ld240, STACK 224 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages1 lbmin4 | 128 | SPILL st408 ld236, STACK 224 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin0 | 168 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin0 | 194 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin0 | 215 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin0 | 214 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin0 | 215 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin3 | 165 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin3 | 168 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin3 | 168 | SPILL st20 ld20, STACK 80 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin3 | 168 | SPILL st20 ld20, STACK 80 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin3 | 168 | SPILL st20 ld20, STACK 80 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin4 | 128 | SPILL st240 ld196, STACK 160 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin4 | 128 | SPILL st240 ld196, STACK 160 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin4 | 128 | SPILL st340 ld212, STACK 192 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin4 | 128 | SPILL st340 ld212, STACK 192 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk12 stages2 lbmin4 | 128 | SPILL st336 ld208, STACK 192 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin0 | 204 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin0 | 204 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin0 | 227 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin0 | 226 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin0 | 227 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin3 | 166 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin3 | 166 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin3 | 168 | SPILL st68 ld56, STACK 96 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin3 | 168 | SPILL st68 ld56, STACK 96 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin3 | 168 | SPILL st68 ld56, STACK 96 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin4 | 128 | SPILL st348 ld236, STACK 208 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin4 | 128 | SPILL st348 ld236, STACK 208 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin4 | 128 | SPILL st456 ld260, STACK 240 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin4 | 128 | SPILL st456 ld260, STACK 240 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages1 lbmin4 | 128 | SPILL st452 ld256, STACK 256 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages2 lbmin0 | 167 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages2 lbmin0 | 195 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages2 lbmin0 | 220 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages2 lbmin0 | 216 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages2 lbmin0 | 219 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages2 lbmin4 | 128 | SPILL st268 ld208, STACK 176 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages2 lbmin4 | 128 | SPILL st272 ld212, STACK 176 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages2 lbmin4 | 128 | SPILL st376 ld228, STACK 208 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages2 lbmin4 | 128 | SPILL st376 ld228, STACK 208 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk16 stages2 lbmin4 | 128 | SPILL st372 ld224, STACK 192 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages1 lbmin0 | 204 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages1 lbmin0 | 204 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages1 lbmin0 | 227 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages1 lbmin0 | 226 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages1 lbmin0 | 227 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages1 lbmin4 | 128 | SPILL st348 ld236, STACK 208 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages1 lbmin4 | 128 | SPILL st348 ld236, STACK 208, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages1 lbmin4 | 128 | SPILL st456 ld260, STACK 240 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages1 lbmin4 | 128 | SPILL st456 ld260, STACK 240, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages1 lbmin4 | 128 | SPILL st452 ld256, STACK 256 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages2 lbmin0 | 197 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages2 lbmin0 | 196 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages2 lbmin0 | 220 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages2 lbmin0 | 216 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages2 lbmin0 | 219 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages2 lbmin4 | 128 | SPILL st268 ld208, STACK 176 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages2 lbmin4 | 128 | SPILL st272 ld212, STACK 176, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages2 lbmin4 | 128 | SPILL st376 ld228, STACK 208 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages2 lbmin4 | 128 | SPILL st376 ld228, STACK 208, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk24 stages2 lbmin4 | 128 | SPILL st372 ld224, STACK 192 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages1 lbmin0 | 195 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages1 lbmin0 | 198 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages1 lbmin0 | 222 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages1 lbmin0 | 221 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages1 lbmin0 | 222 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages1 lbmin4 | 128 | SPILL st272 ld176, STACK 192 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages1 lbmin4 | 128 | SPILL st272 ld176, STACK 192 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages1 lbmin4 | 128 | SPILL st412 ld240, STACK 224 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages1 lbmin4 | 128 | SPILL st412 ld240, STACK 224 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages1 lbmin4 | 128 | SPILL st408 ld236, STACK 224 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages2 lbmin0 | 161 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages2 lbmin0 | 168 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages2 lbmin0 | 215 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages2 lbmin0 | 214 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages2 lbmin0 | 215 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages2 lbmin4 | 128 | SPILL st196 ld160, STACK 144 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages2 lbmin4 | 128 | SPILL st200 ld164, STACK 144 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages2 lbmin4 | 128 | SPILL st340 ld212, STACK 192 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages2 lbmin4 | 128 | SPILL st340 ld212, STACK 192 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg2 cb7 chk8 stages2 lbmin4 | 128 | SPILL st336 ld208, STACK 192 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages1 lbmin0 | 167 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages1 lbmin0 | 167 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages1 lbmin0 | 214 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages1 lbmin0 | 211 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages1 lbmin0 | 213 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages1 lbmin4 | 128 | SPILL st292 ld192, STACK 208 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages1 lbmin4 | 128 | SPILL st292 ld192, STACK 208 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages1 lbmin4 | 128 | SPILL st348 ld208, STACK 224 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages1 lbmin4 | 128 | SPILL st348 ld208, STACK 224 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages1 lbmin4 | 128 | SPILL st356 ld224, STACK 224 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin0 | 160 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin0 | 164 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin0 | 207 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin0 | 166 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin0 | 208 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin3 | 164 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin3 | 164 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin3 | 168 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin3 | 167 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin3 | 166 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin4 | 128 | SPILL st228 ld180, STACK 176 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin4 | 128 | SPILL st228 ld180, STACK 176 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin4 | 128 | SPILL st280 ld184, STACK 192 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin4 | 128 | SPILL st280 ld184, STACK 192 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk12 stages2 lbmin4 | 128 | SPILL st276 ld180, STACK 192 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin0 | 187 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin0 | 196 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin0 | 219 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin0 | 218 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin0 | 222 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin3 | 165 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin3 | 167 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin3 | 168 | SPILL st24 ld24, STACK 80 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin3 | 168 | SPILL st24 ld24, STACK 80 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin3 | 168 | SPILL st28 ld28, STACK 80 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin4 | 128 | SPILL st332 ld208, STACK 224 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin4 | 128 | SPILL st332 ld208, STACK 224 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin4 | 128 | SPILL st392 ld228, STACK 240 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin4 | 128 | SPILL st392 ld228, STACK 240 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages1 lbmin4 | 128 | SPILL st400 ld244, STACK 256 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages2 lbmin0 | 158 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages2 lbmin0 | 158 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages2 lbmin0 | 210 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages2 lbmin0 | 209 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages2 lbmin0 | 211 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages2 lbmin4 | 128 | SPILL st240 ld188, STACK 176 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages2 lbmin4 | 128 | SPILL st240 ld188, STACK 176 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages2 lbmin4 | 128 | SPILL st316 ld200, STACK 208 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages2 lbmin4 | 128 | SPILL st316 ld200, STACK 208 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk16 stages2 lbmin4 | 128 | SPILL st316 ld208, STACK 208 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages1 lbmin0 | 187 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages1 lbmin0 | 196 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages1 lbmin0 | 219 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages1 lbmin0 | 218 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages1 lbmin0 | 222 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages1 lbmin4 | 128 | SPILL st332 ld208, STACK 224 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages1 lbmin4 | 128 | SPILL st332 ld208, STACK 224 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages1 lbmin4 | 128 | SPILL st392 ld228, STACK 240 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages1 lbmin4 | 128 | SPILL st392 ld228, STACK 240 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages1 lbmin4 | 128 | SPILL st400 ld244, STACK 256 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages2 lbmin0 | 160 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages2 lbmin0 | 160 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages2 lbmin0 | 210 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages2 lbmin0 | 209 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages2 lbmin0 | 211 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages2 lbmin4 | 128 | SPILL st240 ld188, STACK 176 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages2 lbmin4 | 128 | SPILL st240 ld188, STACK 176, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages2 lbmin4 | 128 | SPILL st316 ld200, STACK 208 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages2 lbmin4 | 128 | SPILL st316 ld200, STACK 208, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk24 stages2 lbmin4 | 128 | SPILL st312 ld196, STACK 208 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages1 lbmin0 | 167 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages1 lbmin0 | 167 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages1 lbmin0 | 214 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages1 lbmin0 | 211 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages1 lbmin0 | 213 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages1 lbmin4 | 128 | SPILL st296 ld200, STACK 208 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages1 lbmin4 | 128 | SPILL st296 ld200, STACK 208 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages1 lbmin4 | 128 | SPILL st348 ld208, STACK 224 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages1 lbmin4 | 128 | SPILL st348 ld208, STACK 224 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages1 lbmin4 | 128 | SPILL st356 ld224, STACK 224 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages2 lbmin0 | 162 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages2 lbmin0 | 159 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages2 lbmin0 | 207 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages2 lbmin0 | 167 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages2 lbmin0 | 208 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages2 lbmin4 | 128 | SPILL st228 ld180, STACK 176 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages2 lbmin4 | 128 | SPILL st228 ld180, STACK 176 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages2 lbmin4 | 128 | SPILL st280 ld184, STACK 192 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages2 lbmin4 | 128 | SPILL st280 ld184, STACK 192 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb13 chk8 stages2 lbmin4 | 128 | SPILL st280 ld192, STACK 192 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 159 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 157 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 64 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 127 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 156 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 125 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 156 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 153 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 158 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 161 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 123 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 159 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 157 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 159 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 156 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 158 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 160 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 159 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 64 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 123 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 156 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 123 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 160 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 158 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 128 | SPILL st24 ld24, STACK 80 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 128 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 154 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 126 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 128 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 159 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 157 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 123 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 64 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 127 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 156 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin0 | 202 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin0 | 202 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin0 | 228 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin0 | 226 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin0 | 229 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st308 ld220, STACK 192 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st308 ld220, STACK 192 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st412 ld240, STACK 200 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st412 ld240, STACK 200 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st408 ld236, STACK 208 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin0 | 196 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin0 | 196 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin0 | 223 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin0 | 221 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin0 | 223 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin3 | 167 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin3 | 167 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin3 | 168 | SPILL st20 ld20, STACK 56 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin3 | 168 | SPILL st20 ld20, STACK 56 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin3 | 168 | SPILL st20 ld20, STACK 56 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st244 ld200, STACK 160 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st244 ld200, STACK 160 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st340 ld212, STACK 168 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st340 ld212, STACK 168 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st336 ld208, STACK 168 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin3 | 165 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin3 | 165 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin3 | 168 | SPILL st48 ld40, STACK 72 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin3 | 168 | SPILL st48 ld40, STACK 72 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin3 | 168 | SPILL st48 ld40, STACK 72 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st248 ld236, STACK 176 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st248 ld236, STACK 176 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st364 ld268, STACK 224 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st364 ld252, STACK 224 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st360 ld264, STACK 224 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin0 | 238 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin0 | 232 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin0 | 234 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin0 | 229 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin0 | 234 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin3 | 167 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin3 | 167 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin3 | 168 | SPILL st68 ld56, STACK 80 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin3 | 168 | SPILL st68 ld56, STACK 80 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin3 | 168 | SPILL st68 ld56, STACK 80 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st352 ld240, STACK 208 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st352 ld240, STACK 208 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st456 ld260, STACK 216 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st456 ld260, STACK 216 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st452 ld256, STACK 224 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin0 | 198 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin0 | 197 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin0 | 225 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin0 | 225 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin0 | 226 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st272 ld212, STACK 176 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st272 ld212, STACK 176 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st376 ld228, STACK 184 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st376 ld228, STACK 184 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st372 ld224, STACK 176 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin0 | 238 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin0 | 232 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin0 | 234 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin0 | 229 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin0 | 234 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st352 ld240, STACK 208 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st352 ld240, STACK 208, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st456 ld260, STACK 216 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st456 ld260, STACK 216, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st452 ld256, STACK 224 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin0 | 198 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin0 | 198 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin0 | 225 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin0 | 225 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin0 | 226 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st272 ld212, STACK 176 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st272 ld212, STACK 176, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st376 ld228, STACK 184 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st376 ld228, STACK 184, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st372 ld224, STACK 176 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin0 | 168 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin0 | 168 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin0 | 228 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin0 | 226 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin0 | 229 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st272 ld176, STACK 192 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st272 ld176, STACK 192 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st412 ld240, STACK 200 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st412 ld240, STACK 200 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st408 ld236, STACK 208 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin0 | 166 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin0 | 166 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin0 | 223 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin0 | 219 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin0 | 223 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st196 ld160, STACK 144 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st224 ld176, STACK 144 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st340 ld212, STACK 168 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st340 ld212, STACK 168 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg2 cb7 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st336 ld208, STACK 168 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin0 | 165 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin0 | 167 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin0 | 221 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin0 | 217 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin0 | 219 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st292 ld192, STACK 208 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st292 ld192, STACK 208 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st376 ld228, STACK 208 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st376 ld228, STACK 208 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st356 ld224, STACK 216 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin0 | 161 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin0 | 161 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin0 | 213 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin0 | 210 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin0 | 213 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin3 | 161 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin3 | 162 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin3 | 168 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin3 | 168 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin3 | 168 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st228 ld180, STACK 176 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st228 ld180, STACK 176 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st304 ld200, STACK 176 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st304 ld200, STACK 176 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk12 inplace1 stages2 lbmin4 | 128 | SPILL st280 ld192, STACK 168 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin3 | 163 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin3 | 162 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin3 | 168 | SPILL st24 ld24, STACK 56 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin3 | 168 | SPILL st16 ld16, STACK 48 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin3 | 168 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st224 ld204, STACK 184 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st224 ld204, STACK 184 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st316 ld244, STACK 216 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st316 ld228, STACK 216 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st296 ld240, STACK 216 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin0 | 168 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin0 | 167 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin0 | 225 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin0 | 222 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin0 | 225 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin3 | 162 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin3 | 162 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin3 | 168 | SPILL st24 ld24, STACK 56 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin3 | 168 | SPILL st24 ld24, STACK 56 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin3 | 168 | SPILL st24 ld24, STACK 56 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st332 ld208, STACK 224 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st332 ld208, STACK 224 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st420 ld248, STACK 224 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st420 ld248, STACK 224 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st400 ld244, STACK 232 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin0 | 160 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin0 | 158 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin0 | 217 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin0 | 213 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin0 | 217 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st240 ld188, STACK 176 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st240 ld188, STACK 176 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st340 ld216, STACK 192 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st340 ld216, STACK 192 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk16 inplace1 stages2 lbmin4 | 128 | SPILL st316 ld208, STACK 184 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin0 | 168 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin0 | 167 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin0 | 225 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin0 | 222 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin0 | 225 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st332 ld208, STACK 224 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st332 ld208, STACK 224 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st420 ld248, STACK 224 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st420 ld248, STACK 224 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st400 ld244, STACK 232 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin0 | 160 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin0 | 160 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin0 | 217 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin0 | 213 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin0 | 217 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st240 ld188, STACK 176 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st240 ld188, STACK 176, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st340 ld216, STACK 192 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st340 ld216, STACK 192, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk24 inplace1 stages2 lbmin4 | 128 | SPILL st316 ld208, STACK 184 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin0 | 162 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin0 | 166 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin0 | 221 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin0 | 217 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin0 | 219 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st296 ld200, STACK 208 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st296 ld200, STACK 208 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st376 ld228, STACK 208 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st376 ld228, STACK 208 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st356 ld224, STACK 216 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin0 | 160 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin0 | 160 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin0 | 213 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin0 | 210 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin0 | 213 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st228 ld180, STACK 176 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st228 ld180, STACK 176 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st304 ld200, STACK 176 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st304 ld200, STACK 176 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb13 chk8 inplace1 stages2 lbmin4 | 128 | SPILL st280 ld192, STACK 168 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 119 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 120 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 160 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 161 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 127 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 119 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 160 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 157 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 157 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 158 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 152 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 159 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 162 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 123 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | SPILL st4 ld4, STACK 40 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 119 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 120 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 160 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 161 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 158 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 157 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 161 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 162 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 162 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 127 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 123 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 161 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 162 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st24 ld24, STACK 56 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st12 ld12, STACK 48 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 121 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 127 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 160 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 161 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 121 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 127 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 121 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch12 | 248 | STACK 32, LOW_OCC 7w |
| sm_120 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch12 | 252 | STACK 32, LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch12 | 254 | STACK 32, LOW_OCC 7w |
| sm_89 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch12 | 254 | STACK 32, LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch12 | 254 | STACK 32, LOW_OCC 7w |
| sm_100 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch16 | 248 | STACK 32, LOW_OCC 7w |
| sm_120 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch16 | 252 | STACK 32, LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch16 | 254 | STACK 32, LOW_OCC 7w |
| sm_89 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch16 | 254 | STACK 32, LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch16 | 254 | STACK 32, LOW_OCC 7w |
| sm_100 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch24 | 248 | STACK 32, LOW_OCC 7w |
| sm_120 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch24 | 252 | STACK 32, LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch24 | 254 | STACK 32, LOW_OCC 7w |
| sm_89 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch24 | 254 | STACK 32, LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch24 | 254 | STACK 32, LOW_OCC 7w |
| sm_100 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch32 | 248 | STACK 32, LOW_OCC 7w |
| sm_120 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch32 | 252 | STACK 32, LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch32 | 254 | STACK 32, LOW_OCC 7w |
| sm_89 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch32 | 254 | STACK 32, LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt2_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 sch32 | 254 | STACK 32, LOW_OCC 7w |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin0 | 156 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin0 | 156 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin0 | 157 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin0 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin0 | 158 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin4 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin4 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 64 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin4 | 128 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages1 lbmin4 | 128 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin0 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin0 | 127 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin0 | 160 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin0 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin0 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin3 | 161 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin3 | 156 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin3 | 158 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin3 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin3 | 160 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin4 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin4 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin4 | 128 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin4 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk12 stages2 lbmin4 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin0 | 156 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin0 | 158 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin0 | 157 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin0 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin0 | 158 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin3 | 157 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin3 | 162 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin3 | 164 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin3 | 162 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin3 | 162 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin4 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin4 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 64 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin4 | 128 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages1 lbmin4 | 128 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin0 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin0 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin0 | 160 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin0 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin0 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin4 | 128 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin4 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk16 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin0 | 156 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin0 | 158 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin0 | 157 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin0 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin0 | 158 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin4 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin4 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 64 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin4 | 128 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages1 lbmin4 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin0 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin0 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin0 | 160 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin0 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin0 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin4 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin4 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin4 | 128 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin4 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk24 stages2 lbmin4 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin0 | 159 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin0 | 157 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin0 | 157 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin0 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin0 | 158 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin4 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin4 | 126 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 64 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin4 | 128 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages1 lbmin4 | 128 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin0 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin0 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin0 | 160 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin0 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin0 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin4 | 127 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin4 | 128 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin4 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg2 cb4 chk8 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin0 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin0 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin0 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin0 | 121 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin0 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin4 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin4 | 118 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin4 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin4 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages1 lbmin4 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin0 | 79 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin0 | 93 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin0 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin0 | 94 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin0 | 96 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin3 | 148 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin3 | 143 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin3 | 135 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin3 | 133 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin3 | 135 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin4 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin4 | 121 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin4 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk12 stages2 lbmin4 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin0 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin0 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin0 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin0 | 121 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin0 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin3 | 153 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin3 | 148 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin3 | 138 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin3 | 137 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin3 | 141 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin4 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin4 | 118 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin4 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin4 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages1 lbmin4 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin0 | 80 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin0 | 72 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin0 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin0 | 93 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin0 | 96 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin4 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin4 | 121 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin4 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk16 stages2 lbmin4 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin0 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin0 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin0 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin0 | 121 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin0 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin4 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin4 | 118 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin4 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin4 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages1 lbmin4 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin0 | 72 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin0 | 72 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin0 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin0 | 94 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin0 | 96 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin4 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin4 | 121 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin4 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk24 stages2 lbmin4 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin0 | 96 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin0 | 96 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin0 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin0 | 121 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin0 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin4 | 120 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin4 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin4 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages1 lbmin4 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin0 | 80 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin0 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin0 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin0 | 93 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin0 | 96 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin4 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin4 | 120 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin4 | 121 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin4 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb4 chk8 stages2 lbmin4 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 157 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 160 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin0 | 157 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 128 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages1 lbmin4 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 127 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 156 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin0 | 125 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 156 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 153 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 158 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin3 | 161 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 123 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk12 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 157 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 160 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin0 | 157 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 159 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 156 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 160 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 160 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin3 | 161 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 128 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages1 lbmin4 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 123 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 156 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin0 | 123 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 123 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk16 stages2 lbmin4 | 125 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 157 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 160 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin0 | 157 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 128 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages1 lbmin4 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 156 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin0 | 125 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 121 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk24 stages2 lbmin4 | 128 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 157 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 160 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin0 | 157 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 123 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 128 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages1 lbmin4 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 127 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 156 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin0 | 123 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl2_adj | nxg4 cb7 chk8 stages2 lbmin4 | 125 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 128 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 159 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 161 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 162 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin0 | 161 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin0 | 157 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 160 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 157 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 159 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin3 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk12 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 160 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 156 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 158 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin3 | 160 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 125 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 127 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace0 stages1 lbmin4 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 128 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 128 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 161 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 162 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin0 | 161 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 159 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 162 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 162 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 162 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin3 | 161 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 127 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk16 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 128 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 128 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 161 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 162 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin0 | 161 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 121 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin0 | 157 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk24 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 128 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 128 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 161 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 162 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin0 | 161 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages1 lbmin4 | 128 | SPILL st8 ld8, STACK 40 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 127 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 128 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin0 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 127 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg2 cb4 chk8 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 121 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 120 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin0 | 119 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 72 | SPILL st48 ld16, STACK 64 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 93 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 123 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin0 | 119 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 151 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 144 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 138 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 137 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin3 | 140 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk12 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 142 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 136 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 134 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 134 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin3 | 138 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 121 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 121 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace0 stages1 lbmin4 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 121 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 120 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin0 | 119 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 152 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 148 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 146 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 145 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin3 | 148 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 80 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 80 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 123 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin0 | 119 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk16 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 121 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 120 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin0 | 119 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 72 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 72 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 123 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin0 | 119 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk24 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 95 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 95 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 120 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin0 | 119 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 120 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages1 lbmin4 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 80 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 96 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 123 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin0 | 119 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 119 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 121 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb4 chk8 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 119 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 120 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin0 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages1 lbmin4 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 119 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin0 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 160 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 157 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 157 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin3 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk12 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 158 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 152 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 157 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin3 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 123 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 127 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace0 stages1 lbmin4 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 119 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 120 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin0 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 158 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 157 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 160 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 162 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin3 | 161 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages1 lbmin4 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin0 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk16 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 119 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 120 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin0 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages1 lbmin4 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 119 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin0 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk24 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 159 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin0 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 121 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages1 lbmin4 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 124 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 121 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin0 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 126 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 125 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra2_adj | nxg4 cb7 chk8 inplace1 stages2 lbmin4 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 244 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 246 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 254 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 254 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch12 | 244 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 244 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 246 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 254 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 254 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch16 | 244 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 244 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 246 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 254 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 254 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch24 | 244 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 244 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 246 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 254 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 254 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 sch32 | 244 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 248 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 254 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 238 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 242 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch12 | 241 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 250 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 254 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 238 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 242 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch16 | 241 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 250 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 254 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 238 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 242 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch24 | 241 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 248 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 254 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 238 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 242 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | vt2_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 sch32 | 241 | STACK 32 |
