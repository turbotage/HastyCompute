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

### knot1d_o2_cb5 / kl_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 | 128 | 24000 | 119/16 (regs) K48 | 121/16 (regs+smem) K48 | 122/16 (regs) K48 | 128/16 (regs) K48 | 126/16 (regs+smem) K48 |
| nxg1 cb1 chk32 | 128 | 48000 | 124/12 (smem) K48 | 125/8 (smem) K48 | 126/16 (regs+smem) K48 | 139/12 (regs) K48 | 128/8 (smem) K48 |
| nxg1 cb2 chk16 | 128 | 24000 | 150/12 (regs) K48 | 150/12 (regs) K48 | 151/12 (regs) K48 | 156/12 (regs) K48 | 151/12 (regs) K48 |
| nxg1 cb2 chk32 | 128 | 48000 | 155/12 (regs+smem) K48 | 154/8 (smem) K48 | 157/12 (regs) K48 | 156/12 (regs) K48 | 152/8 (smem) K48 |
| nxg2 cb1 chk16 | 128 | 19648 | 102/16 (regs) K48 | 105/16 (regs+smem) K48 | 104/16 (regs) K48 | 119/16 (regs) K48 | 111/16 (regs+smem) K48 |
| nxg2 cb1 chk32 | 128 | 39296 | 102/16 (regs+smem) K48 | 105/8 (smem) K48 | 104/16 (regs) K48 | 119/16 (regs) K48 | 111/8 (smem) K48 |
| nxg2 cb2 chk16 | 128 | 19904 | 117/16 (regs) K48 | 119/16 (regs+smem) K48 | 120/16 (regs) K48 | 133/12 (regs) K48 | 128/16 (regs+smem) K48 |
| nxg2 cb2 chk32 | 128 | 39808 | 124/16 (regs+smem) K48 | 125/8 (smem) K48 | 126/16 (regs) K48 | 139/12 (regs) K48 | 133/8 (smem) K48 |
| nxg2 cb4 chk16 | 128 | 24000 | 150/12 (regs) K48 | 149/12 (regs) K48 | 151/12 (regs) K48 | 156/12 (regs) K48 | 151/12 (regs) K48 |
| nxg2 cb4 chk32 | 128 | 48000 | 156/12 (regs+smem) K48 | 157/8 (smem) K48 | 159/12 (regs) K48 | 157/12 (regs) K48 | 151/8 (smem) K48 |
| nxg4 cb2 chk16 | 128 | 17600 | 99/16 (regs) K48 | 99/16 (regs) K48 | 101/16 (regs) K48 | 117/16 (regs) K48 | 109/16 (regs) K48 |
| nxg4 cb2 chk32 | 128 | 35200 | 99/16 (regs+smem) K48 | 99/8 (smem) K48 | 101/16 (regs) K48 | 117/16 (regs) K48 | 109/8 (smem) K48 |
| nxg4 cb3 chk16 | 128 | 19776 | 114/16 (regs) K48 | 113/16 (regs+smem) K48 | 115/16 (regs) K48 | 125/16 (regs) K48 | 121/16 (regs+smem) K48 |
| nxg4 cb3 chk32 | 128 | 39552 | 114/16 (regs+smem) K48 | 113/8 (smem) K48 | 115/16 (regs) K48 | 125/16 (regs) K48 | 121/8 (smem) K48 |
| nxg4 cb4 chk16 | 128 | 19904 | 117/16 (regs) K48 | 119/16 (regs+smem) K48 | 121/16 (regs) K48 | 133/12 (regs) K48 | 127/16 (regs+smem) K48 |
| nxg4 cb4 chk32 | 128 | 39808 | 124/16 (regs+smem) K48 | 125/8 (smem) K48 | 125/16 (regs) K48 | 138/12 (regs) K48 | 133/8 (smem) K48 |
| nxg4 cb7 chk16 | 128 | 24000 | 145/12 (regs) K48 | 145/12 (regs) K48 | 149/12 (regs) K48 | 154/12 (regs) K48 | 148/12 (regs) K48 |
| nxg4 cb7 chk32 | 128 | 48000 | 151/12 (regs+smem) K48 | 151/8 (smem) K48 | 152/12 (regs) K48 | 152/12 (regs) K48 | 148/8 (smem) K48 |

### knot1d_o2_cb5 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 28032 | 119/16 (regs) K32 | 119/12 (smem) K32 | 120/16 (regs) K32 | 128/16 (regs) K32 | 128/12 (smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 51968 | 122/12 (smem) K32 | 123/4 (smem) K32 | 124/16 (regs+smem) K32 | 136/12 (regs) K32 | 134/4 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 28032 | 147/12 (regs) K32 | 145/12 (regs+smem) K32 | 149/12 (regs) K32 | 158/12 (regs) K32 | 155/12 (regs+smem) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 51968 | 153/12 (regs+smem) K32 | 152/4 (smem) K32 | 154/12 (regs) K32 | 158/12 (regs) K32 | 156/4 (smem) K32 |
| nxg2 cb1 chk16 listmax1024 | 128 | 23680 | 101/16 (regs) K32 | 102/16 (regs+smem) K32 | 103/16 (regs) K32 | 116/16 (regs) K32 | 113/16 (regs+smem) K32 |
| nxg2 cb1 chk32 listmax1024 | 128 | 43264 | 101/12 (smem) K32 | 102/8 (smem) K32 | 103/16 (regs) K32 | 116/16 (regs) K32 | 113/8 (smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 23936 | 114/16 (regs) K32 | 115/16 (regs+smem) K32 | 118/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs+smem) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 43776 | 122/12 (smem) K32 | 124/8 (smem) K32 | 124/16 (regs) K32 | 137/12 (regs) K32 | 134/8 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 28032 | 146/12 (regs) K32 | 147/12 (regs+smem) K32 | 148/12 (regs) K32 | 159/12 (regs) K32 | 158/12 (regs+smem) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 51968 | 153/12 (regs+smem) K32 | 155/4 (smem) K32 | 155/12 (regs) K32 | 158/12 (regs) K32 | 155/4 (smem) K32 |
| nxg4 cb2 chk16 listmax1024 | 128 | 21632 | 96/20 (regs) K32 | 96/16 (smem) K32 | 103/16 (regs) K32 | 111/16 (regs) K32 | 111/16 (regs+smem) K32 |
| nxg4 cb2 chk32 listmax1024 | 128 | 39168 | 96/16 (smem) K32 | 96/8 (smem) K32 | 103/16 (regs) K32 | 111/16 (regs) K32 | 111/8 (smem) K32 |
| nxg4 cb3 chk16 listmax1024 | 128 | 23808 | 112/16 (regs) K32 | 112/16 (regs+smem) K32 | 112/16 (regs) K32 | 122/16 (regs) K32 | 120/16 (regs+smem) K32 |
| nxg4 cb3 chk32 listmax1024 | 128 | 43520 | 112/12 (smem) K32 | 112/8 (smem) K32 | 112/16 (regs) K32 | 122/16 (regs) K32 | 120/8 (smem) K32 |
| nxg4 cb4 chk16 listmax1024 | 128 | 23936 | 114/16 (regs) K32 | 115/16 (regs+smem) K32 | 117/16 (regs) K32 | 128/16 (regs) K32 | 128/16 (regs+smem) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 43776 | 122/12 (smem) K32 | 124/8 (smem) K32 | 124/16 (regs) K32 | 136/12 (regs) K32 | 134/8 (smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 28032 | 143/12 (regs) K32 | 144/12 (regs+smem) K32 | 145/12 (regs) K32 | 158/12 (regs) K32 | 154/12 (regs+smem) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 51968 | 147/12 (regs+smem) K32 | 149/4 (smem) K32 | 148/12 (regs) K32 | 156/12 (regs) K32 | 152/4 (smem) K32 |

### static_o0_10x10 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 15808 | 104/16 (regs) K32 | 105/16 (regs) K32 | 107/16 (regs) K32 | 110/16 (regs) K32 | 104/16 (regs) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 27520 | 109/16 (regs) K32 | 110/12 (smem) K32 | 112/16 (regs) K32 | 116/16 (regs) K32 | 110/12 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 16192 | 131/12 (regs) K32 | 131/12 (regs) K32 | 133/12 (regs) K32 | 138/12 (regs) K32 | 132/12 (regs) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 28288 | 137/12 (regs) K32 | 137/12 (regs+smem) K32 | 139/12 (regs) K32 | 141/12 (regs) K32 | 135/12 (regs+smem) K32 |
| nxg1 cb4 chk16 listmax1024 | 128 | 24640 | 168/12 (regs) S32 K72 | 168/12 (regs+smem) S32 K72 | 168/12 (regs) S96 K104 | 167/12 (regs) K32 | 168/12 (regs+smem) K32 |
| nxg1 cb4 chk32 listmax1024 | 128 | 45184 | 168/12 (regs+smem) S48 K80 | 168/8 (smem) S48 K80 | 168/12 (regs) S112 K112 | 168/12 (regs) S16 K40 | 168/8 (smem) S16 K40 |
| nxg2 cb2 chk16 listmax1024 | 128 | 11712 | 100/16 (regs) K32 | 101/16 (regs) K32 | 104/16 (regs) K32 | 108/16 (regs) K32 | 101/16 (regs) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 19328 | 107/16 (regs) K32 | 108/16 (regs) K32 | 110/16 (regs) K32 | 114/16 (regs) K32 | 108/16 (regs) K32 |
| nxg2 cb3 chk16 listmax1024 | 128 | 15936 | 124/16 (regs) K32 | 125/16 (regs) K32 | 126/16 (regs) K32 | 128/16 (regs) K32 | 121/16 (regs) K32 |
| nxg2 cb3 chk32 listmax1024 | 128 | 27776 | 128/16 (regs) K32 | 128/12 (smem) K32 | 130/12 (regs) K32 | 134/12 (regs) K32 | 128/12 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 16192 | 133/12 (regs) K32 | 133/12 (regs) K32 | 135/12 (regs) K32 | 139/12 (regs) K32 | 133/12 (regs) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 28288 | 139/12 (regs) K32 | 139/12 (regs+smem) K32 | 142/12 (regs) K32 | 143/12 (regs) K32 | 136/12 (regs+smem) K32 |
| nxg2 cb7 chk16 listmax1024 | 128 | 24640 | 168/12 (regs) S16 K48 | 168/12 (regs+smem) S16 K48 | 168/12 (regs) S20 K56 | 164/12 (regs) K32 | 163/12 (regs+smem) K32 |
| nxg2 cb7 chk32 listmax1024 | 128 | 45184 | 168/12 (regs+smem) S28 K64 | 168/8 (smem) S28 K64 | 168/12 (regs) S44 K80 | 161/12 (regs) K32 | 159/8 (smem) K32 |
| nxg4 cb13 chk16 listmax1024 | 128 | 22592 | 168/12 (regs) S20 K56 | 168/12 (regs) S12 K48 | 168/12 (regs) S16 K48 | 160/12 (regs) K32 | 163/12 (regs) K32 |
| nxg4 cb13 chk32 listmax1024 | 128 | 41088 | 168/12 (regs+smem) S32 K64 | 168/8 (smem) S28 K64 | 168/12 (regs) S48 K80 | 162/12 (regs) K32 | 166/8 (smem) K32 |
| nxg4 cb4 chk16 listmax1024 | 128 | 11712 | 100/16 (regs) K32 | 101/16 (regs) K32 | 104/16 (regs) K32 | 107/16 (regs) K32 | 100/16 (regs) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 19328 | 107/16 (regs) K32 | 108/16 (regs) K32 | 110/16 (regs) K32 | 113/16 (regs) K32 | 107/16 (regs) K32 |
| nxg4 cb5 chk16 listmax1024 | 128 | 13760 | 116/16 (regs) K32 | 117/16 (regs) K32 | 118/16 (regs) K32 | 119/16 (regs) K32 | 114/16 (regs) K32 |
| nxg4 cb5 chk32 listmax1024 | 128 | 23424 | 122/16 (regs) K32 | 122/16 (regs+smem) K32 | 123/16 (regs) K32 | 125/16 (regs) K32 | 120/16 (regs+smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 16064 | 136/12 (regs) K32 | 136/12 (regs) K32 | 137/12 (regs) K32 | 138/12 (regs) K32 | 132/12 (regs) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 28032 | 142/12 (regs) K32 | 142/12 (regs+smem) K32 | 144/12 (regs) K32 | 145/12 (regs) K32 | 139/12 (regs+smem) K32 |

### static_o0_48x1 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 18240 | 124/16 (regs) K32 | 124/16 (regs) K32 | 124/16 (regs) K32 | 131/12 (regs) K32 | 125/16 (regs) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 32384 | 142/12 (regs) K32 | 142/12 (regs+smem) K32 | 145/12 (regs) K32 | 140/12 (regs) K32 | 137/12 (regs+smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 20160 | 154/12 (regs) K32 | 154/12 (regs) K32 | 154/12 (regs) K32 | 145/12 (regs) K32 | 147/12 (regs) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 36224 | 168/12 (regs) K32 | 157/8 (smem) K32 | 159/12 (regs) K32 | 156/12 (regs) K32 | 156/8 (smem) K32 |
| nxg2 cb1 chk16 listmax1024 | 128 | 12096 | 97/16 (regs) K32 | 96/20 (regs) K32 | 99/16 (regs) K32 | 112/16 (regs) K32 | 106/16 (regs) K32 |
| nxg2 cb1 chk32 listmax1024 | 128 | 20096 | 107/16 (regs) K32 | 107/16 (regs+smem) K32 | 109/16 (regs) K32 | 124/16 (regs) K32 | 118/16 (regs+smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 14144 | 121/16 (regs) K32 | 121/16 (regs) K32 | 121/16 (regs) K32 | 129/12 (regs) K32 | 124/16 (regs) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 24192 | 138/12 (regs) K32 | 138/12 (regs) K32 | 142/12 (regs) K32 | 139/12 (regs) K32 | 137/12 (regs) K32 |
| nxg2 cb3 chk16 listmax1024 | 128 | 20160 | 147/12 (regs) K32 | 147/12 (regs) K32 | 149/12 (regs) K32 | 143/12 (regs) K32 | 140/12 (regs) K32 |
| nxg2 cb3 chk32 listmax1024 | 128 | 36224 | 166/12 (regs) K32 | 159/8 (smem) K32 | 161/12 (regs) K32 | 156/12 (regs) K32 | 154/8 (smem) K32 |
| nxg4 cb2 chk16 listmax1024 | 128 | 10048 | 95/20 (regs) K32 | 95/20 (regs) K32 | 98/16 (regs) K32 | 103/16 (regs) K32 | 98/16 (regs) K32 |
| nxg4 cb2 chk32 listmax1024 | 128 | 16000 | 106/16 (regs) K32 | 106/16 (regs) K32 | 108/16 (regs) K32 | 115/16 (regs) K32 | 110/16 (regs) K32 |
| nxg4 cb3 chk16 listmax1024 | 128 | 13120 | 114/16 (regs) K32 | 114/16 (regs) K32 | 116/16 (regs) K32 | 116/16 (regs) K32 | 114/16 (regs) K32 |
| nxg4 cb3 chk32 listmax1024 | 128 | 22144 | 129/12 (regs) K32 | 129/12 (regs) K32 | 131/12 (regs) K32 | 138/12 (regs) K32 | 132/12 (regs) K32 |
| nxg4 cb6 chk16 listmax1024 | 128 | 18112 | 146/12 (regs) K32 | 146/12 (regs) K32 | 148/12 (regs) K32 | 145/12 (regs) K32 | 142/12 (regs) K32 |
| nxg4 cb6 chk32 listmax1024 | 128 | 32128 | 168/12 (regs) K32 | 159/12 (regs+smem) K32 | 160/12 (regs) K32 | 152/12 (regs) K32 | 151/12 (regs+smem) K32 |

### static_o2_cb5 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 28032 | 117/16 (regs) K32 | 118/12 (smem) K32 | 121/16 (regs) K32 | 128/16 (regs) K32 | 124/12 (smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 51968 | 122/12 (smem) K32 | 123/4 (smem) K32 | 125/16 (regs+smem) K32 | 136/12 (regs) K32 | 128/4 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 28032 | 147/12 (regs) K32 | 145/12 (regs+smem) K32 | 151/12 (regs) K32 | 161/12 (regs) K32 | 154/12 (regs+smem) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 51968 | 153/12 (regs+smem) K32 | 152/4 (smem) K32 | 155/12 (regs) K32 | 158/12 (regs) K32 | 156/4 (smem) K32 |
| nxg2 cb1 chk16 listmax1024 | 128 | 23680 | 102/16 (regs) K32 | 102/16 (regs+smem) K32 | 104/16 (regs) K32 | 117/16 (regs) K32 | 109/16 (regs+smem) K32 |
| nxg2 cb1 chk32 listmax1024 | 128 | 43264 | 102/12 (smem) K32 | 102/8 (smem) K32 | 104/16 (regs) K32 | 117/16 (regs) K32 | 109/8 (smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 23936 | 114/16 (regs) K32 | 115/16 (regs+smem) K32 | 117/16 (regs) K32 | 128/16 (regs) K32 | 126/16 (regs+smem) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 43776 | 123/12 (smem) K32 | 122/8 (smem) K32 | 125/16 (regs) K32 | 137/12 (regs) K32 | 128/8 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 28032 | 146/12 (regs) K32 | 147/12 (regs+smem) K32 | 151/12 (regs) K32 | 160/12 (regs) K32 | 156/12 (regs+smem) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 51968 | 154/12 (regs+smem) K32 | 155/4 (smem) K32 | 159/12 (regs) K32 | 161/12 (regs) K32 | 155/4 (smem) K32 |
| nxg4 cb2 chk16 listmax1024 | 128 | 21632 | 96/20 (regs) K32 | 96/16 (smem) K32 | 101/16 (regs) K32 | 113/16 (regs) K32 | 107/16 (regs+smem) K32 |
| nxg4 cb2 chk32 listmax1024 | 128 | 39168 | 96/16 (smem) K32 | 96/8 (smem) K32 | 101/16 (regs) K32 | 113/16 (regs) K32 | 107/8 (smem) K32 |
| nxg4 cb3 chk16 listmax1024 | 128 | 23808 | 112/16 (regs) K32 | 112/16 (regs+smem) K32 | 113/16 (regs) K32 | 124/16 (regs) K32 | 119/16 (regs+smem) K32 |
| nxg4 cb3 chk32 listmax1024 | 128 | 43520 | 112/12 (smem) K32 | 112/8 (smem) K32 | 113/16 (regs) K32 | 124/16 (regs) K32 | 119/8 (smem) K32 |
| nxg4 cb4 chk16 listmax1024 | 128 | 23936 | 114/16 (regs) K32 | 115/16 (regs+smem) K32 | 119/16 (regs) K32 | 128/16 (regs) K32 | 125/16 (regs+smem) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 43776 | 123/12 (smem) K32 | 122/8 (smem) K32 | 125/16 (regs) K32 | 137/12 (regs) K32 | 128/8 (smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 28032 | 145/12 (regs) K32 | 145/12 (regs+smem) K32 | 144/12 (regs) K32 | 157/12 (regs) K32 | 151/12 (regs+smem) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 51968 | 150/12 (regs+smem) K32 | 150/4 (smem) K32 | 151/12 (regs) K32 | 157/12 (regs) K32 | 152/4 (smem) K32 |

### vel2d_o3_cb10 / kl_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 | 128 | 26256 | 139/12 (regs) K48 | 139/12 (regs+smem) K48 | 144/12 (regs) K48 | 153/12 (regs) K48 | 146/12 (regs+smem) K48 |
| nxg1 cb1 chk32 | 128 | 52112 | 147/12 (regs+smem) K48 | 147/4 (smem) K48 | 149/12 (regs) K48 | 152/12 (regs) K48 | 148/4 (smem) K48 |
| nxg1 cb2 chk16 | 128 | 26640 | 158/12 (regs) K48 | 160/12 (regs+smem) K48 | 158/12 (regs) K48 | 159/12 (regs) K48 | 160/12 (regs+smem) K48 |
| nxg1 cb2 chk32 | 128 | 52880 | 158/12 (regs+smem) K48 | 162/4 (smem) K48 | 161/12 (regs) K48 | 159/12 (regs) K48 | 159/4 (smem) K48 |
| nxg1 cb4 chk16 | 128 | 35088 | 168/12 (regs) S228 K160 | 168/8 (smem) S228 K160 | 168/12 (regs) S224 K160 | 168/12 (regs) S80 K96 | 168/8 (smem) S80 K96 |
| nxg1 cb4 chk32 | 128 | 69776 | 168/8 (smem) S272 K176 | 168/4 (smem) S272 K176 | 168/12 (regs+smem) S268 K176 | 168/12 (regs+smem) S120 K112 | 168/4 (smem) S120 K112 |
| nxg2 cb2 chk16 | 128 | 22160 | 138/12 (regs) K48 | 137/12 (regs) K48 | 143/12 (regs) K48 | 152/12 (regs) K48 | 147/12 (regs) K48 |
| nxg2 cb2 chk32 | 128 | 43920 | 145/12 (regs+smem) K48 | 144/8 (smem) K48 | 147/12 (regs) K48 | 153/12 (regs) K48 | 148/8 (smem) K48 |
| nxg2 cb3 chk16 | 128 | 26384 | 156/12 (regs) K48 | 157/12 (regs+smem) K48 | 156/12 (regs) K48 | 158/12 (regs) K48 | 151/12 (regs+smem) K48 |
| nxg2 cb3 chk32 | 128 | 52368 | 157/12 (regs+smem) K48 | 157/4 (smem) K48 | 158/12 (regs) K48 | 158/12 (regs) K48 | 156/4 (smem) K48 |
| nxg2 cb4 chk16 | 128 | 26640 | 164/12 (regs) K48 | 162/12 (regs+smem) K48 | 162/12 (regs) K48 | 159/12 (regs) K48 | 157/12 (regs+smem) K48 |
| nxg2 cb4 chk32 | 128 | 52880 | 161/12 (regs+smem) K48 | 163/4 (smem) K48 | 161/12 (regs) K48 | 157/12 (regs) K48 | 160/4 (smem) K48 |
| nxg2 cb7 chk16 | 128 | 35088 | 168/12 (regs) S68 K96 | 168/8 (smem) S68 K96 | 168/12 (regs) S68 K96 | 166/12 (regs) K48 | 166/8 (smem) K48 |
| nxg2 cb7 chk32 | 128 | 69776 | 168/8 (smem) S80 K112 | 168/4 (smem) S80 K112 | 168/12 (regs+smem) S96 K128 | 167/12 (regs+smem) K48 | 168/4 (smem) K48 |
| nxg4 cb13 chk16 | 128 | 33040 | 168/12 (regs) S24 K80 | 168/12 (regs+smem) S24 K80 | 168/12 (regs) S28 K80 | 165/12 (regs) K48 | 167/12 (regs+smem) K48 |
| nxg4 cb13 chk32 | 128 | 65680 | 168/8 (smem) S36 K96 | 168/4 (smem) S36 K96 | 168/12 (regs+smem) S36 K96 | 164/12 (regs+smem) K48 | 164/4 (smem) K48 |
| nxg4 cb4 chk16 | 128 | 22160 | 138/12 (regs) K48 | 137/12 (regs) K48 | 141/12 (regs) K48 | 153/12 (regs) K48 | 148/12 (regs) K48 |
| nxg4 cb4 chk32 | 128 | 43920 | 144/12 (regs+smem) K48 | 144/8 (smem) K48 | 147/12 (regs) K48 | 153/12 (regs) K48 | 148/8 (smem) K48 |
| nxg4 cb5 chk16 | 128 | 24208 | 150/12 (regs) K48 | 151/12 (regs) K48 | 154/12 (regs) K48 | 153/12 (regs) K48 | 150/12 (regs) K48 |
| nxg4 cb5 chk32 | 128 | 48016 | 156/12 (regs+smem) K48 | 157/8 (smem) K48 | 158/12 (regs) K48 | 153/12 (regs) K48 | 150/8 (smem) K48 |
| nxg4 cb7 chk16 | 128 | 26512 | 158/12 (regs) K48 | 160/12 (regs+smem) K48 | 159/12 (regs) K48 | 159/12 (regs) K48 | 156/12 (regs+smem) K48 |
| nxg4 cb7 chk32 | 128 | 52624 | 162/12 (regs+smem) K48 | 162/4 (smem) K48 | 162/12 (regs) K48 | 159/12 (regs) K48 | 156/4 (smem) K48 |

### vel2d_o3_cb10 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 29632 | 134/12 (regs) K32 | 135/12 (regs+smem) K32 | 135/12 (regs) K32 | 144/12 (regs) K32 | 138/12 (regs+smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 55168 | 139/8 (smem) K32 | 141/4 (smem) K32 | 143/12 (regs) K32 | 153/12 (regs) K32 | 141/4 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 30016 | 157/12 (regs) K32 | 159/12 (regs+smem) K32 | 158/12 (regs) K32 | 156/12 (regs) K32 | 153/12 (regs+smem) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 55936 | 159/8 (smem) K32 | 159/4 (smem) K32 | 162/12 (regs) K32 | 160/12 (regs) K32 | 156/4 (smem) K32 |
| nxg1 cb4 chk16 listmax1024 | 128 | 38464 | 168/12 (regs) S124 K120 | 168/8 (smem) S124 K120 | 168/12 (regs) S124 K120 | 168/12 (regs) S16 K40 | 168/8 (smem) S16 K40 |
| nxg1 cb4 chk32 listmax1024 | 128 | 72832 | 168/8 (smem) S152 K136 | 168/4 (smem) S152 K136 | 168/12 (regs+smem) S152 K128 | 168/12 (regs+smem) S32 K48 | 168/4 (smem) S48 K64 |
| nxg2 cb2 chk16 listmax1024 | 128 | 25536 | 133/12 (regs) K32 | 134/12 (regs+smem) K32 | 138/12 (regs) K32 | 144/12 (regs) K32 | 138/12 (regs+smem) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 46976 | 137/12 (regs+smem) K32 | 139/8 (smem) K32 | 140/12 (regs) K32 | 154/12 (regs) K32 | 146/8 (smem) K32 |
| nxg2 cb3 chk16 listmax1024 | 128 | 29760 | 156/12 (regs) K32 | 152/12 (regs+smem) K32 | 156/12 (regs) K32 | 158/12 (regs) K32 | 152/12 (regs+smem) K32 |
| nxg2 cb3 chk32 listmax1024 | 128 | 55424 | 157/8 (smem) K32 | 157/4 (smem) K32 | 158/12 (regs) K32 | 156/12 (regs) K32 | 152/4 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 30016 | 157/12 (regs) K32 | 158/12 (regs+smem) K32 | 160/12 (regs) K32 | 160/12 (regs) K32 | 156/12 (regs+smem) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 55936 | 160/8 (smem) K32 | 160/4 (smem) K32 | 162/12 (regs) K32 | 159/12 (regs) K32 | 158/4 (smem) K32 |
| nxg2 cb7 chk16 listmax1024 | 128 | 38464 | 168/12 (regs) S48 K72 | 168/8 (smem) S48 K72 | 168/12 (regs) S48 K72 | 165/12 (regs) K32 | 165/8 (smem) K32 |
| nxg2 cb7 chk32 listmax1024 | 128 | 72832 | 168/8 (smem) S76 K96 | 168/4 (smem) S76 K96 | 168/12 (regs+smem) S76 K96 | 164/12 (regs+smem) K32 | 167/4 (smem) K32 |
| nxg4 cb13 chk16 listmax1024 | 128 | 36416 | 168/12 (regs) S24 K56 | 168/8 (smem) S16 K48 | 168/12 (regs) K32 | 163/12 (regs) K32 | 162/8 (smem) K32 |
| nxg4 cb13 chk32 listmax1024 | 128 | 68736 | 168/8 (smem) S32 K64 | 168/4 (smem) S32 K64 | 168/12 (regs+smem) S32 K64 | 162/12 (regs+smem) K32 | 162/4 (smem) K32 |
| nxg4 cb4 chk16 listmax1024 | 128 | 25536 | 134/12 (regs) K32 | 134/12 (regs+smem) K32 | 138/12 (regs) K32 | 142/12 (regs) K32 | 136/12 (regs+smem) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 46976 | 137/12 (regs+smem) K32 | 138/8 (smem) K32 | 140/12 (regs) K32 | 152/12 (regs) K32 | 144/8 (smem) K32 |
| nxg4 cb5 chk16 listmax1024 | 128 | 27584 | 143/12 (regs) K32 | 143/12 (regs+smem) K32 | 147/12 (regs) K32 | 154/12 (regs) K32 | 146/12 (regs+smem) K32 |
| nxg4 cb5 chk32 listmax1024 | 128 | 51072 | 148/12 (regs+smem) K32 | 149/4 (smem) K32 | 151/12 (regs) K32 | 154/12 (regs) K32 | 152/4 (smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 29888 | 157/12 (regs) K32 | 159/12 (regs+smem) K32 | 162/12 (regs) K32 | 158/12 (regs) K32 | 152/12 (regs+smem) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 55680 | 159/8 (smem) K32 | 159/4 (smem) K32 | 158/12 (regs) K32 | 156/12 (regs) K32 | 154/4 (smem) K32 |

### vel2d_o3_cb5 / kl_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 | 128 | 26256 | 139/12 (regs) K48 | 139/12 (regs+smem) K48 | 144/12 (regs) K48 | 153/12 (regs) K48 | 146/12 (regs+smem) K48 |
| nxg1 cb1 chk32 | 128 | 52112 | 147/12 (regs+smem) K48 | 147/4 (smem) K48 | 149/12 (regs) K48 | 152/12 (regs) K48 | 148/4 (smem) K48 |
| nxg1 cb2 chk16 | 128 | 26256 | 158/12 (regs) K48 | 160/12 (regs+smem) K48 | 158/12 (regs) K48 | 159/12 (regs) K48 | 159/12 (regs+smem) K48 |
| nxg1 cb2 chk32 | 128 | 52112 | 158/12 (regs+smem) K48 | 162/4 (smem) K48 | 161/12 (regs) K48 | 160/12 (regs) K48 | 161/4 (smem) K48 |
| nxg2 cb1 chk16 | 128 | 21904 | 124/16 (regs) K48 | 124/16 (regs+smem) K48 | 127/16 (regs) K48 | 136/12 (regs) K48 | 128/16 (regs+smem) K48 |
| nxg2 cb1 chk32 | 128 | 43408 | 124/12 (smem) K48 | 124/8 (smem) K48 | 127/16 (regs) K48 | 136/12 (regs) K48 | 128/8 (smem) K48 |
| nxg2 cb2 chk16 | 128 | 22160 | 138/12 (regs) K48 | 137/12 (regs) K48 | 143/12 (regs) K48 | 152/12 (regs) K48 | 147/12 (regs) K48 |
| nxg2 cb2 chk32 | 128 | 43920 | 145/12 (regs+smem) K48 | 144/8 (smem) K48 | 147/12 (regs) K48 | 153/12 (regs) K48 | 148/8 (smem) K48 |
| nxg2 cb4 chk16 | 128 | 26256 | 164/12 (regs) K48 | 162/12 (regs+smem) K48 | 162/12 (regs) K48 | 157/12 (regs) K48 | 162/12 (regs+smem) K48 |
| nxg2 cb4 chk32 | 128 | 52112 | 161/12 (regs+smem) K48 | 163/4 (smem) K48 | 161/12 (regs) K48 | 157/12 (regs) K48 | 158/4 (smem) K48 |
| nxg4 cb2 chk16 | 128 | 19856 | 122/16 (regs) K48 | 122/16 (regs+smem) K48 | 127/16 (regs) K48 | 133/12 (regs) K48 | 128/16 (regs+smem) K48 |
| nxg4 cb2 chk32 | 128 | 39312 | 122/16 (regs+smem) K48 | 122/8 (smem) K48 | 127/16 (regs) K48 | 133/12 (regs) K48 | 128/8 (smem) K48 |
| nxg4 cb3 chk16 | 128 | 22032 | 135/12 (regs) K48 | 138/12 (regs) K48 | 137/12 (regs) K48 | 143/12 (regs) K48 | 138/12 (regs) K48 |
| nxg4 cb3 chk32 | 128 | 43664 | 135/12 (regs+smem) K48 | 138/8 (smem) K48 | 137/12 (regs) K48 | 143/12 (regs) K48 | 138/8 (smem) K48 |
| nxg4 cb4 chk16 | 128 | 22160 | 138/12 (regs) K48 | 137/12 (regs) K48 | 141/12 (regs) K48 | 153/12 (regs) K48 | 148/12 (regs) K48 |
| nxg4 cb4 chk32 | 128 | 43920 | 144/12 (regs+smem) K48 | 144/8 (smem) K48 | 147/12 (regs) K48 | 153/12 (regs) K48 | 148/8 (smem) K48 |
| nxg4 cb7 chk16 | 128 | 26256 | 160/12 (regs) K48 | 160/12 (regs+smem) K48 | 161/12 (regs) K48 | 159/12 (regs) K48 | 156/12 (regs+smem) K48 |
| nxg4 cb7 chk32 | 128 | 52112 | 160/12 (regs+smem) K48 | 160/4 (smem) K48 | 161/12 (regs) K48 | 159/12 (regs) K48 | 156/4 (smem) K48 |

### vel2d_o3_cb5 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 29632 | 134/12 (regs) K32 | 135/12 (regs+smem) K32 | 135/12 (regs) K32 | 144/12 (regs) K32 | 138/12 (regs+smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 55168 | 139/8 (smem) K32 | 141/4 (smem) K32 | 143/12 (regs) K32 | 153/12 (regs) K32 | 141/4 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 29632 | 157/12 (regs) K32 | 159/12 (regs+smem) K32 | 158/12 (regs) K32 | 156/12 (regs) K32 | 153/12 (regs+smem) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 55168 | 159/8 (smem) K32 | 159/4 (smem) K32 | 162/12 (regs) K32 | 161/12 (regs) K32 | 158/4 (smem) K32 |
| nxg2 cb1 chk16 listmax1024 | 128 | 25280 | 118/16 (regs) K32 | 121/12 (smem) K32 | 125/16 (regs) K32 | 128/16 (regs) K32 | 121/12 (smem) K32 |
| nxg2 cb1 chk32 listmax1024 | 128 | 46464 | 118/12 (smem) K32 | 121/8 (smem) K32 | 125/16 (regs+smem) K32 | 128/16 (regs+smem) K32 | 121/8 (smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 25536 | 133/12 (regs) K32 | 134/12 (regs+smem) K32 | 138/12 (regs) K32 | 144/12 (regs) K32 | 138/12 (regs+smem) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 46976 | 137/12 (regs+smem) K32 | 139/8 (smem) K32 | 140/12 (regs) K32 | 154/12 (regs) K32 | 146/8 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 29632 | 157/12 (regs) K32 | 158/12 (regs+smem) K32 | 160/12 (regs) K32 | 160/12 (regs) K32 | 156/12 (regs+smem) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 55168 | 160/8 (smem) K32 | 160/4 (smem) K32 | 162/12 (regs) K32 | 161/12 (regs) K32 | 160/4 (smem) K32 |
| nxg4 cb2 chk16 listmax1024 | 128 | 23232 | 120/16 (regs) K32 | 121/16 (regs+smem) K32 | 125/16 (regs) K32 | 128/16 (regs) K32 | 118/16 (regs+smem) K32 |
| nxg4 cb2 chk32 listmax1024 | 128 | 42368 | 120/12 (smem) K32 | 121/8 (smem) K32 | 125/16 (regs) K32 | 128/16 (regs) K32 | 118/8 (smem) K32 |
| nxg4 cb3 chk16 listmax1024 | 128 | 25408 | 128/16 (regs) K32 | 128/12 (smem) K32 | 132/12 (regs) K32 | 135/12 (regs) K32 | 128/12 (smem) K32 |
| nxg4 cb3 chk32 listmax1024 | 128 | 46720 | 128/12 (smem) K32 | 128/8 (smem) K32 | 132/12 (regs) K32 | 135/12 (regs) K32 | 128/8 (smem) K32 |
| nxg4 cb4 chk16 listmax1024 | 128 | 25536 | 134/12 (regs) K32 | 134/12 (regs+smem) K32 | 138/12 (regs) K32 | 142/12 (regs) K32 | 136/12 (regs+smem) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 46976 | 137/12 (regs+smem) K32 | 138/8 (smem) K32 | 140/12 (regs) K32 | 152/12 (regs) K32 | 144/8 (smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 29632 | 156/12 (regs) K32 | 157/12 (regs+smem) K32 | 158/12 (regs) K32 | 158/12 (regs) K32 | 152/12 (regs+smem) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 55168 | 157/8 (smem) K32 | 157/4 (smem) K32 | 158/12 (regs) K32 | 156/12 (regs) K32 | 154/4 (smem) K32 |

## Flags (spills, stack, no fit, below 8 warps / SM, compile errors)

| arch | case | family | cfg | regs | flags |
|---|---|---|---|---|---|
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 128 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 126 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 119 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 121 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 122 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 139 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 128 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 124 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 125 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 126 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 156 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 151 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 150 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 150 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 151 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 156 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 152 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 155 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 154 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 157 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 119 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 111 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 102 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 105 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 104 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 119 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 111 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 102 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 105 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 104 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 133 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 128 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 117 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 119 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 120 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 139 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 133 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 124 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 125 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 126 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 156 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 151 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 150 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 149 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 151 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 157 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 151 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 156 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 157 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 159 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 117 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 109 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 99 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 99 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 101 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 117 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 109 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 99 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 99 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 101 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 125 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 121 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 114 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 113 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 115 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 125 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 121 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 114 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 113 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 115 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 133 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 117 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 119 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 121 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 138 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 133 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 124 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 125 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 125 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 154 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 148 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 145 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 145 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 149 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 152 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 148 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 151 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 151 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 152 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 128 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 119 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 119 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 120 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 136 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 134 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 122 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 123 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 124 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 158 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 155 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 147 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 145 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 149 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 158 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 156 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 153 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 152 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 154 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 116 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 113 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 101 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 102 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 103 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 116 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 113 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 101 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 102 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 103 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 114 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 115 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 118 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 137 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 134 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 122 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 124 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 124 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 159 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 158 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 146 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 147 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 148 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 158 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 155 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 153 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 155 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 155 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 111 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 111 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 96 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 103 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 111 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 111 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 96 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 103 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 122 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 120 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 112 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 112 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 112 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 122 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 120 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 112 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 112 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 112 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 114 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 115 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 117 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 136 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 134 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 122 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 124 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 124 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 158 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 154 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 143 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 144 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 145 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 156 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 152 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 147 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 149 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 148 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 110 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 104 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 104 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 105 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 107 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 116 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 109 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 110 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 112 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 138 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 132 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 131 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 131 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 133 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 141 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 135 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 137 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 137 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 139 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 167 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 168 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 168 | SPILL st32 ld32, STACK 72 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 168 | SPILL st32 ld32, STACK 72 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 168 | SPILL st96 ld84, STACK 104 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 168 | SPILL st16 ld8, STACK 40 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 168 | SPILL st16 ld8, STACK 40 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 168 | SPILL st48 ld40, STACK 80 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 168 | SPILL st48 ld40, STACK 80 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 168 | SPILL st112 ld92, STACK 112 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 108 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 101 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 100 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 101 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 104 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 114 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 108 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 107 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 108 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 110 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 128 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 121 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 124 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 125 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 126 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 134 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 128 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 128 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 130 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 139 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 133 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 133 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 133 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 135 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 143 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 136 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 139 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 139 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 142 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 164 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 163 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 168 | SPILL st16 ld16, STACK 48 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 168 | SPILL st16 ld16, STACK 48 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 168 | SPILL st20 ld20, STACK 56 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 161 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 159 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 168 | SPILL st28 ld28, STACK 64 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 168 | SPILL st28 ld28, STACK 64 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 168 | SPILL st44 ld44, STACK 80 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 160 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 163 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 168 | SPILL st20 ld20, STACK 56 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 168 | SPILL st12 ld12, STACK 48 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 168 | SPILL st16 ld16, STACK 48 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 162 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 166 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 168 | SPILL st32 ld32, STACK 64 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 168 | SPILL st28 ld28, STACK 64 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 168 | SPILL st48 ld48, STACK 80 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 107 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 100 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 100 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 101 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 104 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 113 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 107 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 107 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 108 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 110 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 119 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 114 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 116 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 117 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 118 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 125 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 120 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 122 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 122 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 123 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 138 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 132 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 136 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 136 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 137 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 145 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 139 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 142 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 142 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 144 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 131 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 125 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 124 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 124 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 124 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 140 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 137 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 142 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 142 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 145 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 145 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 147 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 154 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 154 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 154 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 156 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 156 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 168 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 157 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 159 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 112 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 106 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 97 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 96 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 99 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 124 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 118 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 107 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 107 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 109 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 129 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 124 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 121 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 121 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 121 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 139 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 137 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 138 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 138 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 142 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 143 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 140 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 147 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 147 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 149 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 156 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 154 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 166 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 159 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 161 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 103 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 98 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 95 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 95 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 98 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 115 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 110 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 106 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 106 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 108 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 116 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 114 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 114 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 114 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 116 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 138 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 132 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 129 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 129 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 131 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 145 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 142 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 146 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 146 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 148 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 152 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 151 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 168 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 159 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 160 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 128 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 124 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 117 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 118 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 121 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 136 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 122 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 123 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 161 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 154 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 147 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 145 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 151 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 158 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 156 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 153 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 152 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 155 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 117 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 109 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 102 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 102 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 104 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 117 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 109 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 102 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 102 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 104 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 126 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 114 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 115 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 117 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 137 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 123 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 122 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 160 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 156 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 146 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 147 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 151 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 161 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 155 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 154 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 155 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 159 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 113 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 107 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 96 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 101 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 113 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 107 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 96 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 101 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 124 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 119 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 112 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 112 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 113 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 124 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 119 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 112 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 112 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 113 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 125 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 114 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 115 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 119 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 137 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 128 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 123 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 122 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 157 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 151 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 145 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 145 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 144 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 157 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 152 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 150 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 150 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 151 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 153 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 146 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 139 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 139 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 144 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 152 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 148 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 147 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 147 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 149 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 159 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 160 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 158 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 160 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 158 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 159 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 159 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 158 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 162 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 161 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 168 | SPILL st80 ld40, STACK 96 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 168 | SPILL st80 ld40, STACK 96 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 168 | SPILL st228 ld104, STACK 160 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 168 | SPILL st228 ld104, STACK 160 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 168 | SPILL st224 ld100, STACK 160 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 168 | SPILL st120 ld56, STACK 112 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 168 | SPILL st120 ld56, STACK 112, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 168 | SPILL st272 ld124, STACK 176 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 168 | SPILL st272 ld124, STACK 176, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 168 | SPILL st268 ld120, STACK 176 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 152 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 147 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 138 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 137 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 143 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 153 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 148 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 145 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 144 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 147 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 158 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 151 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 156 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 157 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 156 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 158 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 156 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 157 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 157 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 158 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 159 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 157 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 164 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 162 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 162 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 157 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 160 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 161 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 163 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 161 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 166 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 166 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 168 | SPILL st68 ld56, STACK 96 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 168 | SPILL st68 ld56, STACK 96 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 168 | SPILL st68 ld56, STACK 96 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 167 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 168 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 168 | SPILL st80 ld64, STACK 112 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 168 | SPILL st80 ld64, STACK 112, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 168 | SPILL st96 ld76, STACK 128 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 165 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 167 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 168 | SPILL st24 ld24, STACK 80 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 168 | SPILL st24 ld24, STACK 80 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 168 | SPILL st28 ld28, STACK 80 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 164 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 164 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 168 | SPILL st36 ld36, STACK 96 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 168 | SPILL st36 ld36, STACK 96, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 168 | SPILL st36 ld36, STACK 96 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 153 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 148 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 138 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 137 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 141 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 153 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 148 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 144 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 144 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 147 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 153 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 150 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 150 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 151 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 154 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 153 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 150 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 156 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 157 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 158 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 159 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 156 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 158 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 160 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 159 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 159 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 156 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 162 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 162 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 162 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 144 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 138 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 134 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 135 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 135 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 153 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 141 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 139 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 141 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 143 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 156 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 153 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 159 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 160 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 156 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 159 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 159 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 162 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 168 | SPILL st16 ld8, STACK 40 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 168 | SPILL st16 ld8, STACK 40 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 168 | SPILL st124 ld100, STACK 120 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 168 | SPILL st124 ld84, STACK 120 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 168 | SPILL st124 ld100, STACK 120 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 168 | SPILL st32 ld16, STACK 48 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 168 | SPILL st48 ld32, STACK 64, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 168 | SPILL st152 ld120, STACK 136 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 168 | SPILL st152 ld104, STACK 136, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 168 | SPILL st152 ld120, STACK 128 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 144 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 138 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 133 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 134 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 138 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 154 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 146 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 137 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 139 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 140 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 158 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 152 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 152 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 156 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 156 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 152 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 157 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 160 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 156 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 158 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 160 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 159 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 158 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 160 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 160 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 162 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 165 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 165 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 168 | SPILL st48 ld40, STACK 72 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 168 | SPILL st48 ld40, STACK 72 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 168 | SPILL st48 ld40, STACK 72 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 164 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 167 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 168 | SPILL st76 ld76, STACK 96 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 168 | SPILL st76 ld60, STACK 96, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 168 | SPILL st76 ld76, STACK 96 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 163 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 162 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 168 | SPILL st24 ld24, STACK 56 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 168 | SPILL st16 ld16, STACK 48 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 168 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 162 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 162 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 168 | SPILL st32 ld32, STACK 64 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 168 | SPILL st32 ld32, STACK 64, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 168 | SPILL st32 ld32, STACK 64 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 142 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 136 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 134 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 134 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 138 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 152 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 144 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 137 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 138 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 140 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 154 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 146 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 143 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 143 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 147 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 154 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 152 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 148 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 149 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 151 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 158 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 152 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 159 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 162 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 156 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 154 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 159 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 159 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 153 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 146 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 139 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 139 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 144 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 152 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 148 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 147 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 147 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 149 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 159 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 159 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 158 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 160 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 158 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 160 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 161 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 158 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 162 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 161 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 136 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 128 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 136 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 128 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 152 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 147 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 138 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 137 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 143 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 153 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 148 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 145 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 144 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 147 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 157 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 162 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 164 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 162 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 162 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 157 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 158 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 161 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 163 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 161 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 133 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 128 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 122 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 122 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 133 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 128 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 122 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 122 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 143 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 138 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 135 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 138 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 137 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 143 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 138 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 135 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 138 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 137 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 153 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 148 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 138 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 137 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 141 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 153 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 148 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 144 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 144 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 147 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 159 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 156 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 160 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 160 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 161 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 159 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 156 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 160 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 160 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 161 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 144 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 138 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 134 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 135 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 135 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 153 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 141 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 139 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 141 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 143 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 156 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 153 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 159 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 161 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 158 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 159 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 159 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 162 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 128 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 121 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 118 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 121 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 128 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 121 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 118 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 121 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 144 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 138 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 133 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 134 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 138 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 154 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 146 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 137 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 139 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 140 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 160 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 156 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 158 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 160 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 161 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 160 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 160 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 160 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 162 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 118 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 120 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 121 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 118 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 120 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 121 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 135 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 132 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 135 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 132 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 142 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 136 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 134 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 134 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 138 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 152 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 144 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 137 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 138 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 140 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 158 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 152 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 157 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 156 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 154 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 157 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 158 | STACK 32 |
