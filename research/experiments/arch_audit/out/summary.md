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

### fft / fft_axis

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 1mm x n800 fwd lpb4 tpl100 radices[8, 4, 5, 5] | 400 | 51200 | - | 58/13 (smem) | - | - | 48/13 (smem) |
| 1mm x n800 fwd lpb8 tpl100 radices[8, 4, 5, 5] | 800 | 102400 | 58/25 (regs+smem) | - | 58/25 (regs) | 50/25 (regs) | - |
| 1mm x n800 fwd pruned lpb4 tpl100 radices[8, 4, 5, 5] | 400 | 51200 | - | 44/13 (smem) | - | - | 48/13 (smem) |
| 1mm x n800 fwd pruned lpb8 tpl100 radices[8, 4, 5, 5] | 800 | 102400 | 44/25 (regs+smem) | - | 46/25 (regs) | 46/25 (regs) | - |
| 1mm x n800 inv pruned lpb4 tpl100 radices[8, 4, 5, 5] | 400 | 51200 | - | 47/13 (smem) | - | - | 48/13 (smem) |
| 1mm x n800 inv pruned lpb8 tpl100 radices[8, 4, 5, 5] | 800 | 102400 | 52/25 (regs+smem) | - | 50/25 (regs) | 44/25 (regs) | - |
| 1mm y/z n480 fwd lpb16 tpl60 radices[8, 4, 3, 5] | 960 | 122880 | 59/30 (regs+smem) | - | 58/30 (regs+smem) | 56/30 (regs+smem) | - |
| 1mm y/z n480 fwd lpb8 tpl60 radices[8, 4, 3, 5] | 480 | 61440 | - | 53/15 (smem) | - | - | 40/15 (smem) |
| 1mm y/z n480 fwd pruned lpb16 tpl60 radices[8, 4, 3, 5] | 960 | 122880 | 45/30 (regs+smem) | - | 48/30 (regs+smem) | 32/30 (smem) | - |
| 1mm y/z n480 fwd pruned lpb8 tpl60 radices[8, 4, 3, 5] | 480 | 61440 | - | 40/15 (smem) | - | - | 40/15 (smem) |
| 1mm y/z n480 inv pruned lpb16 tpl60 radices[8, 4, 3, 5] | 960 | 122880 | 50/30 (regs+smem) | - | 50/30 (regs+smem) | 44/30 (regs+smem) | - |
| 1mm y/z n480 inv pruned lpb8 tpl60 radices[8, 4, 3, 5] | 480 | 61440 | - | 40/15 (smem) | - | - | 54/15 (smem) |
| 2mm x/y n288 fwd lpb16 tpl36 radices[8, 4, 3, 3] | 576 | 73728 | 56/36 (regs+smem) | 56/18 (smem) | 56/36 (regs) | 32/54 (regs+smem+threads) | 46/18 (smem) |
| 2mm x/y n288 fwd pruned lpb16 tpl36 radices[8, 4, 3, 3] | 576 | 73728 | 43/36 (regs+smem) | 44/18 (smem) | 46/36 (regs) | 43/36 (regs) | 50/18 (smem) |
| 2mm x/y n288 inv pruned lpb16 tpl36 radices[8, 4, 3, 3] | 576 | 73728 | 48/36 (regs+smem) | 48/18 (smem) | 50/36 (regs) | 43/36 (regs) | 49/18 (smem) |
| 2mm z n140 fwd lpb16 tpl20 radices[4, 5, 7] | 320 | 35840 | 48/40 (regs+smem) | 48/20 (smem) | 44/40 (regs) | 46/40 (regs) | 46/20 (smem) |
| 2mm z n140 fwd pruned lpb16 tpl20 radices[4, 5, 7] | 320 | 35840 | 44/40 (regs+smem) | 48/20 (smem) | 44/40 (regs) | 41/40 (regs) | 48/20 (smem) |
| 2mm z n140 inv pruned lpb16 tpl20 radices[4, 5, 7] | 320 | 35840 | 43/40 (regs+smem) | 45/20 (smem) | 42/40 (regs) | 46/40 (regs) | 48/20 (smem) |

### knot1d_o2_cb5 / kl_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 | 128 | 24000 | 119/16 (regs) K48 | 80/16 (smem) K48 | 96/20 (regs) K48 | 80/24 (regs) K48 | 80/16 (smem) K48 |
| nxg1 cb1 chk32 | 128 | 48000 | 125/12 (smem) K48 | 96/8 (smem) K48 | 96/16 (smem) K48 | 80/16 (smem) K48 | 96/8 (smem) K48 |
| nxg1 cb2 chk16 | 128 | 24000 | 128/16 (regs) K48 | 128/16 (regs+smem) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 127/16 (regs+smem) K48 |
| nxg1 cb2 chk32 | 128 | 48000 | 155/12 (regs+smem) K48 | 128/8 (smem) K48 | 128/16 (regs+smem) K48 | 125/16 (regs+smem) K48 | 124/8 (smem) K48 |
| nxg2 cb1 chk16 | 128 | 19648 | 96/20 (regs) K48 | 64/16 (smem) S12 K64 | 64/32 (regs) S16 K64 | 72/28 (regs) K48 | 64/16 (smem) K48 |
| nxg2 cb1 chk32 | 128 | 39296 | 96/16 (smem) K48 | 64/8 (smem) S12 K64 | 64/20 (smem) S16 K64 | 72/20 (smem) K48 | 64/8 (smem) K48 |
| nxg2 cb2 chk16 | 128 | 19904 | 96/20 (regs) K48 | 80/16 (smem) K48 | 80/24 (regs) K48 | 80/24 (regs) K48 | 72/16 (smem) K48 |
| nxg2 cb2 chk32 | 128 | 39808 | 125/16 (regs+smem) K48 | 96/8 (smem) K48 | 96/20 (regs+smem) K48 | 72/20 (smem) K48 | 72/8 (smem) S20 K64 |
| nxg2 cb4 chk16 | 128 | 24000 | 128/16 (regs) K48 | 128/16 (regs+smem) K48 | 128/16 (regs) K48 | 124/16 (regs) K48 | 127/16 (regs+smem) K48 |
| nxg2 cb4 chk32 | 128 | 48000 | 156/12 (regs+smem) K48 | 128/8 (smem) K48 | 128/16 (regs+smem) K48 | 125/16 (regs+smem) K48 | 124/8 (smem) K48 |
| nxg4 cb2 chk16 | 128 | 17600 | 72/28 (regs) K48 | 64/20 (smem) K48 | 64/32 (regs) K48 | 64/32 (regs) S8 K64 | 64/20 (smem) K48 |
| nxg4 cb2 chk32 | 128 | 35200 | 72/16 (smem) K48 | 64/8 (smem) K48 | 64/24 (smem) K48 | 64/24 (smem) S8 K64 | 64/8 (smem) K48 |
| nxg4 cb3 chk16 | 128 | 19776 | 96/20 (regs) K48 | 72/16 (smem) K48 | 72/28 (regs) S4 K64 | 64/32 (regs) S20 K64 | 72/16 (smem) K48 |
| nxg4 cb3 chk32 | 128 | 39552 | 96/16 (smem) K48 | 72/8 (smem) K48 | 72/20 (smem) S4 K64 | 64/20 (smem) S20 K64 | 72/8 (smem) K48 |
| nxg4 cb4 chk16 | 128 | 19904 | 96/20 (regs) K48 | 80/16 (smem) K48 | 80/24 (regs) K48 | 72/28 (regs) S20 K64 | 72/16 (smem) K48 |
| nxg4 cb4 chk32 | 128 | 39808 | 125/16 (regs+smem) K48 | 96/8 (smem) K48 | 96/20 (regs+smem) K48 | 80/20 (smem) K48 | 72/8 (smem) K48 |
| nxg4 cb7 chk16 | 128 | 24000 | 128/16 (regs) K48 | 127/16 (regs+smem) K48 | 123/16 (regs) K48 | 125/16 (regs) K48 | 127/16 (regs+smem) K48 |
| nxg4 cb7 chk32 | 128 | 48000 | 128/12 (smem) K48 | 128/8 (smem) K48 | 128/16 (regs+smem) K48 | 125/16 (regs+smem) K48 | 127/8 (smem) K48 |

### knot1d_o2_cb5 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 28032 | 96/20 (regs+smem) K32 | 95/12 (smem) K32 | 96/20 (regs) K32 | 96/20 (regs) K32 | 96/12 (smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 51968 | 120/12 (smem) K32 | 95/4 (smem) K32 | 96/16 (smem) K32 | 94/16 (smem) K32 | 94/4 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 28032 | 128/16 (regs) K32 | 123/12 (smem) K32 | 124/16 (regs) K32 | 128/16 (regs) K32 | 128/12 (smem) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 51968 | 153/12 (regs+smem) K32 | 128/4 (smem) K32 | 128/16 (regs+smem) K32 | 128/16 (regs+smem) K32 | 128/4 (smem) K32 |
| nxg2 cb1 chk16 listmax1024 | 128 | 23680 | 80/24 (regs+smem) S16 K48 | 64/16 (smem) K32 | 64/32 (regs) K32 | 64/32 (regs) K32 | 64/16 (smem) K32 |
| nxg2 cb1 chk32 listmax1024 | 128 | 43264 | 80/12 (smem) S16 K48 | 64/8 (smem) K32 | 64/20 (smem) K32 | 64/20 (smem) K32 | 64/8 (smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 23936 | 96/20 (regs) K32 | 80/16 (smem) K32 | 80/24 (regs) K32 | 96/20 (regs) K32 | 96/16 (smem) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 43776 | 122/12 (smem) K32 | 95/8 (smem) K32 | 96/20 (regs+smem) K32 | 80/20 (smem) K32 | 80/8 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 28032 | 128/16 (regs) K32 | 123/12 (smem) K32 | 124/16 (regs) K32 | 128/16 (regs) K32 | 128/12 (smem) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 51968 | 153/12 (regs+smem) K32 | 128/4 (smem) K32 | 128/16 (regs+smem) K32 | 128/16 (regs+smem) K32 | 128/4 (smem) K32 |
| nxg4 cb2 chk16 listmax1024 | 128 | 21632 | 72/28 (regs+smem) S4 K40 | 56/16 (smem) S12 K40 | 64/32 (regs) K32 | 64/32 (regs) K32 | 64/16 (smem) K32 |
| nxg4 cb2 chk32 listmax1024 | 128 | 39168 | 72/16 (smem) S4 K40 | 56/8 (smem) S12 K40 | 64/20 (smem) K32 | 64/20 (smem) K32 | 64/8 (smem) K32 |
| nxg4 cb3 chk16 listmax1024 | 128 | 23808 | 80/24 (regs+smem) S12 K48 | 64/16 (smem) K32 | 72/28 (regs) K32 | 94/20 (regs) K32 | 94/16 (smem) K32 |
| nxg4 cb3 chk32 listmax1024 | 128 | 43520 | 80/12 (smem) S12 K48 | 64/8 (smem) K32 | 72/20 (smem) K32 | 94/20 (regs+smem) K32 | 94/8 (smem) K32 |
| nxg4 cb4 chk16 listmax1024 | 128 | 23936 | 96/20 (regs) K32 | 80/16 (smem) K32 | 80/24 (regs) K32 | 93/20 (regs) K32 | 95/16 (smem) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 43776 | 122/12 (smem) K32 | 95/8 (smem) K32 | 96/20 (regs+smem) K32 | 94/20 (regs+smem) K32 | 94/8 (smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 28032 | 128/16 (regs) K32 | 121/12 (smem) K32 | 128/16 (regs) K32 | 124/16 (regs) K32 | 124/12 (smem) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 51968 | 127/12 (smem) K32 | 128/4 (smem) K32 | 126/16 (regs+smem) K32 | 128/16 (regs+smem) K32 | 128/4 (smem) K32 |

### knot1d_o2_cb5 / tile_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 35328 | 72/16 (smem) K32 | 64/8 (smem) K32 | 72/24 (smem) K32 | 72/24 (smem) S16 K48 | 72/8 (smem) S16 K48 |
| tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 128 | 40448 | 94/16 (smem) K32 | 71/8 (smem) K32 | 77/20 (smem) K32 | 71/20 (smem) K32 | 71/8 (smem) K32 |
| tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 64 | 35328 | 121/8 (smem) K32 | 96/4 (smem) K32 | 121/12 (smem) K32 | 127/12 (smem) K32 | 127/4 (smem) K32 |
| tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 64 | 40448 | 123/8 (smem) K32 | 96/4 (smem) K32 | 125/10 (smem) K32 | 96/10 (smem) K32 | 96/4 (smem) K32 |
| tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 192 | 35584 | 80/24 (regs+smem) K32 | 64/12 (smem) K32 | 64/30 (regs) S12 K48 | 64/30 (regs) S28 K56 | 64/12 (smem) S28 K56 |
| tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 192 | 43264 | 79/18 (smem) K32 | 64/12 (smem) K32 | 74/24 (regs) K32 | 75/24 (regs) K32 | 75/12 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 35840 | 80/24 (regs) K32 | 64/16 (smem) K32 | 73/24 (regs) K32 | 79/24 (regs) K32 | 79/16 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 256 | 46080 | 79/24 (regs+smem) K32 | 64/16 (smem) K32 | 71/24 (regs) K32 | 72/24 (regs) K32 | 72/16 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 128 | 35840 | 96/16 (smem) K32 | 93/8 (smem) K32 | 94/20 (regs) K32 | 119/16 (regs) K32 | 119/8 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 128 | 46080 | 96/12 (smem) K32 | 94/8 (smem) K32 | 95/16 (smem) K32 | 96/16 (smem) K32 | 96/8 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 64 | 35840 | 162/8 (smem) K32 | 164/4 (smem) K32 | 163/12 (regs+smem) K32 | 163/12 (regs+smem) K32 | 163/4 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 64 | 46080 | 163/6 (smem) K32 | 164/4 (smem) K32 | 162/8 (smem) K32 | 164/8 (smem) K32 | 164/4 (smem) K32 |
| tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 320 | 36096 | 83/20 (regs) K32 | 83/20 (regs+smem) K32 | 85/20 (regs) K32 | 96/20 (regs) K32 | 96/20 (regs+smem) K32 |
| tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 320 | 48896 | 64/30 (regs+smem) S16 K48 | 64/20 (smem) S16 K48 | 83/20 (regs) K32 | 84/20 (regs) K32 | 84/20 (regs+smem) K32 |

### knot1d_o2_cb5 / tile_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 32 | 39680 | 78/4 (smem) K32 | 80/2 (smem) K32 | 64/5 (smem) K32 | 66/5 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 32 | 39680 | 79/4 (smem) K32 | 80/2 (smem) K32 | 64/5 (smem) K32 | 66/5 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 32 | 70144 | 127/2 (smem) K32 | 96/1 (smem) K32 | 96/3 (smem) K32 | 96/3 (smem) K32 | 95/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 32 | 70144 | 127/2 (smem) K32 | 96/1 (smem) K32 | 96/3 (smem) K32 | 96/3 (smem) K32 | 95/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | 39936 | 56/8 (smem) K32 | 56/4 (smem) K32 | 63/10 (smem) K32 | 56/10 (smem) K32 | 56/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | 39936 | 64/8 (smem) K32 | 56/4 (smem) K32 | 64/10 (smem) K32 | 61/10 (smem) K32 | 58/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 64 | 70656 | 113/4 (smem) K32 | 80/2 (smem) K32 | 95/6 (smem) K32 | 80/6 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 64 | 70656 | 111/4 (smem) K32 | 79/2 (smem) K32 | 93/6 (smem) K32 | 80/6 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 32 | 70656 | 167/2 (smem) K32 | 143/1 (smem) K32 | 168/3 (smem) K32 | 167/3 (smem) K32 | 167/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 32 | 70656 | 167/2 (smem) K32 | 143/1 (smem) K32 | 167/3 (smem) K32 | 167/3 (smem) K32 | 167/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 40448 | 56/16 (smem) K32 | 48/8 (smem) K32 | 56/20 (smem) K32 | 56/20 (smem) K32 | 48/8 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 128 | 40448 | 64/16 (smem) K32 | 56/8 (smem) K32 | 58/20 (smem) K32 | 56/20 (smem) K32 | 51/8 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 71680 | 95/8 (smem) K32 | 79/4 (smem) K32 | 80/12 (smem) K32 | 80/12 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 128 | 71680 | 95/8 (smem) K32 | 79/4 (smem) K32 | 80/12 (smem) K32 | 80/12 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 71680 | 157/4 (smem) K32 | 128/2 (smem) K32 | 128/6 (smem) K32 | 128/6 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 64 | 71680 | 155/4 (smem) K32 | 128/2 (smem) K32 | 128/6 (smem) K32 | 128/6 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | 45824 | 64/6 (smem) K32 | 56/4 (smem) K32 | 64/8 (smem) K32 | 59/8 (smem) K32 | 60/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | 45824 | 64/6 (smem) S4 K40 | 56/4 (smem) K32 | 64/8 (smem) K32 | 59/8 (smem) K32 | 61/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 64 | 76288 | 113/4 (smem) K32 | 80/2 (smem) K32 | 93/6 (smem) K32 | 80/6 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 64 | 76288 | 96/4 (smem) S8 K40 | 79/2 (smem) K32 | 93/6 (smem) K32 | 80/6 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 32 | 76288 | 167/2 (smem) K32 | 143/1 (smem) K32 | 153/3 (smem) K32 | 167/3 (smem) K32 | 167/1 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 32 | 76288 | 167/2 (smem) K32 | 143/1 (smem) K32 | 167/3 (smem) K32 | 167/3 (smem) K32 | 167/1 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 46080 | 56/12 (smem) K32 | 48/8 (smem) K32 | 56/16 (smem) K32 | 48/16 (smem) K32 | 48/8 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 128 | 46080 | 64/12 (smem) K32 | 48/8 (smem) K32 | 56/16 (smem) K32 | 58/16 (smem) K32 | 56/8 (smem) S32 K48 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 76800 | 95/8 (smem) K32 | 72/4 (smem) K32 | 80/12 (smem) K32 | 79/12 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 128 | 76800 | 95/8 (smem) K32 | 72/4 (smem) K32 | 79/12 (smem) K32 | 79/12 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 76800 | 157/4 (smem) K32 | 128/2 (smem) K32 | 128/6 (smem) K32 | 128/6 (smem) K32 | 128/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 64 | 76800 | 155/4 (smem) K32 | 128/2 (smem) K32 | 128/6 (smem) K32 | 128/6 (smem) K32 | 128/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 46592 | 63/24 (smem) K32 | 48/16 (smem) K32 | 59/32 (regs+smem) K32 | 48/32 (smem) K32 | 48/16 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 256 | 46592 | 63/24 (smem) K32 | 48/16 (smem) K32 | 58/32 (regs+smem) K32 | 61/32 (regs+smem) K32 | 53/16 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 77824 | 80/16 (smem) S4 K40 | 71/8 (smem) K32 | 76/16 (smem) K32 | 80/16 (smem) K32 | 80/8 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 256 | 77824 | 80/16 (smem) S8 K40 | 71/8 (smem) K32 | 80/16 (smem) K32 | 80/16 (smem) K32 | 80/8 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 77824 | 149/8 (smem) K32 | 128/4 (smem) K32 | 128/8 (smem) K32 | 128/8 (smem) K32 | 128/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | 77824 | 128/8 (smem) S4 K40 | 128/4 (smem) K32 | 128/8 (smem) K32 | 127/8 (smem) K32 | 127/4 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 96 | 49920 | 72/9 (smem) K32 | 56/6 (smem) K32 | 56/12 (smem) S36 K56 | 56/12 (smem) K32 | 56/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 96 | 49920 | 56/9 (smem) K32 | 56/6 (smem) K32 | 56/12 (smem) S36 K56 | 56/12 (smem) K32 | 50/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | 80384 | 96/6 (smem) K32 | 72/3 (smem) K32 | 80/6 (smem) K32 | 79/6 (smem) K32 | 79/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | 80384 | 80/6 (smem) S16 K48 | 72/3 (smem) K32 | 80/6 (smem) K32 | 79/6 (smem) K32 | 79/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 192 | 50176 | 56/18 (smem) K32 | 56/12 (smem) K32 | 56/24 (smem) K32 | 56/24 (smem) K32 | 56/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 192 | 50176 | 56/18 (smem) S4 K40 | 64/12 (smem) K32 | 56/24 (smem) K32 | 58/24 (smem) K32 | 56/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 192 | 80896 | 95/12 (smem) K32 | 71/6 (smem) K32 | 80/12 (smem) K32 | 80/12 (smem) K32 | 80/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 192 | 80896 | 80/12 (smem) S8 K40 | 71/6 (smem) K32 | 80/12 (smem) K32 | 80/12 (smem) K32 | 80/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 96 | 80896 | 151/6 (smem) K32 | 128/3 (smem) K32 | 128/6 (smem) K32 | 128/6 (smem) K32 | 128/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 96 | 80896 | 149/6 (smem) K32 | 128/3 (smem) K32 | 128/6 (smem) K32 | 127/6 (smem) K32 | 127/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 384 | 50688 | 56/36 (regs+smem) K32 | 56/12 (smem) K32 | 56/36 (regs) K32 | 56/36 (regs) K32 | 56/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 384 | 50688 | 56/36 (regs+smem) S4 K40 | 55/12 (smem) K32 | 56/36 (regs) K32 | 58/24 (regs) K32 | 56/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 384 | 81920 | 80/24 (regs+smem) S16 K48 | 80/12 (smem) K32 | 80/24 (regs+smem) K32 | 80/24 (regs+smem) K32 | 80/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 384 | 81920 | 80/24 (regs+smem) S8 K40 | 80/12 (smem) K32 | 80/24 (regs+smem) K32 | 80/24 (regs+smem) K32 | 80/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 192 | 81920 | 143/12 (regs+smem) K32 | 143/6 (smem) K32 | 145/12 (regs+smem) K32 | 129/12 (regs+smem) K32 | 129/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 192 | 81920 | 141/12 (regs+smem) K32 | 141/6 (smem) K32 | 145/12 (regs+smem) K32 | 129/12 (regs+smem) K32 | 129/6 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 56064 | 56/8 (smem) K32 | 48/4 (smem) K32 | 56/16 (smem) K32 | 56/16 (smem) K32 | 56/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 128 | 56064 | 56/8 (smem) K32 | 48/4 (smem) K32 | 56/16 (smem) K32 | 56/16 (smem) K32 | 50/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 86528 | 95/4 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 79/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 128 | 86528 | 95/4 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 79/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 86528 | 157/2 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 64 | 86528 | 153/2 (smem) K32 | 128/2 (smem) K32 | 127/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 56320 | 63/16 (smem) K32 | 48/8 (smem) K32 | 59/32 (regs+smem) K32 | 48/32 (smem) K32 | 48/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 256 | 56320 | 63/16 (smem) K32 | 48/8 (smem) K32 | 58/32 (regs+smem) K32 | 61/32 (regs+smem) K32 | 53/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 87040 | 80/8 (smem) S4 K40 | 71/8 (smem) K32 | 76/16 (smem) K32 | 80/16 (smem) K32 | 80/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 256 | 87040 | 80/8 (smem) S8 K40 | 71/8 (smem) K32 | 80/16 (smem) K32 | 80/16 (smem) K32 | 80/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 87040 | 149/4 (smem) K32 | 128/4 (smem) K32 | 128/8 (smem) K32 | 128/8 (smem) K32 | 128/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | 87040 | 128/4 (smem) S4 K40 | 128/4 (smem) K32 | 128/8 (smem) K32 | 127/8 (smem) K32 | 127/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 512 | 56832 | 63/32 (regs+smem) K32 | 64/16 (smem) K32 | 64/32 (regs) K32 | 63/32 (regs) K32 | 64/16 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 512 | 56832 | 63/32 (regs+smem) K32 | 64/16 (smem) K32 | 64/32 (regs) K32 | 64/32 (regs) K32 | 64/16 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 512 | 88064 | 94/16 (regs+smem) K32 | 64/16 (smem) S32 K48 | 97/16 (regs) K32 | 82/16 (regs) K32 | 82/16 (regs+smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 512 | 88064 | 96/16 (regs+smem) K32 | 64/16 (smem) S32 K48 | 96/16 (regs) K32 | 82/16 (regs) K32 | 82/16 (regs+smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 256 | 88064 | 128/8 (smem) K32 | 128/8 (smem) K32 | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | 127/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 256 | 88064 | 128/8 (smem) K32 | 128/8 (smem) K32 | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | 127/8 (smem) K32 |
| tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 160 | 60160 | 64/10 (smem) K32 | 56/5 (smem) K32 | 56/15 (smem) S36 K56 | 56/15 (smem) K32 | 56/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 160 | 60160 | 56/10 (smem) K32 | 56/5 (smem) K32 | 56/15 (smem) S36 K56 | 56/15 (smem) K32 | 50/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 160 | 90624 | 95/5 (smem) K32 | 72/5 (smem) K32 | 84/10 (smem) K32 | 95/10 (smem) K32 | 95/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 160 | 90624 | 95/5 (smem) K32 | 72/5 (smem) K32 | 96/10 (smem) K32 | 95/10 (smem) K32 | 95/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 320 | 60416 | 63/20 (smem) K32 | 48/10 (smem) K32 | 61/30 (regs+smem) K32 | 63/30 (regs+smem) K32 | 64/10 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 320 | 60416 | 64/20 (smem) K32 | 47/10 (smem) K32 | 59/30 (regs+smem) K32 | 64/30 (regs+smem) K32 | 64/10 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 320 | 91136 | 95/10 (smem) K32 | 72/10 (smem) K32 | 96/20 (regs+smem) K32 | 93/20 (regs+smem) K32 | 91/10 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 320 | 91136 | 95/10 (smem) K32 | 72/10 (smem) K32 | 95/20 (regs+smem) K32 | 93/20 (regs+smem) K32 | 91/10 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 160 | 91136 | 128/5 (smem) S4 K40 | 128/5 (smem) K32 | 128/10 (smem) K32 | 128/10 (smem) K32 | 128/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 160 | 91136 | 128/5 (smem) S12 K48 | 128/5 (smem) K32 | 128/10 (smem) K32 | 127/10 (smem) K32 | 127/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 480 | 53120 | 63/30 (regs) K32 | 64/15 (smem) K32 | 64/30 (regs) K32 | 64/30 (regs) K32 | 59/15 (smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv1 | 480 | 53120 | 64/30 (regs) K32 | 64/15 (smem) K32 | 64/30 (regs) K32 | 64/30 (regs) K32 | 64/15 (smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 480 | 76544 | 94/15 (regs) K32 | 72/15 (regs+smem) K32 | 96/15 (regs) K32 | 64/30 (regs) S32 K48 | 64/15 (smem) S32 K48 |
| tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv1 | 480 | 76544 | 94/15 (regs) K32 | 72/15 (regs+smem) K32 | 96/15 (regs) K32 | 82/15 (regs) K32 | 82/15 (regs+smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 320 | 92160 | 143/10 (regs+smem) K32 | 143/10 (regs+smem) K32 | 147/10 (regs) K32 | 129/10 (regs) K32 | 129/10 (regs+smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 320 | 92160 | 141/10 (regs+smem) K32 | 141/10 (regs+smem) K32 | 145/10 (regs) K32 | 129/10 (regs) K32 | 129/10 (regs+smem) K32 |

### knot1d_o2_cb5 / vt_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 2x4x16 smax32 nwc4 nws1 vxpad0 | 128 | 49184 | 167/12 (regs+smem) | 168/8 (smem) | 168/12 (regs) | 168/12 (regs) | 180/8 (regs+smem) |
| 2x4x16 smax32 nwc4 nws2 vxpad0 | 256 | 55584 | 170/8 (regs) | 171/8 (regs+smem) | 172/8 (regs) | 184/8 (regs) | 194/8 (regs+smem) |
| 2x4x8 smax32 nwc4 nws1 vxpad0 | 128 | 48928 | 167/12 (regs+smem) | 168/8 (smem) | 168/12 (regs) | 168/12 (regs) | 180/8 (regs+smem) |
| 2x4x8 smax32 nwc4 nws2 vxpad0 | 256 | 55328 | 170/8 (regs) | 171/8 (regs+smem) | 172/8 (regs) | 184/8 (regs) | 194/8 (regs+smem) |
| 2x4x8 smax48 nwc4 nws1 vxpad0 | 128 | 69664 | 167/8 (smem) | 168/4 (smem) | 168/12 (regs+smem) | 168/12 (regs+smem) | 180/4 (smem) |
| 2x4x8 smax48 nwc4 nws2 vxpad0 | 256 | 76064 | 170/8 (regs) | 171/8 (regs+smem) | 172/8 (regs) | 184/8 (regs) | 194/8 (regs+smem) |
| 4x4x16 smax48 nwc4 nws1 vxpad0 | 128 | 70720 | 254/8 (regs+smem) | 254/4 (smem) | 254/8 (regs) | 252/8 (regs) | 236/4 (smem) |
| 4x4x16 smax48 nwc4 nws2 vxpad0 | 256 | 77120 | 232/8 (regs) | 236/8 (regs+smem) | 242/8 (regs) | 254/8 (regs) | 238/8 (regs+smem) |
| 4x4x16 smax64 nwc4 nws1 vxpad0 | 128 | 91456 | 254/4 (smem) | 254/4 (smem) | 254/8 (regs+smem) | 252/8 (regs+smem) | 236/4 (smem) |
| 4x4x16 smax64 nwc4 nws2 vxpad0 | 256 | 97856 | 232/8 (regs+smem) | 236/8 (regs+smem) | 242/8 (regs) | 254/8 (regs) | 238/8 (regs+smem) |

### knot1d_o2_cb5 / whatif

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 4x4x16 smax128 nwc4 nws1 vxpad0 | 128 | 174400 | - | - | 254/4 (smem) | 252/4 (smem) | - |
| 4x4x16 smax96 nwc4 nws1 vxpad0 | 128 | 132928 | 254/4 (smem) | - | 254/4 (smem) | 252/4 (smem) | - |
| nxg1 cb1 chk16 listmax2048 | 128 | 32128 | 96/20 (regs+smem) K32 | 95/12 (smem) K32 | 96/20 (regs) K32 | 96/20 (regs) K32 | 96/12 (smem) K32 |
| nxg1 cb1 chk16 listmax4096 | 128 | 40320 | 96/16 (smem) K32 | 95/8 (smem) K32 | 96/20 (regs+smem) K32 | 96/20 (regs+smem) K32 | 96/8 (smem) K32 |
| nxg1 cb1 chk48 listmax1024 | 128 | 75904 | 123/8 (smem) K32 | 96/4 (smem) K32 | 96/12 (smem) K32 | 96/12 (smem) K32 | 96/4 (smem) K32 |
| nxg1 cb1 chk64 listmax1024 | 128 | 99840 | 123/4 (smem) K32 | 125/4 (smem) K32 | 96/8 (smem) K32 | 96/8 (smem) K32 | 96/4 (smem) K32 |
| nxg1 cb2 chk16 listmax2048 | 128 | 32128 | 128/16 (regs) K32 | 123/12 (smem) K32 | 124/16 (regs) K32 | 128/16 (regs) K32 | 128/12 (smem) K32 |
| nxg1 cb2 chk16 listmax4096 | 128 | 40320 | 128/16 (regs+smem) K32 | 123/8 (smem) K32 | 124/16 (regs) K32 | 128/16 (regs) K32 | 128/8 (smem) K32 |
| nxg1 cb2 chk48 listmax1024 | 128 | 75904 | 159/8 (smem) K32 | 128/4 (smem) K32 | 128/12 (smem) K32 | 128/12 (smem) K32 | 128/4 (smem) K32 |
| nxg1 cb2 chk64 listmax1024 | 128 | 99840 | 156/4 (smem) K32 | 128/4 (smem) K32 | 128/8 (smem) K32 | 128/8 (smem) K32 | 128/4 (smem) K32 |
| nxg2 cb1 chk16 listmax2048 | 128 | 27776 | 80/20 (smem) S16 K48 | 64/12 (smem) K32 | 64/32 (regs+smem) K32 | 64/32 (regs+smem) K32 | 64/12 (smem) K32 |
| nxg2 cb1 chk16 listmax4096 | 128 | 35968 | 80/16 (smem) S16 K48 | 64/8 (smem) K32 | 64/24 (smem) K32 | 64/24 (smem) K32 | 64/8 (smem) K32 |
| nxg2 cb1 chk48 listmax1024 | 128 | 62848 | 95/8 (smem) K32 | 72/4 (smem) K32 | 72/12 (smem) K32 | 72/12 (smem) S12 K40 | 72/4 (smem) S12 K40 |
| nxg2 cb1 chk64 listmax1024 | 128 | 82432 | 96/8 (smem) K32 | 72/4 (smem) K32 | 72/8 (smem) K32 | 72/8 (smem) S12 K40 | 72/4 (smem) S12 K40 |
| nxg2 cb2 chk16 listmax2048 | 128 | 28032 | 96/20 (regs+smem) K32 | 80/12 (smem) K32 | 80/24 (regs) K32 | 96/20 (regs) K32 | 96/12 (smem) K32 |
| nxg2 cb2 chk16 listmax4096 | 128 | 36224 | 96/16 (smem) K32 | 80/8 (smem) K32 | 80/24 (regs+smem) K32 | 96/20 (regs) K32 | 96/8 (smem) K32 |
| nxg2 cb2 chk48 listmax1024 | 128 | 63616 | 122/8 (smem) K32 | 96/4 (smem) K32 | 96/12 (smem) K32 | 96/12 (smem) K32 | 96/4 (smem) K32 |
| nxg2 cb2 chk64 listmax1024 | 128 | 83456 | 127/4 (smem) K32 | 125/4 (smem) K32 | 96/8 (smem) K32 | 96/8 (smem) K32 | 96/4 (smem) K32 |
| nxg2 cb4 chk16 listmax2048 | 128 | 32128 | 128/16 (regs) K32 | 123/12 (smem) K32 | 124/16 (regs) K32 | 128/16 (regs) K32 | 128/12 (smem) K32 |
| nxg2 cb4 chk16 listmax4096 | 128 | 40320 | 128/16 (regs+smem) K32 | 123/8 (smem) K32 | 124/16 (regs) K32 | 128/16 (regs) K32 | 128/8 (smem) K32 |
| nxg2 cb4 chk48 listmax1024 | 128 | 75904 | 157/8 (smem) K32 | 128/4 (smem) K32 | 128/12 (smem) K32 | 128/12 (smem) K32 | 128/4 (smem) K32 |
| nxg2 cb4 chk64 listmax1024 | 128 | 99840 | 163/4 (smem) K32 | 158/4 (smem) K32 | 128/8 (smem) K32 | 162/8 (smem) K32 | 158/4 (smem) K32 |
| nxg4 cb2 chk16 listmax2048 | 128 | 25728 | 72/24 (smem) S4 K40 | 56/12 (smem) S12 K40 | 64/32 (regs+smem) K32 | 64/32 (regs+smem) K32 | 64/12 (smem) K32 |
| nxg4 cb2 chk16 listmax4096 | 128 | 33920 | 72/16 (smem) S4 K40 | 56/8 (smem) S12 K40 | 64/24 (smem) K32 | 64/24 (smem) K32 | 64/8 (smem) K32 |
| nxg4 cb2 chk48 listmax1024 | 128 | 56704 | 96/8 (smem) K32 | 72/4 (smem) K32 | 72/16 (smem) K32 | 72/16 (smem) S4 K40 | 72/4 (smem) S4 K40 |
| nxg4 cb2 chk64 listmax1024 | 128 | 74240 | 96/8 (smem) K32 | 72/4 (smem) K32 | 72/12 (smem) K32 | 72/12 (smem) S4 K40 | 72/4 (smem) S4 K40 |
| nxg4 cb3 chk16 listmax2048 | 128 | 27904 | 80/20 (smem) S12 K48 | 64/12 (smem) K32 | 72/28 (regs) K32 | 94/20 (regs) K32 | 94/12 (smem) K32 |
| nxg4 cb3 chk16 listmax4096 | 128 | 36096 | 80/16 (smem) S12 K48 | 64/8 (smem) K32 | 72/24 (smem) K32 | 94/20 (regs) K32 | 94/8 (smem) K32 |
| nxg4 cb3 chk48 listmax1024 | 128 | 63232 | 120/8 (smem) K32 | 80/4 (smem) K32 | 96/12 (smem) K32 | 95/12 (smem) K32 | 95/4 (smem) K32 |
| nxg4 cb3 chk64 listmax1024 | 128 | 82944 | 120/8 (smem) K32 | 80/4 (smem) K32 | 95/8 (smem) K32 | 95/8 (smem) K32 | 95/4 (smem) K32 |
| nxg4 cb4 chk16 listmax2048 | 128 | 28032 | 96/20 (regs+smem) K32 | 80/12 (smem) K32 | 80/24 (regs) K32 | 93/20 (regs) K32 | 95/12 (smem) K32 |
| nxg4 cb4 chk16 listmax4096 | 128 | 36224 | 96/16 (smem) K32 | 80/8 (smem) K32 | 80/24 (regs+smem) K32 | 93/20 (regs) K32 | 95/8 (smem) K32 |
| nxg4 cb4 chk48 listmax1024 | 128 | 63616 | 122/8 (smem) K32 | 96/4 (smem) K32 | 96/12 (smem) K32 | 96/12 (smem) K32 | 96/4 (smem) K32 |
| nxg4 cb4 chk64 listmax1024 | 128 | 83456 | 127/4 (smem) K32 | 125/4 (smem) K32 | 96/8 (smem) K32 | 96/8 (smem) K32 | 96/4 (smem) K32 |
| nxg4 cb7 chk16 listmax2048 | 128 | 32128 | 128/16 (regs) K32 | 121/12 (smem) K32 | 128/16 (regs) K32 | 124/16 (regs) K32 | 124/12 (smem) K32 |
| nxg4 cb7 chk16 listmax4096 | 128 | 40320 | 128/16 (regs+smem) K32 | 121/8 (smem) K32 | 128/16 (regs) K32 | 124/16 (regs) K32 | 124/8 (smem) K32 |
| nxg4 cb7 chk48 listmax1024 | 128 | 75904 | 154/8 (smem) K32 | 128/4 (smem) K32 | 127/12 (smem) K32 | 128/12 (smem) K32 | 128/4 (smem) K32 |
| nxg4 cb7 chk64 listmax1024 | 128 | 99840 | 159/4 (smem) K32 | 128/4 (smem) K32 | 128/8 (smem) K32 | 128/8 (smem) K32 | 128/4 (smem) K32 |

### static_o0_10x10 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 15808 | 96/20 (regs) K32 | 80/24 (regs+smem) K32 | 80/24 (regs) S8 K40 | 64/32 (regs) S8 K40 | 64/24 (smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 27520 | 96/20 (regs+smem) S16 K48 | 96/12 (smem) K32 | 80/24 (regs) S12 K48 | 80/24 (regs) S4 K40 | 80/12 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 16192 | 126/16 (regs) K32 | 117/16 (regs) K32 | 118/16 (regs) K32 | 110/16 (regs) K32 | 110/16 (regs) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 28288 | 128/16 (regs) S12 K48 | 117/12 (smem) K32 | 118/16 (regs) K32 | 111/16 (regs) K32 | 111/12 (smem) K32 |
| nxg1 cb4 chk16 listmax1024 | 128 | 24640 | 206/8 (regs) K32 | 204/8 (regs) K32 | 207/8 (regs) K32 | 183/8 (regs) K32 | 182/8 (regs) K32 |
| nxg1 cb4 chk32 listmax1024 | 128 | 45184 | 210/8 (regs) K32 | 208/8 (regs+smem) K32 | 211/8 (regs) K32 | 187/8 (regs) K32 | 186/8 (regs+smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 11712 | 80/24 (regs) S16 K48 | 72/28 (regs) K32 | 72/28 (regs) S8 K40 | 64/32 (regs) K32 | 64/32 (regs+smem) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 19328 | 96/20 (regs) K32 | 80/20 (smem) K32 | 80/24 (regs) S8 K40 | 80/24 (regs) K32 | 80/20 (smem) K32 |
| nxg2 cb3 chk16 listmax1024 | 128 | 15936 | 122/16 (regs) K32 | 96/20 (regs) K32 | 119/16 (regs) K32 | 96/20 (regs) S8 K40 | 95/20 (regs) K32 |
| nxg2 cb3 chk32 listmax1024 | 128 | 27776 | 126/16 (regs) K32 | 121/12 (smem) K32 | 115/16 (regs) K32 | 114/16 (regs) K32 | 110/12 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 16192 | 128/16 (regs) K32 | 118/16 (regs) K32 | 118/16 (regs) K32 | 114/16 (regs) K32 | 112/16 (regs) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 28288 | 128/16 (regs) K32 | 119/12 (smem) K32 | 119/16 (regs) K32 | 116/16 (regs) K32 | 114/12 (smem) K32 |
| nxg2 cb7 chk16 listmax1024 | 128 | 24640 | 198/8 (regs) K32 | 195/8 (regs) K32 | 197/8 (regs) K32 | 161/12 (regs) K32 | 160/12 (regs+smem) K32 |
| nxg2 cb7 chk32 listmax1024 | 128 | 45184 | 202/8 (regs) K32 | 199/8 (regs+smem) K32 | 201/8 (regs) K32 | 158/12 (regs) K32 | 157/8 (smem) K32 |
| nxg4 cb13 chk16 listmax1024 | 128 | 22592 | 200/8 (regs) K32 | 197/8 (regs) K32 | 199/8 (regs) K32 | 158/12 (regs) K32 | 161/12 (regs) K32 |
| nxg4 cb13 chk32 listmax1024 | 128 | 41088 | 204/8 (regs) K32 | 201/8 (regs+smem) K32 | 203/8 (regs) K32 | 160/12 (regs) K32 | 166/8 (smem) K32 |
| nxg4 cb4 chk16 listmax1024 | 128 | 11712 | 80/24 (regs) S16 K48 | 72/28 (regs) K32 | 72/28 (regs) S8 K40 | 64/32 (regs) K32 | 64/32 (regs+smem) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 19328 | 96/20 (regs) K32 | 80/20 (smem) K32 | 80/24 (regs) S8 K40 | 80/24 (regs) K32 | 80/20 (smem) K32 |
| nxg4 cb5 chk16 listmax1024 | 128 | 13760 | 117/16 (regs) K32 | 115/16 (regs) K32 | 118/16 (regs) K32 | 90/20 (regs) K32 | 89/20 (regs) K32 |
| nxg4 cb5 chk32 listmax1024 | 128 | 23424 | 122/16 (regs) K32 | 105/16 (regs+smem) K32 | 110/16 (regs) K32 | 90/20 (regs) K32 | 90/16 (smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 16064 | 128/16 (regs) S16 K48 | 122/16 (regs) K32 | 120/16 (regs) K32 | 118/16 (regs) K32 | 116/16 (regs) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 28032 | 128/16 (regs) S40 K72 | 122/12 (smem) K32 | 120/16 (regs) K32 | 119/16 (regs) K32 | 117/12 (smem) K32 |

### static_o0_10x10 / tile_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 640 | 51712 | 74/20 (regs) K32 | 74/20 (regs+smem) K32 | 74/20 (regs) K32 | 82/20 (regs) K32 | 80/20 (regs+smem) K32 |
| tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 320 | 51712 | 96/20 (regs) K32 | 96/10 (smem) K32 | 96/20 (regs) K32 | 96/20 (regs) K32 | 84/10 (smem) K32 |
| tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 27328 | 64/20 (smem) S12 K48 | 64/12 (smem) K32 | 72/28 (regs) K32 | 64/32 (regs+smem) K32 | 64/12 (smem) K32 |
| tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 64 | 27328 | 96/10 (smem) K32 | 96/6 (smem) K32 | 96/16 (smem) S4 K40 | 96/16 (smem) K32 | 80/6 (smem) K32 |
| tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 192 | 27584 | 64/30 (regs+smem) S12 K48 | 56/18 (smem) K32 | 64/30 (regs) K32 | 64/30 (regs) K32 | 64/18 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 27840 | 64/32 (regs) S12 K48 | 64/24 (smem) K32 | 64/32 (regs) S24 K56 | 64/32 (regs) K32 | 64/24 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 128 | 27840 | 96/20 (regs+smem) K32 | 96/12 (smem) K32 | 96/20 (regs) K32 | 80/24 (regs) K32 | 79/12 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 64 | 27840 | 168/10 (smem) K32 | 168/6 (smem) K32 | 168/12 (regs) K32 | 167/12 (regs) K32 | 157/6 (smem) K32 |
| tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 384 | 31232 | 72/24 (regs) K32 | 72/24 (regs) K32 | 72/24 (regs) K32 | 80/24 (regs) K32 | 66/24 (regs) K32 |
| tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 192 | 31232 | 96/18 (regs) K32 | 96/18 (regs+smem) K32 | 96/18 (regs) K32 | 96/18 (regs) K32 | 80/18 (smem) K32 |
| tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 512 | 41472 | 64/32 (regs) S12 K48 | 64/32 (regs+smem) K32 | 64/32 (regs) S24 K56 | 58/32 (regs) K32 | 64/32 (regs+smem) K32 |
| tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 256 | 41472 | 96/16 (regs) K32 | 98/16 (regs+smem) K32 | 100/16 (regs) K32 | 80/24 (regs) K32 | 80/16 (smem) K32 |
| tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 128 | 41472 | 162/12 (regs+smem) K32 | 128/8 (smem) K32 | 168/12 (regs) K32 | 167/12 (regs) K32 | 128/8 (smem) S8 K40 |

### static_o0_10x10 / tile_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 32 | 31680 | 80/5 (smem) K32 | 80/3 (smem) K32 | 71/7 (smem) K32 | 64/7 (smem) K32 | 78/3 (smem) K32 |
| tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 32 | 54080 | 128/3 (smem) K32 | 96/1 (smem) K32 | 96/4 (smem) K32 | 95/4 (smem) K32 | 80/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | 31936 | 64/10 (smem) K32 | 56/6 (smem) K32 | 64/14 (smem) K32 | 56/14 (smem) K32 | 48/6 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 64 | 54592 | 96/6 (smem) K32 | 79/2 (smem) K32 | 93/8 (smem) K32 | 79/8 (smem) K32 | 79/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 32 | 54592 | 168/3 (smem) K32 | 139/1 (smem) K32 | 168/4 (smem) S4 K40 | 145/4 (smem) K32 | 139/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 32448 | 64/20 (smem) K32 | 56/12 (smem) K32 | 56/24 (smem) K32 | 56/24 (smem) K32 | 48/12 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 55616 | 96/8 (smem) K32 | 79/4 (smem) K32 | 80/16 (smem) K32 | 79/16 (smem) K32 | 71/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 55616 | 154/4 (smem) K32 | 128/2 (smem) K32 | 128/8 (smem) K32 | 127/8 (smem) K32 | 125/2 (smem) K32 |
| tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 320 | 78784 | 64/20 (smem) K32 | 64/10 (smem) K32 | 57/20 (smem) K32 | 64/20 (smem) S8 K40 | 64/10 (smem) K32 |
| tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 320 | 101184 | 94/10 (smem) K32 | 95/10 (smem) K32 | 95/20 (regs+smem) K32 | 93/20 (regs+smem) K32 | 93/10 (smem) K32 |
| tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 160 | 101184 | 128/5 (smem) S8 K40 | 128/5 (smem) K32 | 128/10 (smem) K32 | 127/10 (smem) K32 | 125/5 (smem) K32 |
| tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv0 | 480 | 73376 | 64/30 (regs+smem) K32 | 64/15 (smem) K32 | 64/30 (regs) K32 | 64/30 (regs) K32 | 64/15 (smem) K32 |
| tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 160 | 79040 | - | 121/5 (smem) K32 | - | - | 127/5 (smem) K32 |
| tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv0 | 480 | 90368 | 96/15 (regs+smem) K32 | 94/15 (regs+smem) K32 | 95/15 (regs) K32 | 73/15 (regs) K32 | 64/15 (smem) S52 K72 |
| tt4 cpba1 cpbf10 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 320 | 101696 | 138/10 (regs+smem) K32 | - | 143/10 (regs) K32 | 125/10 (regs) K32 | - |
| tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv0 | 480 | 65072 | 64/30 (regs+smem) K32 | 64/15 (smem) K32 | 64/30 (regs) K32 | 64/30 (regs) K32 | 64/15 (smem) K32 |
| tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv0 | 480 | 73760 | 96/15 (regs) K32 | 74/15 (regs+smem) K32 | 95/15 (regs) K32 | 73/15 (regs) K32 | 64/15 (smem) S52 K72 |
| tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 480 | 91136 | 128/15 (regs+smem) K32 | 128/15 (regs+smem) K32 | 128/15 (regs) K32 | 127/15 (regs) K32 | 127/15 (regs+smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | 37824 | 80/8 (smem) K32 | 56/4 (smem) K32 | 64/12 (smem) K32 | 63/12 (smem) K32 | 56/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 64 | 60224 | 96/4 (smem) K32 | 79/2 (smem) K32 | 95/6 (smem) K32 | 79/6 (smem) K32 | 79/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 32 | 60224 | 168/2 (smem) K32 | 143/1 (smem) K32 | 168/3 (smem) S4 K40 | 128/3 (smem) K32 | 127/1 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 38080 | 64/16 (smem) K32 | 56/8 (smem) K32 | 56/20 (smem) K32 | 56/20 (smem) K32 | 48/8 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 60736 | 96/8 (smem) K32 | 79/4 (smem) K32 | 80/12 (smem) K32 | 79/12 (smem) K32 | 72/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 60736 | 154/4 (smem) K32 | 128/2 (smem) K32 | 128/6 (smem) K32 | 127/6 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 38592 | 63/32 (regs+smem) K32 | 63/16 (smem) K32 | 58/32 (regs) K32 | 64/32 (regs) K32 | 57/16 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 61760 | 80/16 (smem) S4 K40 | 71/8 (smem) K32 | 80/24 (regs+smem) K32 | 72/24 (regs+smem) K32 | 71/8 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 61760 | 128/8 (smem) S8 K40 | 128/4 (smem) K32 | 128/12 (smem) K32 | 127/12 (smem) K32 | 125/4 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 96 | 41920 | 71/9 (smem) K32 | 56/6 (smem) S32 K48 | 56/15 (smem) K32 | 56/15 (smem) K32 | 56/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | 64320 | 96/6 (smem) K32 | 79/3 (smem) K32 | 80/9 (smem) K32 | 72/9 (smem) K32 | 74/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 192 | 42176 | 56/18 (smem) K32 | 56/12 (smem) K32 | 56/30 (smem) K32 | 56/30 (smem) K32 | 49/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 192 | 64832 | 80/12 (smem) S8 K40 | 71/6 (smem) K32 | 80/18 (smem) K32 | 72/18 (smem) K32 | 71/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 96 | 64832 | 142/6 (smem) K32 | 128/3 (smem) K32 | 128/9 (smem) K32 | 127/9 (smem) K32 | 125/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 384 | 42688 | 56/36 (regs+smem) K32 | 56/24 (smem) K32 | 56/36 (regs) K32 | 56/36 (regs) K32 | 49/24 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 384 | 65856 | 80/24 (regs+smem) S8 K40 | 80/12 (smem) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 | 79/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 192 | 65856 | 138/12 (regs+smem) K32 | 139/6 (smem) K32 | 143/12 (regs) K32 | 125/12 (regs) K32 | 125/6 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 48064 | 71/12 (smem) K32 | 56/8 (smem) S32 K48 | 56/16 (smem) K32 | 56/16 (smem) K32 | 56/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 70464 | 96/8 (smem) K32 | 79/4 (smem) K32 | 80/12 (smem) K32 | 74/12 (smem) K32 | 72/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 70464 | 156/4 (smem) K32 | 128/2 (smem) K32 | 128/6 (smem) K32 | 127/6 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 48320 | 63/24 (smem) K32 | 63/16 (smem) K32 | 58/32 (regs+smem) K32 | 64/32 (regs+smem) K32 | 57/16 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 70976 | 80/16 (smem) S4 K40 | 71/8 (smem) K32 | 80/24 (regs+smem) K32 | 72/24 (regs+smem) K32 | 71/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 70976 | 128/8 (smem) S8 K40 | 128/4 (smem) K32 | 128/12 (smem) K32 | 127/12 (smem) K32 | 125/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 512 | 48832 | 63/32 (regs) K32 | 63/32 (regs+smem) K32 | 64/32 (regs) K32 | 64/32 (regs) K32 | 64/32 (regs+smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 512 | 72000 | 94/16 (regs) K32 | 72/16 (regs+smem) K32 | 96/16 (regs) K32 | 79/16 (regs) K32 | 64/16 (smem) S44 K64 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 256 | 72000 | 128/16 (regs+smem) K32 | 128/8 (smem) K32 | 128/16 (regs) K32 | 127/16 (regs) K32 | 127/8 (smem) K32 |
| tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 192 | 58304 | 64/12 (smem) S40 K56 | 56/6 (smem) K32 | 56/18 (smem) S36 K56 | 63/18 (smem) K32 | 56/6 (smem) K32 |
| tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 192 | 80704 | 80/12 (smem) K32 | 71/6 (smem) K32 | 80/12 (smem) K32 | 72/12 (smem) K32 | 71/6 (smem) K32 |
| tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 96 | 80704 | 128/6 (smem) S12 K48 | 128/3 (smem) K32 | 128/6 (smem) K32 | 127/6 (smem) K32 | 125/3 (smem) K32 |
| tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 384 | 58560 | 56/24 (smem) K32 | 55/12 (smem) K32 | 56/36 (regs+smem) K32 | 56/36 (regs+smem) K32 | 50/12 (smem) K32 |
| tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 384 | 81216 | 80/24 (regs+smem) S8 K40 | 80/12 (smem) K32 | 80/24 (regs+smem) K32 | 80/24 (regs+smem) K32 | 79/12 (smem) K32 |
| tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 192 | 81216 | 138/12 (regs+smem) K32 | 139/6 (smem) K32 | 143/12 (regs+smem) K32 | 125/12 (regs+smem) K32 | 125/6 (smem) K32 |
| tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 480 | 50384 | 64/30 (regs) K32 | 64/15 (smem) K32 | 64/30 (regs) K32 | 64/30 (regs) K32 | 64/15 (smem) K32 |
| tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 480 | 64864 | 96/15 (regs) K32 | 74/15 (regs+smem) K32 | 95/15 (regs) K32 | 73/15 (regs) K32 | 64/15 (smem) S52 K72 |
| tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 384 | 82240 | 138/12 (regs) K32 | 139/12 (regs+smem) K32 | 143/12 (regs) K32 | 125/12 (regs) K32 | 125/12 (regs+smem) K32 |
| tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 68544 | 64/16 (smem) S36 K56 | 64/8 (smem) K32 | 57/24 (smem) K32 | 63/24 (smem) K32 | 63/8 (smem) K32 |
| tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 90944 | 80/8 (smem) K32 | 71/8 (smem) K32 | 80/16 (smem) K32 | 72/16 (smem) K32 | 71/8 (smem) K32 |
| tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 90944 | 128/4 (smem) S8 K40 | 128/4 (smem) K32 | 128/8 (smem) K32 | 127/8 (smem) K32 | 125/4 (smem) K32 |
| tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 512 | 68800 | 63/32 (regs+smem) K32 | 63/16 (smem) K32 | 64/32 (regs) K32 | 64/32 (regs) K32 | 64/16 (smem) K32 |
| tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 512 | 91456 | 94/16 (regs+smem) K32 | 94/16 (regs+smem) K32 | 96/16 (regs) K32 | 79/16 (regs) K32 | 64/16 (smem) S44 K64 |
| tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 256 | 91456 | 128/8 (smem) K32 | 128/8 (smem) K32 | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | 127/8 (smem) K32 |
| tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 512 | 57728 | 63/32 (regs+smem) K32 | 63/16 (smem) K32 | 64/32 (regs) K32 | 64/32 (regs) K32 | 64/16 (smem) K32 |
| tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 512 | 69312 | 94/16 (regs) K32 | 72/16 (regs+smem) K32 | 96/16 (regs) K32 | 79/16 (regs) K32 | 64/16 (smem) S44 K64 |
| tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 512 | 92480 | 128/16 (regs+smem) K32 | 128/16 (regs+smem) K32 | 128/16 (regs) K32 | 127/16 (regs) K32 | 128/16 (regs+smem) K32 |

### static_o0_10x10 / vs_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 2x4x16 smax32 nwc7 nws1 vxpad0 | 224 | 37536 | 194/7 (regs) | 186/7 (regs) | 194/7 (regs) | 174/7 (regs) | 198/7 (regs) |
| 2x4x16 smax32 nwc7 nws1 vxpad1 | 224 | 38560 | 192/7 (regs) | 190/7 (regs) | 200/7 (regs) | 204/7 (regs) | 196/7 (regs) |
| 2x4x8 smax32 nwc7 nws1 vxpad0 | 224 | 37280 | 194/7 (regs) | 186/7 (regs) | 194/7 (regs) | 174/7 (regs) | 198/7 (regs) |
| 2x4x8 smax32 nwc7 nws1 vxpad1 | 224 | 38304 | 192/7 (regs) | 190/7 (regs) | 200/7 (regs) | 204/7 (regs) | 196/7 (regs) |
| 2x4x8 smax48 nwc7 nws1 vxpad0 | 224 | 55200 | 194/7 (regs) | 186/7 (regs+smem) | 194/7 (regs) | 174/7 (regs) | 198/7 (regs+smem) |
| 2x4x8 smax48 nwc7 nws1 vxpad1 | 224 | 56736 | 192/7 (regs) | 190/7 (regs+smem) | 200/7 (regs) | 204/7 (regs) | 196/7 (regs+smem) |
| 4x4x16 smax64 nwc7 nws1 vxpad0 | 224 | 74176 | 240/7 (regs) | 240/7 (regs+smem) | 254/7 (regs) | 254/7 (regs) | 248/7 (regs+smem) |
| 4x4x16 smax64 nwc7 nws1 vxpad1 | 224 | 76224 | 242/7 (regs) | 242/7 (regs+smem) | 254/7 (regs) | 254/7 (regs) | 248/7 (regs+smem) |

### static_o0_10x10 / whatif

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 4x4x16 smax128 nwc7 nws1 vxpad0 | 224 | 145856 | 240/7 (regs+smem) | - | 254/7 (regs+smem) | 254/7 (regs+smem) | - |
| 4x4x16 smax192 nwc7 nws1 vxpad0 | 224 | 217536 | - | - | 254/7 (regs+smem) | 254/7 (regs+smem) | - |
| 4x4x16 smax256 nwc5 nws1 vxpad0 | 160 | 223424 | - | - | 254/5 (regs+smem) | 254/5 (regs+smem) | - |
| 4x4x16 smax96 nwc5 nws1 vxpad0 | 160 | 85184 | - | 240/5 (regs+smem) | - | - | 244/5 (regs+smem) |
| 4x4x16 smax96 nwc7 nws1 vxpad0 | 224 | 110016 | 240/7 (regs+smem) | - | 254/7 (regs) | 254/7 (regs) | - |
| nxg1 cb1 chk16 listmax2048 | 128 | 19904 | 96/20 (regs) K32 | 80/16 (smem) K32 | 80/24 (regs) S8 K40 | 64/32 (regs) S8 K40 | 64/16 (smem) K32 |
| nxg1 cb1 chk16 listmax4096 | 128 | 28096 | 96/20 (regs+smem) K32 | 80/12 (smem) K32 | 80/24 (regs) S8 K40 | 64/32 (regs+smem) S8 K40 | 64/12 (smem) K32 |
| nxg1 cb1 chk48 listmax1024 | 128 | 39232 | 116/16 (regs+smem) K32 | 115/8 (smem) K32 | 111/16 (regs) K32 | 87/20 (regs+smem) K32 | 80/8 (smem) S8 K40 |
| nxg1 cb1 chk64 listmax1024 | 128 | 50944 | 114/12 (smem) K32 | 114/4 (smem) K32 | 114/16 (regs+smem) K32 | 88/16 (smem) K32 | 88/4 (smem) K32 |
| nxg1 cb2 chk16 listmax2048 | 128 | 20288 | 126/16 (regs) K32 | 117/16 (regs+smem) K32 | 118/16 (regs) K32 | 110/16 (regs) K32 | 110/16 (regs+smem) K32 |
| nxg1 cb2 chk16 listmax4096 | 128 | 28480 | 126/16 (regs) K32 | 117/12 (smem) K32 | 118/16 (regs) K32 | 110/16 (regs) K32 | 110/12 (smem) K32 |
| nxg1 cb2 chk48 listmax1024 | 128 | 40384 | 150/12 (regs) K32 | 149/8 (smem) K32 | 145/12 (regs) K32 | 117/16 (regs) K32 | 115/8 (smem) K32 |
| nxg1 cb2 chk64 listmax1024 | 128 | 52480 | 150/12 (regs+smem) K32 | 149/4 (smem) K32 | 150/12 (regs) K32 | 146/12 (regs) K32 | 142/4 (smem) K32 |
| nxg1 cb4 chk16 listmax2048 | 128 | 28736 | 206/8 (regs) K32 | 204/8 (regs) K32 | 207/8 (regs) K32 | 183/8 (regs) K32 | 182/8 (regs) K32 |
| nxg1 cb4 chk16 listmax4096 | 128 | 36928 | 206/8 (regs) K32 | 204/8 (regs+smem) K32 | 207/8 (regs) K32 | 183/8 (regs) K32 | 182/8 (regs+smem) K32 |
| nxg1 cb4 chk48 listmax1024 | 128 | 65728 | 222/8 (regs+smem) K32 | 221/4 (smem) K32 | 223/8 (regs) K32 | 200/8 (regs) K32 | 197/4 (smem) K32 |
| nxg1 cb4 chk64 listmax1024 | 128 | 86272 | 227/4 (smem) K32 | 225/4 (smem) K32 | 228/8 (regs+smem) K32 | 223/8 (regs+smem) K32 | 227/4 (smem) K32 |
| nxg2 cb2 chk16 listmax2048 | 128 | 15808 | 80/24 (regs) S16 K48 | 72/24 (smem) K32 | 72/28 (regs) S8 K40 | 64/32 (regs) K32 | 64/24 (smem) K32 |
| nxg2 cb2 chk16 listmax4096 | 128 | 24000 | 80/24 (regs+smem) S16 K48 | 72/16 (smem) K32 | 72/28 (regs) S8 K40 | 64/32 (regs) K32 | 64/16 (smem) K32 |
| nxg2 cb2 chk48 listmax1024 | 128 | 26944 | 116/16 (regs) K32 | 113/12 (smem) K32 | 116/16 (regs) K32 | 88/20 (regs) K32 | 80/12 (smem) S12 K48 |
| nxg2 cb2 chk64 listmax1024 | 128 | 34560 | 114/16 (regs+smem) K32 | 117/8 (smem) K32 | 114/16 (regs) K32 | 86/20 (regs) K32 | 80/8 (smem) K32 |
| nxg2 cb3 chk16 listmax2048 | 128 | 20032 | 122/16 (regs) K32 | 96/16 (smem) K32 | 119/16 (regs) K32 | 96/20 (regs) S8 K40 | 95/16 (smem) K32 |
| nxg2 cb3 chk16 listmax4096 | 128 | 28224 | 122/16 (regs) K32 | 96/12 (smem) K32 | 119/16 (regs) K32 | 96/20 (regs) S8 K40 | 95/12 (smem) K32 |
| nxg2 cb3 chk48 listmax1024 | 128 | 39616 | 128/16 (regs+smem) S12 K48 | 125/8 (smem) K32 | 124/16 (regs) K32 | 110/16 (regs) K32 | 114/8 (smem) K32 |
| nxg2 cb3 chk64 listmax1024 | 128 | 51456 | 128/12 (smem) S12 K48 | 125/4 (smem) K32 | 124/16 (regs+smem) K32 | 110/16 (regs+smem) K32 | 114/4 (smem) K32 |
| nxg2 cb4 chk16 listmax2048 | 128 | 20288 | 128/16 (regs) K32 | 118/16 (regs+smem) K32 | 118/16 (regs) K32 | 114/16 (regs) K32 | 112/16 (regs+smem) K32 |
| nxg2 cb4 chk16 listmax4096 | 128 | 28480 | 128/16 (regs) K32 | 118/12 (smem) K32 | 118/16 (regs) K32 | 114/16 (regs) K32 | 112/12 (smem) K32 |
| nxg2 cb4 chk48 listmax1024 | 128 | 40384 | 152/12 (regs) K32 | 150/8 (smem) K32 | 153/12 (regs) K32 | 147/12 (regs) K32 | 123/8 (smem) K32 |
| nxg2 cb4 chk64 listmax1024 | 128 | 52480 | 153/12 (regs+smem) K32 | 151/4 (smem) K32 | 150/12 (regs) K32 | 147/12 (regs) K32 | 128/4 (smem) K32 |
| nxg2 cb7 chk16 listmax2048 | 128 | 28736 | 198/8 (regs) K32 | 195/8 (regs) K32 | 197/8 (regs) K32 | 161/12 (regs) K32 | 160/12 (regs+smem) K32 |
| nxg2 cb7 chk16 listmax4096 | 128 | 36928 | 198/8 (regs) K32 | 195/8 (regs+smem) K32 | 197/8 (regs) K32 | 161/12 (regs) K32 | 160/8 (smem) K32 |
| nxg2 cb7 chk48 listmax1024 | 128 | 65728 | 214/8 (regs+smem) K32 | 211/4 (smem) K32 | 213/8 (regs) K32 | 186/8 (regs) K32 | 185/4 (smem) K32 |
| nxg2 cb7 chk64 listmax1024 | 128 | 86272 | 221/4 (smem) K32 | 216/4 (smem) K32 | 218/8 (regs+smem) K32 | 190/8 (regs+smem) K32 | 189/4 (smem) K32 |
| nxg4 cb13 chk16 listmax2048 | 128 | 26688 | 200/8 (regs) K32 | 197/8 (regs) K32 | 199/8 (regs) K32 | 158/12 (regs) K32 | 161/12 (regs+smem) K32 |
| nxg4 cb13 chk16 listmax4096 | 128 | 34880 | 200/8 (regs) K32 | 197/8 (regs+smem) K32 | 199/8 (regs) K32 | 158/12 (regs) K32 | 161/8 (smem) K32 |
| nxg4 cb13 chk48 listmax1024 | 128 | 59584 | 216/8 (regs+smem) K32 | 213/4 (smem) K32 | 215/8 (regs) K32 | 188/8 (regs) K32 | 187/4 (smem) K32 |
| nxg4 cb13 chk64 listmax1024 | 128 | 78080 | 221/8 (regs+smem) K32 | 218/4 (smem) K32 | 220/8 (regs+smem) K32 | 192/8 (regs+smem) K32 | 191/4 (smem) K32 |
| nxg4 cb4 chk16 listmax2048 | 128 | 15808 | 80/24 (regs) S16 K48 | 72/24 (smem) K32 | 72/28 (regs) S8 K40 | 64/32 (regs) K32 | 64/24 (smem) K32 |
| nxg4 cb4 chk16 listmax4096 | 128 | 24000 | 80/24 (regs+smem) S16 K48 | 72/16 (smem) K32 | 72/28 (regs) S8 K40 | 64/32 (regs) K32 | 64/16 (smem) K32 |
| nxg4 cb4 chk48 listmax1024 | 128 | 26944 | 116/16 (regs) K32 | 113/12 (smem) K32 | 116/16 (regs) K32 | 89/20 (regs) K32 | 80/12 (smem) S12 K48 |
| nxg4 cb4 chk64 listmax1024 | 128 | 34560 | 114/16 (regs+smem) K32 | 117/8 (smem) K32 | 113/16 (regs) K32 | 89/20 (regs) K32 | 88/8 (smem) K32 |
| nxg4 cb5 chk16 listmax2048 | 128 | 17856 | 117/16 (regs) K32 | 115/16 (regs) K32 | 118/16 (regs) K32 | 90/20 (regs) K32 | 89/20 (regs+smem) K32 |
| nxg4 cb5 chk16 listmax4096 | 128 | 26048 | 117/16 (regs) K32 | 115/12 (smem) K32 | 118/16 (regs) K32 | 90/20 (regs) K32 | 89/12 (smem) K32 |
| nxg4 cb5 chk48 listmax1024 | 128 | 33088 | 126/16 (regs+smem) K32 | 118/12 (smem) K32 | 116/16 (regs) K32 | 115/16 (regs) K32 | 112/12 (smem) K32 |
| nxg4 cb5 chk64 listmax1024 | 128 | 42752 | 128/12 (smem) K32 | 124/8 (smem) K32 | 122/16 (regs) K32 | 91/20 (regs+smem) K32 | 91/8 (smem) K32 |
| nxg4 cb7 chk16 listmax2048 | 128 | 20160 | 128/16 (regs) S16 K48 | 122/16 (regs+smem) K32 | 120/16 (regs) K32 | 118/16 (regs) K32 | 116/16 (regs+smem) K32 |
| nxg4 cb7 chk16 listmax4096 | 128 | 28352 | 128/16 (regs) S16 K48 | 122/12 (smem) K32 | 120/16 (regs) K32 | 118/16 (regs) K32 | 116/12 (smem) K32 |
| nxg4 cb7 chk48 listmax1024 | 128 | 40000 | 153/12 (regs) K32 | 150/8 (smem) K32 | 152/12 (regs) K32 | 119/16 (regs) K32 | 118/8 (smem) K32 |
| nxg4 cb7 chk64 listmax1024 | 128 | 51968 | 156/12 (regs+smem) K32 | 155/4 (smem) K32 | 152/12 (regs) K32 | 118/16 (regs+smem) K32 | 118/4 (smem) K32 |

### static_o0_48x1 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 18240 | 122/16 (regs) K32 | 122/16 (regs) K32 | 120/16 (regs) K32 | 90/20 (regs) K32 | 90/20 (regs+smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 32384 | 128/16 (regs) S4 K40 | 120/12 (smem) K32 | 120/16 (regs) K32 | 120/16 (regs) K32 | 118/12 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 20160 | 153/12 (regs) K32 | 153/12 (regs) K32 | 153/12 (regs) K32 | 145/12 (regs) K32 | 147/12 (regs) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 36224 | 168/12 (regs) S4 K40 | 155/8 (smem) K32 | 156/12 (regs) K32 | 155/12 (regs) K32 | 155/8 (smem) K32 |
| nxg2 cb1 chk16 listmax1024 | 128 | 12096 | 95/20 (regs) K32 | 72/28 (regs+smem) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 | 72/28 (regs+smem) K32 |
| nxg2 cb1 chk32 listmax1024 | 128 | 20096 | 96/20 (regs) S8 K40 | 80/16 (smem) K32 | 80/24 (regs) S8 K40 | 80/24 (regs) K32 | 80/16 (smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 14144 | 119/16 (regs) K32 | 117/16 (regs) K32 | 116/16 (regs) K32 | 91/20 (regs) K32 | 89/20 (regs) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 24192 | 128/16 (regs) K32 | 119/16 (regs+smem) K32 | 120/16 (regs) K32 | 121/16 (regs) K32 | 117/16 (regs+smem) K32 |
| nxg2 cb3 chk16 listmax1024 | 128 | 20160 | 147/12 (regs) K32 | 128/16 (regs+smem) S20 K56 | 128/16 (regs) S24 K56 | 121/16 (regs) K32 | 116/16 (regs+smem) K32 |
| nxg2 cb3 chk32 listmax1024 | 128 | 36224 | 168/12 (regs) K32 | 151/8 (smem) K32 | 153/12 (regs) K32 | 125/16 (regs) K32 | 123/8 (smem) K32 |
| nxg4 cb2 chk16 listmax1024 | 128 | 10048 | 94/20 (regs) K32 | 72/28 (regs) K32 | 72/28 (regs) S4 K40 | 72/28 (regs) K32 | 72/28 (regs) K32 |
| nxg4 cb2 chk32 listmax1024 | 128 | 16000 | 96/20 (regs) S4 K40 | 80/24 (regs+smem) K32 | 80/24 (regs) S12 K48 | 72/28 (regs) K32 | 72/24 (smem) K32 |
| nxg4 cb3 chk16 listmax1024 | 128 | 13120 | 114/16 (regs) K32 | 111/16 (regs) K32 | 114/16 (regs) K32 | 80/24 (regs) K32 | 72/28 (regs+smem) S4 K40 |
| nxg4 cb3 chk32 listmax1024 | 128 | 22144 | 128/16 (regs) K32 | 111/16 (regs+smem) K32 | 111/16 (regs) K32 | 80/24 (regs) K32 | 90/16 (smem) K32 |
| nxg4 cb6 chk16 listmax1024 | 128 | 18112 | 146/12 (regs) K32 | 122/16 (regs) K32 | 118/16 (regs) K32 | 120/16 (regs) K32 | 119/16 (regs) K32 |
| nxg4 cb6 chk32 listmax1024 | 128 | 32128 | 168/12 (regs) K32 | 150/12 (regs+smem) K32 | 152/12 (regs) K32 | 120/16 (regs) K32 | 120/12 (smem) K32 |

### static_o0_48x1 / tile_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| tt4 cpba12 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 768 | 27584 | 40/48 (regs+threads) S12 K48 | 40/48 (regs+threads) S12 K48 | 40/48 (regs+threads) S32 K72 | 61/24 (regs) K32 | 40/48 (regs+threads) K32 |
| tt4 cpba12 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 384 | 27584 | 52/36 (regs) K32 | 55/36 (regs+smem) K32 | 56/36 (regs) K32 | 40/48 (regs) S16 K48 | 40/36 (smem) K32 |
| tt4 cpba12 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 192 | 27584 | 64/30 (regs+smem) K32 | 56/18 (smem) K32 | 56/36 (regs) S8 K40 | 55/36 (regs) K32 | 50/18 (smem) K32 |
| tt4 cpba16 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 1024 | 28608 | 55/32 (regs) K32 | 55/32 (regs+threads) K32 | 56/32 (regs) K32 | 55/32 (regs) K32 | 53/32 (regs+threads) K32 |
| tt4 cpba16 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 512 | 28608 | 58/32 (regs) K32 | 57/32 (regs) K32 | 59/32 (regs) K32 | 40/48 (regs) S12 K48 | 40/48 (regs+smem+threads) K32 |
| tt4 cpba16 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 256 | 28608 | 64/32 (regs) K32 | 48/24 (smem) S8 K40 | 64/32 (regs) K32 | 48/40 (regs) K32 | 48/24 (smem) K32 |
| tt4 cpba16 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 128 | 28608 | 93/20 (regs+smem) K32 | 71/12 (smem) K32 | 72/28 (regs+smem) K32 | 64/28 (smem) K32 | 56/12 (smem) K32 |
| tt4 cpba24 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 768 | 30656 | 57/24 (regs) K32 | 57/24 (regs) K32 | 59/24 (regs) K32 | 62/24 (regs) K32 | 40/48 (regs+threads) K32 |
| tt4 cpba24 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 384 | 30656 | 72/24 (regs) K32 | 72/24 (regs) K32 | 72/24 (regs) K32 | 56/36 (regs) K32 | 51/36 (regs+smem) K32 |
| tt4 cpba24 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 192 | 30656 | 93/18 (regs) K32 | 64/18 (smem) K32 | 64/30 (regs) S16 K48 | 64/30 (regs) K32 | 64/18 (smem) K32 |
| tt4 cpba32 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 1024 | 32704 | 58/32 (regs) K32 | 57/32 (regs+threads) K32 | 59/32 (regs) K32 | 57/32 (regs) K32 | 56/32 (regs+threads) K32 |
| tt4 cpba32 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 512 | 32704 | 64/32 (regs) K32 | 58/32 (regs) K32 | 64/32 (regs) K32 | 51/32 (regs) K32 | 58/32 (regs) K32 |
| tt4 cpba32 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 256 | 32704 | 80/24 (regs) S16 K48 | 64/24 (smem) S4 K40 | 64/32 (regs) S16 K48 | 64/32 (regs) K32 | 62/24 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 25536 | 48/40 (regs) K32 | 40/24 (smem) S4 K40 | 40/48 (regs) S8 K40 | 48/40 (regs) K32 | 40/24 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 128 | 25536 | 48/24 (smem) K32 | 48/12 (smem) K32 | 48/32 (smem) S4 K40 | 46/32 (smem) K32 | 40/12 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 64 | 25536 | 72/12 (smem) K32 | 64/6 (smem) K32 | 56/16 (smem) S8 K40 | 56/16 (smem) K32 | 48/6 (smem) K32 |
| tt4 cpba48 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 768 | 36800 | 72/24 (regs) K32 | 72/24 (regs) K32 | 74/24 (regs) K32 | 66/24 (regs) K32 | 62/24 (regs) K32 |
| tt4 cpba48 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 384 | 36800 | 80/24 (regs) S16 K48 | 72/24 (regs+smem) K32 | 75/24 (regs) K32 | 56/36 (regs) S28 K64 | 78/24 (regs+smem) K32 |
| tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 384 | 26048 | 40/48 (regs) S12 K48 | 40/36 (smem) K32 | 40/48 (regs) S8 K40 | 40/48 (regs) K32 | 38/36 (smem) K32 |
| tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 192 | 26048 | 52/36 (regs+smem) K32 | 53/18 (smem) K32 | 56/36 (regs) K32 | 40/48 (regs+smem) S16 K48 | 40/18 (smem) K32 |
| tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 512 | 26560 | 55/32 (regs) K32 | 55/32 (regs) K32 | 56/32 (regs) K32 | 40/48 (regs) K32 | 40/48 (regs+smem+threads) K32 |
| tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 256 | 26560 | 48/40 (regs) K32 | 48/24 (smem) K32 | 48/40 (regs) S4 K40 | 48/40 (regs) K32 | 40/24 (smem) K32 |
| tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 128 | 26560 | 56/24 (smem) K32 | 48/12 (smem) S8 K40 | 56/32 (smem) S8 K40 | 55/32 (smem) K32 | 48/12 (smem) K32 |
| tt4 cpba8 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 64 | 26560 | 95/12 (smem) K32 | 64/6 (smem) S4 K40 | 80/16 (smem) K32 | 72/16 (smem) K32 | 72/6 (smem) K32 |

### static_o0_48x1 / tile_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 384 | 31424 | 56/36 (regs) K32 | 40/36 (smem) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 39/36 (smem) K32 |
| tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 192 | 31424 | 56/30 (smem) K32 | 40/18 (smem) K32 | 40/42 (smem) K32 | 56/36 (regs) K32 | 40/18 (smem) K32 |
| tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 96 | 31424 | 48/15 (smem) K32 | 48/9 (smem) K32 | 48/21 (smem) K32 | 48/21 (smem) K32 | 40/9 (smem) K32 |
| tt4 cpba1 cpbf12 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 384 | 51520 | 40/36 (smem) K32 | 40/12 (smem) K32 | 40/48 (regs+smem) K32 | 40/48 (regs+smem) K32 | 40/12 (smem) K32 |
| tt4 cpba1 cpbf12 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 192 | 51520 | 56/18 (smem) K32 | 40/6 (smem) K32 | 56/24 (smem) K32 | 56/24 (smem) K32 | 52/6 (smem) K32 |
| tt4 cpba1 cpbf12 ppf2 chunk20 acb1 fsb1 fcb1 kv0 | 480 | 24048 | 40/45 (regs) S4 K40 | 40/45 (regs+threads) K32 | 32/60 (regs+threads) K32 | 40/45 (regs) K32 | 39/45 (regs+threads) K32 |
| tt4 cpba1 cpbf12 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 384 | 31680 | 55/36 (regs) K32 | 40/36 (smem) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/36 (smem) K32 |
| tt4 cpba1 cpbf12 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 192 | 31680 | 56/30 (smem) K32 | 56/18 (smem) K32 | 56/36 (regs) K32 | 56/36 (regs) K32 | 40/18 (smem) K32 |
| tt4 cpba1 cpbf12 ppf2 chunk40 acb1 fsb2 fcb1 kv0 | 480 | 36768 | 40/45 (regs) K32 | 40/30 (smem) K32 | 40/45 (regs) K32 | 40/45 (regs) K32 | 40/30 (smem) K32 |
| tt4 cpba1 cpbf12 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 384 | 52032 | 56/36 (regs+smem) K32 | 40/12 (smem) K32 | 56/36 (regs) K32 | 56/36 (regs) K32 | 40/12 (smem) K32 |
| tt4 cpba1 cpbf12 ppf4 chunk10 acb1 fsb1 fcb1 kv0 | 480 | 17856 | 40/45 (regs) S4 K40 | 40/45 (regs+threads) K32 | 32/60 (regs+threads) K32 | 40/45 (regs) K32 | 38/45 (regs+threads) K32 |
| tt4 cpba1 cpbf12 ppf4 chunk20 acb1 fsb2 fcb1 kv0 | 480 | 24368 | 40/45 (regs) K32 | 40/45 (regs+threads) K32 | 40/45 (regs) K32 | 40/45 (regs) K32 | 40/45 (regs+threads) K32 |
| tt4 cpba1 cpbf12 ppf4 chunk40 acb1 fsb4 fcb1 kv0 | 480 | 37408 | 56/30 (regs) K32 | 40/30 (smem) K32 | 59/30 (regs) K32 | 57/30 (regs) K32 | 40/30 (smem) K32 |
| tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 512 | 33472 | 40/48 (regs) K32 | 40/32 (smem) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/32 (smem) K32 |
| tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 256 | 33472 | 40/32 (smem) K32 | 40/16 (smem) K32 | 40/48 (regs+smem) K32 | 40/48 (regs+smem) K32 | 40/16 (smem) K32 |
| tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 128 | 33472 | 48/16 (smem) K32 | 40/8 (smem) K32 | 48/24 (smem) K32 | 48/24 (smem) K32 | 40/8 (smem) K32 |
| tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 64 | 33472 | 64/8 (smem) K32 | 64/4 (smem) K32 | 64/12 (smem) K32 | 72/12 (smem) K32 | 56/4 (smem) K32 |
| tt4 cpba1 cpbf16 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 512 | 53568 | 40/48 (regs+smem) K32 | 40/16 (smem) K32 | 40/48 (regs) K32 | 64/32 (regs) K32 | 40/16 (smem) K32 |
| tt4 cpba1 cpbf16 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 256 | 53568 | 48/24 (smem) K32 | 40/8 (smem) K32 | 40/32 (smem) S32 K48 | 48/32 (smem) K32 | 40/8 (smem) K32 |
| tt4 cpba1 cpbf16 ppf2 chunk16 acb1 fsb1 fcb1 kv0 | 512 | 23552 | 62/32 (regs) K32 | 39/48 (regs+threads) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/48 (regs+threads) K32 |
| tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 512 | 33728 | 57/32 (regs) K32 | 40/32 (smem) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 39/32 (smem) K32 |
| tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 256 | 33728 | 63/32 (regs+smem) K32 | 40/16 (smem) K32 | 40/48 (regs+smem) K32 | 64/32 (regs) K32 | 40/16 (smem) K32 |
| tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 128 | 33728 | 56/16 (smem) K32 | 56/8 (smem) K32 | 64/24 (smem) K32 | 59/24 (smem) K32 | 56/8 (smem) K32 |
| tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb2 fcb1 kv0 | 512 | 33728 | 40/48 (regs) K32 | 40/32 (smem) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/32 (smem) K32 |
| tt4 cpba1 cpbf16 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 512 | 54080 | 56/32 (regs) K32 | 40/16 (smem) K32 | 59/32 (regs) K32 | 57/32 (regs) K32 | 40/16 (smem) K32 |
| tt4 cpba1 cpbf16 ppf4 chunk16 acb1 fsb2 fcb1 kv0 | 512 | 23808 | 40/48 (regs) K32 | 40/48 (regs+threads) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/48 (regs+threads) K32 |
| tt4 cpba1 cpbf16 ppf4 chunk32 acb1 fsb4 fcb1 kv0 | 512 | 34240 | 56/32 (regs) K32 | 40/32 (smem) K32 | 59/32 (regs) K32 | 57/32 (regs) K32 | 40/32 (smem) K32 |
| tt4 cpba1 cpbf16 ppf4 chunk8 acb1 fsb1 fcb1 kv0 | 512 | 18592 | 62/32 (regs) K32 | 39/48 (regs+threads) K32 | 40/48 (regs) K32 | 39/48 (regs) K32 | 39/48 (regs+threads) K32 |
| tt4 cpba1 cpbf24 ppf1 chunk20 acb1 fsb1 fcb1 kv0 | 480 | 30032 | 40/45 (regs) S4 K40 | 40/45 (regs+smem+threads) K32 | 40/45 (regs) K32 | 40/45 (regs) K32 | 39/45 (regs+smem+threads) K32 |
| tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 384 | 37568 | 56/36 (regs) K32 | 40/24 (smem) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/24 (smem) K32 |
| tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 192 | 37568 | 56/24 (smem) K32 | 56/12 (smem) K32 | 52/36 (regs+smem) K32 | 56/36 (regs+smem) K32 | 40/12 (smem) K32 |
| tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 96 | 37568 | 56/12 (smem) K32 | 56/6 (smem) K32 | 56/18 (smem) K32 | 56/18 (smem) K32 | 48/6 (smem) K32 |
| tt4 cpba1 cpbf24 ppf1 chunk40 acb1 fsb2 fcb1 kv0 | 480 | 42592 | 40/45 (regs+smem) K32 | 40/30 (smem) K32 | 40/45 (regs) K32 | 40/45 (regs) K32 | 40/30 (smem) K32 |
| tt4 cpba1 cpbf24 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 384 | 57664 | 56/24 (smem) K32 | 40/12 (smem) K32 | 56/36 (regs+smem) K32 | 56/36 (regs+smem) K32 | 40/12 (smem) K32 |
| tt4 cpba1 cpbf24 ppf2 chunk10 acb1 fsb1 fcb1 kv0 | 480 | 23840 | 40/45 (regs) S4 K40 | 40/45 (regs+threads) K32 | 32/60 (regs+threads) K32 | 40/45 (regs) K32 | 38/45 (regs+threads) K32 |
| tt4 cpba1 cpbf24 ppf2 chunk20 acb1 fsb1 fcb2 kv0 | 480 | 30192 | 57/30 (regs) K32 | 40/45 (regs+smem+threads) K32 | 40/45 (regs) K32 | 40/45 (regs) K32 | 40/45 (regs+smem+threads) K32 |
| tt4 cpba1 cpbf24 ppf2 chunk20 acb1 fsb2 fcb1 kv0 | 480 | 30192 | 40/45 (regs) K32 | 40/45 (regs+smem+threads) K32 | 40/45 (regs) K32 | 40/45 (regs) K32 | 40/45 (regs+smem+threads) K32 |
| tt4 cpba1 cpbf24 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 384 | 37824 | 56/36 (regs) K32 | 40/24 (smem) K32 | 40/48 (regs) K32 | 56/36 (regs) K32 | 40/24 (smem) K32 |
| tt4 cpba1 cpbf24 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 192 | 37824 | 56/24 (smem) K32 | 56/12 (smem) K32 | 64/30 (regs) K32 | 56/36 (regs+smem) K32 | 55/12 (smem) K32 |
| tt4 cpba1 cpbf24 ppf2 chunk40 acb1 fsb4 fcb1 kv0 | 480 | 42912 | 56/30 (regs) K32 | 56/30 (regs+smem) K32 | 59/30 (regs) K32 | 56/30 (regs) K32 | 40/30 (smem) K32 |
| tt4 cpba1 cpbf24 ppf4 chunk10 acb1 fsb2 fcb1 kv0 | 480 | 24000 | 40/45 (regs) K32 | 40/45 (regs+threads) K32 | 40/45 (regs) K32 | 40/45 (regs) K32 | 40/45 (regs+threads) K32 |
| tt4 cpba1 cpbf24 ppf4 chunk20 acb1 fsb4 fcb1 kv0 | 480 | 30512 | 56/30 (regs) K32 | 40/45 (regs+smem+threads) K32 | 59/30 (regs) K32 | 56/30 (regs) K32 | 40/45 (regs+smem+threads) K32 |
| tt4 cpba1 cpbf32 ppf1 chunk16 acb1 fsb1 fcb1 kv0 | 512 | 31616 | 40/48 (regs) K32 | 40/48 (regs+smem+threads) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/48 (regs+smem+threads) K32 |
| tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 512 | 41664 | 40/48 (regs+smem) K32 | 40/32 (smem) K32 | 40/48 (regs) K32 | 55/32 (regs) K32 | 40/32 (smem) K32 |
| tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 256 | 41664 | 48/24 (smem) K32 | 40/16 (smem) K32 | 48/40 (regs+smem) K32 | 64/32 (regs) K32 | 40/16 (smem) K32 |
| tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 128 | 41664 | 56/12 (smem) K32 | 56/8 (smem) K32 | 56/20 (smem) K32 | 56/20 (smem) K32 | 48/8 (smem) K32 |
| tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb2 fcb1 kv0 | 512 | 41664 | 40/48 (regs+smem) K32 | 40/32 (smem) K32 | 40/48 (regs) K32 | 64/32 (regs) K32 | 40/32 (smem) K32 |
| tt4 cpba1 cpbf32 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 512 | 61760 | 56/32 (regs+smem) K32 | 40/16 (smem) K32 | 60/32 (regs) K32 | 64/32 (regs) K32 | 40/16 (smem) S12 K48 |
| tt4 cpba1 cpbf32 ppf2 chunk16 acb1 fsb1 fcb2 kv0 | 512 | 31744 | 57/32 (regs) K32 | 40/48 (regs+smem+threads) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 39/48 (regs+smem+threads) K32 |
| tt4 cpba1 cpbf32 ppf2 chunk16 acb1 fsb2 fcb1 kv0 | 512 | 31744 | 40/48 (regs) K32 | 40/48 (regs+smem+threads) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/48 (regs+smem+threads) K32 |
| tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 512 | 41920 | 56/32 (regs) K32 | 40/32 (smem) K32 | 59/32 (regs) K32 | 56/32 (regs) K32 | 56/32 (regs+smem) K32 |
| tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 256 | 41920 | 64/24 (smem) K32 | 64/16 (smem) K32 | 64/32 (regs) K32 | 59/32 (regs) K32 | 59/16 (smem) K32 |
| tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 512 | 41920 | 56/32 (regs) K32 | 40/32 (smem) K32 | 59/32 (regs) K32 | 57/32 (regs) K32 | 40/32 (smem) K32 |
| tt4 cpba1 cpbf32 ppf2 chunk8 acb1 fsb1 fcb1 kv0 | 512 | 26656 | 62/32 (regs) K32 | 39/48 (regs+smem+threads) K32 | 40/48 (regs) K32 | 39/48 (regs) K32 | 39/48 (regs+smem+threads) K32 |
| tt4 cpba1 cpbf32 ppf4 chunk16 acb1 fsb4 fcb1 kv0 | 512 | 32000 | 56/32 (regs) K32 | 40/48 (regs+smem+threads) K32 | 59/32 (regs) K32 | 57/32 (regs) K32 | 40/48 (regs+smem+threads) K32 |
| tt4 cpba1 cpbf32 ppf4 chunk8 acb1 fsb2 fcb1 kv0 | 512 | 26784 | 40/48 (regs) K32 | 40/48 (regs+smem+threads) K32 | 40/48 (regs) K32 | 32/64 (regs+threads) K32 | 39/48 (regs+smem+threads) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 27328 | 40/20 (smem) K32 | 40/12 (smem) K32 | 40/32 (smem) K32 | 40/32 (smem) K32 | 40/12 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 64 | 27328 | 56/10 (smem) K32 | 40/6 (smem) K32 | 54/16 (smem) K32 | 56/16 (smem) K32 | 40/6 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 32 | 27328 | 69/5 (smem) K32 | 80/3 (smem) K32 | 66/8 (smem) K32 | 65/8 (smem) K32 | 79/3 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 47424 | 48/12 (smem) K32 | 48/8 (smem) K32 | 40/16 (smem) S4 K40 | 40/16 (smem) K32 | 40/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 47424 | 64/6 (smem) K32 | 48/4 (smem) K32 | 56/8 (smem) K32 | 56/8 (smem) K32 | 40/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 27584 | 40/40 (smem) K32 | 39/24 (smem) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/24 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 128 | 27584 | 48/20 (smem) K32 | 40/12 (smem) K32 | 40/32 (smem) K32 | 40/32 (smem) K32 | 38/12 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 64 | 27584 | 56/10 (smem) K32 | 48/6 (smem) K32 | 48/16 (smem) K32 | 48/16 (smem) K32 | 40/6 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 47936 | 40/24 (smem) K32 | 40/16 (smem) K32 | 40/32 (smem) K32 | 40/32 (smem) K32 | 38/16 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 47936 | 48/12 (smem) K32 | 48/8 (smem) K32 | 48/16 (smem) K32 | 48/16 (smem) K32 | 40/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 512 | 28096 | 62/32 (regs) K32 | 39/48 (regs+smem+threads) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/48 (regs+smem+threads) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 512 | 48960 | 40/48 (regs+smem) K32 | 40/32 (smem) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 38/32 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 256 | 48960 | 64/24 (smem) K32 | 40/16 (smem) K32 | 48/32 (smem) K32 | 40/32 (smem) S4 K40 | 39/16 (smem) K32 |
| tt4 cpba1 cpbf48 ppf1 chunk10 acb1 fsb1 fcb1 kv0 | 480 | 36048 | 40/45 (regs) S4 K40 | 40/30 (smem) K32 | 40/45 (regs) K32 | 40/45 (regs) K32 | 39/30 (smem) K32 |
| tt4 cpba1 cpbf48 ppf1 chunk20 acb1 fsb1 fcb2 kv0 | 480 | 42320 | 40/45 (regs+smem) K32 | 40/30 (smem) K32 | 40/45 (regs) K32 | 55/30 (regs) K32 | 40/30 (smem) K32 |
| tt4 cpba1 cpbf48 ppf1 chunk20 acb1 fsb2 fcb1 kv0 | 480 | 42320 | 40/45 (regs+smem) K32 | 40/30 (smem) K32 | 40/45 (regs) K32 | 40/45 (regs) K32 | 40/30 (smem) K32 |
| tt4 cpba1 cpbf48 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 384 | 49856 | 56/36 (regs+smem) K32 | 40/24 (smem) K32 | 50/36 (regs) K32 | 56/36 (regs) K32 | 40/24 (smem) K32 |
| tt4 cpba1 cpbf48 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 192 | 49856 | 56/18 (smem) K32 | 56/12 (smem) K32 | 56/24 (smem) K32 | 64/24 (smem) K32 | 56/12 (smem) K32 |
| tt4 cpba1 cpbf48 ppf1 chunk40 acb1 fsb4 fcb1 kv0 | 480 | 54880 | 55/30 (regs) K32 | 60/15 (smem) K32 | 58/30 (regs) K32 | 64/30 (regs) K32 | 40/15 (smem) S12 K48 |
| tt4 cpba1 cpbf48 ppf2 chunk10 acb1 fsb1 fcb2 kv0 | 480 | 36128 | 57/30 (regs) K32 | 40/30 (smem) K32 | 40/45 (regs) K32 | 40/45 (regs) K32 | 40/30 (smem) K32 |
| tt4 cpba1 cpbf48 ppf2 chunk10 acb1 fsb2 fcb1 kv0 | 480 | 36128 | 40/45 (regs) K32 | 40/30 (smem) K32 | 40/45 (regs) K32 | 40/45 (regs) K32 | 40/30 (smem) K32 |
| tt4 cpba1 cpbf48 ppf2 chunk20 acb1 fsb1 fcb4 kv0 | 480 | 42480 | 58/30 (regs) K32 | 40/30 (smem) K32 | 60/30 (regs) K32 | 56/30 (regs) K32 | 56/30 (regs+smem) K32 |
| tt4 cpba1 cpbf48 ppf2 chunk20 acb1 fsb4 fcb1 kv0 | 480 | 42480 | 56/30 (regs) K32 | 56/30 (regs+smem) K32 | 59/30 (regs) K32 | 56/30 (regs) K32 | 40/30 (smem) K32 |
| tt4 cpba1 cpbf48 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 384 | 50112 | 56/36 (regs+smem) K32 | 56/24 (smem) K32 | 56/36 (regs) K32 | 56/36 (regs) K32 | 55/24 (smem) K32 |
| tt4 cpba1 cpbf48 ppf4 chunk8 acb1 fsb4 fcb1 kv0 | 384 | 34976 | 56/36 (regs) K32 | 40/24 (smem) K32 | 56/36 (regs) K32 | 51/36 (regs) K32 | 40/24 (smem) K32 |
| tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 192 | 27328 | 56/30 (smem) K32 | 40/18 (smem) K32 | 40/48 (regs+smem) K32 | 40/48 (regs+smem) K32 | 40/18 (smem) K32 |
| tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 96 | 27328 | 48/15 (smem) K32 | 40/9 (smem) K32 | 40/24 (smem) K32 | 40/24 (smem) K32 | 40/9 (smem) K32 |
| tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 192 | 47424 | 40/18 (smem) K32 | 40/12 (smem) K32 | 40/24 (smem) K32 | 56/24 (smem) K32 | 40/12 (smem) K32 |
| tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 96 | 47424 | 56/9 (smem) K32 | 48/6 (smem) K32 | 48/12 (smem) K32 | 56/12 (smem) K32 | 40/6 (smem) K32 |
| tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 384 | 27584 | 56/36 (regs) K32 | 39/36 (smem) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/36 (smem) K32 |
| tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 192 | 27584 | 55/30 (smem) K32 | 40/18 (smem) K32 | 40/48 (regs+smem) K32 | 40/48 (regs+smem) K32 | 40/18 (smem) K32 |
| tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 384 | 47936 | 40/36 (smem) K32 | 40/24 (smem) K32 | 40/48 (regs+smem) K32 | 54/36 (regs) K32 | 38/24 (smem) K32 |
| tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 192 | 47936 | 56/18 (smem) K32 | 40/12 (smem) K32 | 56/24 (smem) K32 | 55/24 (smem) K32 | 39/12 (smem) K32 |
| tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 480 | 20272 | 40/45 (regs) S4 K40 | 40/45 (regs+threads) K32 | 32/60 (regs+threads) K32 | 40/45 (regs) K32 | 38/45 (regs+threads) K32 |
| tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 480 | 33312 | 40/45 (regs) K32 | 40/30 (smem) K32 | 40/45 (regs) K32 | 40/45 (regs) K32 | 39/30 (smem) K32 |
| tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 384 | 48960 | 56/36 (regs+smem) K32 | 40/24 (smem) K32 | 56/36 (regs) K32 | 40/48 (regs+smem) S4 K40 | 40/24 (smem) K32 |
| tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 29376 | 40/40 (smem) K32 | 40/24 (smem) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/24 (smem) K32 |
| tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 128 | 29376 | 48/20 (smem) K32 | 40/12 (smem) K32 | 40/28 (smem) K32 | 40/28 (smem) K32 | 40/12 (smem) K32 |
| tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 64 | 29376 | 56/10 (smem) K32 | 56/6 (smem) K32 | 56/14 (smem) S4 K40 | 56/14 (smem) K32 | 40/6 (smem) K32 |
| tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 32 | 29376 | 78/5 (smem) K32 | 80/3 (smem) K32 | 78/7 (smem) K32 | 65/7 (smem) K32 | 79/3 (smem) K32 |
| tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 49472 | 40/24 (smem) K32 | 40/16 (smem) K32 | 40/32 (smem) K32 | 40/32 (smem) K32 | 40/16 (smem) K32 |
| tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 49472 | 56/12 (smem) K32 | 48/8 (smem) K32 | 48/16 (smem) K32 | 48/16 (smem) K32 | 40/8 (smem) K32 |
| tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 512 | 29632 | 62/32 (regs) K32 | 39/48 (regs+smem+threads) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/48 (regs+smem+threads) K32 |
| tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 256 | 29632 | 40/40 (smem) K32 | 40/24 (smem) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/24 (smem) K32 |
| tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 128 | 29632 | 48/20 (smem) K32 | 40/12 (smem) K32 | 40/28 (smem) K32 | 48/28 (smem) K32 | 40/12 (smem) K32 |
| tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 64 | 29632 | 72/10 (smem) K32 | 64/6 (smem) K32 | 66/14 (smem) K32 | 64/14 (smem) K32 | 56/6 (smem) K32 |
| tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 512 | 49984 | 40/48 (regs+smem) K32 | 40/32 (smem) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 38/32 (smem) K32 |
| tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 256 | 49984 | 64/24 (smem) K32 | 40/16 (smem) K32 | 48/32 (smem) K32 | 40/32 (smem) K32 | 40/16 (smem) K32 |
| tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 512 | 19712 | 62/32 (regs) K32 | 39/48 (regs+threads) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 40/48 (regs+threads) K32 |
| tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 512 | 30144 | 40/48 (regs) K32 | 40/48 (regs+smem+threads) K32 | 40/48 (regs) K32 | 40/48 (regs) K32 | 38/48 (regs+smem+threads) K32 |
| tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 512 | 51008 | 56/32 (regs) K32 | 40/16 (smem) K32 | 59/32 (regs) K32 | 55/32 (regs) K32 | 40/16 (smem) K32 |

### static_o0_48x1 / vs_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 2x4x16 smax32 nwc1 nws1 vxpad0 | 32 | 12192 | 166/12 (regs+smem) | 166/7 (smem) | 159/12 (regs) | 163/12 (regs) | 182/7 (smem) |
| 2x4x16 smax32 nwc1 nws1 vxpad1 | 32 | 13216 | 168/11 (smem) | 168/7 (smem) | 168/12 (regs) | 167/12 (regs) | 182/7 (smem) |
| 2x4x16 smax32 nwc1 nws2 vxpad0 | 64 | 12192 | 168/12 (regs) | 164/12 (regs) | 164/12 (regs) | 166/12 (regs) | 192/8 (regs) |
| 2x4x16 smax32 nwc1 nws2 vxpad1 | 64 | 13216 | 168/12 (regs) | 166/12 (regs) | 164/12 (regs) | 166/12 (regs) | 198/8 (regs) |
| 2x4x16 smax32 nwc2 nws1 vxpad0 | 64 | 16416 | 168/12 (regs) | 168/10 (smem) | 165/12 (regs) | 167/12 (regs) | 196/8 (regs) |
| 2x4x16 smax32 nwc2 nws1 vxpad1 | 64 | 17440 | 168/12 (regs) | 168/10 (smem) | 168/12 (regs) | 168/12 (regs) | 194/8 (regs) |
| 2x4x16 smax32 nwc2 nws2 vxpad0 | 128 | 16416 | 164/12 (regs) | 166/12 (regs) | 162/12 (regs) | 208/8 (regs) | 254/8 (regs) |
| 2x4x16 smax32 nwc2 nws2 vxpad1 | 128 | 17440 | 166/12 (regs) | 168/12 (regs) | 162/12 (regs) | 168/12 (regs) | 192/8 (regs) |
| 2x4x16 smax32 nwc3 nws1 vxpad0 | 96 | 20640 | 168/12 (regs) | 168/12 (regs+smem) | 165/12 (regs) | 167/12 (regs) | 200/6 (regs) |
| 2x4x16 smax32 nwc3 nws1 vxpad1 | 96 | 21664 | 168/12 (regs) | 168/12 (regs+smem) | 168/12 (regs) | 168/12 (regs) | 198/6 (regs) |
| 2x4x16 smax32 nwc3 nws2 vxpad0 | 192 | 20640 | 164/12 (regs) | 166/12 (regs) | 168/12 (regs) | 168/12 (regs) | 254/6 (regs) |
| 2x4x16 smax32 nwc3 nws2 vxpad1 | 192 | 21664 | 166/12 (regs) | 168/12 (regs) | 168/12 (regs) | 168/12 (regs) | 190/6 (regs) |
| 2x4x8 smax32 nwc1 nws1 vxpad0 | 32 | 11936 | 166/12 (regs+smem) | 166/7 (smem) | 159/12 (regs) | 163/12 (regs) | 182/7 (smem) |
| 2x4x8 smax32 nwc1 nws1 vxpad1 | 32 | 12960 | 168/12 (regs+smem) | 168/7 (smem) | 168/12 (regs) | 167/12 (regs) | 182/7 (smem) |
| 2x4x8 smax32 nwc1 nws2 vxpad0 | 64 | 11936 | 168/12 (regs) | 164/12 (regs) | 164/12 (regs) | 166/12 (regs) | 192/8 (regs) |
| 2x4x8 smax32 nwc1 nws2 vxpad1 | 64 | 12960 | 168/12 (regs) | 166/12 (regs) | 164/12 (regs) | 166/12 (regs) | 198/8 (regs) |
| 2x4x8 smax32 nwc2 nws1 vxpad0 | 64 | 16160 | 168/12 (regs) | 168/10 (smem) | 165/12 (regs) | 167/12 (regs) | 196/8 (regs) |
| 2x4x8 smax32 nwc2 nws1 vxpad1 | 64 | 17184 | 168/12 (regs) | 168/10 (smem) | 168/12 (regs) | 168/12 (regs) | 194/8 (regs) |
| 2x4x8 smax32 nwc2 nws2 vxpad0 | 128 | 16160 | 164/12 (regs) | 166/12 (regs) | 162/12 (regs) | 208/8 (regs) | 254/8 (regs) |
| 2x4x8 smax32 nwc2 nws2 vxpad1 | 128 | 17184 | 166/12 (regs) | 168/12 (regs) | 162/12 (regs) | 168/12 (regs) | 192/8 (regs) |
| 2x4x8 smax32 nwc3 nws1 vxpad0 | 96 | 20384 | 168/12 (regs) | 168/12 (regs+smem) | 165/12 (regs) | 167/12 (regs) | 200/6 (regs) |
| 2x4x8 smax32 nwc3 nws1 vxpad1 | 96 | 21408 | 168/12 (regs) | 168/12 (regs+smem) | 168/12 (regs) | 168/12 (regs) | 198/6 (regs) |
| 2x4x8 smax32 nwc3 nws2 vxpad0 | 192 | 20384 | 164/12 (regs) | 166/12 (regs) | 168/12 (regs) | 168/12 (regs) | 254/6 (regs) |
| 2x4x8 smax32 nwc3 nws2 vxpad1 | 192 | 21408 | 166/12 (regs) | 168/12 (regs) | 168/12 (regs) | 168/12 (regs) | 190/6 (regs) |
| 2x4x8 smax48 nwc1 nws1 vxpad0 | 32 | 17568 | 166/9 (smem) | 166/5 (smem) | 159/12 (regs+smem) | 163/12 (regs+smem) | 182/5 (smem) |
| 2x4x8 smax48 nwc1 nws1 vxpad1 | 32 | 19104 | 168/8 (smem) | 168/5 (smem) | 168/11 (smem) | 167/11 (smem) | 182/5 (smem) |
| 2x4x8 smax48 nwc1 nws2 vxpad0 | 64 | 17568 | 168/12 (regs) | 164/10 (smem) | 164/12 (regs) | 166/12 (regs) | 192/8 (regs) |
| 2x4x8 smax48 nwc1 nws2 vxpad1 | 64 | 19104 | 168/12 (regs) | 166/10 (smem) | 164/12 (regs) | 166/12 (regs) | 198/8 (regs) |
| 2x4x8 smax48 nwc2 nws1 vxpad0 | 64 | 23840 | 168/12 (regs+smem) | 168/8 (smem) | 165/12 (regs) | 167/12 (regs) | 196/8 (regs+smem) |
| 2x4x8 smax48 nwc2 nws1 vxpad1 | 64 | 25376 | 168/12 (regs+smem) | 168/6 (smem) | 168/12 (regs) | 168/12 (regs) | 194/6 (smem) |
| 2x4x8 smax48 nwc2 nws2 vxpad0 | 128 | 23840 | 164/12 (regs) | 166/12 (regs) | 162/12 (regs) | 208/8 (regs) | 254/8 (regs) |
| 2x4x8 smax48 nwc2 nws2 vxpad1 | 128 | 25376 | 166/12 (regs) | 168/12 (regs+smem) | 162/12 (regs) | 168/12 (regs) | 192/8 (regs) |
| 2x4x8 smax48 nwc3 nws1 vxpad0 | 96 | 30112 | 168/12 (regs) | 168/9 (smem) | 165/12 (regs) | 167/12 (regs) | 200/6 (regs) |
| 2x4x8 smax48 nwc3 nws1 vxpad1 | 96 | 31648 | 168/12 (regs) | 168/9 (smem) | 168/12 (regs) | 168/12 (regs) | 198/6 (regs) |
| 2x4x8 smax48 nwc3 nws2 vxpad0 | 192 | 30112 | 164/12 (regs) | 166/12 (regs) | 168/12 (regs) | 168/12 (regs) | 254/6 (regs) |
| 2x4x8 smax48 nwc3 nws2 vxpad1 | 192 | 31648 | 166/12 (regs) | 168/12 (regs) | 168/12 (regs) | 168/12 (regs) | 190/6 (regs) |
| 4x4x16 smax32 nwc1 nws1 vxpad0 | 32 | 12992 | 250/8 (regs) | 252/7 (smem) | 252/8 (regs) | 238/8 (regs) | 248/7 (smem) |
| 4x4x16 smax32 nwc1 nws1 vxpad1 | 32 | 14016 | 246/8 (regs) | 254/6 (smem) | 254/8 (regs) | 172/8 (regs) | 250/6 (smem) |
| 4x4x16 smax32 nwc1 nws2 vxpad0 | 64 | 12992 | 220/8 (regs) | 228/8 (regs) | 234/8 (regs) | 204/8 (regs) | 168/12 (regs) |
| 4x4x16 smax32 nwc1 nws2 vxpad1 | 64 | 14016 | 232/8 (regs) | 226/8 (regs) | 234/8 (regs) | 204/8 (regs) | 168/12 (regs+smem) |
| 4x4x16 smax32 nwc2 nws1 vxpad0 | 64 | 17216 | 242/8 (regs) | 238/8 (regs) | 254/8 (regs) | 234/8 (regs) | 242/8 (regs) |
| 4x4x16 smax32 nwc2 nws1 vxpad1 | 64 | 18240 | 242/8 (regs) | 242/8 (regs) | 254/8 (regs) | 238/8 (regs) | 240/8 (regs) |
| 4x4x16 smax32 nwc2 nws2 vxpad0 | 128 | 17216 | 236/8 (regs) | 226/8 (regs) | 234/8 (regs) | 230/8 (regs) | 252/8 (regs) |
| 4x4x16 smax32 nwc2 nws2 vxpad1 | 128 | 18240 | 168/12 (regs) | 228/8 (regs) | 236/8 (regs) | 230/8 (regs) | 252/8 (regs) |
| 4x4x16 smax32 nwc3 nws1 vxpad0 | 96 | 21440 | 240/6 (regs) | 240/6 (regs) | 254/6 (regs) | 254/6 (regs) | 252/6 (regs) |
| 4x4x16 smax32 nwc3 nws1 vxpad1 | 96 | 22464 | 240/6 (regs) | 242/6 (regs) | 254/6 (regs) | 254/6 (regs) | 250/6 (regs) |
| 4x4x16 smax32 nwc3 nws2 vxpad0 | 192 | 21440 | 234/6 (regs) | 228/6 (regs) | 234/6 (regs) | 222/6 (regs) | 242/6 (regs) |
| 4x4x16 smax32 nwc3 nws2 vxpad1 | 192 | 22464 | 168/12 (regs) | 230/6 (regs) | 236/6 (regs) | 254/6 (regs) | 254/6 (regs) S28 K32 |
| 4x4x16 smax64 nwc1 nws1 vxpad0 | 32 | 24256 | 250/6 (smem) | 252/4 (smem) | 252/8 (regs) | 238/8 (regs) | 248/4 (smem) |
| 4x4x16 smax64 nwc1 nws1 vxpad1 | 32 | 26304 | 246/6 (smem) | 254/3 (smem) | 254/8 (regs+smem) | 172/8 (regs+smem) | 250/3 (smem) |
| 4x4x16 smax64 nwc1 nws2 vxpad0 | 64 | 24256 | 220/8 (regs) | 228/8 (regs+smem) | 234/8 (regs) | 204/8 (regs) | 168/8 (smem) |
| 4x4x16 smax64 nwc1 nws2 vxpad1 | 64 | 26304 | 232/8 (regs) | 226/6 (smem) | 234/8 (regs) | 204/8 (regs) | 168/6 (smem) |
| 4x4x16 smax64 nwc2 nws1 vxpad0 | 64 | 32576 | 242/8 (regs+smem) | 238/6 (smem) | 254/8 (regs) | 234/8 (regs) | 242/6 (smem) |
| 4x4x16 smax64 nwc2 nws1 vxpad1 | 64 | 34624 | 242/8 (regs+smem) | 242/4 (smem) | 254/8 (regs) | 238/8 (regs) | 240/4 (smem) |
| 4x4x16 smax64 nwc2 nws2 vxpad0 | 128 | 32576 | 236/8 (regs) | 226/8 (regs) | 234/8 (regs) | 230/8 (regs) | 250/8 (regs) |
| 4x4x16 smax64 nwc2 nws2 vxpad1 | 128 | 34624 | 168/12 (regs) | 228/8 (regs+smem) | 236/8 (regs) | 230/8 (regs) | 252/8 (regs+smem) |
| 4x4x16 smax64 nwc3 nws1 vxpad0 | 96 | 40896 | 240/6 (regs) | 240/6 (regs+smem) | 254/6 (regs) | 254/6 (regs) | 252/6 (regs+smem) |
| 4x4x16 smax64 nwc3 nws1 vxpad1 | 96 | 42944 | 240/6 (regs) | 242/6 (regs+smem) | 254/6 (regs) | 254/6 (regs) | 250/6 (regs+smem) |
| 4x4x16 smax64 nwc3 nws2 vxpad0 | 192 | 40896 | 234/6 (regs) | 228/6 (regs) | 234/6 (regs) | 224/6 (regs) | 242/6 (regs) |
| 4x4x16 smax64 nwc3 nws2 vxpad1 | 192 | 42944 | 168/12 (regs) | 230/6 (regs) | 236/6 (regs) | 254/6 (regs) | 254/6 (regs) S28 K32 |

### static_o0_48x1 / whatif

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 4x4x16 smax128 nwc1 nws1 vxpad0 | 32 | 46784 | 250/3 (smem) | 252/2 (smem) | 252/4 (smem) | 238/4 (smem) | 248/2 (smem) |
| 4x4x16 smax128 nwc2 nws1 vxpad0 | 64 | 63296 | 242/4 (smem) | 238/2 (smem) | 254/6 (smem) | 234/6 (smem) | 242/2 (smem) |
| 4x4x16 smax128 nwc3 nws1 vxpad0 | 96 | 79808 | 240/6 (regs+smem) | 240/3 (smem) | 254/6 (regs+smem) | 254/6 (regs+smem) | 252/3 (smem) |
| 4x4x16 smax192 nwc1 nws1 vxpad0 | 32 | 69312 | 250/2 (smem) | 252/1 (smem) | 252/3 (smem) | 238/3 (smem) | 248/1 (smem) |
| 4x4x16 smax192 nwc2 nws1 vxpad0 | 64 | 94016 | 242/2 (smem) | 238/2 (smem) | 254/4 (smem) | 234/4 (smem) | 242/2 (smem) |
| 4x4x16 smax192 nwc3 nws1 vxpad0 | 96 | 118720 | 240/3 (smem) | - | 254/3 (smem) | 254/3 (smem) | - |
| 4x4x16 smax256 nwc1 nws1 vxpad0 | 32 | 91840 | 250/1 (smem) | 252/1 (smem) | 252/2 (smem) | 238/2 (smem) | 248/1 (smem) |
| 4x4x16 smax256 nwc2 nws1 vxpad0 | 64 | 124736 | 242/2 (smem) | - | 254/2 (smem) | 234/2 (smem) | - |
| 4x4x16 smax256 nwc3 nws1 vxpad0 | 96 | 157632 | 240/3 (smem) | - | 254/3 (smem) | 254/3 (smem) | - |
| 4x4x16 smax96 nwc1 nws1 vxpad0 | 32 | 35520 | 250/4 (smem) | 252/2 (smem) | 252/6 (smem) | 238/6 (smem) | 248/2 (smem) |
| 4x4x16 smax96 nwc2 nws1 vxpad0 | 64 | 47936 | 242/6 (smem) | 238/4 (smem) | 254/8 (regs+smem) | 234/8 (regs+smem) | 242/4 (smem) |
| 4x4x16 smax96 nwc3 nws1 vxpad0 | 96 | 60352 | 240/6 (regs+smem) | 240/3 (smem) | 254/6 (regs) | 254/6 (regs) | 252/3 (smem) |
| nxg1 cb1 chk16 listmax2048 | 128 | 22336 | 122/16 (regs) K32 | 122/16 (regs+smem) K32 | 120/16 (regs) K32 | 90/20 (regs) K32 | 90/16 (smem) K32 |
| nxg1 cb1 chk16 listmax4096 | 128 | 30528 | 122/16 (regs) K32 | 122/12 (smem) K32 | 120/16 (regs) K32 | 90/20 (regs) K32 | 90/12 (smem) K32 |
| nxg1 cb1 chk48 listmax1024 | 128 | 46528 | 162/12 (regs+smem) K32 | 128/8 (smem) K32 | 148/12 (regs) K32 | 125/16 (regs+smem) K32 | 125/8 (smem) K32 |
| nxg1 cb1 chk64 listmax1024 | 128 | 60672 | 168/8 (smem) S4 K40 | 159/4 (smem) K32 | 161/12 (regs+smem) K32 | 127/12 (smem) K32 | 121/4 (smem) K32 |
| nxg1 cb2 chk16 listmax2048 | 128 | 24256 | 153/12 (regs) K32 | 153/12 (regs) K32 | 153/12 (regs) K32 | 145/12 (regs) K32 | 147/12 (regs) K32 |
| nxg1 cb2 chk16 listmax4096 | 128 | 32448 | 153/12 (regs) K32 | 153/12 (regs+smem) K32 | 153/12 (regs) K32 | 145/12 (regs) K32 | 147/12 (regs+smem) K32 |
| nxg1 cb2 chk48 listmax1024 | 128 | 52288 | 223/8 (regs) K32 | 222/4 (smem) K32 | 225/8 (regs) K32 | 226/8 (regs) K32 | 223/4 (smem) K32 |
| nxg1 cb2 chk64 listmax1024 | 128 | 68352 | 248/8 (regs+smem) K32 | 241/4 (smem) K32 | 243/8 (regs) K32 | 246/8 (regs) K32 | 226/4 (smem) K32 |
| nxg2 cb1 chk16 listmax2048 | 128 | 16192 | 95/20 (regs) K32 | 72/20 (smem) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 | 72/20 (smem) K32 |
| nxg2 cb1 chk16 listmax4096 | 128 | 24384 | 95/20 (regs) K32 | 72/16 (smem) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 | 72/16 (smem) K32 |
| nxg2 cb1 chk48 listmax1024 | 128 | 28096 | 118/16 (regs) K32 | 96/12 (smem) K32 | 111/16 (regs) K32 | 87/20 (regs) K32 | 87/12 (smem) K32 |
| nxg2 cb1 chk64 listmax1024 | 128 | 36096 | 128/16 (regs+smem) K32 | 117/8 (smem) K32 | 120/16 (regs) K32 | 118/16 (regs) K32 | 116/8 (smem) K32 |
| nxg2 cb2 chk16 listmax2048 | 128 | 18240 | 119/16 (regs) K32 | 117/16 (regs) K32 | 116/16 (regs) K32 | 91/20 (regs) K32 | 89/20 (regs+smem) K32 |
| nxg2 cb2 chk16 listmax4096 | 128 | 26432 | 119/16 (regs) K32 | 117/12 (smem) K32 | 116/16 (regs) K32 | 91/20 (regs) K32 | 89/12 (smem) K32 |
| nxg2 cb2 chk48 listmax1024 | 128 | 34240 | 160/12 (regs) K32 | 128/8 (smem) S4 K40 | 128/16 (regs) S4 K40 | 128/16 (regs) K32 | 128/8 (smem) K32 |
| nxg2 cb2 chk64 listmax1024 | 128 | 44288 | 168/12 (regs+smem) K32 | 128/8 (smem) S28 K64 | 128/16 (regs) S8 K40 | 162/12 (regs) K32 | 158/8 (smem) K32 |
| nxg2 cb3 chk16 listmax2048 | 128 | 24256 | 147/12 (regs) K32 | 128/16 (regs+smem) S20 K56 | 128/16 (regs) S24 K56 | 121/16 (regs) K32 | 116/16 (regs+smem) K32 |
| nxg2 cb3 chk16 listmax4096 | 128 | 32448 | 147/12 (regs) K32 | 128/12 (smem) S20 K56 | 128/16 (regs) S24 K56 | 121/16 (regs) K32 | 116/12 (smem) K32 |
| nxg2 cb3 chk48 listmax1024 | 128 | 52288 | 168/12 (regs+smem) S32 K64 | 168/4 (smem) S32 K64 | 216/8 (regs) K32 | 221/8 (regs) K32 | 218/4 (smem) K32 |
| nxg2 cb3 chk64 listmax1024 | 128 | 68352 | 239/8 (regs+smem) K32 | 239/4 (smem) K32 | 247/8 (regs) K32 | 244/8 (regs) K32 | 254/4 (smem) K32 |
| nxg4 cb2 chk16 listmax2048 | 128 | 14144 | 94/20 (regs) K32 | 72/24 (smem) K32 | 72/28 (regs) S4 K40 | 72/28 (regs) K32 | 72/24 (smem) K32 |
| nxg4 cb2 chk16 listmax4096 | 128 | 22336 | 94/20 (regs) K32 | 72/16 (smem) K32 | 72/28 (regs) S4 K40 | 72/28 (regs) K32 | 72/16 (smem) K32 |
| nxg4 cb2 chk48 listmax1024 | 128 | 21952 | 117/16 (regs) K32 | 80/16 (smem) K32 | 108/16 (regs) K32 | 80/24 (regs) K32 | 78/16 (smem) K32 |
| nxg4 cb2 chk64 listmax1024 | 128 | 27904 | 128/16 (regs) K32 | 115/12 (smem) K32 | 119/16 (regs) K32 | 113/16 (regs) K32 | 111/12 (smem) K32 |
| nxg4 cb3 chk16 listmax2048 | 128 | 17216 | 114/16 (regs) K32 | 111/16 (regs) K32 | 114/16 (regs) K32 | 80/24 (regs) K32 | 72/20 (smem) S4 K40 |
| nxg4 cb3 chk16 listmax4096 | 128 | 25408 | 114/16 (regs) K32 | 111/12 (smem) K32 | 114/16 (regs) K32 | 80/24 (regs) K32 | 72/12 (smem) S4 K40 |
| nxg4 cb3 chk48 listmax1024 | 128 | 31168 | 150/12 (regs) K32 | 147/12 (regs+smem) K32 | 151/12 (regs) K32 | 119/16 (regs) K32 | 117/12 (smem) K32 |
| nxg4 cb3 chk64 listmax1024 | 128 | 40192 | 158/12 (regs) K32 | 128/8 (smem) S4 K40 | 147/12 (regs) K32 | 125/16 (regs) K32 | 122/8 (smem) K32 |
| nxg4 cb6 chk16 listmax2048 | 128 | 22208 | 146/12 (regs) K32 | 122/16 (regs+smem) K32 | 118/16 (regs) K32 | 120/16 (regs) K32 | 119/16 (regs+smem) K32 |
| nxg4 cb6 chk16 listmax4096 | 128 | 30400 | 146/12 (regs) K32 | 122/12 (smem) K32 | 118/16 (regs) K32 | 120/16 (regs) K32 | 119/12 (smem) K32 |
| nxg4 cb6 chk48 listmax1024 | 128 | 46144 | 168/12 (regs+smem) S12 K48 | 168/8 (smem) S12 K48 | 214/8 (regs) K32 | 220/8 (regs) K32 | 217/8 (regs+smem) K32 |
| nxg4 cb6 chk64 listmax1024 | 128 | 60160 | 239/8 (regs+smem) K32 | 239/4 (smem) K32 | 240/8 (regs) K32 | 248/8 (regs) K32 | 246/4 (smem) K32 |

### static_o2_cb5 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 28032 | 96/20 (regs+smem) K32 | 72/12 (smem) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 | 80/12 (smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 51968 | 96/12 (smem) K32 | 80/4 (smem) K32 | 80/16 (smem) S4 K40 | 72/16 (smem) K32 | 72/4 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 28032 | 128/16 (regs) K32 | 123/12 (smem) K32 | 124/16 (regs) K32 | 122/16 (regs) K32 | 122/12 (smem) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 51968 | 128/12 (smem) K32 | 128/4 (smem) K32 | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | 127/4 (smem) K32 |
| nxg2 cb1 chk16 listmax1024 | 128 | 23680 | 72/24 (smem) K32 | 56/16 (smem) K32 | 56/36 (regs+smem) K32 | 56/36 (regs+smem) K32 | 56/16 (smem) K32 |
| nxg2 cb1 chk32 listmax1024 | 128 | 43264 | 72/12 (smem) K32 | 56/8 (smem) K32 | 56/20 (smem) K32 | 56/20 (smem) K32 | 56/8 (smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 23936 | 96/20 (regs) K32 | 72/16 (smem) K32 | 72/28 (regs) K32 | 80/24 (regs) K32 | 80/16 (smem) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 43776 | 96/12 (smem) K32 | 72/8 (smem) K32 | 80/20 (smem) K32 | 80/20 (smem) K32 | 72/8 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 28032 | 128/16 (regs) K32 | 123/12 (smem) K32 | 125/16 (regs) K32 | 124/16 (regs) K32 | 124/12 (smem) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 51968 | 128/12 (smem) K32 | 128/4 (smem) K32 | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | 127/4 (smem) K32 |
| nxg4 cb2 chk16 listmax1024 | 128 | 21632 | 64/28 (smem) K32 | 56/16 (smem) K32 | 64/32 (regs) K32 | 56/36 (regs) K32 | 56/16 (smem) K32 |
| nxg4 cb2 chk32 listmax1024 | 128 | 39168 | 64/16 (smem) K32 | 56/8 (smem) K32 | 64/20 (smem) K32 | 56/20 (smem) K32 | 56/8 (smem) K32 |
| nxg4 cb3 chk16 listmax1024 | 128 | 23808 | 96/20 (regs) K32 | 64/16 (smem) K32 | 72/28 (regs) K32 | 80/24 (regs) K32 | 64/16 (smem) S8 K40 |
| nxg4 cb3 chk32 listmax1024 | 128 | 43520 | 96/12 (smem) K32 | 64/8 (smem) K32 | 72/20 (smem) K32 | 80/20 (smem) K32 | 64/8 (smem) S8 K40 |
| nxg4 cb4 chk16 listmax1024 | 128 | 23936 | 96/20 (regs) K32 | 72/16 (smem) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 | 80/16 (smem) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 43776 | 96/12 (smem) K32 | 72/8 (smem) K32 | 80/20 (smem) K32 | 72/20 (smem) K32 | 72/8 (smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 28032 | 128/16 (regs) K32 | 121/12 (smem) K32 | 123/16 (regs) K32 | 121/16 (regs) K32 | 121/12 (smem) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 51968 | 128/12 (smem) K32 | 128/4 (smem) K32 | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | 127/4 (smem) K32 |

### static_o2_cb5 / tile_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 35328 | 72/16 (smem) K32 | 72/8 (smem) K32 | 72/24 (smem) K32 | 64/24 (smem) K32 | 64/8 (smem) K32 |
| tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 64 | 35328 | 123/8 (smem) K32 | 96/4 (smem) K32 | 122/12 (smem) K32 | 125/12 (smem) K32 | 96/4 (smem) K32 |
| tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 192 | 35584 | 64/24 (smem) S12 K48 | 64/12 (smem) K32 | 80/24 (regs) K32 | 64/30 (regs) K32 | 64/12 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 35840 | 80/24 (regs) K32 | 64/16 (smem) S8 K40 | 73/24 (regs) K32 | 64/32 (regs) K32 | 61/16 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 128 | 35840 | 96/16 (smem) K32 | 80/8 (smem) K32 | 96/20 (regs) K32 | 96/20 (regs) K32 | 96/8 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 64 | 35840 | 164/8 (smem) K32 | 163/4 (smem) K32 | 164/12 (regs+smem) K32 | 157/12 (regs+smem) K32 | 153/4 (smem) K32 |
| tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 320 | 36096 | 84/20 (regs) K32 | 84/20 (regs+smem) K32 | 81/20 (regs) K32 | 84/20 (regs) K32 | 77/20 (regs+smem) K32 |

### static_o2_cb5 / tile_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 32 | 39680 | 80/4 (smem) K32 | 80/2 (smem) K32 | 71/5 (smem) K32 | 64/5 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 32 | 70144 | 128/2 (smem) K32 | 96/1 (smem) K32 | 96/3 (smem) K32 | 95/3 (smem) K32 | 93/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | 39936 | 64/8 (smem) K32 | 56/4 (smem) K32 | 64/10 (smem) K32 | 56/10 (smem) K32 | 48/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 64 | 70656 | 96/4 (smem) K32 | 79/2 (smem) K32 | 93/6 (smem) K32 | 80/6 (smem) K32 | 79/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 32 | 70656 | 167/2 (smem) K32 | 143/1 (smem) K32 | 168/3 (smem) S4 K40 | 147/3 (smem) K32 | 143/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 40448 | 64/16 (smem) K32 | 48/8 (smem) K32 | 56/20 (smem) K32 | 48/20 (smem) K32 | 40/8 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 71680 | 96/8 (smem) K32 | 79/4 (smem) K32 | 80/12 (smem) K32 | 79/12 (smem) K32 | 72/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 71680 | 157/4 (smem) K32 | 128/2 (smem) K32 | 128/6 (smem) K32 | 127/6 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | 45824 | 72/6 (smem) K32 | 56/4 (smem) K32 | 64/8 (smem) K32 | 56/8 (smem) K32 | 58/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 64 | 76288 | 96/4 (smem) K32 | 79/2 (smem) K32 | 95/6 (smem) K32 | 72/6 (smem) K32 | 79/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 32 | 76288 | 167/2 (smem) K32 | 143/1 (smem) K32 | 168/3 (smem) S4 K40 | 128/3 (smem) K32 | 128/1 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 46080 | 64/12 (smem) K32 | 48/8 (smem) K32 | 56/16 (smem) K32 | 48/16 (smem) K32 | 48/8 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 76800 | 96/8 (smem) K32 | 72/4 (smem) K32 | 80/12 (smem) K32 | 79/12 (smem) K32 | 72/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 76800 | 159/4 (smem) K32 | 128/2 (smem) K32 | 128/6 (smem) K32 | 128/6 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 46592 | 63/24 (smem) K32 | 48/16 (smem) K32 | 58/32 (regs+smem) K32 | 48/32 (smem) K32 | 48/16 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 77824 | 80/16 (smem) S4 K40 | 71/8 (smem) K32 | 80/16 (smem) K32 | 80/16 (smem) K32 | 71/8 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 77824 | 128/8 (smem) S8 K40 | 128/4 (smem) K32 | 128/8 (smem) K32 | 127/8 (smem) K32 | 127/4 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 96 | 49920 | 70/9 (smem) K32 | 48/6 (smem) K32 | 64/12 (smem) S36 K56 | 56/12 (smem) K32 | 50/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | 80384 | 96/6 (smem) K32 | 72/3 (smem) K32 | 80/6 (smem) K32 | 72/6 (smem) K32 | 72/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 192 | 50176 | 56/18 (smem) K32 | 56/12 (smem) K32 | 56/24 (smem) K32 | 56/24 (smem) K32 | 56/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 192 | 80896 | 80/12 (smem) S8 K40 | 71/6 (smem) K32 | 80/12 (smem) K32 | 80/12 (smem) K32 | 71/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 96 | 80896 | 147/6 (smem) K32 | 128/3 (smem) K32 | 128/6 (smem) K32 | 127/6 (smem) K32 | 127/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 384 | 50688 | 56/36 (regs+smem) K32 | 56/12 (smem) K32 | 56/36 (regs) K32 | 56/36 (regs) K32 | 56/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 384 | 81920 | 80/24 (regs+smem) S8 K40 | 80/12 (smem) K32 | 80/24 (regs+smem) K32 | 80/24 (regs+smem) K32 | 79/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 192 | 81920 | 143/12 (regs+smem) K32 | 143/6 (smem) K32 | 147/12 (regs+smem) K32 | 131/12 (regs+smem) K32 | 127/6 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 56064 | 64/8 (smem) S4 K40 | 48/4 (smem) K32 | 64/16 (smem) S36 K56 | 56/16 (smem) K32 | 51/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 86528 | 96/4 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 71/8 (smem) K32 | 72/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 86528 | 157/2 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 127/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 56320 | 63/16 (smem) K32 | 48/8 (smem) K32 | 58/32 (regs+smem) K32 | 48/32 (smem) K32 | 48/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 87040 | 80/8 (smem) S4 K40 | 71/8 (smem) K32 | 80/16 (smem) K32 | 80/16 (smem) K32 | 71/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 87040 | 128/4 (smem) S8 K40 | 128/4 (smem) K32 | 128/8 (smem) K32 | 127/8 (smem) K32 | 127/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 512 | 56832 | 63/32 (regs+smem) K32 | 64/16 (smem) K32 | 64/32 (regs) K32 | 60/32 (regs) K32 | 64/16 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 512 | 88064 | 94/16 (regs+smem) K32 | 64/16 (smem) S32 K48 | 96/16 (regs) K32 | 80/16 (regs) K32 | 64/16 (smem) S12 K48 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 256 | 88064 | 127/8 (smem) K32 | 128/8 (smem) K32 | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | 127/8 (smem) K32 |
| tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 160 | 60160 | 64/10 (smem) S4 K40 | 48/5 (smem) K32 | 64/15 (smem) S36 K56 | 56/15 (smem) K32 | 50/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 160 | 90624 | 96/5 (smem) K32 | 72/5 (smem) K32 | 96/10 (smem) K32 | 71/10 (smem) K32 | 72/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 320 | 60416 | 64/20 (smem) K32 | 48/10 (smem) K32 | 58/30 (regs+smem) K32 | 48/30 (smem) K32 | 48/10 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 320 | 91136 | 95/10 (smem) K32 | 95/10 (smem) K32 | 95/20 (regs+smem) K32 | 93/20 (regs+smem) K32 | 91/10 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 160 | 91136 | 128/5 (smem) S8 K40 | 128/5 (smem) K32 | 128/10 (smem) K32 | 127/10 (smem) K32 | 127/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 480 | 53120 | 64/30 (regs) K32 | 64/15 (smem) K32 | 64/30 (regs) K32 | 62/30 (regs) K32 | 64/15 (smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 480 | 76544 | 94/15 (regs) K32 | 76/15 (regs+smem) K32 | 96/15 (regs) K32 | 74/15 (regs) K32 | 92/15 (regs+smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 320 | 92160 | 143/10 (regs+smem) K32 | 143/10 (regs+smem) K32 | 147/10 (regs) K32 | 131/10 (regs) K32 | 127/10 (regs+smem) K32 |

### static_o2_cb5 / vt_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 2x4x16 smax32 nwc4 nws1 vxpad0 | 128 | 49184 | 167/12 (regs+smem) | 168/8 (smem) | 168/12 (regs) | 168/12 (regs) | 180/8 (regs+smem) |
| 2x4x16 smax32 nwc4 nws2 vxpad0 | 256 | 55584 | 170/8 (regs) | 171/8 (regs+smem) | 172/8 (regs) | 184/8 (regs) | 194/8 (regs+smem) |
| 2x4x8 smax32 nwc4 nws1 vxpad0 | 128 | 48928 | 167/12 (regs+smem) | 168/8 (smem) | 168/12 (regs) | 168/12 (regs) | 180/8 (regs+smem) |
| 2x4x8 smax32 nwc4 nws2 vxpad0 | 256 | 55328 | 170/8 (regs) | 171/8 (regs+smem) | 172/8 (regs) | 184/8 (regs) | 194/8 (regs+smem) |
| 2x4x8 smax48 nwc4 nws1 vxpad0 | 128 | 69664 | 167/8 (smem) | 168/4 (smem) | 168/12 (regs+smem) | 168/12 (regs+smem) | 180/4 (smem) |
| 2x4x8 smax48 nwc4 nws2 vxpad0 | 256 | 76064 | 170/8 (regs) | 171/8 (regs+smem) | 172/8 (regs) | 184/8 (regs) | 194/8 (regs+smem) |
| 4x4x16 smax48 nwc4 nws1 vxpad0 | 128 | 70720 | 254/8 (regs+smem) | 254/4 (smem) | 254/8 (regs) | 252/8 (regs) | 236/4 (smem) |
| 4x4x16 smax48 nwc4 nws2 vxpad0 | 256 | 77120 | 232/8 (regs) | 236/8 (regs+smem) | 242/8 (regs) | 254/8 (regs) | 238/8 (regs+smem) |
| 4x4x16 smax64 nwc4 nws1 vxpad0 | 128 | 91456 | 254/4 (smem) | 254/4 (smem) | 254/8 (regs+smem) | 252/8 (regs+smem) | 236/4 (smem) |
| 4x4x16 smax64 nwc4 nws2 vxpad0 | 256 | 97856 | 232/8 (regs+smem) | 236/8 (regs+smem) | 242/8 (regs) | 254/8 (regs) | 238/8 (regs+smem) |

### static_o2_cb5 / whatif

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 4x4x16 smax128 nwc4 nws1 vxpad0 | 128 | 174400 | - | - | 254/4 (smem) | 252/4 (smem) | - |
| 4x4x16 smax96 nwc4 nws1 vxpad0 | 128 | 132928 | 254/4 (smem) | - | 254/4 (smem) | 252/4 (smem) | - |
| nxg1 cb1 chk16 listmax2048 | 128 | 32128 | 96/20 (regs+smem) K32 | 72/12 (smem) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 | 80/12 (smem) K32 |
| nxg1 cb1 chk16 listmax4096 | 128 | 40320 | 96/16 (smem) K32 | 72/8 (smem) K32 | 80/20 (smem) K32 | 80/20 (smem) K32 | 80/8 (smem) K32 |
| nxg1 cb1 chk48 listmax1024 | 128 | 75904 | 124/8 (smem) K32 | 96/4 (smem) K32 | 96/12 (smem) K32 | 95/12 (smem) K32 | 96/4 (smem) K32 |
| nxg1 cb1 chk64 listmax1024 | 128 | 99840 | 123/4 (smem) K32 | 128/4 (smem) K32 | 123/8 (smem) K32 | 96/8 (smem) K32 | 95/4 (smem) K32 |
| nxg1 cb2 chk16 listmax2048 | 128 | 32128 | 128/16 (regs) K32 | 123/12 (smem) K32 | 124/16 (regs) K32 | 122/16 (regs) K32 | 122/12 (smem) K32 |
| nxg1 cb2 chk16 listmax4096 | 128 | 40320 | 128/16 (regs+smem) K32 | 123/8 (smem) K32 | 124/16 (regs) K32 | 122/16 (regs) K32 | 122/8 (smem) K32 |
| nxg1 cb2 chk48 listmax1024 | 128 | 75904 | 159/8 (smem) K32 | 128/4 (smem) K32 | 128/12 (smem) K32 | 128/12 (smem) K32 | 127/4 (smem) K32 |
| nxg1 cb2 chk64 listmax1024 | 128 | 99840 | 159/4 (smem) K32 | 160/4 (smem) K32 | 128/8 (smem) K32 | 127/8 (smem) K32 | 125/4 (smem) K32 |
| nxg2 cb1 chk16 listmax2048 | 128 | 27776 | 72/20 (smem) K32 | 56/12 (smem) K32 | 56/32 (smem) K32 | 56/32 (smem) K32 | 56/12 (smem) K32 |
| nxg2 cb1 chk16 listmax4096 | 128 | 35968 | 72/16 (smem) K32 | 56/8 (smem) K32 | 56/24 (smem) K32 | 56/24 (smem) K32 | 56/8 (smem) K32 |
| nxg2 cb1 chk48 listmax1024 | 128 | 62848 | 96/8 (smem) K32 | 64/4 (smem) K32 | 72/12 (smem) K32 | 80/12 (smem) K32 | 72/4 (smem) K32 |
| nxg2 cb1 chk64 listmax1024 | 128 | 82432 | 96/8 (smem) K32 | 72/4 (smem) K32 | 72/8 (smem) K32 | 80/8 (smem) K32 | 72/4 (smem) K32 |
| nxg2 cb2 chk16 listmax2048 | 128 | 28032 | 96/20 (regs+smem) K32 | 72/12 (smem) K32 | 72/28 (regs) K32 | 80/24 (regs) K32 | 80/12 (smem) K32 |
| nxg2 cb2 chk16 listmax4096 | 128 | 36224 | 96/16 (smem) K32 | 72/8 (smem) K32 | 72/24 (smem) K32 | 80/24 (regs+smem) K32 | 80/8 (smem) K32 |
| nxg2 cb2 chk48 listmax1024 | 128 | 63616 | 124/8 (smem) K32 | 96/4 (smem) K32 | 96/12 (smem) K32 | 96/12 (smem) K32 | 80/4 (smem) K32 |
| nxg2 cb2 chk64 listmax1024 | 128 | 83456 | 124/4 (smem) K32 | 126/4 (smem) K32 | 124/8 (smem) K32 | 80/8 (smem) K32 | 80/4 (smem) K32 |
| nxg2 cb4 chk16 listmax2048 | 128 | 32128 | 128/16 (regs) K32 | 123/12 (smem) K32 | 125/16 (regs) K32 | 124/16 (regs) K32 | 124/12 (smem) K32 |
| nxg2 cb4 chk16 listmax4096 | 128 | 40320 | 128/16 (regs+smem) K32 | 123/8 (smem) K32 | 125/16 (regs) K32 | 124/16 (regs) K32 | 124/8 (smem) K32 |
| nxg2 cb4 chk48 listmax1024 | 128 | 75904 | 157/8 (smem) K32 | 128/4 (smem) K32 | 128/12 (smem) K32 | 128/12 (smem) K32 | 124/4 (smem) K32 |
| nxg2 cb4 chk64 listmax1024 | 128 | 99840 | 160/4 (smem) K32 | 158/4 (smem) K32 | 160/8 (smem) K32 | 128/8 (smem) K32 | 127/4 (smem) K32 |
| nxg4 cb2 chk16 listmax2048 | 128 | 25728 | 64/24 (smem) K32 | 56/12 (smem) K32 | 64/32 (regs+smem) K32 | 56/32 (smem) K32 | 56/12 (smem) K32 |
| nxg4 cb2 chk16 listmax4096 | 128 | 33920 | 64/16 (smem) K32 | 56/8 (smem) K32 | 64/24 (smem) K32 | 56/24 (smem) K32 | 56/8 (smem) K32 |
| nxg4 cb2 chk48 listmax1024 | 128 | 56704 | 96/8 (smem) K32 | 64/4 (smem) K32 | 64/16 (smem) K32 | 80/16 (smem) K32 | 72/4 (smem) K32 |
| nxg4 cb2 chk64 listmax1024 | 128 | 74240 | 96/8 (smem) K32 | 64/4 (smem) K32 | 64/12 (smem) K32 | 80/12 (smem) K32 | 72/4 (smem) K32 |
| nxg4 cb3 chk16 listmax2048 | 128 | 27904 | 96/20 (regs+smem) K32 | 64/12 (smem) K32 | 72/28 (regs) K32 | 80/24 (regs) K32 | 64/12 (smem) S8 K40 |
| nxg4 cb3 chk16 listmax4096 | 128 | 36096 | 96/16 (smem) K32 | 64/8 (smem) K32 | 72/24 (smem) K32 | 80/24 (regs+smem) K32 | 64/8 (smem) S8 K40 |
| nxg4 cb3 chk48 listmax1024 | 128 | 63232 | 123/8 (smem) K32 | 96/4 (smem) K32 | 96/12 (smem) K32 | 72/12 (smem) S4 K40 | 72/4 (smem) K32 |
| nxg4 cb3 chk64 listmax1024 | 128 | 82944 | 123/8 (smem) K32 | 96/4 (smem) K32 | 95/8 (smem) K32 | 72/8 (smem) S4 K40 | 72/4 (smem) K32 |
| nxg4 cb4 chk16 listmax2048 | 128 | 28032 | 96/20 (regs+smem) K32 | 72/12 (smem) K32 | 80/24 (regs) K32 | 80/24 (regs) K32 | 80/12 (smem) K32 |
| nxg4 cb4 chk16 listmax4096 | 128 | 36224 | 96/16 (smem) K32 | 72/8 (smem) K32 | 80/24 (regs+smem) K32 | 80/24 (regs+smem) K32 | 80/8 (smem) K32 |
| nxg4 cb4 chk48 listmax1024 | 128 | 63616 | 124/8 (smem) K32 | 96/4 (smem) K32 | 96/12 (smem) K32 | 96/12 (smem) K32 | 80/4 (smem) K32 |
| nxg4 cb4 chk64 listmax1024 | 128 | 83456 | 124/4 (smem) K32 | 126/4 (smem) K32 | 124/8 (smem) K32 | 80/8 (smem) K32 | 80/4 (smem) K32 |
| nxg4 cb7 chk16 listmax2048 | 128 | 32128 | 128/16 (regs) K32 | 121/12 (smem) K32 | 123/16 (regs) K32 | 121/16 (regs) K32 | 121/12 (smem) K32 |
| nxg4 cb7 chk16 listmax4096 | 128 | 40320 | 128/16 (regs+smem) K32 | 121/8 (smem) K32 | 123/16 (regs) K32 | 121/16 (regs) K32 | 121/8 (smem) K32 |
| nxg4 cb7 chk48 listmax1024 | 128 | 75904 | 157/8 (smem) K32 | 128/4 (smem) K32 | 128/12 (smem) K32 | 127/12 (smem) K32 | 127/4 (smem) K32 |
| nxg4 cb7 chk64 listmax1024 | 128 | 99840 | 157/4 (smem) K32 | 128/4 (smem) K32 | 128/8 (smem) K32 | 125/8 (smem) K32 | 127/4 (smem) K32 |

### vel2d_o3_cb10 / kl_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 | 128 | 26256 | 126/16 (regs) K48 | 121/12 (smem) K48 | 119/16 (regs) K48 | 95/20 (regs) K48 | 93/12 (smem) K48 |
| nxg1 cb1 chk32 | 128 | 52112 | 126/12 (smem) K48 | 118/4 (smem) K48 | 121/16 (regs+smem) K48 | 96/16 (smem) K48 | 96/4 (smem) K48 |
| nxg1 cb2 chk16 | 128 | 26640 | 158/12 (regs) K48 | 158/12 (regs+smem) K48 | 159/12 (regs) K48 | 158/12 (regs) K48 | 156/12 (regs+smem) K48 |
| nxg1 cb2 chk32 | 128 | 52880 | 160/12 (regs+smem) K48 | 159/4 (smem) K48 | 161/12 (regs) K48 | 160/12 (regs) K48 | 156/4 (smem) K48 |
| nxg1 cb4 chk16 | 128 | 35088 | 239/8 (regs) K48 | 238/8 (regs+smem) K48 | 236/8 (regs) K48 | 239/8 (regs) K48 | 235/8 (regs+smem) K48 |
| nxg1 cb4 chk32 | 128 | 69776 | 238/8 (regs+smem) K48 | 239/4 (smem) K48 | 237/8 (regs) K48 | 241/8 (regs) K48 | 242/4 (smem) K48 |
| nxg2 cb2 chk16 | 128 | 22160 | 124/16 (regs) K48 | 121/16 (regs+smem) K48 | 119/16 (regs) K48 | 123/16 (regs) K48 | 125/16 (regs+smem) K48 |
| nxg2 cb2 chk32 | 128 | 43920 | 126/12 (smem) K48 | 119/8 (smem) K48 | 119/16 (regs) K48 | 96/20 (regs+smem) K48 | 96/8 (smem) K48 |
| nxg2 cb3 chk16 | 128 | 26384 | 154/12 (regs) K48 | 125/12 (smem) K48 | 126/16 (regs) K48 | 125/16 (regs) K48 | 120/12 (smem) K48 |
| nxg2 cb3 chk32 | 128 | 52368 | 155/12 (regs+smem) K48 | 125/4 (smem) K48 | 125/16 (regs+smem) K48 | 123/16 (regs+smem) K48 | 123/4 (smem) K48 |
| nxg2 cb4 chk16 | 128 | 26640 | 157/12 (regs) K48 | 158/12 (regs+smem) K48 | 158/12 (regs) K48 | 156/12 (regs) K48 | 154/12 (regs+smem) K48 |
| nxg2 cb4 chk32 | 128 | 52880 | 159/12 (regs+smem) K48 | 159/4 (smem) K48 | 159/12 (regs) K48 | 156/12 (regs) K48 | 158/4 (smem) K48 |
| nxg2 cb7 chk16 | 128 | 35088 | 227/8 (regs) K48 | 226/8 (regs+smem) K48 | 227/8 (regs) K48 | 204/8 (regs) K48 | 204/8 (regs+smem) K48 |
| nxg2 cb7 chk32 | 128 | 69776 | 232/8 (regs+smem) K48 | 228/4 (smem) K48 | 232/8 (regs) K48 | 236/8 (regs) K48 | 234/4 (smem) K48 |
| nxg4 cb13 chk16 | 128 | 33040 | 219/8 (regs) K48 | 218/8 (regs) K48 | 222/8 (regs) K48 | 187/8 (regs) K48 | 196/8 (regs) K48 |
| nxg4 cb13 chk32 | 128 | 65680 | 224/8 (regs+smem) K48 | 221/4 (smem) K48 | 225/8 (regs) K48 | 201/8 (regs) K48 | 200/4 (smem) K48 |
| nxg4 cb4 chk16 | 128 | 22160 | 124/16 (regs) K48 | 121/16 (regs+smem) K48 | 119/16 (regs) K48 | 121/16 (regs) K48 | 122/16 (regs+smem) K48 |
| nxg4 cb4 chk32 | 128 | 43920 | 126/12 (smem) K48 | 119/8 (smem) K48 | 119/16 (regs) K48 | 96/20 (regs+smem) K48 | 96/8 (smem) K48 |
| nxg4 cb5 chk16 | 128 | 24208 | 128/16 (regs) K48 | 124/16 (regs+smem) K48 | 124/16 (regs) K48 | 122/16 (regs) K48 | 124/16 (regs+smem) K48 |
| nxg4 cb5 chk32 | 128 | 48016 | 153/12 (regs+smem) K48 | 124/8 (smem) K48 | 124/16 (regs+smem) K48 | 122/16 (regs+smem) K48 | 123/8 (smem) K48 |
| nxg4 cb7 chk16 | 128 | 26512 | 159/12 (regs) K48 | 158/12 (regs+smem) K48 | 157/12 (regs) K48 | 123/16 (regs) K48 | 125/12 (smem) K48 |
| nxg4 cb7 chk32 | 128 | 52624 | 160/12 (regs+smem) K48 | 158/4 (smem) K48 | 158/12 (regs) K48 | 126/16 (regs+smem) K48 | 125/4 (smem) K48 |

### vel2d_o3_cb10 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 29632 | 122/16 (regs) K32 | 93/12 (smem) K32 | 95/20 (regs) K32 | 79/24 (regs) K32 | 78/12 (smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 55168 | 124/8 (smem) K32 | 119/4 (smem) K32 | 119/16 (regs+smem) K32 | 80/16 (smem) K32 | 80/4 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 30016 | 156/12 (regs) K32 | 126/12 (smem) K32 | 125/16 (regs) K32 | 125/16 (regs) K32 | 125/12 (smem) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 55936 | 157/8 (smem) K32 | 161/4 (smem) K32 | 159/12 (regs) K32 | 123/16 (regs+smem) K32 | 123/4 (smem) K32 |
| nxg1 cb4 chk16 listmax1024 | 128 | 38464 | 233/8 (regs) K32 | 231/8 (regs+smem) K32 | 234/8 (regs) K32 | 240/8 (regs) K32 | 238/8 (regs+smem) K32 |
| nxg1 cb4 chk32 listmax1024 | 128 | 72832 | 236/8 (regs+smem) K32 | 235/4 (smem) K32 | 240/8 (regs) K32 | 242/8 (regs) K32 | 236/4 (smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 25536 | 124/16 (regs) K32 | 93/12 (smem) K32 | 94/20 (regs) K32 | 72/28 (regs) K32 | 72/12 (smem) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 46976 | 124/12 (smem) K32 | 124/8 (smem) K32 | 124/16 (regs+smem) K32 | 80/16 (smem) K32 | 80/8 (smem) K32 |
| nxg2 cb3 chk16 listmax1024 | 128 | 29760 | 156/12 (regs) K32 | 126/12 (smem) K32 | 123/16 (regs) K32 | 123/16 (regs) K32 | 124/12 (smem) K32 |
| nxg2 cb3 chk32 listmax1024 | 128 | 55424 | 154/8 (smem) K32 | 127/4 (smem) K32 | 124/16 (regs+smem) K32 | 123/16 (regs+smem) K32 | 124/4 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 30016 | 160/12 (regs) K32 | 126/12 (smem) K32 | 125/16 (regs) K32 | 124/16 (regs) K32 | 124/12 (smem) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 55936 | 157/8 (smem) K32 | 158/4 (smem) K32 | 159/12 (regs) K32 | 123/16 (regs+smem) K32 | 123/4 (smem) K32 |
| nxg2 cb7 chk16 listmax1024 | 128 | 38464 | 221/8 (regs) K32 | 219/8 (regs+smem) K32 | 221/8 (regs) K32 | 197/8 (regs) K32 | 196/8 (regs+smem) K32 |
| nxg2 cb7 chk32 listmax1024 | 128 | 72832 | 224/8 (regs+smem) K32 | 223/4 (smem) K32 | 225/8 (regs) K32 | 201/8 (regs) K32 | 200/4 (smem) K32 |
| nxg4 cb13 chk16 listmax1024 | 128 | 36416 | 212/8 (regs) K32 | 209/8 (regs+smem) K32 | 214/8 (regs) K32 | 161/12 (regs) K32 | 161/8 (smem) K32 |
| nxg4 cb13 chk32 listmax1024 | 128 | 68736 | 216/8 (regs+smem) K32 | 213/4 (smem) K32 | 219/8 (regs) K32 | 160/12 (regs+smem) K32 | 161/4 (smem) K32 |
| nxg4 cb4 chk16 listmax1024 | 128 | 25536 | 124/16 (regs) K32 | 93/12 (smem) K32 | 96/20 (regs) K32 | 78/24 (regs) K32 | 72/12 (smem) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 46976 | 124/12 (smem) K32 | 124/8 (smem) K32 | 125/16 (regs+smem) K32 | 80/16 (smem) K32 | 80/8 (smem) K32 |
| nxg4 cb5 chk16 listmax1024 | 128 | 27584 | 127/16 (regs) K32 | 119/12 (smem) K32 | 122/16 (regs) K32 | 95/20 (regs) K32 | 80/12 (smem) K32 |
| nxg4 cb5 chk32 listmax1024 | 128 | 51072 | 128/12 (smem) K32 | 126/4 (smem) K32 | 123/16 (regs+smem) K32 | 125/16 (regs+smem) K32 | 127/4 (smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 29888 | 156/12 (regs) K32 | 128/12 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 122/12 (smem) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 55680 | 158/8 (smem) K32 | 126/4 (smem) K32 | 125/16 (regs+smem) K32 | 123/16 (regs+smem) K32 | 123/4 (smem) K32 |

### vel2d_o3_cb10 / tile_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 640 | 51712 | 92/20 (regs) K32 | 93/20 (regs+smem) K32 | 94/20 (regs) K32 | 83/20 (regs) K32 | 87/20 (regs+smem) K32 |
| tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 640 | 66176 | 92/20 (regs) K32 | 94/20 (regs+smem) K32 | 94/20 (regs) K32 | 87/20 (regs) K32 | 83/20 (regs+smem) K32 |
| tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 320 | 51712 | 121/10 (regs) K32 | 93/10 (smem) K32 | 123/10 (regs) K32 | 94/20 (regs) K32 | 92/10 (smem) K32 |
| tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 320 | 66176 | 122/10 (regs) K32 | 94/10 (smem) K32 | 120/10 (regs) K32 | 95/20 (regs) K32 | 92/10 (smem) K32 |
| tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 38528 | 90/16 (smem) K32 | 72/8 (smem) K32 | 80/20 (smem) K32 | 72/20 (smem) S20 K56 | 64/8 (smem) S24 K56 |
| tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 128 | 43648 | 93/12 (smem) K32 | 72/8 (smem) K32 | 80/20 (smem) K32 | 71/20 (smem) K32 | 72/8 (smem) K32 |
| tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 64 | 38528 | 122/8 (smem) K32 | 122/4 (smem) K32 | 125/10 (smem) K32 | 122/10 (smem) K32 | 122/4 (smem) K32 |
| tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 64 | 43648 | 123/6 (smem) K32 | 123/4 (smem) K32 | 125/10 (smem) K32 | 122/10 (smem) K32 | 122/4 (smem) K32 |
| tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 192 | 38784 | 94/18 (regs) K32 | 72/12 (smem) K32 | 96/18 (regs) K32 | 64/30 (regs+smem) S8 K40 | 56/12 (smem) S40 K56 |
| tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 192 | 46464 | 90/18 (regs+smem) K32 | 71/12 (smem) K32 | 78/24 (regs+smem) K32 | 72/24 (regs+smem) K32 | 75/12 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 39040 | 94/16 (regs) K32 | 77/16 (smem) K32 | 95/16 (regs) K32 | 94/16 (regs) K32 | 79/16 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 256 | 49280 | 94/16 (regs) K32 | 71/16 (smem) K32 | 80/24 (regs) K32 | 64/32 (regs+smem) S8 K40 | 80/16 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 128 | 39040 | 122/16 (regs+smem) K32 | 93/8 (smem) K32 | 123/16 (regs) K32 | 120/16 (regs) K32 | 93/8 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 128 | 49280 | 122/12 (smem) K32 | 122/8 (smem) K32 | 123/16 (regs+smem) K32 | 122/16 (regs+smem) K32 | 94/8 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 64 | 39040 | 165/8 (smem) K32 | 158/4 (smem) K32 | 159/10 (smem) K32 | 160/10 (smem) K32 | 157/4 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 64 | 49280 | 163/6 (smem) K32 | 161/4 (smem) K32 | 157/8 (smem) K32 | 160/8 (smem) K32 | 156/4 (smem) K32 |
| tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 384 | 39552 | 80/24 (regs) S16 K48 | 77/24 (regs+smem) K32 | 80/24 (regs) S24 K56 | 56/36 (regs) S40 K56 | 56/24 (smem) S36 K56 |
| tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 384 | 54912 | 80/24 (regs+smem) S16 K48 | 71/12 (smem) K32 | 79/24 (regs) K32 | 67/24 (regs) K32 | 67/12 (smem) K32 |
| tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 192 | 39552 | 121/12 (regs) K32 | 93/12 (smem) K32 | 123/12 (regs) K32 | 93/18 (regs) K32 | 96/12 (smem) K32 |
| tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 192 | 54912 | 122/12 (regs+smem) K32 | 124/6 (smem) K32 | 122/12 (regs) K32 | 95/18 (regs) K32 | 96/6 (smem) K32 |
| tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 512 | 41472 | 94/16 (regs) K32 | 96/16 (regs) K32 | 95/16 (regs) K32 | 64/32 (regs) S144 K144 | 64/32 (regs+smem) S136 K128 |
| tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 512 | 60544 | 94/16 (regs) K32 | 98/16 (regs+smem) K32 | 95/16 (regs) K32 | 96/16 (regs) K32 | 96/16 (regs+smem) K32 |
| tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 256 | 41472 | 122/16 (regs) K32 | 120/16 (regs+smem) K32 | 123/16 (regs) K32 | 120/16 (regs) K32 | 120/16 (regs+smem) K32 |
| tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 256 | 60544 | 122/16 (regs+smem) K32 | 124/8 (smem) K32 | 123/16 (regs) K32 | 122/16 (regs) K32 | 122/8 (smem) K32 |
| tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 128 | 41472 | 161/12 (regs+smem) K32 | 161/8 (smem) K32 | 161/12 (regs) K32 | 158/12 (regs) K32 | 158/8 (smem) K32 |
| tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 128 | 60544 | 158/8 (smem) K32 | 158/4 (smem) K32 | 162/12 (regs+smem) K32 | 158/12 (regs+smem) K32 | 158/4 (smem) K32 |

### vel2d_o3_cb10 / tile_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 32 | 42880 | 94/3 (smem) K32 | 79/2 (smem) K32 | 64/5 (smem) K32 | 64/5 (smem) K32 | 79/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 32 | 42880 | 94/3 (smem) K32 | 79/2 (smem) K32 | 64/5 (smem) K32 | 64/5 (smem) K32 | 79/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 32 | 76544 | 127/2 (smem) K32 | 96/1 (smem) K32 | 96/3 (smem) K32 | 95/3 (smem) K32 | 96/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 32 | 76544 | 127/2 (smem) K32 | 96/1 (smem) K32 | 96/3 (smem) K32 | 95/3 (smem) K32 | 96/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | 43136 | 64/6 (smem) K32 | 56/4 (smem) K32 | 64/10 (smem) K32 | 64/10 (smem) K32 | 56/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | 43136 | 72/6 (smem) K32 | 56/4 (smem) K32 | 64/10 (smem) K32 | 64/10 (smem) K32 | 64/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 64 | 77056 | 115/4 (smem) K32 | 80/2 (smem) K32 | 95/4 (smem) K32 | 80/4 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 64 | 77056 | 115/4 (smem) K32 | 80/2 (smem) K32 | 95/4 (smem) K32 | 80/4 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 32 | 77056 | 167/2 (smem) K32 | 143/1 (smem) K32 | 155/2 (smem) K32 | 168/2 (smem) K32 | 161/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 32 | 77056 | 167/2 (smem) K32 | 143/1 (smem) K32 | 155/2 (smem) K32 | 149/2 (smem) K32 | 147/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 43648 | 64/12 (smem) K32 | 55/8 (smem) K32 | 56/20 (smem) K32 | 56/20 (smem) K32 | 56/8 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 128 | 43648 | 70/12 (smem) K32 | 56/8 (smem) K32 | 56/20 (smem) K32 | 58/20 (smem) K32 | 56/8 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 78080 | 96/8 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 80/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 128 | 78080 | 96/8 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 80/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 78080 | 157/4 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 64 | 78080 | 157/4 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 320 | 89984 | 64/10 (smem) K32 | 64/10 (smem) K32 | 64/20 (smem) K32 | 64/20 (smem) K32 | 64/10 (smem) K32 |
| tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 320 | 89984 | 64/10 (smem) K32 | 64/10 (smem) K32 | 64/20 (smem) K32 | 64/20 (smem) K32 | 64/10 (smem) K32 |
| tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb2 fcb1 kv0 | 160 | 89984 | - | 71/5 (smem) K32 | - | - | 72/5 (smem) S36 K56 |
| tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb2 fcb1 kv1 | 160 | 89984 | - | 71/5 (smem) K32 | - | - | 72/5 (smem) S36 K56 |
| tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 320 | 123648 | 95/10 (smem) K32 | - | 96/10 (smem) K32 | 95/10 (smem) K32 | - |
| tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 320 | 123648 | 95/10 (smem) K32 | - | 96/10 (smem) K32 | 95/10 (smem) K32 | - |
| tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 160 | 123648 | 151/5 (smem) K32 | - | 128/5 (smem) K32 | 128/5 (smem) K32 | - |
| tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 160 | 123648 | 151/5 (smem) K32 | - | 128/5 (smem) K32 | 127/5 (smem) K32 | - |
| tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv0 | 480 | 81760 | 64/30 (regs+smem) K32 | 63/15 (smem) K32 | 64/30 (regs+smem) K32 | 61/30 (regs+smem) K32 | 61/15 (smem) K32 |
| tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv1 | 480 | 81760 | 64/30 (regs+smem) S32 K48 | 64/15 (smem) K32 | 64/30 (regs+smem) S12 K48 | 64/30 (regs+smem) S4 K40 | 64/15 (smem) K32 |
| tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb2 fcb1 kv0 | 320 | 90240 | - | 95/10 (smem) K32 | - | - | 95/10 (smem) K32 |
| tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb2 fcb1 kv1 | 320 | 90240 | - | 95/10 (smem) K32 | - | - | 95/10 (smem) K32 |
| tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 160 | 90240 | - | 125/5 (smem) K32 | - | - | 127/5 (smem) K32 |
| tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb4 fcb1 kv1 | 160 | 90240 | - | 125/5 (smem) K32 | - | - | 127/5 (smem) K32 |
| tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv0 | 480 | 107200 | 95/15 (regs+smem) K32 | - | 95/15 (regs) K32 | 82/15 (regs) K32 | - |
| tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv1 | 480 | 107200 | 95/15 (regs+smem) K32 | - | 95/15 (regs) K32 | 82/15 (regs) K32 | - |
| tt4 cpba1 cpbf10 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 320 | 124160 | 143/10 (regs+smem) K32 | - | 147/10 (regs+smem) K32 | 131/10 (regs+smem) K32 | - |
| tt4 cpba1 cpbf10 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 320 | 124160 | 143/10 (regs+smem) K32 | - | 147/10 (regs+smem) K32 | 131/10 (regs+smem) K32 | - |
| tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv0 | 480 | 69232 | 64/30 (regs+smem) K32 | 63/15 (smem) K32 | 64/30 (regs) K32 | 61/30 (regs) K32 | 61/15 (smem) K32 |
| tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv1 | 480 | 69232 | 64/30 (regs+smem) S32 K48 | 64/15 (smem) K32 | 64/30 (regs) S12 K48 | 64/30 (regs) S4 K40 | 64/15 (smem) K32 |
| tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv0 | 480 | 82144 | 95/15 (regs) K32 | 76/15 (regs+smem) K32 | 95/15 (regs) K32 | 82/15 (regs) K32 | 82/15 (regs+smem) K32 |
| tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv1 | 480 | 82144 | 95/15 (regs) K32 | 76/15 (regs+smem) K32 | 95/15 (regs) K32 | 82/15 (regs) K32 | 82/15 (regs+smem) K32 |
| tt4 cpba1 cpbf10 ppf4 chunk32 acb1 fsb4 fcb1 kv0 | 320 | 90752 | - | 143/10 (regs+smem) K32 | - | - | 131/10 (regs+smem) K32 |
| tt4 cpba1 cpbf10 ppf4 chunk32 acb1 fsb4 fcb1 kv1 | 320 | 90752 | - | 143/10 (regs+smem) K32 | - | - | 131/10 (regs+smem) K32 |
| tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 480 | 107968 | 128/15 (regs+smem) K32 | - | 128/15 (regs) K32 | 128/15 (regs) K32 | - |
| tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv1 | 480 | 107968 | 128/15 (regs+smem) K32 | - | 128/15 (regs) K32 | 128/15 (regs) K32 | - |
| tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | 49024 | 80/6 (smem) K32 | 56/4 (smem) K32 | 64/8 (smem) K32 | 64/8 (smem) K32 | 66/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | 49024 | 80/6 (smem) K32 | 56/4 (smem) K32 | 64/8 (smem) K32 | 64/8 (smem) K32 | 66/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 64 | 82688 | 115/4 (smem) K32 | 80/2 (smem) K32 | 95/4 (smem) K32 | 80/4 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 64 | 82688 | 115/4 (smem) K32 | 80/2 (smem) K32 | 95/4 (smem) K32 | 80/4 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 32 | 82688 | 167/2 (smem) K32 | 143/1 (smem) K32 | 153/2 (smem) K32 | 167/2 (smem) K32 | 159/1 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 32 | 82688 | 167/2 (smem) K32 | 143/1 (smem) K32 | 153/2 (smem) K32 | 149/2 (smem) K32 | 143/1 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 49280 | 64/12 (smem) K32 | 56/8 (smem) K32 | 56/16 (smem) K32 | 56/16 (smem) K32 | 56/8 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 128 | 49280 | 70/12 (smem) K32 | 56/8 (smem) K32 | 56/16 (smem) K32 | 56/16 (smem) K32 | 64/8 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 83200 | 96/4 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 79/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 128 | 83200 | 96/4 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 79/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 83200 | 157/2 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 64 | 83200 | 157/2 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 49792 | 64/24 (smem) K32 | 64/16 (smem) K32 | 64/32 (regs+smem) K32 | 64/32 (regs+smem) K32 | 60/16 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 256 | 49792 | 72/24 (regs+smem) K32 | 64/16 (smem) K32 | 64/32 (regs+smem) K32 | 64/32 (regs+smem) K32 | 64/16 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 84224 | 96/8 (smem) K32 | 71/8 (smem) K32 | 80/16 (smem) K32 | 78/16 (smem) K32 | 64/8 (smem) S32 K48 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 256 | 84224 | 96/8 (smem) K32 | 71/8 (smem) K32 | 80/16 (smem) K32 | 78/16 (smem) K32 | 64/8 (smem) S32 K48 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 84224 | 151/4 (smem) K32 | 128/4 (smem) K32 | 128/8 (smem) K32 | 128/8 (smem) K32 | 127/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | 84224 | 151/4 (smem) K32 | 128/4 (smem) K32 | 128/8 (smem) K32 | 128/8 (smem) K32 | 127/4 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 96 | 53120 | 64/9 (smem) S24 K56 | 56/3 (smem) K32 | 56/12 (smem) K32 | 56/12 (smem) K32 | 56/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 96 | 53120 | 72/9 (smem) K32 | 56/3 (smem) K32 | 56/12 (smem) K32 | 56/12 (smem) K32 | 56/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | 86784 | 96/3 (smem) K32 | 72/3 (smem) K32 | 80/6 (smem) K32 | 80/6 (smem) K32 | 72/3 (smem) S32 K48 |
| tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | 86784 | 96/3 (smem) K32 | 72/3 (smem) K32 | 80/6 (smem) K32 | 80/6 (smem) K32 | 72/3 (smem) S32 K48 |
| tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 192 | 53376 | 64/18 (smem) K32 | 56/6 (smem) K32 | 56/24 (smem) K32 | 56/24 (smem) K32 | 56/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 192 | 53376 | 64/18 (smem) S36 K56 | 56/6 (smem) K32 | 56/24 (smem) K32 | 58/24 (smem) K32 | 56/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 192 | 87296 | 96/6 (smem) K32 | 71/6 (smem) K32 | 79/12 (smem) K32 | 78/12 (smem) K32 | 76/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 192 | 87296 | 96/6 (smem) K32 | 71/6 (smem) K32 | 79/12 (smem) K32 | 78/12 (smem) K32 | 76/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 96 | 87296 | 153/3 (smem) K32 | 128/3 (smem) K32 | 128/6 (smem) K32 | 128/6 (smem) K32 | 127/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 96 | 87296 | 153/3 (smem) K32 | 128/3 (smem) K32 | 128/6 (smem) K32 | 127/6 (smem) K32 | 127/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 384 | 53888 | 72/24 (regs) K32 | 56/12 (smem) K32 | 56/36 (regs) K32 | 56/36 (regs) K32 | 56/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 384 | 53888 | 73/24 (regs) K32 | 56/12 (smem) K32 | 56/36 (regs) K32 | 58/24 (regs) K32 | 56/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 384 | 88320 | 100/12 (regs+smem) K32 | 80/12 (smem) K32 | 80/24 (regs+smem) K32 | 80/24 (regs+smem) K32 | 79/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 384 | 88320 | 100/12 (regs+smem) K32 | 80/12 (smem) K32 | 80/24 (regs+smem) K32 | 80/24 (regs+smem) K32 | 79/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 192 | 88320 | 145/6 (smem) K32 | 145/6 (smem) K32 | 147/12 (regs+smem) K32 | 131/12 (regs+smem) K32 | 131/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 192 | 88320 | 145/6 (smem) K32 | 145/6 (smem) K32 | 147/12 (regs+smem) K32 | 131/12 (regs+smem) K32 | 131/6 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 59264 | 64/8 (smem) K32 | 56/4 (smem) K32 | 56/12 (smem) K32 | 56/12 (smem) K32 | 56/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 128 | 59264 | 72/8 (smem) K32 | 56/4 (smem) K32 | 56/12 (smem) K32 | 56/12 (smem) K32 | 56/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 92928 | 96/4 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 79/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 128 | 92928 | 96/4 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 79/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 92928 | 157/2 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 64 | 92928 | 157/2 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 59520 | 64/16 (smem) K32 | 64/8 (smem) K32 | 64/24 (smem) K32 | 64/24 (smem) K32 | 60/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 256 | 59520 | 72/16 (smem) K32 | 64/8 (smem) K32 | 64/24 (smem) K32 | 64/24 (smem) K32 | 64/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 93440 | 80/8 (smem) S8 K40 | 71/8 (smem) K32 | 80/16 (smem) K32 | 78/16 (smem) K32 | 64/8 (smem) S32 K48 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 256 | 93440 | 80/8 (smem) S8 K40 | 71/8 (smem) K32 | 80/16 (smem) K32 | 78/16 (smem) K32 | 64/8 (smem) S32 K48 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 93440 | 151/4 (smem) K32 | 127/4 (smem) K32 | 128/8 (smem) K32 | 128/8 (smem) K32 | 127/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | 93440 | 151/4 (smem) K32 | 127/4 (smem) K32 | 128/8 (smem) K32 | 128/8 (smem) K32 | 127/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 512 | 60032 | 64/32 (regs+smem) K32 | 64/16 (smem) K32 | 63/32 (regs) K32 | 64/32 (regs) K32 | 64/16 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 512 | 60032 | 64/32 (regs+smem) K32 | 64/16 (smem) K32 | 64/32 (regs) K32 | 64/32 (regs) K32 | 64/16 (smem) S36 K56 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 512 | 94464 | 96/16 (regs+smem) K32 | 95/16 (regs+smem) K32 | 95/16 (regs) K32 | 80/16 (regs) K32 | 80/16 (regs+smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 512 | 94464 | 96/16 (regs+smem) K32 | 95/16 (regs+smem) K32 | 95/16 (regs) K32 | 80/16 (regs) K32 | 80/16 (regs+smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 256 | 94464 | 128/8 (smem) K32 | 128/8 (smem) K32 | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | 127/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 256 | 94464 | 128/8 (smem) K32 | 128/8 (smem) K32 | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | 127/8 (smem) K32 |
| tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 192 | 69504 | 64/12 (smem) S24 K56 | 56/6 (smem) K32 | 56/18 (smem) K32 | 56/18 (smem) K32 | 58/6 (smem) K32 |
| tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 192 | 69504 | 73/12 (smem) K32 | 56/6 (smem) K32 | 56/18 (smem) K32 | 56/18 (smem) K32 | 58/6 (smem) K32 |
| tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb2 fcb1 kv0 | 96 | 69504 | - | 71/3 (smem) K32 | - | - | 72/3 (smem) K32 |
| tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb2 fcb1 kv1 | 96 | 69504 | - | 71/3 (smem) K32 | - | - | 72/3 (smem) K32 |
| tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 192 | 103168 | 96/6 (smem) K32 | - | 79/12 (smem) K32 | 76/12 (smem) K32 | - |
| tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 192 | 103168 | 96/6 (smem) K32 | - | 79/12 (smem) K32 | 76/12 (smem) K32 | - |
| tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 96 | 103168 | 151/3 (smem) K32 | - | 128/6 (smem) K32 | 128/6 (smem) K32 | - |
| tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 96 | 103168 | 151/3 (smem) K32 | - | 128/6 (smem) K32 | 127/6 (smem) K32 | - |
| tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 384 | 69760 | 72/24 (regs+smem) K32 | 56/12 (smem) K32 | 56/36 (regs+smem) K32 | 56/36 (regs+smem) K32 | 56/12 (smem) K32 |
| tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 384 | 69760 | 73/24 (regs+smem) K32 | 56/12 (smem) K32 | 56/36 (regs+smem) K32 | 58/24 (regs) K32 | 56/12 (smem) K32 |
| tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 96 | 69760 | - | 125/3 (smem) K32 | - | - | 127/3 (smem) K32 |
| tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb4 fcb1 kv1 | 96 | 69760 | - | 125/3 (smem) K32 | - | - | 127/3 (smem) K32 |
| tt4 cpba1 cpbf6 ppf2 chunk48 acb1 fsb2 fcb1 kv0 | 288 | 86720 | - | 72/9 (smem) K32 | - | - | 72/9 (smem) S36 K56 |
| tt4 cpba1 cpbf6 ppf2 chunk48 acb1 fsb2 fcb1 kv1 | 288 | 86720 | - | 72/9 (smem) K32 | - | - | 72/9 (smem) S36 K56 |
| tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 384 | 103680 | 100/12 (regs+smem) K32 | - | 80/24 (regs+smem) K32 | 80/24 (regs+smem) K32 | - |
| tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 384 | 103680 | 100/12 (regs+smem) K32 | - | 80/24 (regs+smem) K32 | 80/24 (regs+smem) K32 | - |
| tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 192 | 103680 | 145/6 (smem) K32 | - | 147/12 (regs+smem) K32 | 131/12 (regs+smem) K32 | - |
| tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 192 | 103680 | 145/6 (smem) K32 | - | 147/12 (regs+smem) K32 | 131/12 (regs+smem) K32 | - |
| tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 480 | 57360 | 64/30 (regs+smem) K32 | 64/15 (smem) K32 | 64/30 (regs) K32 | 61/30 (regs) K32 | 61/15 (smem) K32 |
| tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv1 | 480 | 57360 | 64/30 (regs+smem) S44 K64 | 64/15 (smem) K32 | 64/30 (regs) S40 K56 | 64/30 (regs) S4 K40 | 64/15 (smem) K32 |
| tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 480 | 78880 | 96/15 (regs) K32 | 76/15 (regs+smem) K32 | 96/15 (regs) K32 | 82/15 (regs) K32 | 82/15 (regs+smem) K32 |
| tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv1 | 480 | 78880 | 96/15 (regs) K32 | 74/15 (regs+smem) K32 | 96/15 (regs) K32 | 82/15 (regs) K32 | 82/15 (regs+smem) K32 |
| tt4 cpba1 cpbf6 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 288 | 87488 | - | 145/9 (regs+smem) K32 | - | - | 129/9 (regs+smem) K32 |
| tt4 cpba1 cpbf6 ppf4 chunk48 acb1 fsb4 fcb1 kv1 | 288 | 87488 | - | 145/9 (regs+smem) K32 | - | - | 129/9 (regs+smem) K32 |
| tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 384 | 104704 | 145/12 (regs+smem) K32 | - | 147/12 (regs) K32 | 131/12 (regs) K32 | - |
| tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 384 | 104704 | 145/12 (regs+smem) K32 | - | 147/12 (regs) K32 | 131/12 (regs) K32 | - |
| tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 79744 | 64/16 (smem) K32 | 64/8 (smem) K32 | 64/16 (smem) S4 K40 | 64/16 (smem) K32 | 64/8 (smem) S4 K40 |
| tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 256 | 79744 | 64/16 (smem) K32 | 64/8 (smem) K32 | 64/16 (smem) S36 K56 | 64/16 (smem) K32 | 64/8 (smem) S4 K40 |
| tt4 cpba1 cpbf8 ppf1 chunk48 acb1 fsb2 fcb1 kv0 | 192 | 96576 | - | 71/6 (smem) K32 | - | - | 78/6 (smem) K32 |
| tt4 cpba1 cpbf8 ppf1 chunk48 acb1 fsb2 fcb1 kv1 | 192 | 96576 | - | 71/6 (smem) K32 | - | - | 78/6 (smem) K32 |
| tt4 cpba1 cpbf8 ppf1 chunk48 acb1 fsb4 fcb1 kv0 | 96 | 96576 | - | 128/3 (smem) K32 | - | - | 127/3 (smem) K32 |
| tt4 cpba1 cpbf8 ppf1 chunk48 acb1 fsb4 fcb1 kv1 | 96 | 96576 | - | 128/3 (smem) K32 | - | - | 127/3 (smem) K32 |
| tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 113408 | 96/8 (smem) K32 | - | 80/16 (smem) K32 | 76/16 (smem) K32 | - |
| tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 256 | 113408 | 80/8 (smem) S8 K40 | - | 80/16 (smem) K32 | 76/16 (smem) K32 | - |
| tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 113408 | 151/4 (smem) K32 | - | 128/8 (smem) K32 | 128/8 (smem) K32 | - |
| tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 128 | 113408 | 151/4 (smem) K32 | - | 128/8 (smem) K32 | 127/8 (smem) K32 | - |
| tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 512 | 80000 | 64/32 (regs+smem) K32 | 64/16 (smem) K32 | 63/32 (regs+smem) K32 | 64/32 (regs+smem) K32 | 64/16 (smem) K32 |
| tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 512 | 80000 | 64/32 (regs+smem) K32 | 64/16 (smem) K32 | 64/32 (regs+smem) K32 | 64/32 (regs+smem) K32 | 64/16 (smem) S36 K56 |
| tt4 cpba1 cpbf8 ppf2 chunk48 acb1 fsb4 fcb1 kv0 | 192 | 96960 | - | 143/6 (smem) K32 | - | - | 129/6 (smem) K32 |
| tt4 cpba1 cpbf8 ppf2 chunk48 acb1 fsb4 fcb1 kv1 | 192 | 96960 | - | 143/6 (smem) K32 | - | - | 129/6 (smem) K32 |
| tt4 cpba1 cpbf8 ppf2 chunk52 acb1 fsb2 fcb1 kv0 | 416 | 101200 | - | 72/13 (smem) K32 | - | - | 72/13 (smem) S36 K56 |
| tt4 cpba1 cpbf8 ppf2 chunk52 acb1 fsb2 fcb1 kv1 | 416 | 101200 | - | 72/13 (smem) K32 | - | - | 72/13 (smem) S36 K56 |
| tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 512 | 113920 | 96/16 (regs+smem) K32 | - | 95/16 (regs) K32 | 80/16 (regs) K32 | - |
| tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 512 | 113920 | 96/16 (regs+smem) K32 | - | 95/16 (regs) K32 | 80/16 (regs) K32 | - |
| tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 256 | 113920 | 128/8 (smem) K32 | - | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | - |
| tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 256 | 113920 | 128/8 (smem) K32 | - | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | - |
| tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 512 | 63296 | 64/32 (regs+smem) K32 | 64/16 (smem) K32 | 63/32 (regs) K32 | 64/32 (regs) K32 | 64/16 (smem) K32 |
| tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv1 | 512 | 63296 | 64/32 (regs+smem) K32 | 64/16 (smem) K32 | 64/32 (regs) K32 | 64/32 (regs) K32 | 64/16 (smem) S36 K56 |
| tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 512 | 80512 | 96/16 (regs) K32 | 95/16 (regs+smem) K32 | 95/16 (regs) K32 | 80/16 (regs) K32 | 80/16 (regs+smem) K32 |
| tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv1 | 512 | 80512 | 96/16 (regs) K32 | 95/16 (regs+smem) K32 | 95/16 (regs) K32 | 80/16 (regs) K32 | 80/16 (regs+smem) K32 |
| tt4 cpba1 cpbf8 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 384 | 97728 | - | 143/12 (regs+smem) K32 | - | - | 129/12 (regs+smem) K32 |
| tt4 cpba1 cpbf8 ppf4 chunk48 acb1 fsb4 fcb1 kv1 | 384 | 97728 | - | 143/12 (regs+smem) K32 | - | - | 129/12 (regs+smem) K32 |
| tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 512 | 114944 | 128/16 (regs+smem) K32 | - | 127/16 (regs) K32 | 127/16 (regs) K32 | - |
| tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 512 | 114944 | 128/16 (regs+smem) K32 | - | 127/16 (regs) K32 | 127/16 (regs) K32 | - |

### vel2d_o3_cb10 / vt_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 2x4x16 smax32 nwc7 nws1 vxpad0 | 224 | 70240 | 198/7 (regs) | 198/7 (regs+smem) | 178/7 (regs) | 174/7 (regs) | 190/7 (regs+smem) |
| 2x4x8 smax32 nwc7 nws1 vxpad0 | 224 | 69984 | 198/7 (regs) | 198/7 (regs+smem) | 178/7 (regs) | 174/7 (regs) | 190/7 (regs+smem) |
| 2x4x8 smax48 nwc7 nws1 vxpad0 | 224 | 98656 | 198/7 (regs+smem) | 198/7 (regs+smem) | 178/7 (regs) | 174/7 (regs) | 190/7 (regs+smem) |
| 4x4x16 smax48 nwc7 nws1 vxpad0 | 224 | 99712 | 250/7 (regs+smem) | 252/7 (regs+smem) | 254/7 (regs) | 249/7 (regs) | 254/7 (regs+smem) |
| 4x4x16 smax64 nwc7 nws1 vxpad0 | 224 | 128384 | 250/7 (regs+smem) | - | 254/7 (regs+smem) | 249/7 (regs+smem) | - |

### vel2d_o3_cb10 / whatif

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 4x4x16 smax128 nwc5 nws1 vxpad0 | 160 | 206848 | - | - | 254/5 (regs+smem) | 254/5 (regs+smem) | - |
| 4x4x16 smax96 nwc5 nws1 vxpad0 | 160 | 157696 | 254/5 (regs+smem) | - | - | - | - |
| 4x4x16 smax96 nwc7 nws1 vxpad0 | 224 | 185728 | - | - | 254/7 (regs+smem) | 249/7 (regs+smem) | - |
| nxg1 cb1 chk16 listmax2048 | 128 | 33728 | 122/16 (regs+smem) K32 | 93/8 (smem) K32 | 95/20 (regs) K32 | 79/24 (regs+smem) K32 | 78/8 (smem) K32 |
| nxg1 cb1 chk16 listmax4096 | 128 | 41920 | 122/12 (smem) K32 | 93/8 (smem) K32 | 95/20 (regs+smem) K32 | 79/20 (smem) K32 | 78/8 (smem) K32 |
| nxg1 cb1 chk48 listmax1024 | 128 | 80704 | 127/8 (smem) K32 | 121/4 (smem) K32 | 120/8 (smem) K32 | 123/8 (smem) K32 | 122/4 (smem) K32 |
| nxg1 cb1 chk64 listmax1024 | 128 | 106240 | 128/4 (smem) K32 | - | 122/8 (smem) K32 | 96/8 (smem) K32 | - |
| nxg1 cb2 chk16 listmax2048 | 128 | 34112 | 156/12 (regs) K32 | 126/8 (smem) K32 | 125/16 (regs) K32 | 125/16 (regs) K32 | 125/8 (smem) K32 |
| nxg1 cb2 chk16 listmax4096 | 128 | 42304 | 156/12 (regs+smem) K32 | 126/8 (smem) K32 | 125/16 (regs) K32 | 125/16 (regs) K32 | 125/8 (smem) K32 |
| nxg1 cb2 chk48 listmax1024 | 128 | 81856 | 166/8 (smem) K32 | 162/4 (smem) K32 | 162/8 (smem) K32 | 122/8 (smem) K32 | 125/4 (smem) K32 |
| nxg1 cb2 chk64 listmax1024 | 128 | 107776 | 164/4 (smem) K32 | - | 157/8 (smem) K32 | 126/8 (smem) K32 | - |
| nxg1 cb4 chk16 listmax2048 | 128 | 42560 | 233/8 (regs) K32 | 231/8 (regs+smem) K32 | 234/8 (regs) K32 | 240/8 (regs) K32 | 238/8 (regs+smem) K32 |
| nxg1 cb4 chk16 listmax4096 | 128 | 50752 | 233/8 (regs) K32 | 231/4 (smem) K32 | 234/8 (regs) K32 | 240/8 (regs) K32 | 238/4 (smem) K32 |
| nxg1 cb4 chk48 listmax1024 | 128 | 107200 | 239/4 (smem) K32 | - | 242/8 (regs+smem) K32 | 233/8 (regs+smem) K32 | - |
| nxg1 cb4 chk64 listmax1024 | 128 | 141568 | 242/4 (smem) K32 | - | 246/4 (smem) K32 | 240/4 (smem) K32 | - |
| nxg2 cb2 chk16 listmax2048 | 128 | 29632 | 124/16 (regs) K32 | 93/12 (smem) K32 | 94/20 (regs) K32 | 72/28 (regs+smem) K32 | 72/12 (smem) K32 |
| nxg2 cb2 chk16 listmax4096 | 128 | 37824 | 124/16 (regs+smem) K32 | 93/8 (smem) K32 | 94/20 (regs) K32 | 72/24 (smem) K32 | 72/8 (smem) K32 |
| nxg2 cb2 chk48 listmax1024 | 128 | 68416 | 127/8 (smem) K32 | 119/4 (smem) K32 | 120/12 (smem) K32 | 123/12 (smem) K32 | 122/4 (smem) K32 |
| nxg2 cb2 chk64 listmax1024 | 128 | 89856 | 128/4 (smem) K32 | 126/4 (smem) K32 | 126/8 (smem) K32 | 123/8 (smem) K32 | 122/4 (smem) K32 |
| nxg2 cb3 chk16 listmax2048 | 128 | 33856 | 156/12 (regs) K32 | 126/8 (smem) K32 | 123/16 (regs) K32 | 123/16 (regs) K32 | 124/8 (smem) K32 |
| nxg2 cb3 chk16 listmax4096 | 128 | 42048 | 156/12 (regs+smem) K32 | 126/8 (smem) K32 | 123/16 (regs) K32 | 123/16 (regs) K32 | 124/8 (smem) K32 |
| nxg2 cb3 chk48 listmax1024 | 128 | 81088 | 158/8 (smem) K32 | 126/4 (smem) K32 | 125/8 (smem) K32 | 125/8 (smem) K32 | 125/4 (smem) K32 |
| nxg2 cb3 chk64 listmax1024 | 128 | 106752 | 158/4 (smem) K32 | - | 125/8 (smem) K32 | 125/8 (smem) K32 | - |
| nxg2 cb4 chk16 listmax2048 | 128 | 34112 | 160/12 (regs) K32 | 126/8 (smem) K32 | 125/16 (regs) K32 | 124/16 (regs) K32 | 124/8 (smem) K32 |
| nxg2 cb4 chk16 listmax4096 | 128 | 42304 | 160/12 (regs+smem) K32 | 126/8 (smem) K32 | 125/16 (regs) K32 | 124/16 (regs) K32 | 124/8 (smem) K32 |
| nxg2 cb4 chk48 listmax1024 | 128 | 81856 | 167/8 (smem) K32 | 155/4 (smem) K32 | 156/8 (smem) K32 | 122/8 (smem) K32 | 125/4 (smem) K32 |
| nxg2 cb4 chk64 listmax1024 | 128 | 107776 | 166/4 (smem) K32 | - | 156/8 (smem) K32 | 126/8 (smem) K32 | - |
| nxg2 cb7 chk16 listmax2048 | 128 | 42560 | 221/8 (regs) K32 | 219/8 (regs+smem) K32 | 221/8 (regs) K32 | 197/8 (regs) K32 | 196/8 (regs+smem) K32 |
| nxg2 cb7 chk16 listmax4096 | 128 | 50752 | 221/8 (regs) K32 | 219/4 (smem) K32 | 221/8 (regs) K32 | 197/8 (regs) K32 | 196/4 (smem) K32 |
| nxg2 cb7 chk48 listmax1024 | 128 | 107200 | 236/4 (smem) K32 | - | 239/8 (regs+smem) K32 | 243/8 (regs+smem) K32 | - |
| nxg2 cb7 chk64 listmax1024 | 128 | 141568 | 237/4 (smem) K32 | - | 241/4 (smem) K32 | 241/4 (smem) K32 | - |
| nxg4 cb13 chk16 listmax2048 | 128 | 40512 | 212/8 (regs) K32 | 209/8 (regs+smem) K32 | 214/8 (regs) K32 | 161/12 (regs) K32 | 161/8 (smem) K32 |
| nxg4 cb13 chk16 listmax4096 | 128 | 48704 | 212/8 (regs) K32 | 209/8 (regs+smem) K32 | 214/8 (regs) K32 | 161/12 (regs) K32 | 161/8 (smem) K32 |
| nxg4 cb13 chk48 listmax1024 | 128 | 101056 | 228/4 (smem) K32 | 225/4 (smem) K32 | 228/8 (regs+smem) K32 | 238/8 (regs+smem) K32 | 232/4 (smem) K32 |
| nxg4 cb13 chk64 listmax1024 | 128 | 133376 | 234/4 (smem) K32 | - | 235/4 (smem) K32 | 237/4 (smem) K32 | - |
| nxg4 cb4 chk16 listmax2048 | 128 | 29632 | 124/16 (regs) K32 | 93/12 (smem) K32 | 96/20 (regs) K32 | 78/24 (regs) K32 | 72/12 (smem) K32 |
| nxg4 cb4 chk16 listmax4096 | 128 | 37824 | 124/16 (regs+smem) K32 | 93/8 (smem) K32 | 96/20 (regs) K32 | 78/24 (regs+smem) K32 | 72/8 (smem) K32 |
| nxg4 cb4 chk48 listmax1024 | 128 | 68416 | 127/8 (smem) K32 | 119/4 (smem) K32 | 120/12 (smem) K32 | 121/12 (smem) K32 | 122/4 (smem) K32 |
| nxg4 cb4 chk64 listmax1024 | 128 | 89856 | 128/4 (smem) K32 | 126/4 (smem) K32 | 126/8 (smem) K32 | 122/8 (smem) K32 | 123/4 (smem) K32 |
| nxg4 cb5 chk16 listmax2048 | 128 | 31680 | 127/16 (regs) K32 | 119/12 (smem) K32 | 122/16 (regs) K32 | 95/20 (regs) K32 | 80/12 (smem) K32 |
| nxg4 cb5 chk16 listmax4096 | 128 | 39872 | 127/16 (regs+smem) K32 | 119/8 (smem) K32 | 122/16 (regs) K32 | 95/20 (regs+smem) K32 | 80/8 (smem) K32 |
| nxg4 cb5 chk48 listmax1024 | 128 | 74560 | 154/8 (smem) K32 | 127/4 (smem) K32 | 124/12 (smem) K32 | 123/12 (smem) K32 | 127/4 (smem) K32 |
| nxg4 cb5 chk64 listmax1024 | 128 | 98048 | 152/4 (smem) K32 | 126/4 (smem) K32 | 124/8 (smem) K32 | 126/8 (smem) K32 | 125/4 (smem) K32 |
| nxg4 cb7 chk16 listmax2048 | 128 | 33984 | 156/12 (regs) K32 | 128/8 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 122/8 (smem) K32 |
| nxg4 cb7 chk16 listmax4096 | 128 | 42176 | 156/12 (regs+smem) K32 | 128/8 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 122/8 (smem) K32 |
| nxg4 cb7 chk48 listmax1024 | 128 | 81472 | 162/8 (smem) K32 | 164/4 (smem) K32 | 160/8 (smem) K32 | 124/8 (smem) K32 | 124/4 (smem) K32 |
| nxg4 cb7 chk64 listmax1024 | 128 | 107264 | 163/4 (smem) K32 | - | 158/8 (smem) K32 | 128/8 (smem) K32 | - |

### vel2d_o3_cb5 / kl_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 | 128 | 26256 | 126/16 (regs) K48 | 121/12 (smem) K48 | 119/16 (regs) K48 | 95/20 (regs) K48 | 93/12 (smem) K48 |
| nxg1 cb1 chk32 | 128 | 52112 | 126/12 (smem) K48 | 118/4 (smem) K48 | 121/16 (regs+smem) K48 | 96/16 (smem) K48 | 96/4 (smem) K48 |
| nxg1 cb2 chk16 | 128 | 26256 | 158/12 (regs) K48 | 158/12 (regs+smem) K48 | 159/12 (regs) K48 | 157/12 (regs) K48 | 156/12 (regs+smem) K48 |
| nxg1 cb2 chk32 | 128 | 52112 | 160/12 (regs+smem) K48 | 159/4 (smem) K48 | 161/12 (regs) K48 | 158/12 (regs) K48 | 159/4 (smem) K48 |
| nxg2 cb1 chk16 | 128 | 21904 | 119/16 (regs) K48 | 93/16 (smem) K48 | 94/20 (regs) K48 | 72/28 (regs) K48 | 72/16 (smem) K48 |
| nxg2 cb1 chk32 | 128 | 43408 | 119/12 (smem) K48 | 93/8 (smem) K48 | 94/20 (regs+smem) K48 | 72/20 (smem) K48 | 72/8 (smem) K48 |
| nxg2 cb2 chk16 | 128 | 22160 | 124/16 (regs) K48 | 121/16 (regs+smem) K48 | 119/16 (regs) K48 | 123/16 (regs) K48 | 125/16 (regs+smem) K48 |
| nxg2 cb2 chk32 | 128 | 43920 | 126/12 (smem) K48 | 119/8 (smem) K48 | 119/16 (regs) K48 | 96/20 (regs+smem) K48 | 96/8 (smem) K48 |
| nxg2 cb4 chk16 | 128 | 26256 | 157/12 (regs) K48 | 158/12 (regs+smem) K48 | 158/12 (regs) K48 | 156/12 (regs) K48 | 158/12 (regs+smem) K48 |
| nxg2 cb4 chk32 | 128 | 52112 | 159/12 (regs+smem) K48 | 159/4 (smem) K48 | 159/12 (regs) K48 | 156/12 (regs) K48 | 158/4 (smem) K48 |
| nxg4 cb2 chk16 | 128 | 19856 | 120/16 (regs) K48 | 80/16 (smem) K48 | 91/20 (regs) K48 | 72/28 (regs) K48 | 71/16 (smem) K48 |
| nxg4 cb2 chk32 | 128 | 39312 | 120/16 (regs+smem) K48 | 80/8 (smem) K48 | 91/20 (regs+smem) K48 | 72/20 (smem) K48 | 71/8 (smem) K48 |
| nxg4 cb3 chk16 | 128 | 22032 | 124/16 (regs) K48 | 123/16 (regs+smem) K48 | 123/16 (regs) K48 | 80/24 (regs) K48 | 80/16 (smem) K48 |
| nxg4 cb3 chk32 | 128 | 43664 | 124/12 (smem) K48 | 123/8 (smem) K48 | 123/16 (regs) K48 | 80/20 (smem) K48 | 80/8 (smem) K48 |
| nxg4 cb4 chk16 | 128 | 22160 | 124/16 (regs) K48 | 121/16 (regs+smem) K48 | 119/16 (regs) K48 | 121/16 (regs) K48 | 122/16 (regs+smem) K48 |
| nxg4 cb4 chk32 | 128 | 43920 | 126/12 (smem) K48 | 119/8 (smem) K48 | 119/16 (regs) K48 | 96/20 (regs+smem) K48 | 96/8 (smem) K48 |
| nxg4 cb7 chk16 | 128 | 26256 | 157/12 (regs) K48 | 160/12 (regs+smem) K48 | 157/12 (regs) K48 | 123/16 (regs) K48 | 125/12 (smem) K48 |
| nxg4 cb7 chk32 | 128 | 52112 | 158/12 (regs+smem) K48 | 157/4 (smem) K48 | 160/12 (regs) K48 | 126/16 (regs+smem) K48 | 125/4 (smem) K48 |

### vel2d_o3_cb5 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 29632 | 122/16 (regs) K32 | 93/12 (smem) K32 | 95/20 (regs) K32 | 79/24 (regs) K32 | 78/12 (smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 55168 | 124/8 (smem) K32 | 119/4 (smem) K32 | 119/16 (regs+smem) K32 | 80/16 (smem) K32 | 80/4 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 29632 | 156/12 (regs) K32 | 126/12 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 123/12 (smem) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 55168 | 157/8 (smem) K32 | 161/4 (smem) K32 | 159/12 (regs) K32 | 126/16 (regs+smem) K32 | 126/4 (smem) K32 |
| nxg2 cb1 chk16 listmax1024 | 128 | 25280 | 119/16 (regs) K32 | 79/12 (smem) K32 | 80/24 (regs) K32 | 64/32 (regs+smem) K32 | 64/12 (smem) K32 |
| nxg2 cb1 chk32 listmax1024 | 128 | 46464 | 119/12 (smem) K32 | 79/8 (smem) K32 | 79/16 (smem) K32 | 64/16 (smem) K32 | 64/8 (smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 25536 | 124/16 (regs) K32 | 93/12 (smem) K32 | 94/20 (regs) K32 | 72/28 (regs) K32 | 72/12 (smem) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 46976 | 124/12 (smem) K32 | 124/8 (smem) K32 | 124/16 (regs+smem) K32 | 80/16 (smem) K32 | 80/8 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 29632 | 160/12 (regs) K32 | 126/12 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 121/12 (smem) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 55168 | 157/8 (smem) K32 | 158/4 (smem) K32 | 159/12 (regs) K32 | 126/16 (regs+smem) K32 | 126/4 (smem) K32 |
| nxg4 cb2 chk16 listmax1024 | 128 | 23232 | 117/16 (regs) K32 | 72/16 (smem) S12 K48 | 72/28 (regs) S12 K48 | 64/32 (regs) K32 | 64/16 (smem) K32 |
| nxg4 cb2 chk32 listmax1024 | 128 | 42368 | 117/12 (smem) K32 | 72/8 (smem) S12 K48 | 72/20 (smem) S12 K48 | 64/20 (smem) K32 | 64/8 (smem) K32 |
| nxg4 cb3 chk16 listmax1024 | 128 | 25408 | 120/16 (regs) K32 | 93/12 (smem) K32 | 94/20 (regs) K32 | 71/28 (regs) K32 | 71/12 (smem) K32 |
| nxg4 cb3 chk32 listmax1024 | 128 | 46720 | 120/12 (smem) K32 | 93/8 (smem) K32 | 94/16 (smem) K32 | 71/16 (smem) K32 | 71/8 (smem) K32 |
| nxg4 cb4 chk16 listmax1024 | 128 | 25536 | 124/16 (regs) K32 | 93/12 (smem) K32 | 96/20 (regs) K32 | 78/24 (regs) K32 | 72/12 (smem) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 46976 | 124/12 (smem) K32 | 124/8 (smem) K32 | 125/16 (regs+smem) K32 | 80/16 (smem) K32 | 80/8 (smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 29632 | 156/12 (regs) K32 | 128/12 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 122/12 (smem) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 55168 | 160/8 (smem) K32 | 126/4 (smem) K32 | 125/16 (regs+smem) K32 | 123/16 (regs+smem) K32 | 123/4 (smem) K32 |

### vel2d_o3_cb5 / tile_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 38528 | 90/16 (smem) K32 | 72/8 (smem) K32 | 80/20 (smem) K32 | 72/20 (smem) S20 K56 | 64/8 (smem) S24 K56 |
| tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 128 | 43648 | 93/12 (smem) K32 | 72/8 (smem) K32 | 80/20 (smem) K32 | 71/20 (smem) K32 | 72/8 (smem) K32 |
| tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 64 | 38528 | 122/8 (smem) K32 | 122/4 (smem) K32 | 125/10 (smem) K32 | 122/10 (smem) K32 | 122/4 (smem) K32 |
| tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 64 | 43648 | 123/6 (smem) K32 | 123/4 (smem) K32 | 125/10 (smem) K32 | 122/10 (smem) K32 | 122/4 (smem) K32 |
| tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 192 | 38784 | 94/18 (regs) K32 | 72/12 (smem) K32 | 96/18 (regs) K32 | 64/30 (regs+smem) S8 K40 | 56/12 (smem) S40 K56 |
| tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 192 | 46464 | 90/18 (regs+smem) K32 | 71/12 (smem) K32 | 78/24 (regs+smem) K32 | 72/24 (regs+smem) K32 | 75/12 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 39040 | 94/16 (regs) K32 | 77/16 (smem) K32 | 95/16 (regs) K32 | 94/16 (regs) K32 | 79/16 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 256 | 49280 | 94/16 (regs) K32 | 71/16 (smem) K32 | 80/24 (regs) K32 | 64/32 (regs+smem) S8 K40 | 80/16 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 128 | 39040 | 122/16 (regs+smem) K32 | 93/8 (smem) K32 | 123/16 (regs) K32 | 120/16 (regs) K32 | 93/8 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 128 | 49280 | 122/12 (smem) K32 | 122/8 (smem) K32 | 123/16 (regs+smem) K32 | 122/16 (regs+smem) K32 | 94/8 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 64 | 39040 | 165/8 (smem) K32 | 158/4 (smem) K32 | 159/10 (smem) K32 | 160/10 (smem) K32 | 157/4 (smem) K32 |
| tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 64 | 49280 | 163/6 (smem) K32 | 161/4 (smem) K32 | 157/8 (smem) K32 | 160/8 (smem) K32 | 156/4 (smem) K32 |
| tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 320 | 39296 | 90/20 (regs) K32 | 91/20 (regs+smem) K32 | 94/20 (regs) K32 | 84/20 (regs) K32 | 82/20 (regs+smem) K32 |
| tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 320 | 52096 | 94/20 (regs) K32 | 92/10 (smem) K32 | 94/20 (regs) K32 | 85/20 (regs) K32 | 85/10 (smem) K32 |

### vel2d_o3_cb5 / tile_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 32 | 42880 | 94/3 (smem) K32 | 79/2 (smem) K32 | 64/5 (smem) K32 | 64/5 (smem) K32 | 79/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 32 | 42880 | 94/3 (smem) K32 | 79/2 (smem) K32 | 64/5 (smem) K32 | 64/5 (smem) K32 | 79/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 32 | 76544 | 127/2 (smem) K32 | 96/1 (smem) K32 | 96/3 (smem) K32 | 95/3 (smem) K32 | 96/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 32 | 76544 | 127/2 (smem) K32 | 96/1 (smem) K32 | 96/3 (smem) K32 | 95/3 (smem) K32 | 96/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | 43136 | 64/6 (smem) K32 | 56/4 (smem) K32 | 64/10 (smem) K32 | 64/10 (smem) K32 | 56/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | 43136 | 72/6 (smem) K32 | 56/4 (smem) K32 | 64/10 (smem) K32 | 64/10 (smem) K32 | 64/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 64 | 77056 | 115/4 (smem) K32 | 80/2 (smem) K32 | 95/4 (smem) K32 | 80/4 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 64 | 77056 | 115/4 (smem) K32 | 80/2 (smem) K32 | 95/4 (smem) K32 | 80/4 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 32 | 77056 | 167/2 (smem) K32 | 143/1 (smem) K32 | 155/2 (smem) K32 | 168/2 (smem) K32 | 161/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 32 | 77056 | 167/2 (smem) K32 | 143/1 (smem) K32 | 155/2 (smem) K32 | 149/2 (smem) K32 | 147/1 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 43648 | 64/12 (smem) K32 | 55/8 (smem) K32 | 56/20 (smem) K32 | 56/20 (smem) K32 | 56/8 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 128 | 43648 | 70/12 (smem) K32 | 56/8 (smem) K32 | 56/20 (smem) K32 | 58/20 (smem) K32 | 56/8 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 78080 | 96/8 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 80/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 128 | 78080 | 96/8 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 80/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 78080 | 157/4 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 64 | 78080 | 157/4 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | 49024 | 80/6 (smem) K32 | 56/4 (smem) K32 | 64/8 (smem) K32 | 64/8 (smem) K32 | 66/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | 49024 | 80/6 (smem) K32 | 56/4 (smem) K32 | 64/8 (smem) K32 | 64/8 (smem) K32 | 66/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 64 | 82688 | 115/4 (smem) K32 | 80/2 (smem) K32 | 95/4 (smem) K32 | 80/4 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 64 | 82688 | 115/4 (smem) K32 | 80/2 (smem) K32 | 95/4 (smem) K32 | 80/4 (smem) K32 | 80/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 32 | 82688 | 167/2 (smem) K32 | 143/1 (smem) K32 | 153/2 (smem) K32 | 167/2 (smem) K32 | 159/1 (smem) K32 |
| tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 32 | 82688 | 167/2 (smem) K32 | 143/1 (smem) K32 | 153/2 (smem) K32 | 149/2 (smem) K32 | 143/1 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 49280 | 64/12 (smem) K32 | 56/8 (smem) K32 | 56/16 (smem) K32 | 56/16 (smem) K32 | 56/8 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 128 | 49280 | 70/12 (smem) K32 | 56/8 (smem) K32 | 56/16 (smem) K32 | 56/16 (smem) K32 | 64/8 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 83200 | 96/4 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 79/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 128 | 83200 | 96/4 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 79/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 83200 | 157/2 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 64 | 83200 | 157/2 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 49792 | 64/24 (smem) K32 | 64/16 (smem) K32 | 64/32 (regs+smem) K32 | 64/32 (regs+smem) K32 | 60/16 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 256 | 49792 | 72/24 (regs+smem) K32 | 64/16 (smem) K32 | 64/32 (regs+smem) K32 | 64/32 (regs+smem) K32 | 64/16 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 84224 | 96/8 (smem) K32 | 71/8 (smem) K32 | 80/16 (smem) K32 | 78/16 (smem) K32 | 64/8 (smem) S32 K48 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 256 | 84224 | 96/8 (smem) K32 | 71/8 (smem) K32 | 80/16 (smem) K32 | 78/16 (smem) K32 | 64/8 (smem) S32 K48 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 84224 | 151/4 (smem) K32 | 128/4 (smem) K32 | 128/8 (smem) K32 | 128/8 (smem) K32 | 127/4 (smem) K32 |
| tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | 84224 | 151/4 (smem) K32 | 128/4 (smem) K32 | 128/8 (smem) K32 | 128/8 (smem) K32 | 127/4 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 96 | 53120 | 64/9 (smem) S24 K56 | 56/3 (smem) K32 | 56/12 (smem) K32 | 56/12 (smem) K32 | 56/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 96 | 53120 | 72/9 (smem) K32 | 56/3 (smem) K32 | 56/12 (smem) K32 | 56/12 (smem) K32 | 56/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | 86784 | 96/3 (smem) K32 | 72/3 (smem) K32 | 80/6 (smem) K32 | 80/6 (smem) K32 | 72/3 (smem) S32 K48 |
| tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | 86784 | 96/3 (smem) K32 | 72/3 (smem) K32 | 80/6 (smem) K32 | 80/6 (smem) K32 | 72/3 (smem) S32 K48 |
| tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 192 | 53376 | 64/18 (smem) K32 | 56/6 (smem) K32 | 56/24 (smem) K32 | 56/24 (smem) K32 | 56/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 192 | 53376 | 64/18 (smem) S36 K56 | 56/6 (smem) K32 | 56/24 (smem) K32 | 58/24 (smem) K32 | 56/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 192 | 87296 | 96/6 (smem) K32 | 71/6 (smem) K32 | 79/12 (smem) K32 | 78/12 (smem) K32 | 76/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 192 | 87296 | 96/6 (smem) K32 | 71/6 (smem) K32 | 79/12 (smem) K32 | 78/12 (smem) K32 | 76/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 96 | 87296 | 153/3 (smem) K32 | 128/3 (smem) K32 | 128/6 (smem) K32 | 128/6 (smem) K32 | 127/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 96 | 87296 | 153/3 (smem) K32 | 128/3 (smem) K32 | 128/6 (smem) K32 | 127/6 (smem) K32 | 127/3 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 384 | 53888 | 72/24 (regs) K32 | 56/12 (smem) K32 | 56/36 (regs) K32 | 56/36 (regs) K32 | 56/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 384 | 53888 | 73/24 (regs) K32 | 56/12 (smem) K32 | 56/36 (regs) K32 | 58/24 (regs) K32 | 56/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 384 | 88320 | 100/12 (regs+smem) K32 | 80/12 (smem) K32 | 80/24 (regs+smem) K32 | 80/24 (regs+smem) K32 | 79/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 384 | 88320 | 100/12 (regs+smem) K32 | 80/12 (smem) K32 | 80/24 (regs+smem) K32 | 80/24 (regs+smem) K32 | 79/12 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 192 | 88320 | 145/6 (smem) K32 | 145/6 (smem) K32 | 147/12 (regs+smem) K32 | 131/12 (regs+smem) K32 | 131/6 (smem) K32 |
| tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 192 | 88320 | 145/6 (smem) K32 | 145/6 (smem) K32 | 147/12 (regs+smem) K32 | 131/12 (regs+smem) K32 | 131/6 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 128 | 59264 | 64/8 (smem) K32 | 56/4 (smem) K32 | 56/12 (smem) K32 | 56/12 (smem) K32 | 56/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 128 | 59264 | 72/8 (smem) K32 | 56/4 (smem) K32 | 56/12 (smem) K32 | 56/12 (smem) K32 | 56/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 128 | 92928 | 96/4 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 79/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 128 | 92928 | 96/4 (smem) K32 | 72/4 (smem) K32 | 80/8 (smem) K32 | 79/8 (smem) K32 | 79/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 64 | 92928 | 157/2 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 64 | 92928 | 157/2 (smem) K32 | 128/2 (smem) K32 | 128/4 (smem) K32 | 128/4 (smem) K32 | 127/2 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 256 | 59520 | 64/16 (smem) K32 | 64/8 (smem) K32 | 64/24 (smem) K32 | 64/24 (smem) K32 | 60/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 256 | 59520 | 72/16 (smem) K32 | 64/8 (smem) K32 | 64/24 (smem) K32 | 64/24 (smem) K32 | 64/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 256 | 93440 | 80/8 (smem) S8 K40 | 71/8 (smem) K32 | 80/16 (smem) K32 | 78/16 (smem) K32 | 64/8 (smem) S32 K48 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 256 | 93440 | 80/8 (smem) S8 K40 | 71/8 (smem) K32 | 80/16 (smem) K32 | 78/16 (smem) K32 | 64/8 (smem) S32 K48 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | 93440 | 151/4 (smem) K32 | 127/4 (smem) K32 | 128/8 (smem) K32 | 128/8 (smem) K32 | 127/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | 93440 | 151/4 (smem) K32 | 127/4 (smem) K32 | 128/8 (smem) K32 | 128/8 (smem) K32 | 127/4 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 512 | 60032 | 64/32 (regs+smem) K32 | 64/16 (smem) K32 | 63/32 (regs) K32 | 64/32 (regs) K32 | 64/16 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 512 | 60032 | 64/32 (regs+smem) K32 | 64/16 (smem) K32 | 64/32 (regs) K32 | 64/32 (regs) K32 | 64/16 (smem) S36 K56 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 512 | 94464 | 96/16 (regs+smem) K32 | 95/16 (regs+smem) K32 | 95/16 (regs) K32 | 80/16 (regs) K32 | 80/16 (regs+smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 512 | 94464 | 96/16 (regs+smem) K32 | 95/16 (regs+smem) K32 | 95/16 (regs) K32 | 80/16 (regs) K32 | 80/16 (regs+smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 256 | 94464 | 128/8 (smem) K32 | 128/8 (smem) K32 | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | 127/8 (smem) K32 |
| tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 256 | 94464 | 128/8 (smem) K32 | 128/8 (smem) K32 | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | 127/8 (smem) K32 |
| tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 160 | 63360 | 64/10 (smem) K32 | 56/5 (smem) K32 | 56/15 (smem) K32 | 56/15 (smem) K32 | 56/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 160 | 63360 | 72/10 (smem) K32 | 56/5 (smem) K32 | 56/15 (smem) K32 | 56/15 (smem) K32 | 56/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 160 | 97024 | 96/5 (smem) K32 | 72/5 (smem) K32 | 86/10 (smem) K32 | 96/10 (smem) K32 | 95/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 160 | 97024 | 96/5 (smem) K32 | 72/5 (smem) K32 | 86/10 (smem) K32 | 96/10 (smem) K32 | 95/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 320 | 63616 | 64/20 (smem) K32 | 64/10 (smem) K32 | 64/30 (regs+smem) K32 | 64/30 (regs+smem) K32 | 60/10 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 320 | 63616 | 64/20 (smem) K32 | 64/10 (smem) K32 | 64/30 (regs+smem) K32 | 66/20 (regs) K32 | 64/10 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 320 | 97536 | 95/10 (smem) K32 | 95/10 (smem) K32 | 96/20 (regs+smem) K32 | 95/20 (regs+smem) K32 | 95/10 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 320 | 97536 | 95/10 (smem) K32 | 95/10 (smem) K32 | 96/20 (regs+smem) K32 | 95/20 (regs+smem) K32 | 95/10 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 160 | 97536 | 151/5 (smem) K32 | 128/5 (smem) K32 | 128/10 (smem) K32 | 128/10 (smem) K32 | 127/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 160 | 97536 | 151/5 (smem) K32 | 128/5 (smem) K32 | 128/10 (smem) K32 | 127/10 (smem) K32 | 127/5 (smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 480 | 55520 | 64/30 (regs+smem) K32 | 63/15 (smem) K32 | 64/30 (regs) K32 | 61/30 (regs) K32 | 61/15 (smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv1 | 480 | 55520 | 64/30 (regs+smem) S32 K48 | 64/15 (smem) K32 | 64/30 (regs) S12 K48 | 64/30 (regs) K32 | 64/15 (smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 480 | 81344 | 95/15 (regs) K32 | 76/15 (regs+smem) K32 | 95/15 (regs) K32 | 82/15 (regs) K32 | 82/15 (regs+smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv1 | 480 | 81344 | 95/15 (regs) K32 | 74/15 (regs+smem) K32 | 95/15 (regs) K32 | 82/15 (regs) K32 | 82/15 (regs+smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 320 | 98560 | 143/10 (regs+smem) K32 | 143/10 (regs+smem) K32 | 147/10 (regs) K32 | 131/10 (regs) K32 | 131/10 (regs+smem) K32 |
| tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 320 | 98560 | 143/10 (regs+smem) K32 | 143/10 (regs+smem) K32 | 147/10 (regs) K32 | 131/10 (regs) K32 | 131/10 (regs+smem) K32 |

### vel2d_o3_cb5 / vt_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 2x4x16 smax32 nwc4 nws1 vxpad0 | 128 | 52768 | 181/8 (regs) | 178/4 (smem) | 178/8 (regs) | 168/12 (regs) S8 K8 | 168/4 (smem) S8 K8 |
| 2x4x16 smax32 nwc4 nws2 vxpad0 | 256 | 59168 | 172/8 (regs) | 173/8 (regs+smem) | 174/8 (regs) | 184/8 (regs) | 194/8 (regs+smem) |
| 2x4x8 smax32 nwc4 nws1 vxpad0 | 128 | 52512 | 181/8 (regs) | 178/4 (smem) | 178/8 (regs) | 168/12 (regs) S8 K8 | 168/4 (smem) S8 K8 |
| 2x4x8 smax32 nwc4 nws2 vxpad0 | 256 | 58912 | 172/8 (regs) | 173/8 (regs+smem) | 174/8 (regs) | 184/8 (regs) | 194/8 (regs+smem) |
| 2x4x8 smax48 nwc4 nws1 vxpad0 | 128 | 75040 | 181/8 (regs+smem) | 178/4 (smem) | 178/8 (regs) | 168/12 (regs+smem) S8 K8 | 168/4 (smem) S8 K8 |
| 2x4x8 smax48 nwc4 nws2 vxpad0 | 256 | 81440 | 172/8 (regs) | 173/8 (regs+smem) | 174/8 (regs) | 184/8 (regs) | 194/8 (regs+smem) |
| 4x4x16 smax48 nwc4 nws1 vxpad0 | 128 | 76096 | 254/8 (regs+smem) | 254/4 (smem) | 254/8 (regs) | 254/8 (regs) | 236/4 (smem) |
| 4x4x16 smax48 nwc4 nws2 vxpad0 | 256 | 82496 | 232/8 (regs) | 236/8 (regs+smem) | 240/8 (regs) | 252/8 (regs) | 238/8 (regs+smem) |
| 4x4x16 smax64 nwc4 nws1 vxpad0 | 128 | 98624 | 254/4 (smem) | 254/4 (smem) | 254/8 (regs+smem) | 254/8 (regs+smem) | 236/4 (smem) |
| 4x4x16 smax64 nwc4 nws2 vxpad0 | 256 | 105024 | 232/8 (regs+smem) | - | 240/8 (regs) | 252/8 (regs) | - |

### vel2d_o3_cb5 / whatif

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 4x4x16 smax128 nwc4 nws1 vxpad0 | 128 | 188736 | - | - | 254/4 (smem) | 254/4 (smem) | - |
| 4x4x16 smax96 nwc4 nws1 vxpad0 | 128 | 143680 | 254/4 (smem) | - | 254/4 (smem) | 254/4 (smem) | - |
| nxg1 cb1 chk16 listmax2048 | 128 | 33728 | 122/16 (regs+smem) K32 | 93/8 (smem) K32 | 95/20 (regs) K32 | 79/24 (regs+smem) K32 | 78/8 (smem) K32 |
| nxg1 cb1 chk16 listmax4096 | 128 | 41920 | 122/12 (smem) K32 | 93/8 (smem) K32 | 95/20 (regs+smem) K32 | 79/20 (smem) K32 | 78/8 (smem) K32 |
| nxg1 cb1 chk48 listmax1024 | 128 | 80704 | 127/8 (smem) K32 | 121/4 (smem) K32 | 120/8 (smem) K32 | 123/8 (smem) K32 | 122/4 (smem) K32 |
| nxg1 cb1 chk64 listmax1024 | 128 | 106240 | 128/4 (smem) K32 | - | 122/8 (smem) K32 | 96/8 (smem) K32 | - |
| nxg1 cb2 chk16 listmax2048 | 128 | 33728 | 156/12 (regs) K32 | 126/8 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 123/8 (smem) K32 |
| nxg1 cb2 chk16 listmax4096 | 128 | 41920 | 156/12 (regs+smem) K32 | 126/8 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 123/8 (smem) K32 |
| nxg1 cb2 chk48 listmax1024 | 128 | 80704 | 159/8 (smem) K32 | 159/4 (smem) K32 | 161/8 (smem) K32 | 125/8 (smem) K32 | 126/4 (smem) K32 |
| nxg1 cb2 chk64 listmax1024 | 128 | 106240 | 166/4 (smem) K32 | - | 162/8 (smem) K32 | 122/8 (smem) K32 | - |
| nxg2 cb1 chk16 listmax2048 | 128 | 29376 | 119/16 (regs) K32 | 79/12 (smem) K32 | 80/24 (regs) K32 | 64/28 (smem) K32 | 64/12 (smem) K32 |
| nxg2 cb1 chk16 listmax4096 | 128 | 37568 | 119/16 (regs+smem) K32 | 79/8 (smem) K32 | 80/24 (regs+smem) K32 | 64/24 (smem) K32 | 64/8 (smem) K32 |
| nxg2 cb1 chk48 listmax1024 | 128 | 67648 | 120/8 (smem) K32 | 94/4 (smem) K32 | 96/12 (smem) K32 | 71/12 (smem) K32 | 70/4 (smem) K32 |
| nxg2 cb1 chk64 listmax1024 | 128 | 88832 | 120/4 (smem) K32 | 94/4 (smem) K32 | 96/8 (smem) K32 | 71/8 (smem) K32 | 70/4 (smem) K32 |
| nxg2 cb2 chk16 listmax2048 | 128 | 29632 | 124/16 (regs) K32 | 93/12 (smem) K32 | 94/20 (regs) K32 | 72/28 (regs+smem) K32 | 72/12 (smem) K32 |
| nxg2 cb2 chk16 listmax4096 | 128 | 37824 | 124/16 (regs+smem) K32 | 93/8 (smem) K32 | 94/20 (regs) K32 | 72/24 (smem) K32 | 72/8 (smem) K32 |
| nxg2 cb2 chk48 listmax1024 | 128 | 68416 | 127/8 (smem) K32 | 119/4 (smem) K32 | 120/12 (smem) K32 | 123/12 (smem) K32 | 122/4 (smem) K32 |
| nxg2 cb2 chk64 listmax1024 | 128 | 89856 | 128/4 (smem) K32 | 126/4 (smem) K32 | 126/8 (smem) K32 | 123/8 (smem) K32 | 122/4 (smem) K32 |
| nxg2 cb4 chk16 listmax2048 | 128 | 33728 | 160/12 (regs) K32 | 126/8 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 121/8 (smem) K32 |
| nxg2 cb4 chk16 listmax4096 | 128 | 41920 | 160/12 (regs+smem) K32 | 126/8 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 121/8 (smem) K32 |
| nxg2 cb4 chk48 listmax1024 | 128 | 80704 | 162/8 (smem) K32 | 162/4 (smem) K32 | 159/8 (smem) K32 | 125/8 (smem) K32 | 126/4 (smem) K32 |
| nxg2 cb4 chk64 listmax1024 | 128 | 106240 | 167/4 (smem) K32 | - | 156/8 (smem) K32 | 122/8 (smem) K32 | - |
| nxg4 cb2 chk16 listmax2048 | 128 | 27328 | 117/16 (regs) K32 | 72/12 (smem) S12 K48 | 72/28 (regs) S12 K48 | 64/32 (regs+smem) K32 | 64/12 (smem) K32 |
| nxg4 cb2 chk16 listmax4096 | 128 | 35520 | 117/16 (regs+smem) K32 | 72/8 (smem) S12 K48 | 72/24 (smem) S12 K48 | 64/24 (smem) K32 | 64/8 (smem) K32 |
| nxg4 cb2 chk48 listmax1024 | 128 | 61504 | 119/8 (smem) K32 | 94/4 (smem) K32 | 94/12 (smem) K32 | 72/12 (smem) K32 | 64/4 (smem) K32 |
| nxg4 cb2 chk64 listmax1024 | 128 | 80640 | 119/8 (smem) K32 | 94/4 (smem) K32 | 94/8 (smem) K32 | 72/8 (smem) K32 | 64/4 (smem) K32 |
| nxg4 cb3 chk16 listmax2048 | 128 | 29504 | 120/16 (regs) K32 | 93/12 (smem) K32 | 94/20 (regs) K32 | 71/28 (regs+smem) K32 | 71/12 (smem) K32 |
| nxg4 cb3 chk16 listmax4096 | 128 | 37696 | 120/16 (regs+smem) K32 | 93/8 (smem) K32 | 94/20 (regs) K32 | 71/24 (smem) K32 | 71/8 (smem) K32 |
| nxg4 cb3 chk48 listmax1024 | 128 | 68032 | 124/8 (smem) K32 | 118/4 (smem) K32 | 96/12 (smem) K32 | 72/12 (smem) K32 | 72/4 (smem) K32 |
| nxg4 cb3 chk64 listmax1024 | 128 | 89344 | 124/4 (smem) K32 | 118/4 (smem) K32 | 96/8 (smem) K32 | 72/8 (smem) K32 | 72/4 (smem) K32 |
| nxg4 cb4 chk16 listmax2048 | 128 | 29632 | 124/16 (regs) K32 | 93/12 (smem) K32 | 96/20 (regs) K32 | 78/24 (regs) K32 | 72/12 (smem) K32 |
| nxg4 cb4 chk16 listmax4096 | 128 | 37824 | 124/16 (regs+smem) K32 | 93/8 (smem) K32 | 96/20 (regs) K32 | 78/24 (regs+smem) K32 | 72/8 (smem) K32 |
| nxg4 cb4 chk48 listmax1024 | 128 | 68416 | 127/8 (smem) K32 | 119/4 (smem) K32 | 120/12 (smem) K32 | 121/12 (smem) K32 | 122/4 (smem) K32 |
| nxg4 cb4 chk64 listmax1024 | 128 | 89856 | 128/4 (smem) K32 | 126/4 (smem) K32 | 126/8 (smem) K32 | 122/8 (smem) K32 | 123/4 (smem) K32 |
| nxg4 cb7 chk16 listmax2048 | 128 | 33728 | 156/12 (regs) K32 | 128/8 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 122/8 (smem) K32 |
| nxg4 cb7 chk16 listmax4096 | 128 | 41920 | 156/12 (regs+smem) K32 | 128/8 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 122/8 (smem) K32 |
| nxg4 cb7 chk48 listmax1024 | 128 | 80704 | 159/8 (smem) K32 | 159/4 (smem) K32 | 159/8 (smem) K32 | 126/8 (smem) K32 | 126/4 (smem) K32 |
| nxg4 cb7 chk64 listmax1024 | 128 | 106240 | 162/4 (smem) K32 | - | 161/8 (smem) K32 | 124/8 (smem) K32 | - |

## Flags (spills, stack, no fit, below 8 warps / SM, compile errors)

| arch | case | family | cfg | regs | flags |
|---|---|---|---|---|---|
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 80 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 80 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 119 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 80 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 96 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 80 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 96 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 125 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 96 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 96 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 124 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 125 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 124 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 155 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 72 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 64 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 64 | SPILL st12 ld12, STACK 64 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 64 | SPILL st16 ld4, STACK 64 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 72 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 64 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 64 | SPILL st12 ld12, STACK 64 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 64 | SPILL st16 ld4, STACK 64 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 80 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 72 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 80 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 80 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 72 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 72 | SPILL st20 ld8, STACK 64 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 125 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 96 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 96 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 124 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 125 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 124 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 156 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 64 | SPILL st8 ld8, STACK 64 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 64 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 72 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 64 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 64 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 64 | SPILL st8 ld8, STACK 64 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 64 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 72 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 64 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 64 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 64 | SPILL st20 ld8, STACK 64 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 72 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 72 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 72 | SPILL st4 ld4, STACK 64 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 64 | SPILL st20 ld8, STACK 64 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 72 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 72 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 72 | SPILL st4 ld4, STACK 64 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 72 | SPILL st20 ld8, STACK 64 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 72 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 96 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 80 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 80 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 80 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 72 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 125 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 96 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 96 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 125 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 127 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 123 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 125 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 127 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 96 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 95 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 94 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 94 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 120 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 95 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 123 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 124 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 153 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 64 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 64 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 80 | SPILL st16 ld32, STACK 48 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 64 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 64 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 80 | SPILL st16 ld32, STACK 48 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 96 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 80 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 80 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 80 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 122 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 95 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 123 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 124 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 153 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 64 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 64 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 72 | SPILL st4 ld20, STACK 40 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 56 | SPILL st12 ld28, STACK 40 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 64 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 64 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 72 | SPILL st4 ld20, STACK 40 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 56 | SPILL st12 ld28, STACK 40 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 94 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 94 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 80 | SPILL st12 ld12, STACK 48 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 72 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 94 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 94 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 80 | SPILL st12 ld12, STACK 48 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 72 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 93 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 95 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 80 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 94 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 94 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 122 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 95 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 124 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 124 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 121 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 127 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 126 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | SPILL st16 ld16, STACK 48 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | SPILL st16 ld16, STACK 48 |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 71 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 71 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 94 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 71 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 77 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 121 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 121 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 96 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 123 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 96 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 125 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st28 ld44, STACK 56 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st28 ld44, STACK 56 |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st12 ld12, STACK 48 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 75 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 75 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 79 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 74 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 79 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 73 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 79 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 71 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 119 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 119 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 93 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 94 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 96 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 94 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 95 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 163 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 163 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 162 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 164 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 163 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 164 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 164 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 163 | STACK 32, LOW_OCC 6w |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 164 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 162 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 83 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 83 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 85 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 84 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 84 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | SPILL st16 ld16, STACK 48 |
| sm_89 | knot1d_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | SPILL st16 ld16, STACK 48 |
| sm_90 | knot1d_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 83 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 66 | STACK 32, LOW_OCC 5w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 78 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32, LOW_OCC 5w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 66 | STACK 32, LOW_OCC 5w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 79 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 80 | STACK 32, LOW_OCC 2w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32, LOW_OCC 5w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 3w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 1w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 1w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 3w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 3w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32, LOW_OCC 1w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 127 | STACK 32, LOW_OCC 2w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 1w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 3w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 61 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 6w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 113 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 6w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 6w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 111 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 2w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 93 | STACK 32, LOW_OCC 6w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 3w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 1w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 168 | STACK 32, LOW_OCC 3w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 167 | STACK 32, LOW_OCC 3w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 167 | STACK 32, LOW_OCC 1w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 167 | STACK 32, LOW_OCC 3w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 51 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 157 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 6w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 155 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 59 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 60 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32, LOW_OCC 6w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 59 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 61 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | SPILL st4 ld4, STACK 40, LOW_OCC 6w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 6w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 113 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 93 | STACK 32, LOW_OCC 6w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 6w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | SPILL st8 ld8, STACK 40, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 2w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 93 | STACK 32, LOW_OCC 6w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 3w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 1w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 153 | STACK 32, LOW_OCC 3w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 167 | STACK 32, LOW_OCC 3w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 167 | STACK 32, LOW_OCC 1w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 167 | STACK 32, LOW_OCC 3w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | SPILL st32 ld16, STACK 48 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 48 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 157 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 6w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 2w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 155 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 59 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 61 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 53 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 63 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 48 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st4 ld4, STACK 40 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 76 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | SPILL st8 ld8, STACK 40 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 71 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 149 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | SPILL st4 ld4, STACK 40 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | SPILL st36 ld20, STACK 56 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 50 | STACK 32, LOW_OCC 6w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 6w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | SPILL st36 ld20, STACK 56 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 6w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 3w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 6w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 3w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 6w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 6w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 3w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | SPILL st16 ld16, STACK 48, LOW_OCC 6w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 3w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 6w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | SPILL st4 ld4, STACK 40 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 6w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32, LOW_OCC 6w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 6w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | SPILL st8 ld8, STACK 40 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 71 | STACK 32, LOW_OCC 6w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 3w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 151 | STACK 32, LOW_OCC 6w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 3w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 6w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 3w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 149 | STACK 32, LOW_OCC 6w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 3w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | SPILL st4 ld4, STACK 40 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 55 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st16 ld16, STACK 48 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | SPILL st8 ld8, STACK 40 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 129 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 129 | STACK 32, LOW_OCC 6w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32, LOW_OCC 6w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 145 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 129 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 129 | STACK 32, LOW_OCC 6w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 141 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 141 | STACK 32, LOW_OCC 6w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 145 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 50 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 48 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 157 | STACK 32, LOW_OCC 2w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 153 | STACK 32, LOW_OCC 2w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 4w |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 59 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 61 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 53 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 63 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 48 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st4 ld4, STACK 40 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 76 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | SPILL st8 ld8, STACK 40 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 71 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 149 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | SPILL st4 ld4, STACK 40, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 63 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 82 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 82 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 94 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 64 | SPILL st32 ld16, STACK 48 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 97 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 82 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 82 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 64 | SPILL st32 ld16, STACK 48 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 5w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 5w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | SPILL st36 ld20, STACK 56 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 50 | STACK 32, LOW_OCC 5w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 5w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | SPILL st36 ld20, STACK 56 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 5w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 5w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 5w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 84 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32, LOW_OCC 5w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32, LOW_OCC 5w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 5w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 61 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 47 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 59 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 93 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 91 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 93 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 91 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 5w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | SPILL st4 ld4, STACK 40, LOW_OCC 5w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 5w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 5w |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | SPILL st12 ld12, STACK 48, LOW_OCC 5w |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 5w |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 59 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 64 | SPILL st32 ld16, STACK 48 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 64 | SPILL st32 ld16, STACK 48 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 94 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 72 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv1 | 82 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv1 | 82 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv1 | 94 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv1 | 72 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 129 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 129 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 147 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 129 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 129 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 141 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 141 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 145 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws1 vxpad0 | 180 | LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws1 vxpad0 | 168 | LOW_OCC 4w |
| sm_120 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 | 236 | LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_120 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 236 | LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_100 | knot1d_o2_cb5 | whatif | 4x4x16 smax128 nwc4 nws1 vxpad0 | 252 | LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | 4x4x16 smax128 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_100 | knot1d_o2_cb5 | whatif | 4x4x16 smax96 nwc4 nws1 vxpad0 | 252 | LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | 4x4x16 smax96 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | 4x4x16 smax96 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_100 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 96 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 95 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 96 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 95 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 123 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk64 listmax1024 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk64 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk64 listmax1024 | 123 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk64 listmax1024 | 125 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg1 cb1 chk64 listmax1024 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 128 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 123 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 124 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 128 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 123 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 124 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 159 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk64 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk64 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk64 listmax1024 | 156 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk64 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg1 cb2 chk64 listmax1024 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 64 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 64 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 80 | SPILL st16 ld32, STACK 48 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 64 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 64 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 80 | SPILL st16 ld32, STACK 48 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 72 | SPILL st12 ld16, STACK 40 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 72 | SPILL st12 ld16, STACK 40, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 95 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 72 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 72 | SPILL st12 ld16, STACK 40 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 72 | SPILL st12 ld16, STACK 40, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 72 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 96 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 80 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 96 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 80 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 122 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 125 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 128 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 123 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 124 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 128 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 123 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 124 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 157 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk64 listmax1024 | 162 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk64 listmax1024 | 158 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk64 listmax1024 | 163 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk64 listmax1024 | 158 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg2 cb4 chk64 listmax1024 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 64 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 64 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 72 | SPILL st4 ld20, STACK 40 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 56 | SPILL st12 ld28, STACK 40 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 64 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 64 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 72 | SPILL st4 ld20, STACK 40 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 56 | SPILL st12 ld28, STACK 40 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 64 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 72 | SPILL st4 ld4, STACK 40 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 72 | SPILL st4 ld4, STACK 40, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 72 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 72 | SPILL st4 ld4, STACK 40 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 72 | SPILL st4 ld4, STACK 40, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 72 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 94 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 94 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 80 | SPILL st12 ld12, STACK 48 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 72 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 94 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 94 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 80 | SPILL st12 ld12, STACK 48 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 64 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 72 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 95 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 95 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 120 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 80 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 95 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 95 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 120 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 80 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 95 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 93 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 95 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 80 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 93 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 95 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 96 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 80 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 80 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 122 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 96 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 125 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 96 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 124 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 124 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 121 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 124 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 124 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 121 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 154 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 127 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk64 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk64 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk64 listmax1024 | 159 | STACK 32, LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk64 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | whatif | nxg4 cb7 chk64 listmax1024 | 128 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 64 | SPILL st8 ld8, STACK 40 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 80 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 80 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 80 | SPILL st4 ld4, STACK 40 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 80 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 96 | SPILL st16 ld32, STACK 48 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 96 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 80 | SPILL st12 ld12, STACK 48 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 110 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 126 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 117 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 118 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 111 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 111 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | SPILL st12 ld28, STACK 48 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 117 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 118 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 183 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 182 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 206 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 204 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 207 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 187 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 186 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 210 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 208 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 211 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 80 | SPILL st16 ld16, STACK 48 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 72 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 72 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 80 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 80 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 80 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 96 | SPILL st8 ld8, STACK 40 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 95 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 122 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 96 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 119 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 114 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 126 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 121 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 115 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 114 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 112 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 118 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 118 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 116 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 114 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 119 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 119 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 161 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 160 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 198 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 195 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 197 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 158 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 157 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 202 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 199 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 201 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 158 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 161 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 200 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 197 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 199 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 160 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 166 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 204 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 201 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 203 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 80 | SPILL st16 ld16, STACK 48 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 72 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 72 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 80 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 80 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 80 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 90 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 89 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 117 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 115 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 118 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 90 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 90 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 122 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 105 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 110 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 118 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 116 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | SPILL st16 ld16, STACK 48 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 122 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 120 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 119 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 117 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | SPILL st40 ld56, STACK 72 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 122 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 120 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 82 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 74 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 74 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 74 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 84 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st12 ld12, STACK 48 |
| sm_89 | static_o0_10x10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 80 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_10x10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_10x10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o0_10x10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st12 ld12, STACK 48 |
| sm_89 | static_o0_10x10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st12 ld12, STACK 48 |
| sm_89 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st24 ld24, STACK 56 |
| sm_100 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 79 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 167 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 157 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 168 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 168 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_10x10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 168 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 66 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 58 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st12 ld12, STACK 48 |
| sm_89 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st24 ld24, STACK 56 |
| sm_100 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 98 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 100 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 167 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 128 | SPILL st8 ld8, STACK 40 |
| sm_80 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 162 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 128 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 168 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32, LOW_OCC 7w |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 78 | STACK 32, LOW_OCC 3w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32, LOW_OCC 5w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32, LOW_OCC 3w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 71 | STACK 32, LOW_OCC 7w |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 4w |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 1w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 128 | STACK 32, LOW_OCC 3w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 1w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 4w |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 2w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 2w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 93 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 145 | STACK 32, LOW_OCC 4w |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 139 | STACK 32, LOW_OCC 1w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 168 | STACK 32, LOW_OCC 3w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 139 | STACK 32, LOW_OCC 1w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 168 | SPILL st4 ld4, STACK 40, LOW_OCC 4w |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32, LOW_OCC 2w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 154 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st8 ld8, STACK 40 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 57 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 93 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 93 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 94 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32, LOW_OCC 5w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | SPILL st8 ld8, STACK 40, LOW_OCC 5w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 5w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 5w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 121 | STACK 32, LOW_OCC 5w |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv0 | 73 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv0 | 64 | SPILL st52 ld36, STACK 72 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv0 | 94 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 138 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv0 | 73 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv0 | 64 | SPILL st52 ld36, STACK 72 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv0 | 74 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 2w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 2w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 3w |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 1w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 168 | STACK 32, LOW_OCC 2w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 168 | SPILL st4 ld4, STACK 40, LOW_OCC 3w |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 154 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 57 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 58 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 71 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | SPILL st32 ld16, STACK 48, LOW_OCC 6w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 74 | STACK 32, LOW_OCC 3w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 3w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 49 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st8 ld8, STACK 40 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32, LOW_OCC 3w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 142 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 3w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 49 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st8 ld8, STACK 40 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 138 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 139 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 71 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | SPILL st32 ld16, STACK 48 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 74 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 156 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 57 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 58 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 64 | SPILL st44 ld28, STACK 64 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 94 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st40 ld24, STACK 56 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | SPILL st36 ld20, STACK 56 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32, LOW_OCC 3w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | SPILL st12 ld12, STACK 48, LOW_OCC 6w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 3w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 50 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 55 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st8 ld8, STACK 40 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 138 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 139 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 73 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 64 | SPILL st52 ld36, STACK 72 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 74 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 138 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 139 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st36 ld20, STACK 56 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 57 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 125 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | SPILL st8 ld8, STACK 40, LOW_OCC 4w |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 64 | SPILL st44 ld28, STACK 64 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 94 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 94 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 64 | SPILL st44 ld28, STACK 64 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 94 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 72 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_80 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_90 | static_o0_10x10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 174 | LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 198 | LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 194 | LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 186 | LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 194 | LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad1 | 204 | LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad1 | 196 | LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad1 | 192 | LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad1 | 190 | LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad1 | 200 | LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 174 | LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 198 | LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 194 | LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 186 | LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 194 | LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad1 | 204 | LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad1 | 196 | LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad1 | 192 | LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad1 | 190 | LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad1 | 200 | LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 174 | LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 198 | LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 194 | LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 186 | LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 194 | LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad1 | 204 | LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad1 | 196 | LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad1 | 192 | LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad1 | 190 | LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad1 | 200 | LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 248 | LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 240 | LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 240 | LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_100 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 | 254 | LOW_OCC 7w |
| sm_120 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 | 248 | LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 | 242 | LOW_OCC 7w |
| sm_89 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 | 242 | LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 | 254 | LOW_OCC 7w |
| sm_100 | static_o0_10x10 | whatif | 4x4x16 smax128 nwc7 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_80 | static_o0_10x10 | whatif | 4x4x16 smax128 nwc7 nws1 vxpad0 | 240 | LOW_OCC 7w |
| sm_90 | static_o0_10x10 | whatif | 4x4x16 smax128 nwc7 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_100 | static_o0_10x10 | whatif | 4x4x16 smax192 nwc7 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_90 | static_o0_10x10 | whatif | 4x4x16 smax192 nwc7 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_100 | static_o0_10x10 | whatif | 4x4x16 smax256 nwc5 nws1 vxpad0 | 254 | LOW_OCC 5w |
| sm_90 | static_o0_10x10 | whatif | 4x4x16 smax256 nwc5 nws1 vxpad0 | 254 | LOW_OCC 5w |
| sm_120 | static_o0_10x10 | whatif | 4x4x16 smax96 nwc5 nws1 vxpad0 | 244 | LOW_OCC 5w |
| sm_89 | static_o0_10x10 | whatif | 4x4x16 smax96 nwc5 nws1 vxpad0 | 240 | LOW_OCC 5w |
| sm_100 | static_o0_10x10 | whatif | 4x4x16 smax96 nwc7 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_80 | static_o0_10x10 | whatif | 4x4x16 smax96 nwc7 nws1 vxpad0 | 240 | LOW_OCC 7w |
| sm_90 | static_o0_10x10 | whatif | 4x4x16 smax96 nwc7 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_100 | static_o0_10x10 | whatif | nxg1 cb1 chk16 listmax2048 | 64 | SPILL st8 ld8, STACK 40 |
| sm_120 | static_o0_10x10 | whatif | nxg1 cb1 chk16 listmax2048 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg1 cb1 chk16 listmax2048 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg1 cb1 chk16 listmax2048 | 80 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg1 cb1 chk16 listmax2048 | 80 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_10x10 | whatif | nxg1 cb1 chk16 listmax4096 | 64 | SPILL st8 ld8, STACK 40 |
| sm_120 | static_o0_10x10 | whatif | nxg1 cb1 chk16 listmax4096 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg1 cb1 chk16 listmax4096 | 96 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg1 cb1 chk16 listmax4096 | 80 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg1 cb1 chk16 listmax4096 | 80 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_10x10 | whatif | nxg1 cb1 chk48 listmax1024 | 87 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg1 cb1 chk48 listmax1024 | 80 | SPILL st8 ld8, STACK 40 |
| sm_80 | static_o0_10x10 | whatif | nxg1 cb1 chk48 listmax1024 | 116 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg1 cb1 chk48 listmax1024 | 115 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg1 cb1 chk48 listmax1024 | 111 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg1 cb1 chk64 listmax1024 | 88 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg1 cb1 chk64 listmax1024 | 88 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | whatif | nxg1 cb1 chk64 listmax1024 | 114 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg1 cb1 chk64 listmax1024 | 114 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | whatif | nxg1 cb1 chk64 listmax1024 | 114 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg1 cb2 chk16 listmax2048 | 110 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg1 cb2 chk16 listmax2048 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg1 cb2 chk16 listmax2048 | 126 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg1 cb2 chk16 listmax2048 | 117 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg1 cb2 chk16 listmax2048 | 118 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg1 cb2 chk16 listmax4096 | 110 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg1 cb2 chk16 listmax4096 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg1 cb2 chk16 listmax4096 | 126 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg1 cb2 chk16 listmax4096 | 117 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg1 cb2 chk16 listmax4096 | 118 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg1 cb2 chk48 listmax1024 | 117 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg1 cb2 chk48 listmax1024 | 115 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg1 cb2 chk48 listmax1024 | 150 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg1 cb2 chk48 listmax1024 | 149 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg1 cb2 chk48 listmax1024 | 145 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg1 cb2 chk64 listmax1024 | 146 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg1 cb2 chk64 listmax1024 | 142 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | whatif | nxg1 cb2 chk64 listmax1024 | 150 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg1 cb2 chk64 listmax1024 | 149 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | whatif | nxg1 cb2 chk64 listmax1024 | 150 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg1 cb4 chk16 listmax2048 | 183 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg1 cb4 chk16 listmax2048 | 182 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg1 cb4 chk16 listmax2048 | 206 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg1 cb4 chk16 listmax2048 | 204 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg1 cb4 chk16 listmax2048 | 207 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg1 cb4 chk16 listmax4096 | 183 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg1 cb4 chk16 listmax4096 | 182 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg1 cb4 chk16 listmax4096 | 206 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg1 cb4 chk16 listmax4096 | 204 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg1 cb4 chk16 listmax4096 | 207 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg1 cb4 chk48 listmax1024 | 200 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg1 cb4 chk48 listmax1024 | 197 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | whatif | nxg1 cb4 chk48 listmax1024 | 222 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg1 cb4 chk48 listmax1024 | 221 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | whatif | nxg1 cb4 chk48 listmax1024 | 223 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg1 cb4 chk64 listmax1024 | 223 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg1 cb4 chk64 listmax1024 | 227 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | whatif | nxg1 cb4 chk64 listmax1024 | 227 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o0_10x10 | whatif | nxg1 cb4 chk64 listmax1024 | 225 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | whatif | nxg1 cb4 chk64 listmax1024 | 228 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb2 chk16 listmax2048 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb2 chk16 listmax2048 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb2 chk16 listmax2048 | 80 | SPILL st16 ld16, STACK 48 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb2 chk16 listmax2048 | 72 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb2 chk16 listmax2048 | 72 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb2 chk16 listmax4096 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb2 chk16 listmax4096 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb2 chk16 listmax4096 | 80 | SPILL st16 ld16, STACK 48 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb2 chk16 listmax4096 | 72 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb2 chk16 listmax4096 | 72 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb2 chk48 listmax1024 | 88 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb2 chk48 listmax1024 | 80 | SPILL st12 ld12, STACK 48 |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb2 chk48 listmax1024 | 116 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb2 chk48 listmax1024 | 113 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb2 chk48 listmax1024 | 116 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb2 chk64 listmax1024 | 86 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb2 chk64 listmax1024 | 80 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb2 chk64 listmax1024 | 114 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb2 chk64 listmax1024 | 117 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb2 chk64 listmax1024 | 114 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb3 chk16 listmax2048 | 96 | SPILL st8 ld8, STACK 40 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb3 chk16 listmax2048 | 95 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb3 chk16 listmax2048 | 122 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb3 chk16 listmax2048 | 96 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb3 chk16 listmax2048 | 119 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb3 chk16 listmax4096 | 96 | SPILL st8 ld8, STACK 40 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb3 chk16 listmax4096 | 95 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb3 chk16 listmax4096 | 122 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb3 chk16 listmax4096 | 96 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb3 chk16 listmax4096 | 119 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb3 chk48 listmax1024 | 110 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb3 chk48 listmax1024 | 114 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb3 chk48 listmax1024 | 128 | SPILL st12 ld28, STACK 48 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb3 chk48 listmax1024 | 125 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb3 chk48 listmax1024 | 124 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb3 chk64 listmax1024 | 110 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb3 chk64 listmax1024 | 114 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb3 chk64 listmax1024 | 128 | SPILL st12 ld28, STACK 48 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb3 chk64 listmax1024 | 125 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb3 chk64 listmax1024 | 124 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb4 chk16 listmax2048 | 114 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb4 chk16 listmax2048 | 112 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb4 chk16 listmax2048 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb4 chk16 listmax2048 | 118 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb4 chk16 listmax2048 | 118 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb4 chk16 listmax4096 | 114 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb4 chk16 listmax4096 | 112 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb4 chk16 listmax4096 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb4 chk16 listmax4096 | 118 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb4 chk16 listmax4096 | 118 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb4 chk48 listmax1024 | 147 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb4 chk48 listmax1024 | 123 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb4 chk48 listmax1024 | 152 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb4 chk48 listmax1024 | 150 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb4 chk48 listmax1024 | 153 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb4 chk64 listmax1024 | 147 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb4 chk64 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb4 chk64 listmax1024 | 153 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb4 chk64 listmax1024 | 151 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb4 chk64 listmax1024 | 150 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb7 chk16 listmax2048 | 161 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb7 chk16 listmax2048 | 160 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb7 chk16 listmax2048 | 198 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb7 chk16 listmax2048 | 195 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb7 chk16 listmax2048 | 197 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb7 chk16 listmax4096 | 161 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb7 chk16 listmax4096 | 160 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb7 chk16 listmax4096 | 198 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb7 chk16 listmax4096 | 195 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb7 chk16 listmax4096 | 197 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb7 chk48 listmax1024 | 186 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb7 chk48 listmax1024 | 185 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb7 chk48 listmax1024 | 214 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb7 chk48 listmax1024 | 211 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb7 chk48 listmax1024 | 213 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg2 cb7 chk64 listmax1024 | 190 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg2 cb7 chk64 listmax1024 | 189 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | whatif | nxg2 cb7 chk64 listmax1024 | 221 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o0_10x10 | whatif | nxg2 cb7 chk64 listmax1024 | 216 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | whatif | nxg2 cb7 chk64 listmax1024 | 218 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb13 chk16 listmax2048 | 158 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb13 chk16 listmax2048 | 161 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb13 chk16 listmax2048 | 200 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb13 chk16 listmax2048 | 197 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb13 chk16 listmax2048 | 199 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb13 chk16 listmax4096 | 158 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb13 chk16 listmax4096 | 161 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb13 chk16 listmax4096 | 200 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb13 chk16 listmax4096 | 197 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb13 chk16 listmax4096 | 199 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb13 chk48 listmax1024 | 188 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb13 chk48 listmax1024 | 187 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb13 chk48 listmax1024 | 216 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb13 chk48 listmax1024 | 213 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb13 chk48 listmax1024 | 215 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb13 chk64 listmax1024 | 192 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb13 chk64 listmax1024 | 191 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb13 chk64 listmax1024 | 221 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb13 chk64 listmax1024 | 218 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb13 chk64 listmax1024 | 220 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb4 chk16 listmax2048 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb4 chk16 listmax2048 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb4 chk16 listmax2048 | 80 | SPILL st16 ld16, STACK 48 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb4 chk16 listmax2048 | 72 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb4 chk16 listmax2048 | 72 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb4 chk16 listmax4096 | 64 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb4 chk16 listmax4096 | 64 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb4 chk16 listmax4096 | 80 | SPILL st16 ld16, STACK 48 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb4 chk16 listmax4096 | 72 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb4 chk16 listmax4096 | 72 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb4 chk48 listmax1024 | 89 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb4 chk48 listmax1024 | 80 | SPILL st12 ld12, STACK 48 |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb4 chk48 listmax1024 | 116 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb4 chk48 listmax1024 | 113 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb4 chk48 listmax1024 | 116 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb4 chk64 listmax1024 | 89 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb4 chk64 listmax1024 | 88 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb4 chk64 listmax1024 | 114 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb4 chk64 listmax1024 | 117 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb4 chk64 listmax1024 | 113 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb5 chk16 listmax2048 | 90 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb5 chk16 listmax2048 | 89 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb5 chk16 listmax2048 | 117 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb5 chk16 listmax2048 | 115 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb5 chk16 listmax2048 | 118 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb5 chk16 listmax4096 | 90 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb5 chk16 listmax4096 | 89 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb5 chk16 listmax4096 | 117 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb5 chk16 listmax4096 | 115 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb5 chk16 listmax4096 | 118 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb5 chk48 listmax1024 | 115 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb5 chk48 listmax1024 | 112 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb5 chk48 listmax1024 | 126 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb5 chk48 listmax1024 | 118 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb5 chk48 listmax1024 | 116 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb5 chk64 listmax1024 | 91 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb5 chk64 listmax1024 | 91 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb5 chk64 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb5 chk64 listmax1024 | 124 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb5 chk64 listmax1024 | 122 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb7 chk16 listmax2048 | 118 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb7 chk16 listmax2048 | 116 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb7 chk16 listmax2048 | 128 | SPILL st16 ld16, STACK 48 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb7 chk16 listmax2048 | 122 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb7 chk16 listmax2048 | 120 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb7 chk16 listmax4096 | 118 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb7 chk16 listmax4096 | 116 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb7 chk16 listmax4096 | 128 | SPILL st16 ld16, STACK 48 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb7 chk16 listmax4096 | 122 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb7 chk16 listmax4096 | 120 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb7 chk48 listmax1024 | 119 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb7 chk48 listmax1024 | 118 | STACK 32 |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb7 chk48 listmax1024 | 153 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb7 chk48 listmax1024 | 150 | STACK 32 |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb7 chk48 listmax1024 | 152 | STACK 32 |
| sm_100 | static_o0_10x10 | whatif | nxg4 cb7 chk64 listmax1024 | 118 | STACK 32 |
| sm_120 | static_o0_10x10 | whatif | nxg4 cb7 chk64 listmax1024 | 118 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_10x10 | whatif | nxg4 cb7 chk64 listmax1024 | 156 | STACK 32 |
| sm_89 | static_o0_10x10 | whatif | nxg4 cb7 chk64 listmax1024 | 155 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_10x10 | whatif | nxg4 cb7 chk64 listmax1024 | 152 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 90 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 90 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 122 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 122 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 120 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 120 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 118 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 128 | SPILL st4 ld20, STACK 40 |
| sm_89 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 120 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 120 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 145 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 147 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 153 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 153 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 153 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 155 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 155 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 168 | SPILL st4 ld20, STACK 40 |
| sm_89 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 155 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 156 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 72 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 95 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 72 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 80 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 80 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 96 | SPILL st8 ld24, STACK 40 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 80 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 80 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 91 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 89 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 119 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 117 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 116 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 121 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 117 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 119 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 120 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 121 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 116 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 147 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 128 | SPILL st20 ld36, STACK 56 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 128 | SPILL st24 ld40, STACK 56 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 125 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 123 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 168 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 151 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 153 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 72 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 72 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 94 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 72 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 72 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 72 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 72 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 96 | SPILL st4 ld20, STACK 40 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 80 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 80 | SPILL st12 ld28, STACK 48 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 72 | SPILL st4 ld4, STACK 40 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 114 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 111 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 114 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 90 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 111 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 111 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 120 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 119 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 146 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 122 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 118 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 120 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 120 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 168 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 150 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 152 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 61 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | SPILL st12 ld12, STACK 48 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | SPILL st12 ld12, STACK 48 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | SPILL st32 ld32, STACK 72 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 40 | SPILL st16 ld16, STACK 48 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 52 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 55 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 55 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 50 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba12 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 56 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 55 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 53 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 55 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 55 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 40 | SPILL st12 ld12, STACK 48 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 58 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 57 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 59 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 48 | SPILL st8 ld8, STACK 40 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 93 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 71 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba16 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 62 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 57 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 57 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 59 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 51 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 93 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba24 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 64 | SPILL st16 ld16, STACK 48 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 57 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 58 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 57 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 59 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 51 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 58 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 58 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 62 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 80 | SPILL st16 ld16, STACK 48 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 64 | SPILL st4 ld4, STACK 40 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba32 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 64 | SPILL st16 ld16, STACK 48 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | SPILL st4 ld4, STACK 40 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 46 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 48 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 48 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 64 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 56 | SPILL st8 ld12, STACK 40 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba48 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 66 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba48 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 62 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba48 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba48 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba48 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 74 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba48 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 56 | SPILL st28 ld28, STACK 64 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba48 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 78 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba48 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 80 | SPILL st16 ld16, STACK 48 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba48 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba48 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 75 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 38 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | SPILL st12 ld12, STACK 48 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 40 | SPILL st16 ld16, STACK 48 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 52 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 53 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 55 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 55 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 48 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 55 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 48 | SPILL st8 ld8, STACK 40 |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 56 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 72 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 95 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 64 | SPILL st4 ld4, STACK 40, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb8 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 52 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk20 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk20 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk20 acb1 fsb1 fcb1 kv0 | 40 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk20 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk20 acb1 fsb1 fcb1 kv0 | 32 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 55 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk10 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk10 acb1 fsb1 fcb1 kv0 | 38 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk10 acb1 fsb1 fcb1 kv0 | 40 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk10 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk10 acb1 fsb1 fcb1 kv0 | 32 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk40 acb1 fsb4 fcb1 kv0 | 57 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk40 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk40 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk40 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf12 ppf4 chunk40 acb1 fsb4 fcb1 kv0 | 59 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 72 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 64 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 40 | SPILL st32 ld16, STACK 48 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk16 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk16 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk16 acb1 fsb1 fcb1 kv0 | 62 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk16 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk16 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 39 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 57 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 63 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 59 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk32 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 57 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 59 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk16 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk16 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk16 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk16 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk16 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk32 acb1 fsb4 fcb1 kv0 | 57 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk32 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk32 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk32 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk32 acb1 fsb4 fcb1 kv0 | 59 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk8 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk8 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk8 acb1 fsb1 fcb1 kv0 | 62 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk8 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf16 ppf4 chunk8 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk20 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk20 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk20 acb1 fsb1 fcb1 kv0 | 40 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk20 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk20 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 52 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 48 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk10 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk10 acb1 fsb1 fcb1 kv0 | 38 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk10 acb1 fsb1 fcb1 kv0 | 40 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk10 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk10 acb1 fsb1 fcb1 kv0 | 32 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk20 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk20 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk20 acb1 fsb1 fcb2 kv0 | 57 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk20 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk20 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 55 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk40 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk40 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk40 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk40 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf2 chunk40 acb1 fsb4 fcb1 kv0 | 59 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf4 chunk10 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf4 chunk10 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf4 chunk10 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf4 chunk10 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf4 chunk10 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf4 chunk20 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf4 chunk20 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf4 chunk20 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf4 chunk20 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf24 ppf4 chunk20 acb1 fsb4 fcb1 kv0 | 59 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk16 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk16 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk16 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk16 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk16 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 55 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 48 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb2 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk32 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 40 | SPILL st12 ld12, STACK 48 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 60 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk16 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk16 acb1 fsb1 fcb2 kv0 | 39 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk16 acb1 fsb1 fcb2 kv0 | 57 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk16 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk16 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk16 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk16 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk16 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk16 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk16 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 59 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 59 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 59 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 64 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 64 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 57 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 59 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk8 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk8 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk8 acb1 fsb1 fcb1 kv0 | 62 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk8 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf2 chunk8 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf4 chunk16 acb1 fsb4 fcb1 kv0 | 57 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf4 chunk16 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf4 chunk16 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf4 chunk16 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf4 chunk16 acb1 fsb4 fcb1 kv0 | 59 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf4 chunk8 acb1 fsb2 fcb1 kv0 | 32 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf4 chunk8 acb1 fsb2 fcb1 kv0 | 39 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf4 chunk8 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf4 chunk8 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf32 ppf4 chunk8 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 54 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 65 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 79 | STACK 32, LOW_OCC 3w |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 69 | STACK 32, LOW_OCC 5w |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 80 | STACK 32, LOW_OCC 3w |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 66 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 48 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 64 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 38 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 48 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 38 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 62 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 38 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 40 | SPILL st4 ld4, STACK 40 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 39 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk10 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk10 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk10 acb1 fsb1 fcb1 kv0 | 40 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk10 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk10 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk20 acb1 fsb1 fcb2 kv0 | 55 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk20 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk20 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk20 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk20 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk20 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 50 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk40 acb1 fsb4 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk40 acb1 fsb4 fcb1 kv0 | 40 | SPILL st12 ld12, STACK 48 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk40 acb1 fsb4 fcb1 kv0 | 55 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk40 acb1 fsb4 fcb1 kv0 | 60 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf1 chunk40 acb1 fsb4 fcb1 kv0 | 58 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk10 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk10 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk10 acb1 fsb1 fcb2 kv0 | 57 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk10 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk10 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk10 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk10 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk10 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk10 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk10 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk20 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk20 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk20 acb1 fsb1 fcb4 kv0 | 58 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk20 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk20 acb1 fsb1 fcb4 kv0 | 60 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk20 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk20 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk20 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk20 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk20 acb1 fsb4 fcb1 kv0 | 59 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 55 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf4 chunk8 acb1 fsb4 fcb1 kv0 | 51 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf4 chunk8 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf4 chunk8 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf4 chunk8 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf48 ppf4 chunk8 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 48 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 55 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 54 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 38 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 55 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 39 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 38 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 40 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 32 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 39 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 40 | SPILL st4 ld4, STACK 40 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 48 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb4 kv0 | 56 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 65 | STACK 32, LOW_OCC 7w |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 79 | STACK 32, LOW_OCC 3w |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 78 | STACK 32, LOW_OCC 5w |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 80 | STACK 32, LOW_OCC 3w |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb8 kv0 | 78 | STACK 32, LOW_OCC 7w |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 62 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb2 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 48 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb4 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 64 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 72 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 64 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb8 kv0 | 66 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 38 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 48 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 62 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 39 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 38 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 40 | STACK 32 |
| sm_100 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 55 | STACK 32 |
| sm_120 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 40 | STACK 32 |
| sm_90 | static_o0_48x1 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 59 | STACK 32 |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc1 nws1 vxpad0 | 182 | LOW_OCC 7w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc1 nws1 vxpad0 | 166 | LOW_OCC 7w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc1 nws1 vxpad1 | 182 | LOW_OCC 7w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc1 nws1 vxpad1 | 168 | LOW_OCC 7w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws1 vxpad0 | 200 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws1 vxpad1 | 198 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws2 vxpad0 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws2 vxpad1 | 190 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc1 nws1 vxpad0 | 182 | LOW_OCC 7w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc1 nws1 vxpad0 | 166 | LOW_OCC 7w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc1 nws1 vxpad1 | 182 | LOW_OCC 7w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc1 nws1 vxpad1 | 168 | LOW_OCC 7w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws1 vxpad0 | 200 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws1 vxpad1 | 198 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws2 vxpad0 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws2 vxpad1 | 190 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc1 nws1 vxpad0 | 182 | LOW_OCC 5w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc1 nws1 vxpad0 | 166 | LOW_OCC 5w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc1 nws1 vxpad1 | 182 | LOW_OCC 5w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc1 nws1 vxpad1 | 168 | LOW_OCC 5w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc2 nws1 vxpad1 | 194 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc2 nws1 vxpad1 | 168 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws1 vxpad0 | 200 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws1 vxpad1 | 198 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws2 vxpad0 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws2 vxpad1 | 190 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 | 248 | LOW_OCC 7w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 | 252 | LOW_OCC 7w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 | 250 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 | 252 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 | 240 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 | 240 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 | 254 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 | 250 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 | 240 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 | 242 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 | 222 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 | 242 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 | 234 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 | 228 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 | 234 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 | 254 | SPILL st28 ld28, STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 | 230 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 | 236 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws1 vxpad0 | 248 | LOW_OCC 4w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws1 vxpad0 | 250 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws1 vxpad0 | 252 | LOW_OCC 4w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws1 vxpad1 | 250 | LOW_OCC 3w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws1 vxpad1 | 246 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws1 vxpad1 | 254 | LOW_OCC 3w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws2 vxpad1 | 168 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws2 vxpad1 | 226 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc2 nws1 vxpad0 | 242 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc2 nws1 vxpad0 | 238 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc2 nws1 vxpad1 | 240 | LOW_OCC 4w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc2 nws1 vxpad1 | 242 | LOW_OCC 4w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad0 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad0 | 252 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad0 | 240 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad0 | 240 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad0 | 254 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad1 | 250 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad1 | 240 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad1 | 242 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad0 | 224 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad0 | 242 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad0 | 234 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad0 | 228 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad0 | 234 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad1 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad1 | 254 | SPILL st28 ld28, STACK 32, LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad1 | 230 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad1 | 236 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc1 nws1 vxpad0 | 238 | LOW_OCC 4w |
| sm_120 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc1 nws1 vxpad0 | 248 | LOW_OCC 2w |
| sm_80 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc1 nws1 vxpad0 | 250 | LOW_OCC 3w |
| sm_89 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc1 nws1 vxpad0 | 252 | LOW_OCC 2w |
| sm_90 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc1 nws1 vxpad0 | 252 | LOW_OCC 4w |
| sm_100 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc2 nws1 vxpad0 | 234 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc2 nws1 vxpad0 | 242 | LOW_OCC 2w |
| sm_80 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc2 nws1 vxpad0 | 242 | LOW_OCC 4w |
| sm_89 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc2 nws1 vxpad0 | 238 | LOW_OCC 2w |
| sm_90 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc2 nws1 vxpad0 | 254 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc3 nws1 vxpad0 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc3 nws1 vxpad0 | 252 | LOW_OCC 3w |
| sm_80 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc3 nws1 vxpad0 | 240 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc3 nws1 vxpad0 | 240 | LOW_OCC 3w |
| sm_90 | static_o0_48x1 | whatif | 4x4x16 smax128 nwc3 nws1 vxpad0 | 254 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | whatif | 4x4x16 smax192 nwc1 nws1 vxpad0 | 238 | LOW_OCC 3w |
| sm_120 | static_o0_48x1 | whatif | 4x4x16 smax192 nwc1 nws1 vxpad0 | 248 | LOW_OCC 1w |
| sm_80 | static_o0_48x1 | whatif | 4x4x16 smax192 nwc1 nws1 vxpad0 | 250 | LOW_OCC 2w |
| sm_89 | static_o0_48x1 | whatif | 4x4x16 smax192 nwc1 nws1 vxpad0 | 252 | LOW_OCC 1w |
| sm_90 | static_o0_48x1 | whatif | 4x4x16 smax192 nwc1 nws1 vxpad0 | 252 | LOW_OCC 3w |
| sm_100 | static_o0_48x1 | whatif | 4x4x16 smax192 nwc2 nws1 vxpad0 | 234 | LOW_OCC 4w |
| sm_120 | static_o0_48x1 | whatif | 4x4x16 smax192 nwc2 nws1 vxpad0 | 242 | LOW_OCC 2w |
| sm_80 | static_o0_48x1 | whatif | 4x4x16 smax192 nwc2 nws1 vxpad0 | 242 | LOW_OCC 2w |
| sm_89 | static_o0_48x1 | whatif | 4x4x16 smax192 nwc2 nws1 vxpad0 | 238 | LOW_OCC 2w |
| sm_90 | static_o0_48x1 | whatif | 4x4x16 smax192 nwc2 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_100 | static_o0_48x1 | whatif | 4x4x16 smax192 nwc3 nws1 vxpad0 | 254 | LOW_OCC 3w |
| sm_80 | static_o0_48x1 | whatif | 4x4x16 smax192 nwc3 nws1 vxpad0 | 240 | LOW_OCC 3w |
| sm_90 | static_o0_48x1 | whatif | 4x4x16 smax192 nwc3 nws1 vxpad0 | 254 | LOW_OCC 3w |
| sm_100 | static_o0_48x1 | whatif | 4x4x16 smax256 nwc1 nws1 vxpad0 | 238 | LOW_OCC 2w |
| sm_120 | static_o0_48x1 | whatif | 4x4x16 smax256 nwc1 nws1 vxpad0 | 248 | LOW_OCC 1w |
| sm_80 | static_o0_48x1 | whatif | 4x4x16 smax256 nwc1 nws1 vxpad0 | 250 | LOW_OCC 1w |
| sm_89 | static_o0_48x1 | whatif | 4x4x16 smax256 nwc1 nws1 vxpad0 | 252 | LOW_OCC 1w |
| sm_90 | static_o0_48x1 | whatif | 4x4x16 smax256 nwc1 nws1 vxpad0 | 252 | LOW_OCC 2w |
| sm_100 | static_o0_48x1 | whatif | 4x4x16 smax256 nwc2 nws1 vxpad0 | 234 | LOW_OCC 2w |
| sm_80 | static_o0_48x1 | whatif | 4x4x16 smax256 nwc2 nws1 vxpad0 | 242 | LOW_OCC 2w |
| sm_90 | static_o0_48x1 | whatif | 4x4x16 smax256 nwc2 nws1 vxpad0 | 254 | LOW_OCC 2w |
| sm_100 | static_o0_48x1 | whatif | 4x4x16 smax256 nwc3 nws1 vxpad0 | 254 | LOW_OCC 3w |
| sm_80 | static_o0_48x1 | whatif | 4x4x16 smax256 nwc3 nws1 vxpad0 | 240 | LOW_OCC 3w |
| sm_90 | static_o0_48x1 | whatif | 4x4x16 smax256 nwc3 nws1 vxpad0 | 254 | LOW_OCC 3w |
| sm_100 | static_o0_48x1 | whatif | 4x4x16 smax96 nwc1 nws1 vxpad0 | 238 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | whatif | 4x4x16 smax96 nwc1 nws1 vxpad0 | 248 | LOW_OCC 2w |
| sm_80 | static_o0_48x1 | whatif | 4x4x16 smax96 nwc1 nws1 vxpad0 | 250 | LOW_OCC 4w |
| sm_89 | static_o0_48x1 | whatif | 4x4x16 smax96 nwc1 nws1 vxpad0 | 252 | LOW_OCC 2w |
| sm_90 | static_o0_48x1 | whatif | 4x4x16 smax96 nwc1 nws1 vxpad0 | 252 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | whatif | 4x4x16 smax96 nwc2 nws1 vxpad0 | 242 | LOW_OCC 4w |
| sm_80 | static_o0_48x1 | whatif | 4x4x16 smax96 nwc2 nws1 vxpad0 | 242 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | whatif | 4x4x16 smax96 nwc2 nws1 vxpad0 | 238 | LOW_OCC 4w |
| sm_100 | static_o0_48x1 | whatif | 4x4x16 smax96 nwc3 nws1 vxpad0 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | whatif | 4x4x16 smax96 nwc3 nws1 vxpad0 | 252 | LOW_OCC 3w |
| sm_80 | static_o0_48x1 | whatif | 4x4x16 smax96 nwc3 nws1 vxpad0 | 240 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | whatif | 4x4x16 smax96 nwc3 nws1 vxpad0 | 240 | LOW_OCC 3w |
| sm_90 | static_o0_48x1 | whatif | 4x4x16 smax96 nwc3 nws1 vxpad0 | 254 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | whatif | nxg1 cb1 chk16 listmax2048 | 90 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg1 cb1 chk16 listmax2048 | 90 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg1 cb1 chk16 listmax2048 | 122 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg1 cb1 chk16 listmax2048 | 122 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg1 cb1 chk16 listmax2048 | 120 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg1 cb1 chk16 listmax4096 | 90 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg1 cb1 chk16 listmax4096 | 90 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg1 cb1 chk16 listmax4096 | 122 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg1 cb1 chk16 listmax4096 | 122 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg1 cb1 chk16 listmax4096 | 120 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg1 cb1 chk48 listmax1024 | 125 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg1 cb1 chk48 listmax1024 | 125 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg1 cb1 chk48 listmax1024 | 162 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg1 cb1 chk48 listmax1024 | 128 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg1 cb1 chk48 listmax1024 | 148 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg1 cb1 chk64 listmax1024 | 127 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg1 cb1 chk64 listmax1024 | 121 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_48x1 | whatif | nxg1 cb1 chk64 listmax1024 | 168 | SPILL st4 ld20, STACK 40 |
| sm_89 | static_o0_48x1 | whatif | nxg1 cb1 chk64 listmax1024 | 159 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_48x1 | whatif | nxg1 cb1 chk64 listmax1024 | 161 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg1 cb2 chk16 listmax2048 | 145 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg1 cb2 chk16 listmax2048 | 147 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg1 cb2 chk16 listmax2048 | 153 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg1 cb2 chk16 listmax2048 | 153 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg1 cb2 chk16 listmax2048 | 153 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg1 cb2 chk16 listmax4096 | 145 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg1 cb2 chk16 listmax4096 | 147 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg1 cb2 chk16 listmax4096 | 153 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg1 cb2 chk16 listmax4096 | 153 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg1 cb2 chk16 listmax4096 | 153 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg1 cb2 chk48 listmax1024 | 226 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg1 cb2 chk48 listmax1024 | 223 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_48x1 | whatif | nxg1 cb2 chk48 listmax1024 | 223 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg1 cb2 chk48 listmax1024 | 222 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_48x1 | whatif | nxg1 cb2 chk48 listmax1024 | 225 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg1 cb2 chk64 listmax1024 | 246 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg1 cb2 chk64 listmax1024 | 226 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_48x1 | whatif | nxg1 cb2 chk64 listmax1024 | 248 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg1 cb2 chk64 listmax1024 | 241 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_48x1 | whatif | nxg1 cb2 chk64 listmax1024 | 243 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg2 cb1 chk16 listmax2048 | 80 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg2 cb1 chk16 listmax2048 | 72 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg2 cb1 chk16 listmax2048 | 95 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg2 cb1 chk16 listmax2048 | 72 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg2 cb1 chk16 listmax2048 | 80 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg2 cb1 chk16 listmax4096 | 80 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg2 cb1 chk16 listmax4096 | 72 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg2 cb1 chk16 listmax4096 | 95 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg2 cb1 chk16 listmax4096 | 72 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg2 cb1 chk16 listmax4096 | 80 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg2 cb1 chk48 listmax1024 | 87 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg2 cb1 chk48 listmax1024 | 87 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg2 cb1 chk48 listmax1024 | 118 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg2 cb1 chk48 listmax1024 | 96 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg2 cb1 chk48 listmax1024 | 111 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg2 cb1 chk64 listmax1024 | 118 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg2 cb1 chk64 listmax1024 | 116 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg2 cb1 chk64 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg2 cb1 chk64 listmax1024 | 117 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg2 cb1 chk64 listmax1024 | 120 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg2 cb2 chk16 listmax2048 | 91 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg2 cb2 chk16 listmax2048 | 89 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg2 cb2 chk16 listmax2048 | 119 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg2 cb2 chk16 listmax2048 | 117 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg2 cb2 chk16 listmax2048 | 116 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg2 cb2 chk16 listmax4096 | 91 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg2 cb2 chk16 listmax4096 | 89 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg2 cb2 chk16 listmax4096 | 119 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg2 cb2 chk16 listmax4096 | 117 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg2 cb2 chk16 listmax4096 | 116 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg2 cb2 chk48 listmax1024 | 128 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg2 cb2 chk48 listmax1024 | 128 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg2 cb2 chk48 listmax1024 | 160 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg2 cb2 chk48 listmax1024 | 128 | SPILL st4 ld20, STACK 40 |
| sm_90 | static_o0_48x1 | whatif | nxg2 cb2 chk48 listmax1024 | 128 | SPILL st4 ld20, STACK 40 |
| sm_100 | static_o0_48x1 | whatif | nxg2 cb2 chk64 listmax1024 | 162 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg2 cb2 chk64 listmax1024 | 158 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg2 cb2 chk64 listmax1024 | 168 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg2 cb2 chk64 listmax1024 | 128 | SPILL st28 ld28, STACK 64 |
| sm_90 | static_o0_48x1 | whatif | nxg2 cb2 chk64 listmax1024 | 128 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_48x1 | whatif | nxg2 cb3 chk16 listmax2048 | 121 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg2 cb3 chk16 listmax2048 | 116 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg2 cb3 chk16 listmax2048 | 147 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg2 cb3 chk16 listmax2048 | 128 | SPILL st20 ld36, STACK 56 |
| sm_90 | static_o0_48x1 | whatif | nxg2 cb3 chk16 listmax2048 | 128 | SPILL st24 ld40, STACK 56 |
| sm_100 | static_o0_48x1 | whatif | nxg2 cb3 chk16 listmax4096 | 121 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg2 cb3 chk16 listmax4096 | 116 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg2 cb3 chk16 listmax4096 | 147 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg2 cb3 chk16 listmax4096 | 128 | SPILL st20 ld36, STACK 56 |
| sm_90 | static_o0_48x1 | whatif | nxg2 cb3 chk16 listmax4096 | 128 | SPILL st24 ld40, STACK 56 |
| sm_100 | static_o0_48x1 | whatif | nxg2 cb3 chk48 listmax1024 | 221 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg2 cb3 chk48 listmax1024 | 218 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_48x1 | whatif | nxg2 cb3 chk48 listmax1024 | 168 | SPILL st32 ld48, STACK 64 |
| sm_89 | static_o0_48x1 | whatif | nxg2 cb3 chk48 listmax1024 | 168 | SPILL st32 ld48, STACK 64, LOW_OCC 4w |
| sm_90 | static_o0_48x1 | whatif | nxg2 cb3 chk48 listmax1024 | 216 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg2 cb3 chk64 listmax1024 | 244 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg2 cb3 chk64 listmax1024 | 254 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_48x1 | whatif | nxg2 cb3 chk64 listmax1024 | 239 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg2 cb3 chk64 listmax1024 | 239 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_48x1 | whatif | nxg2 cb3 chk64 listmax1024 | 247 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg4 cb2 chk16 listmax2048 | 72 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg4 cb2 chk16 listmax2048 | 72 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg4 cb2 chk16 listmax2048 | 94 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg4 cb2 chk16 listmax2048 | 72 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg4 cb2 chk16 listmax2048 | 72 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o0_48x1 | whatif | nxg4 cb2 chk16 listmax4096 | 72 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg4 cb2 chk16 listmax4096 | 72 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg4 cb2 chk16 listmax4096 | 94 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg4 cb2 chk16 listmax4096 | 72 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg4 cb2 chk16 listmax4096 | 72 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o0_48x1 | whatif | nxg4 cb2 chk48 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg4 cb2 chk48 listmax1024 | 78 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg4 cb2 chk48 listmax1024 | 117 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg4 cb2 chk48 listmax1024 | 80 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg4 cb2 chk48 listmax1024 | 108 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg4 cb2 chk64 listmax1024 | 113 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg4 cb2 chk64 listmax1024 | 111 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg4 cb2 chk64 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg4 cb2 chk64 listmax1024 | 115 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg4 cb2 chk64 listmax1024 | 119 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg4 cb3 chk16 listmax2048 | 80 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg4 cb3 chk16 listmax2048 | 72 | SPILL st4 ld4, STACK 40 |
| sm_80 | static_o0_48x1 | whatif | nxg4 cb3 chk16 listmax2048 | 114 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg4 cb3 chk16 listmax2048 | 111 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg4 cb3 chk16 listmax2048 | 114 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg4 cb3 chk16 listmax4096 | 80 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg4 cb3 chk16 listmax4096 | 72 | SPILL st4 ld4, STACK 40 |
| sm_80 | static_o0_48x1 | whatif | nxg4 cb3 chk16 listmax4096 | 114 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg4 cb3 chk16 listmax4096 | 111 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg4 cb3 chk16 listmax4096 | 114 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg4 cb3 chk48 listmax1024 | 119 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg4 cb3 chk48 listmax1024 | 117 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg4 cb3 chk48 listmax1024 | 150 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg4 cb3 chk48 listmax1024 | 147 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg4 cb3 chk48 listmax1024 | 151 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg4 cb3 chk64 listmax1024 | 125 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg4 cb3 chk64 listmax1024 | 122 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg4 cb3 chk64 listmax1024 | 158 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg4 cb3 chk64 listmax1024 | 128 | SPILL st4 ld20, STACK 40 |
| sm_90 | static_o0_48x1 | whatif | nxg4 cb3 chk64 listmax1024 | 147 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg4 cb6 chk16 listmax2048 | 120 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg4 cb6 chk16 listmax2048 | 119 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg4 cb6 chk16 listmax2048 | 146 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg4 cb6 chk16 listmax2048 | 122 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg4 cb6 chk16 listmax2048 | 118 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg4 cb6 chk16 listmax4096 | 120 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg4 cb6 chk16 listmax4096 | 119 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg4 cb6 chk16 listmax4096 | 146 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg4 cb6 chk16 listmax4096 | 122 | STACK 32 |
| sm_90 | static_o0_48x1 | whatif | nxg4 cb6 chk16 listmax4096 | 118 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg4 cb6 chk48 listmax1024 | 220 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg4 cb6 chk48 listmax1024 | 217 | STACK 32 |
| sm_80 | static_o0_48x1 | whatif | nxg4 cb6 chk48 listmax1024 | 168 | SPILL st12 ld28, STACK 48 |
| sm_89 | static_o0_48x1 | whatif | nxg4 cb6 chk48 listmax1024 | 168 | SPILL st12 ld28, STACK 48 |
| sm_90 | static_o0_48x1 | whatif | nxg4 cb6 chk48 listmax1024 | 214 | STACK 32 |
| sm_100 | static_o0_48x1 | whatif | nxg4 cb6 chk64 listmax1024 | 248 | STACK 32 |
| sm_120 | static_o0_48x1 | whatif | nxg4 cb6 chk64 listmax1024 | 246 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o0_48x1 | whatif | nxg4 cb6 chk64 listmax1024 | 239 | STACK 32 |
| sm_89 | static_o0_48x1 | whatif | nxg4 cb6 chk64 listmax1024 | 239 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o0_48x1 | whatif | nxg4 cb6 chk64 listmax1024 | 240 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 72 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 80 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 80 | SPILL st4 ld4, STACK 40 |
| sm_100 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 122 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 122 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 123 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 124 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 56 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 72 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 56 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 56 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 56 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 72 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 56 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 56 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 72 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 72 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 124 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 124 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 123 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 56 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 64 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 56 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 64 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 56 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 64 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 56 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 64 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 64 | SPILL st8 ld8, STACK 40 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 64 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 72 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 64 | SPILL st8 ld8, STACK 40 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 64 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 72 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 72 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 72 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 121 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 121 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 121 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 123 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 125 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 123 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 122 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st12 ld12, STACK 48 |
| sm_89 | static_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 61 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st8 ld8, STACK 40 |
| sm_90 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 73 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 157 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 153 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 164 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 163 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 164 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 84 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 77 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 84 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 84 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 81 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32, LOW_OCC 5w |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 71 | STACK 32, LOW_OCC 5w |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 3w |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 93 | STACK 32, LOW_OCC 1w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 1w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 3w |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 2w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 2w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 93 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 147 | STACK 32, LOW_OCC 3w |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32, LOW_OCC 1w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 168 | SPILL st4 ld4, STACK 40, LOW_OCC 3w |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 40 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 157 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 58 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 2w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 2w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 3w |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 1w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 168 | SPILL st4 ld4, STACK 40, LOW_OCC 3w |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 159 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 58 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 50 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 70 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st36 ld20, STACK 56 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 3w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 3w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st8 ld8, STACK 40 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 6w |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 3w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 147 | STACK 32, LOW_OCC 6w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 3w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st8 ld8, STACK 40 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 131 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 6w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32, LOW_OCC 6w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 147 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 51 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st36 ld20, STACK 56 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 4w |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 157 | STACK 32, LOW_OCC 2w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 58 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | SPILL st8 ld8, STACK 40, LOW_OCC 4w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 60 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 64 | SPILL st12 ld12, STACK 48 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 94 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 64 | SPILL st32 ld16, STACK 48 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 50 | STACK 32, LOW_OCC 5w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st4 ld4, STACK 40 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32, LOW_OCC 5w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st36 ld20, STACK 56 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 5w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 5w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 5w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 48 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 58 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 93 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 91 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 5w |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | SPILL st8 ld8, STACK 40, LOW_OCC 5w |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 5w |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 62 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 74 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 92 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 94 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 76 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 131 | STACK 32 |
| sm_120 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_80 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_89 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_90 | static_o2_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 147 | STACK 32 |
| sm_120 | static_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws1 vxpad0 | 180 | LOW_OCC 4w |
| sm_89 | static_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws1 vxpad0 | 168 | LOW_OCC 4w |
| sm_120 | static_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 | 236 | LOW_OCC 4w |
| sm_89 | static_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_120 | static_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 236 | LOW_OCC 4w |
| sm_80 | static_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_89 | static_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_100 | static_o2_cb5 | whatif | 4x4x16 smax128 nwc4 nws1 vxpad0 | 252 | LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | 4x4x16 smax128 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_100 | static_o2_cb5 | whatif | 4x4x16 smax96 nwc4 nws1 vxpad0 | 252 | LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | 4x4x16 smax96 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | 4x4x16 smax96 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_100 | static_o2_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 95 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 124 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 96 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg1 cb1 chk64 listmax1024 | 96 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg1 cb1 chk64 listmax1024 | 95 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg1 cb1 chk64 listmax1024 | 123 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o2_cb5 | whatif | nxg1 cb1 chk64 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg1 cb1 chk64 listmax1024 | 123 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 122 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 122 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 123 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 124 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 122 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 122 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 123 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 124 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 128 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 159 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg1 cb2 chk64 listmax1024 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg1 cb2 chk64 listmax1024 | 125 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg1 cb2 chk64 listmax1024 | 159 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o2_cb5 | whatif | nxg1 cb2 chk64 listmax1024 | 160 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg1 cb2 chk64 listmax1024 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 56 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 72 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 56 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 56 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 56 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 72 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 56 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 56 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 64 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 72 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 72 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 72 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 72 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 96 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 80 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 124 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 96 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 80 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 124 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o2_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 124 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 124 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 124 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 123 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 125 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 124 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 124 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 123 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 125 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 128 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 124 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 157 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg2 cb4 chk64 listmax1024 | 128 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg2 cb4 chk64 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg2 cb4 chk64 listmax1024 | 160 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o2_cb5 | whatif | nxg2 cb4 chk64 listmax1024 | 158 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg2 cb4 chk64 listmax1024 | 160 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 56 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 64 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 56 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 64 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 56 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 56 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 64 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 56 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 64 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 64 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 64 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 64 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 64 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 64 | SPILL st8 ld8, STACK 40 |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 64 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 72 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 64 | SPILL st8 ld8, STACK 40 |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 64 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 72 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 72 | SPILL st4 ld4, STACK 40 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 123 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 96 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 72 | SPILL st4 ld4, STACK 40 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 123 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 95 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 80 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 96 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 72 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 80 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 96 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 80 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 124 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 96 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 96 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 80 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 80 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 124 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 124 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 121 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 121 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 121 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 123 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 121 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 121 | STACK 32 |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 121 | STACK 32 |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 123 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 157 | STACK 32 |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | whatif | nxg4 cb7 chk64 listmax1024 | 125 | STACK 32 |
| sm_120 | static_o2_cb5 | whatif | nxg4 cb7 chk64 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | whatif | nxg4 cb7 chk64 listmax1024 | 157 | STACK 32, LOW_OCC 4w |
| sm_89 | static_o2_cb5 | whatif | nxg4 cb7 chk64 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | whatif | nxg4 cb7 chk64 listmax1024 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 95 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 93 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 121 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 96 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 96 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 118 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 121 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 158 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 156 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 158 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 159 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 160 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 156 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 160 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 159 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 161 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 239 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 235 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 239 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 238 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 236 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 241 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 242 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 238 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 239 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 237 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 121 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 96 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 96 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 119 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 120 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 154 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 123 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 155 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 125 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 125 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 156 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 154 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 157 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 158 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 156 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 158 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 159 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 159 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 159 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 204 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 204 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 227 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 226 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 227 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 236 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 234 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 232 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 228 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 232 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 187 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 196 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 219 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 218 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 222 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 201 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 200 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 224 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 221 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 225 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 121 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 96 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 96 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 119 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 128 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 123 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 153 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 159 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 157 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 125 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 160 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 158 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 158 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 78 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 80 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 119 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 119 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 125 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 125 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 123 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 161 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 159 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 240 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 238 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 233 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 231 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 234 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 242 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 236 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 236 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 235 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 240 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 124 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 124 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 154 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 124 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 124 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 160 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 123 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 158 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 159 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 197 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 196 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 221 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 219 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 221 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 201 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 200 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 224 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 223 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 225 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 161 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 161 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 212 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 209 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 214 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 160 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 161 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 216 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 213 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 219 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 95 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 127 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 119 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 122 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 125 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 122 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 123 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 158 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 83 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 87 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 92 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 87 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 83 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 92 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 94 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 94 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 92 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 121 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 95 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 92 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 94 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba10 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 120 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | SPILL st20 ld20, STACK 56 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st24 ld24, STACK 56 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 90 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 71 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 93 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 122 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 123 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 123 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st8 ld8, STACK 40 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | SPILL st40 ld40, STACK 56 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 94 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 75 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 90 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 71 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 78 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 94 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 79 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 94 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 77 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | SPILL st8 ld8, STACK 40 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 94 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 71 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 120 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 93 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 94 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 160 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 157 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 165 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 158 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 159 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 160 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 156 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 163 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 161 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 157 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | SPILL st40 ld40, STACK 56 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | SPILL st36 ld36, STACK 56 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 80 | SPILL st16 ld16, STACK 48 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 77 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 80 | SPILL st24 ld24, STACK 56 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 67 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 67 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 80 | SPILL st16 ld16, STACK 48 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 71 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 79 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 93 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 121 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 95 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 96 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 124 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba6 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st144 ld224, STACK 144 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st136 ld220, STACK 128 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 94 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 96 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 96 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 94 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 98 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 120 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 120 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 120 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 158 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 158 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 161 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 161 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 161 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 158 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 158 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 158 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 158 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_adj | tt4 cpba8 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 162 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32, LOW_OCC 5w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 79 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 94 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 79 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32, LOW_OCC 5w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32, LOW_OCC 5w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 79 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 94 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 79 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32, LOW_OCC 5w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 3w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 1w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 1w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 3w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32, LOW_OCC 3w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 1w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 127 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 1w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 3w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 115 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 115 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 168 | STACK 32, LOW_OCC 2w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 161 | STACK 32, LOW_OCC 1w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 155 | STACK 32, LOW_OCC 2w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 149 | STACK 32, LOW_OCC 2w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 147 | STACK 32, LOW_OCC 1w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 155 | STACK 32, LOW_OCC 2w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 55 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 70 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 157 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 157 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb2 fcb1 kv0 | 72 | SPILL st36 ld36, STACK 56, LOW_OCC 5w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb2 fcb1 kv0 | 71 | STACK 32, LOW_OCC 5w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb2 fcb1 kv1 | 72 | SPILL st36 ld36, STACK 56, LOW_OCC 5w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk32 acb1 fsb2 fcb1 kv1 | 71 | STACK 32, LOW_OCC 5w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 5w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 151 | STACK 32, LOW_OCC 5w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 5w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 5w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 151 | STACK 32, LOW_OCC 5w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 5w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv0 | 61 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv0 | 61 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv1 | 64 | SPILL st4 ld4, STACK 40 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv1 | 64 | SPILL st32 ld32, STACK 48 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk24 acb1 fsb1 fcb1 kv1 | 64 | SPILL st12 ld12, STACK 48 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 5w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 125 | STACK 32, LOW_OCC 5w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 5w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk32 acb1 fsb4 fcb1 kv1 | 125 | STACK 32, LOW_OCC 5w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv0 | 82 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv1 | 82 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk48 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 131 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 147 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 131 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 143 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 147 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv0 | 61 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv0 | 61 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv1 | 64 | SPILL st4 ld4, STACK 40 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv1 | 64 | SPILL st32 ld32, STACK 48 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk12 acb1 fsb1 fcb1 kv1 | 64 | SPILL st12 ld12, STACK 48 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv0 | 82 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv0 | 82 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv0 | 76 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv1 | 82 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv1 | 82 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv1 | 76 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk24 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk32 acb1 fsb4 fcb1 kv0 | 131 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk32 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk32 acb1 fsb4 fcb1 kv1 | 131 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk32 acb1 fsb4 fcb1 kv1 | 143 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf10 ppf4 chunk48 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 66 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 66 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 80 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 115 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 115 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 2w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 159 | STACK 32, LOW_OCC 1w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 153 | STACK 32, LOW_OCC 2w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 149 | STACK 32, LOW_OCC 2w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 143 | STACK 32, LOW_OCC 1w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 153 | STACK 32, LOW_OCC 2w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 70 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 157 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 157 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 60 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 64 | SPILL st32 ld32, STACK 48 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 64 | SPILL st32 ld32, STACK 48 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 71 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 151 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 151 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 3w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st24 ld24, STACK 56 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 3w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 6w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | SPILL st32 ld32, STACK 48, LOW_OCC 3w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 6w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 6w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 72 | SPILL st32 ld32, STACK 48, LOW_OCC 3w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 6w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | SPILL st36 ld36, STACK 56 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 76 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 76 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 71 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 3w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 153 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 6w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 3w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 153 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 73 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 100 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 100 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 131 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 131 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 145 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 145 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 147 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 131 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 131 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 145 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 145 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 147 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 157 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 157 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 60 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 64 | SPILL st32 ld32, STACK 48 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st8 ld8, STACK 40 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 64 | SPILL st32 ld32, STACK 48 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | SPILL st8 ld8, STACK 40 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 71 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 151 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 151 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | SPILL st36 ld36, STACK 56 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 58 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st24 ld24, STACK 56 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 73 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb2 fcb1 kv0 | 71 | STACK 32, LOW_OCC 3w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk32 acb1 fsb2 fcb1 kv1 | 71 | STACK 32, LOW_OCC 3w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 76 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 76 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 151 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 151 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 73 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb4 fcb1 kv0 | 125 | STACK 32, LOW_OCC 3w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk32 acb1 fsb4 fcb1 kv1 | 125 | STACK 32, LOW_OCC 3w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk48 acb1 fsb2 fcb1 kv0 | 72 | SPILL st36 ld36, STACK 56 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk48 acb1 fsb2 fcb1 kv0 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk48 acb1 fsb2 fcb1 kv1 | 72 | SPILL st36 ld36, STACK 56 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk48 acb1 fsb2 fcb1 kv1 | 72 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 100 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 100 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 131 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 145 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 147 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 131 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 145 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 147 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 61 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 61 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv1 | 64 | SPILL st4 ld4, STACK 40 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv1 | 64 | SPILL st44 ld44, STACK 64 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk20 acb1 fsb1 fcb1 kv1 | 64 | SPILL st40 ld40, STACK 56 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 82 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 82 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 76 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv1 | 82 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv1 | 82 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv1 | 74 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk40 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 129 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 145 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk48 acb1 fsb4 fcb1 kv1 | 129 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk48 acb1 fsb4 fcb1 kv1 | 145 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 131 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 145 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 147 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 131 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 145 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf6 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 147 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st4 ld4, STACK 40 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st4 ld4, STACK 40 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | SPILL st4 ld4, STACK 40 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | SPILL st36 ld36, STACK 56 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk48 acb1 fsb2 fcb1 kv0 | 78 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk48 acb1 fsb2 fcb1 kv0 | 71 | STACK 32, LOW_OCC 6w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk48 acb1 fsb2 fcb1 kv1 | 78 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk48 acb1 fsb2 fcb1 kv1 | 71 | STACK 32, LOW_OCC 6w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk48 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk48 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 3w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk48 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk48 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 3w |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 76 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 76 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | SPILL st8 ld8, STACK 40 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 151 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 151 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | SPILL st36 ld36, STACK 56 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk48 acb1 fsb4 fcb1 kv0 | 129 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk48 acb1 fsb4 fcb1 kv0 | 143 | STACK 32, LOW_OCC 6w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk48 acb1 fsb4 fcb1 kv1 | 129 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk48 acb1 fsb4 fcb1 kv1 | 143 | STACK 32, LOW_OCC 6w |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk52 acb1 fsb2 fcb1 kv0 | 72 | SPILL st36 ld36, STACK 56 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk52 acb1 fsb2 fcb1 kv0 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk52 acb1 fsb2 fcb1 kv1 | 72 | SPILL st36 ld36, STACK 56 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk52 acb1 fsb2 fcb1 kv1 | 72 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv1 | 64 | SPILL st36 ld36, STACK 56 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk16 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk32 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 129 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk48 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk48 acb1 fsb4 fcb1 kv1 | 129 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk48 acb1 fsb4 fcb1 kv1 | 143 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | tile_fwd | tt4 cpba1 cpbf8 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | vt_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 174 | LOW_OCC 7w |
| sm_120 | vel2d_o3_cb10 | vt_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 190 | LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 198 | LOW_OCC 7w |
| sm_89 | vel2d_o3_cb10 | vt_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 198 | LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 178 | LOW_OCC 7w |
| sm_100 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 174 | LOW_OCC 7w |
| sm_120 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 190 | LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 198 | LOW_OCC 7w |
| sm_89 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 198 | LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 178 | LOW_OCC 7w |
| sm_100 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 174 | LOW_OCC 7w |
| sm_120 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 190 | LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 198 | LOW_OCC 7w |
| sm_89 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 198 | LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 178 | LOW_OCC 7w |
| sm_100 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 | 249 | LOW_OCC 7w |
| sm_120 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 | 250 | LOW_OCC 7w |
| sm_89 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 | 252 | LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_100 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 249 | LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 250 | LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_100 | vel2d_o3_cb10 | whatif | 4x4x16 smax128 nwc5 nws1 vxpad0 | 254 | LOW_OCC 5w |
| sm_90 | vel2d_o3_cb10 | whatif | 4x4x16 smax128 nwc5 nws1 vxpad0 | 254 | LOW_OCC 5w |
| sm_80 | vel2d_o3_cb10 | whatif | 4x4x16 smax96 nwc5 nws1 vxpad0 | 254 | LOW_OCC 5w |
| sm_100 | vel2d_o3_cb10 | whatif | 4x4x16 smax96 nwc7 nws1 vxpad0 | 249 | LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | whatif | 4x4x16 smax96 nwc7 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_100 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk16 listmax2048 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk16 listmax2048 | 78 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk16 listmax2048 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk16 listmax2048 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk16 listmax2048 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk16 listmax4096 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk16 listmax4096 | 78 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk16 listmax4096 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk16 listmax4096 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk16 listmax4096 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk48 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk48 listmax1024 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk48 listmax1024 | 127 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk48 listmax1024 | 121 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk48 listmax1024 | 120 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk64 listmax1024 | 96 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk64 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg1 cb1 chk64 listmax1024 | 122 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk16 listmax2048 | 125 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk16 listmax2048 | 125 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk16 listmax2048 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk16 listmax2048 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk16 listmax2048 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk16 listmax4096 | 125 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk16 listmax4096 | 125 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk16 listmax4096 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk16 listmax4096 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk16 listmax4096 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk48 listmax1024 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk48 listmax1024 | 125 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk48 listmax1024 | 166 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk48 listmax1024 | 162 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk48 listmax1024 | 162 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk64 listmax1024 | 126 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk64 listmax1024 | 164 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg1 cb2 chk64 listmax1024 | 157 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk16 listmax2048 | 240 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk16 listmax2048 | 238 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk16 listmax2048 | 233 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk16 listmax2048 | 231 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk16 listmax2048 | 234 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk16 listmax4096 | 240 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk16 listmax4096 | 238 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk16 listmax4096 | 233 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk16 listmax4096 | 231 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk16 listmax4096 | 234 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk48 listmax1024 | 233 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk48 listmax1024 | 239 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk48 listmax1024 | 242 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk64 listmax1024 | 240 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk64 listmax1024 | 242 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg1 cb4 chk64 listmax1024 | 246 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk16 listmax2048 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk16 listmax2048 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk16 listmax2048 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk16 listmax2048 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk16 listmax2048 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk16 listmax4096 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk16 listmax4096 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk16 listmax4096 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk16 listmax4096 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk16 listmax4096 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk48 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk48 listmax1024 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk48 listmax1024 | 127 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk48 listmax1024 | 119 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk48 listmax1024 | 120 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk64 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk64 listmax1024 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk64 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk64 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb2 chk64 listmax1024 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk16 listmax2048 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk16 listmax2048 | 124 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk16 listmax2048 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk16 listmax2048 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk16 listmax2048 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk16 listmax4096 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk16 listmax4096 | 124 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk16 listmax4096 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk16 listmax4096 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk16 listmax4096 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk48 listmax1024 | 125 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk48 listmax1024 | 125 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk48 listmax1024 | 158 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk48 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk48 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk64 listmax1024 | 125 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk64 listmax1024 | 158 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb3 chk64 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk16 listmax2048 | 124 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk16 listmax2048 | 124 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk16 listmax2048 | 160 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk16 listmax2048 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk16 listmax2048 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk16 listmax4096 | 124 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk16 listmax4096 | 124 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk16 listmax4096 | 160 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk16 listmax4096 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk16 listmax4096 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk48 listmax1024 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk48 listmax1024 | 125 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk48 listmax1024 | 167 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk48 listmax1024 | 155 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk48 listmax1024 | 156 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk64 listmax1024 | 126 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk64 listmax1024 | 166 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb4 chk64 listmax1024 | 156 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk16 listmax2048 | 197 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk16 listmax2048 | 196 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk16 listmax2048 | 221 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk16 listmax2048 | 219 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk16 listmax2048 | 221 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk16 listmax4096 | 197 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk16 listmax4096 | 196 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk16 listmax4096 | 221 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk16 listmax4096 | 219 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk16 listmax4096 | 221 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk48 listmax1024 | 243 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk48 listmax1024 | 236 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk48 listmax1024 | 239 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk64 listmax1024 | 241 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk64 listmax1024 | 237 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg2 cb7 chk64 listmax1024 | 241 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk16 listmax2048 | 161 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk16 listmax2048 | 161 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk16 listmax2048 | 212 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk16 listmax2048 | 209 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk16 listmax2048 | 214 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk16 listmax4096 | 161 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk16 listmax4096 | 161 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk16 listmax4096 | 212 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk16 listmax4096 | 209 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk16 listmax4096 | 214 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk48 listmax1024 | 238 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk48 listmax1024 | 232 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk48 listmax1024 | 228 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk48 listmax1024 | 225 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk48 listmax1024 | 228 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk64 listmax1024 | 237 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk64 listmax1024 | 234 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb13 chk64 listmax1024 | 235 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk16 listmax2048 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk16 listmax2048 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk16 listmax2048 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk16 listmax2048 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk16 listmax2048 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk16 listmax4096 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk16 listmax4096 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk16 listmax4096 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk16 listmax4096 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk16 listmax4096 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk48 listmax1024 | 121 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk48 listmax1024 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk48 listmax1024 | 127 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk48 listmax1024 | 119 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk48 listmax1024 | 120 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk64 listmax1024 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk64 listmax1024 | 123 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk64 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk64 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb4 chk64 listmax1024 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk16 listmax2048 | 95 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk16 listmax2048 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk16 listmax2048 | 127 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk16 listmax2048 | 119 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk16 listmax2048 | 122 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk16 listmax4096 | 95 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk16 listmax4096 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk16 listmax4096 | 127 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk16 listmax4096 | 119 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk16 listmax4096 | 122 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk48 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk48 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk48 listmax1024 | 154 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk48 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk48 listmax1024 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk64 listmax1024 | 126 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk64 listmax1024 | 125 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk64 listmax1024 | 152 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk64 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb5 chk64 listmax1024 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk16 listmax2048 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk16 listmax2048 | 122 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk16 listmax2048 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk16 listmax2048 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk16 listmax2048 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk16 listmax4096 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk16 listmax4096 | 122 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk16 listmax4096 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk16 listmax4096 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk16 listmax4096 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk48 listmax1024 | 124 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk48 listmax1024 | 124 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk48 listmax1024 | 162 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk48 listmax1024 | 164 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk48 listmax1024 | 160 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk64 listmax1024 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk64 listmax1024 | 163 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | whatif | nxg4 cb7 chk64 listmax1024 | 158 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 95 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 93 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 121 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 96 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 96 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 118 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 121 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 157 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 156 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 158 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 159 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 158 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 159 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 160 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 159 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 161 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 72 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 72 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 119 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 93 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 94 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 72 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 72 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 119 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 93 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 94 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 121 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 96 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 96 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 119 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 156 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 158 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 157 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 158 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 158 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 156 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 158 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 159 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 159 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 159 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 72 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 71 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 120 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 80 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 91 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 72 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 71 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 120 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 80 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 91 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 80 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 80 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 123 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 80 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 80 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 123 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 123 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 121 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 96 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 96 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 119 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 119 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 157 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 160 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 157 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 125 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 158 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 157 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 160 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 78 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 80 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 119 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 119 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 123 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 126 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 161 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 159 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 119 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 79 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 119 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 79 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 79 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 121 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 160 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 126 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 157 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 158 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 159 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 117 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 72 | SPILL st12 ld28, STACK 48 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 72 | SPILL st12 ld12, STACK 48 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 117 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 72 | SPILL st12 ld28, STACK 48 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 72 | SPILL st12 ld12, STACK 48 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 71 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 71 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 120 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 71 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 71 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 120 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 122 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 123 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 160 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | SPILL st20 ld20, STACK 56 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st24 ld24, STACK 56 |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 90 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 71 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 93 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 122 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 123 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 123 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba2 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st8 ld8, STACK 40 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | SPILL st40 ld40, STACK 56 |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 94 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 75 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 90 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 71 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba3 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 78 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 94 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 79 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 94 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 77 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | SPILL st8 ld8, STACK 40 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 94 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 71 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 120 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 93 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv0 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 94 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 122 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb2 fsb1 fcb1 kv1 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 160 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 157 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 165 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 158 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv0 | 159 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 160 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 156 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 163 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 161 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba4 cpbf1 ppf2 chunk32 acb4 fsb1 fcb1 kv1 | 157 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 84 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 82 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 90 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 91 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 85 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 85 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 94 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 92 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_adj | tt4 cpba5 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32, LOW_OCC 5w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 79 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 94 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 79 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32, LOW_OCC 5w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32, LOW_OCC 5w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 79 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 94 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 79 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32, LOW_OCC 5w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 3w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 1w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 1w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 3w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32, LOW_OCC 3w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 1w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 127 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 1w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 3w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 115 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 115 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 168 | STACK 32, LOW_OCC 2w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 161 | STACK 32, LOW_OCC 1w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 155 | STACK 32, LOW_OCC 2w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 149 | STACK 32, LOW_OCC 2w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 147 | STACK 32, LOW_OCC 1w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 155 | STACK 32, LOW_OCC 2w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 55 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 70 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 157 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 157 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf1 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 66 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 80 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 66 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 80 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 115 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 115 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 2w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 159 | STACK 32, LOW_OCC 1w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 153 | STACK 32, LOW_OCC 2w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 149 | STACK 32, LOW_OCC 2w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 143 | STACK 32, LOW_OCC 1w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 167 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 143 | STACK 32, LOW_OCC 1w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 153 | STACK 32, LOW_OCC 2w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 70 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 157 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 157 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 60 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 64 | SPILL st32 ld32, STACK 48 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 64 | SPILL st32 ld32, STACK 48 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 71 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 151 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 151 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf2 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 3w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | SPILL st24 ld24, STACK 56 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 3w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 6w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | SPILL st32 ld32, STACK 48, LOW_OCC 3w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32, LOW_OCC 6w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 6w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 72 | SPILL st32 ld32, STACK 48, LOW_OCC 3w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32, LOW_OCC 6w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | SPILL st36 ld36, STACK 56 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 76 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 76 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 71 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 3w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 153 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 6w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 3w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 153 | STACK 32, LOW_OCC 3w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 3w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 6w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 72 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 58 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 73 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 100 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 100 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 131 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 131 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 145 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 145 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 147 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 131 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 131 | STACK 32, LOW_OCC 6w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 145 | STACK 32, LOW_OCC 6w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 145 | STACK 32, LOW_OCC 6w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf3 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 147 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 79 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 157 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 2w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 157 | STACK 32, LOW_OCC 2w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 2w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf1 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 60 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 64 | SPILL st32 ld32, STACK 48 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | SPILL st8 ld8, STACK 40 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 71 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 64 | SPILL st32 ld32, STACK 48 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | SPILL st8 ld8, STACK 40 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 71 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 151 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 151 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | SPILL st36 ld36, STACK 56 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 80 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf4 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 5w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32, LOW_OCC 5w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv0 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 5w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 72 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32, LOW_OCC 5w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk32 acb1 fsb1 fcb1 kv1 | 56 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32, LOW_OCC 5w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32, LOW_OCC 5w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 72 | STACK 32, LOW_OCC 5w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv0 | 86 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32, LOW_OCC 5w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32, LOW_OCC 5w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 72 | STACK 32, LOW_OCC 5w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf1 chunk64 acb1 fsb2 fcb1 kv1 | 86 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 60 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 66 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk32 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv0 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb2 fcb1 kv1 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 127 | STACK 32, LOW_OCC 5w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 151 | STACK 32, LOW_OCC 5w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32, LOW_OCC 5w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv0 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 127 | STACK 32, LOW_OCC 5w |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 151 | STACK 32, LOW_OCC 5w |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32, LOW_OCC 5w |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf2 chunk64 acb1 fsb4 fcb1 kv1 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 61 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 61 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 63 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv0 | 64 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv1 | 64 | SPILL st32 ld32, STACK 48 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv1 | 64 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk24 acb1 fsb1 fcb1 kv1 | 64 | SPILL st12 ld12, STACK 48 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 82 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 82 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 76 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv0 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv1 | 82 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv1 | 82 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv1 | 74 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk48 acb1 fsb2 fcb1 kv1 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 131 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 131 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 143 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv0 | 147 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 131 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 131 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 143 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 143 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | tile_fwd | tt4 cpba1 cpbf5 ppf4 chunk64 acb1 fsb4 fcb1 kv1 | 147 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws1 vxpad0 | 168 | SPILL st8 ld8, STACK 8 |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws1 vxpad0 | 168 | SPILL st8 ld8, STACK 8, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws1 vxpad0 | 178 | LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws1 vxpad0 | 168 | SPILL st8 ld8, STACK 8 |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws1 vxpad0 | 168 | SPILL st8 ld8, STACK 8, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws1 vxpad0 | 178 | LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws1 vxpad0 | 168 | SPILL st8 ld8, STACK 8 |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws1 vxpad0 | 168 | SPILL st8 ld8, STACK 8, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws1 vxpad0 | 178 | LOW_OCC 4w |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 | 236 | LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 236 | LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | whatif | 4x4x16 smax128 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | 4x4x16 smax128 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | whatif | 4x4x16 smax96 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | 4x4x16 smax96 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | 4x4x16 smax96 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 78 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk16 listmax2048 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 79 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 78 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk16 listmax4096 | 95 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 127 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 121 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk48 listmax1024 | 120 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk64 listmax1024 | 96 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk64 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg1 cb1 chk64 listmax1024 | 122 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 123 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk16 listmax2048 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 123 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk16 listmax4096 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 125 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 159 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 159 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk48 listmax1024 | 161 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk64 listmax1024 | 122 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk64 listmax1024 | 166 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg1 cb2 chk64 listmax1024 | 162 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 119 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 79 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk16 listmax2048 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 119 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 79 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk16 listmax4096 | 80 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 71 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 70 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 120 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 94 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk48 listmax1024 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 71 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 70 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 120 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 94 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg2 cb1 chk64 listmax1024 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk16 listmax2048 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk16 listmax4096 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 127 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 119 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk48 listmax1024 | 120 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg2 cb2 chk64 listmax1024 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 121 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 160 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk16 listmax2048 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 121 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 160 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk16 listmax4096 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 125 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 162 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 162 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk48 listmax1024 | 159 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk64 listmax1024 | 122 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk64 listmax1024 | 167 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg2 cb4 chk64 listmax1024 | 156 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 117 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 72 | SPILL st12 ld28, STACK 48 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk16 listmax2048 | 72 | SPILL st12 ld12, STACK 48 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 64 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 64 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 117 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 72 | SPILL st12 ld28, STACK 48 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk16 listmax4096 | 72 | SPILL st12 ld12, STACK 48 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 64 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 119 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 94 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk48 listmax1024 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 64 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 119 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 94 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb2 chk64 listmax1024 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 71 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 71 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 120 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk16 listmax2048 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 71 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 71 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 120 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk16 listmax4096 | 94 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 118 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk48 listmax1024 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 72 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 72 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 124 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 118 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb3 chk64 listmax1024 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk16 listmax2048 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 78 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 72 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 93 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk16 listmax4096 | 96 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 121 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 127 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 119 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk48 listmax1024 | 120 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 123 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb4 chk64 listmax1024 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 122 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk16 listmax2048 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 122 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 156 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk16 listmax4096 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 126 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 159 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 159 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk48 listmax1024 | 159 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk64 listmax1024 | 124 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk64 listmax1024 | 162 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | whatif | nxg4 cb7 chk64 listmax1024 | 161 | STACK 32 |
