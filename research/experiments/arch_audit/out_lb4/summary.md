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
| nxg1 cb1 chk16 | 128 | 24000 | 119/16 (regs) K48 | 121/16 (regs+smem) K48 | 122/16 (regs) K48 | 119/16 (regs) K48 | 113/16 (regs+smem) K48 |
| nxg1 cb1 chk32 | 128 | 48000 | 125/12 (smem) K48 | 123/8 (smem) K48 | 124/16 (regs+smem) K48 | 123/16 (regs+smem) K48 | 119/8 (smem) K48 |
| nxg1 cb2 chk16 | 128 | 24000 | 128/16 (regs) K48 | 128/16 (regs+smem) K48 | 128/16 (regs) K48 | 127/16 (regs) K48 | 126/16 (regs+smem) K48 |
| nxg1 cb2 chk32 | 128 | 48000 | 128/12 (smem) K48 | 128/8 (smem) K48 | 128/16 (regs+smem) K48 | 128/16 (regs+smem) K48 | 126/8 (smem) K48 |
| nxg2 cb1 chk16 | 128 | 19648 | 102/16 (regs) K48 | 105/16 (regs+smem) K48 | 104/16 (regs) K48 | 111/16 (regs) K48 | 107/16 (regs+smem) K48 |
| nxg2 cb1 chk32 | 128 | 39296 | 102/16 (regs+smem) K48 | 105/8 (smem) K48 | 104/16 (regs) K48 | 111/16 (regs) K48 | 107/8 (smem) K48 |
| nxg2 cb2 chk16 | 128 | 19904 | 117/16 (regs) K48 | 119/16 (regs+smem) K48 | 120/16 (regs) K48 | 119/16 (regs) K48 | 117/16 (regs+smem) K48 |
| nxg2 cb2 chk32 | 128 | 39808 | 125/16 (regs+smem) K48 | 123/8 (smem) K48 | 124/16 (regs) K48 | 123/16 (regs) K48 | 117/8 (smem) K48 |
| nxg2 cb4 chk16 | 128 | 24000 | 128/16 (regs) K48 | 128/16 (regs+smem) K48 | 128/16 (regs) K48 | 127/16 (regs) K48 | 126/16 (regs+smem) K48 |
| nxg2 cb4 chk32 | 128 | 48000 | 127/12 (smem) K48 | 128/8 (smem) K48 | 128/16 (regs+smem) K48 | 128/16 (regs+smem) K48 | 126/8 (smem) K48 |
| nxg4 cb2 chk16 | 128 | 17600 | 99/16 (regs) K48 | 99/16 (regs) K48 | 101/16 (regs) K48 | 116/16 (regs) K48 | 109/16 (regs) K48 |
| nxg4 cb2 chk32 | 128 | 35200 | 99/16 (regs+smem) K48 | 99/8 (smem) K48 | 101/16 (regs) K48 | 116/16 (regs) K48 | 109/8 (smem) K48 |
| nxg4 cb3 chk16 | 128 | 19776 | 114/16 (regs) K48 | 113/16 (regs+smem) K48 | 115/16 (regs) K48 | 112/16 (regs) K48 | 111/16 (regs+smem) K48 |
| nxg4 cb3 chk32 | 128 | 39552 | 114/16 (regs+smem) K48 | 113/8 (smem) K48 | 115/16 (regs) K48 | 112/16 (regs) K48 | 111/8 (smem) K48 |
| nxg4 cb4 chk16 | 128 | 19904 | 117/16 (regs) K48 | 119/16 (regs+smem) K48 | 121/16 (regs) K48 | 117/16 (regs) K48 | 112/16 (regs+smem) K48 |
| nxg4 cb4 chk32 | 128 | 39808 | 125/16 (regs+smem) K48 | 123/8 (smem) K48 | 122/16 (regs) K48 | 122/16 (regs) K48 | 117/8 (smem) K48 |
| nxg4 cb7 chk16 | 128 | 24000 | 128/16 (regs) K48 | 128/16 (regs+smem) K48 | 128/16 (regs) K48 | 126/16 (regs) K48 | 125/16 (regs+smem) K48 |
| nxg4 cb7 chk32 | 128 | 48000 | 128/12 (smem) K48 | 128/8 (smem) K48 | 128/16 (regs+smem) K48 | 124/16 (regs+smem) K48 | 124/8 (smem) K48 |

### knot1d_o2_cb5 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 28032 | 119/16 (regs) K32 | 119/12 (smem) K32 | 120/16 (regs) K32 | 118/16 (regs) K32 | 114/12 (smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 51968 | 123/12 (smem) K32 | 124/4 (smem) K32 | 122/16 (regs+smem) K32 | 120/16 (regs+smem) K32 | 119/4 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 28032 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 127/16 (regs) K32 | 127/12 (smem) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 51968 | 128/12 (smem) K32 | 128/4 (smem) K32 | 128/16 (regs+smem) K32 | 128/16 (regs+smem) K32 | 128/4 (smem) K32 |
| nxg2 cb1 chk16 listmax1024 | 128 | 23680 | 101/16 (regs) K32 | 102/16 (regs+smem) K32 | 103/16 (regs) K32 | 112/16 (regs) K32 | 112/16 (regs+smem) K32 |
| nxg2 cb1 chk32 listmax1024 | 128 | 43264 | 101/12 (smem) K32 | 102/8 (smem) K32 | 103/16 (regs) K32 | 112/16 (regs) K32 | 112/8 (smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 23936 | 114/16 (regs) K32 | 115/16 (regs+smem) K32 | 118/16 (regs) K32 | 120/16 (regs) K32 | 119/16 (regs+smem) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 43776 | 122/12 (smem) K32 | 124/8 (smem) K32 | 123/16 (regs) K32 | 120/16 (regs) K32 | 118/8 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 28032 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 127/16 (regs) K32 | 127/12 (smem) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 51968 | 128/12 (smem) K32 | 128/4 (smem) K32 | 128/16 (regs+smem) K32 | 128/16 (regs+smem) K32 | 128/4 (smem) K32 |
| nxg4 cb2 chk16 listmax1024 | 128 | 21632 | 96/20 (regs) K32 | 96/16 (smem) K32 | 103/16 (regs) K32 | 111/16 (regs) K32 | 111/16 (regs+smem) K32 |
| nxg4 cb2 chk32 listmax1024 | 128 | 39168 | 96/16 (smem) K32 | 96/8 (smem) K32 | 103/16 (regs) K32 | 111/16 (regs) K32 | 111/8 (smem) K32 |
| nxg4 cb3 chk16 listmax1024 | 128 | 23808 | 112/16 (regs) K32 | 112/16 (regs+smem) K32 | 112/16 (regs) K32 | 116/16 (regs) K32 | 114/16 (regs+smem) K32 |
| nxg4 cb3 chk32 listmax1024 | 128 | 43520 | 112/12 (smem) K32 | 112/8 (smem) K32 | 112/16 (regs) K32 | 116/16 (regs) K32 | 114/8 (smem) K32 |
| nxg4 cb4 chk16 listmax1024 | 128 | 23936 | 114/16 (regs) K32 | 115/16 (regs+smem) K32 | 117/16 (regs) K32 | 119/16 (regs) K32 | 116/16 (regs+smem) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 43776 | 122/12 (smem) K32 | 124/8 (smem) K32 | 127/16 (regs) K32 | 120/16 (regs) K32 | 119/8 (smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 28032 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 124/16 (regs) K32 | 124/12 (smem) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 51968 | 128/12 (smem) K32 | 128/4 (smem) K32 | 128/16 (regs+smem) K32 | 125/16 (regs+smem) K32 | 125/4 (smem) K32 |

### static_o0_10x10 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 15808 | 104/16 (regs) K32 | 105/16 (regs) K32 | 107/16 (regs) K32 | 108/16 (regs) K32 | 102/16 (regs) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 27520 | 109/16 (regs) K32 | 110/12 (smem) K32 | 112/16 (regs) K32 | 108/16 (regs) K32 | 104/12 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 16192 | 126/16 (regs) K32 | 119/16 (regs) K32 | 119/16 (regs) K32 | 110/16 (regs) K32 | 110/16 (regs) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 28288 | 128/16 (regs) K32 | 119/12 (smem) K32 | 120/16 (regs) K32 | 111/16 (regs) K32 | 111/12 (smem) K32 |
| nxg1 cb4 chk16 listmax1024 | 128 | 24640 | 128/16 (regs) S328 K224 | 128/12 (smem) S328 K224 | 128/16 (regs) S432 K248 | 128/16 (regs) S384 K216 | 128/12 (smem) S384 K216 |
| nxg1 cb4 chk32 listmax1024 | 128 | 45184 | 128/12 (smem) S344 K224 | 128/8 (smem) S344 K224 | 128/16 (regs) S452 K264 | 128/16 (regs) S432 K240 | 128/8 (smem) S432 K240 |
| nxg2 cb2 chk16 listmax1024 | 128 | 11712 | 100/16 (regs) K32 | 101/16 (regs) K32 | 104/16 (regs) K32 | 108/16 (regs) K32 | 100/16 (regs) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 19328 | 107/16 (regs) K32 | 108/16 (regs) K32 | 110/16 (regs) K32 | 107/16 (regs) K32 | 102/16 (regs) K32 |
| nxg2 cb3 chk16 listmax1024 | 128 | 15936 | 122/16 (regs) K32 | 123/16 (regs) K32 | 121/16 (regs) K32 | 112/16 (regs) K32 | 108/16 (regs) K32 |
| nxg2 cb3 chk32 listmax1024 | 128 | 27776 | 126/16 (regs) K32 | 125/12 (smem) K32 | 123/16 (regs) K32 | 114/16 (regs) K32 | 110/12 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 16192 | 128/16 (regs) K32 | 120/16 (regs) K32 | 120/16 (regs) K32 | 116/16 (regs) K32 | 115/16 (regs) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 28288 | 128/16 (regs) K32 | 120/12 (smem) K32 | 121/16 (regs) K32 | 116/16 (regs) K32 | 115/12 (smem) K32 |
| nxg2 cb7 chk16 listmax1024 | 128 | 24640 | 128/16 (regs) S212 K176 | 128/12 (smem) S212 K176 | 128/16 (regs) S220 K176 | 128/16 (regs) S232 K176 | 128/12 (smem) S228 K176 |
| nxg2 cb7 chk32 listmax1024 | 128 | 45184 | 128/12 (smem) S244 K200 | 128/8 (smem) S244 K200 | 128/16 (regs) S292 K232 | 128/16 (regs) S260 K192 | 128/8 (smem) S256 K192 |
| nxg4 cb13 chk16 listmax1024 | 128 | 22592 | 128/16 (regs) S260 K208 | 128/16 (regs+smem) S260 K208 | 128/16 (regs) S316 K240 | 128/16 (regs) S276 K208 | 128/16 (regs+smem) S272 K208 |
| nxg4 cb13 chk32 listmax1024 | 128 | 41088 | 128/12 (smem) S300 K240 | 128/8 (smem) S300 K240 | 128/16 (regs) S348 K264 | 128/16 (regs) S300 K224 | 128/8 (smem) S296 K224 |
| nxg4 cb4 chk16 listmax1024 | 128 | 11712 | 100/16 (regs) K32 | 101/16 (regs) K32 | 104/16 (regs) K32 | 107/16 (regs) K32 | 100/16 (regs) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 19328 | 107/16 (regs) K32 | 108/16 (regs) K32 | 110/16 (regs) K32 | 108/16 (regs) K32 | 104/16 (regs) K32 |
| nxg4 cb5 chk16 listmax1024 | 128 | 13760 | 116/16 (regs) K32 | 117/16 (regs) K32 | 114/16 (regs) K32 | 109/16 (regs) K32 | 106/16 (regs) K32 |
| nxg4 cb5 chk32 listmax1024 | 128 | 23424 | 120/16 (regs) K32 | 120/16 (regs+smem) K32 | 119/16 (regs) K32 | 110/16 (regs) K32 | 107/16 (regs+smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 16064 | 128/16 (regs) S8 K40 | 125/16 (regs) K32 | 122/16 (regs) K32 | 119/16 (regs) K32 | 116/16 (regs) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 28032 | 128/16 (regs) S12 K48 | 124/12 (smem) K32 | 123/16 (regs) K32 | 119/16 (regs) K32 | 117/12 (smem) K32 |

### static_o0_48x1 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 18240 | 122/16 (regs) K32 | 122/16 (regs) K32 | 121/16 (regs) K32 | 112/16 (regs) K32 | 110/16 (regs) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 32384 | 128/16 (regs) K32 | 121/12 (smem) K32 | 122/16 (regs) K32 | 117/16 (regs) K32 | 117/12 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 20160 | 128/16 (regs) K32 | 128/16 (regs+smem) K32 | 128/16 (regs) S8 K40 | 122/16 (regs) K32 | 123/16 (regs+smem) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 36224 | 128/16 (regs+smem) S40 K80 | 128/8 (smem) S48 K88 | 128/16 (regs) S72 K112 | 128/16 (regs) K32 | 128/8 (smem) K32 |
| nxg2 cb1 chk16 listmax1024 | 128 | 12096 | 97/16 (regs) K32 | 96/20 (regs) K32 | 99/16 (regs) K32 | 100/16 (regs) K32 | 97/16 (regs) K32 |
| nxg2 cb1 chk32 listmax1024 | 128 | 20096 | 107/16 (regs) K32 | 107/16 (regs+smem) K32 | 109/16 (regs) K32 | 101/16 (regs) K32 | 98/16 (regs+smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 14144 | 119/16 (regs) K32 | 119/16 (regs) K32 | 118/16 (regs) K32 | 110/16 (regs) K32 | 107/16 (regs) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 24192 | 128/16 (regs) K32 | 121/16 (regs+smem) K32 | 125/16 (regs) K32 | 121/16 (regs) K32 | 117/16 (regs+smem) K32 |
| nxg2 cb3 chk16 listmax1024 | 128 | 20160 | 128/16 (regs) K32 | 126/16 (regs+smem) K32 | 127/16 (regs) K32 | 118/16 (regs) K32 | 116/16 (regs+smem) K32 |
| nxg2 cb3 chk32 listmax1024 | 128 | 36224 | 128/16 (regs+smem) S20 K56 | 128/8 (smem) S20 K56 | 128/16 (regs) S36 K72 | 128/16 (regs) K32 | 128/8 (smem) K32 |
| nxg4 cb2 chk16 listmax1024 | 128 | 10048 | 95/20 (regs) K32 | 95/20 (regs) K32 | 98/16 (regs) K32 | 100/16 (regs) K32 | 97/16 (regs) K32 |
| nxg4 cb2 chk32 listmax1024 | 128 | 16000 | 106/16 (regs) K32 | 106/16 (regs) K32 | 108/16 (regs) K32 | 101/16 (regs) K32 | 97/16 (regs) K32 |
| nxg4 cb3 chk16 listmax1024 | 128 | 13120 | 114/16 (regs) K32 | 114/16 (regs) K32 | 116/16 (regs) K32 | 106/16 (regs) K32 | 103/16 (regs) K32 |
| nxg4 cb3 chk32 listmax1024 | 128 | 22144 | 124/16 (regs) K32 | 121/16 (regs+smem) K32 | 119/16 (regs) K32 | 111/16 (regs) K32 | 109/16 (regs+smem) K32 |
| nxg4 cb6 chk16 listmax1024 | 128 | 18112 | 128/16 (regs) K32 | 120/16 (regs) K32 | 120/16 (regs) K32 | 119/16 (regs) K32 | 124/16 (regs) K32 |
| nxg4 cb6 chk32 listmax1024 | 128 | 32128 | 128/16 (regs) S12 K48 | 128/12 (smem) S12 K48 | 128/16 (regs) S12 K48 | 124/16 (regs) K32 | 123/12 (smem) K32 |

### static_o2_cb5 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 28032 | 117/16 (regs) K32 | 118/12 (smem) K32 | 121/16 (regs) K32 | 118/16 (regs) K32 | 112/12 (smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 51968 | 123/12 (smem) K32 | 124/4 (smem) K32 | 124/16 (regs+smem) K32 | 120/16 (regs+smem) K32 | 114/4 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 28032 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 122/16 (regs) K32 | 125/12 (smem) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 51968 | 128/12 (smem) K32 | 128/4 (smem) K32 | 128/16 (regs+smem) K32 | 125/16 (regs+smem) K32 | 125/4 (smem) K32 |
| nxg2 cb1 chk16 listmax1024 | 128 | 23680 | 102/16 (regs) K32 | 102/16 (regs+smem) K32 | 104/16 (regs) K32 | 116/16 (regs) K32 | 107/16 (regs+smem) K32 |
| nxg2 cb1 chk32 listmax1024 | 128 | 43264 | 102/12 (smem) K32 | 102/8 (smem) K32 | 104/16 (regs) K32 | 116/16 (regs) K32 | 107/8 (smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 23936 | 114/16 (regs) K32 | 115/16 (regs+smem) K32 | 117/16 (regs) K32 | 120/16 (regs) K32 | 116/16 (regs+smem) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 43776 | 123/12 (smem) K32 | 122/8 (smem) K32 | 125/16 (regs) K32 | 122/16 (regs) K32 | 117/8 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 28032 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 123/16 (regs) K32 | 123/12 (smem) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 51968 | 128/12 (smem) K32 | 128/4 (smem) K32 | 128/16 (regs+smem) K32 | 127/16 (regs+smem) K32 | 127/4 (smem) K32 |
| nxg4 cb2 chk16 listmax1024 | 128 | 21632 | 96/20 (regs) K32 | 96/16 (smem) K32 | 101/16 (regs) K32 | 113/16 (regs) K32 | 107/16 (regs+smem) K32 |
| nxg4 cb2 chk32 listmax1024 | 128 | 39168 | 96/16 (smem) K32 | 96/8 (smem) K32 | 101/16 (regs) K32 | 113/16 (regs) K32 | 107/8 (smem) K32 |
| nxg4 cb3 chk16 listmax1024 | 128 | 23808 | 112/16 (regs) K32 | 112/16 (regs+smem) K32 | 113/16 (regs) K32 | 120/16 (regs) K32 | 114/16 (regs+smem) K32 |
| nxg4 cb3 chk32 listmax1024 | 128 | 43520 | 112/12 (smem) K32 | 112/8 (smem) K32 | 113/16 (regs) K32 | 120/16 (regs) K32 | 114/8 (smem) K32 |
| nxg4 cb4 chk16 listmax1024 | 128 | 23936 | 114/16 (regs) K32 | 115/16 (regs+smem) K32 | 119/16 (regs) K32 | 120/16 (regs) K32 | 116/16 (regs+smem) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 43776 | 123/12 (smem) K32 | 122/8 (smem) K32 | 123/16 (regs) K32 | 120/16 (regs) K32 | 116/8 (smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 28032 | 128/16 (regs) K32 | 128/12 (smem) K32 | 128/16 (regs) K32 | 123/16 (regs) K32 | 123/12 (smem) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 51968 | 128/12 (smem) K32 | 128/4 (smem) K32 | 128/16 (regs+smem) K32 | 125/16 (regs+smem) K32 | 122/4 (smem) K32 |

### vel2d_o3_cb10 / kl_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 | 128 | 26256 | 126/16 (regs) K48 | 124/12 (smem) K48 | 124/16 (regs) K48 | 121/16 (regs) K48 | 124/12 (smem) K48 |
| nxg1 cb1 chk32 | 128 | 52112 | 126/12 (smem) K48 | 127/4 (smem) K48 | 127/16 (regs+smem) K48 | 123/16 (regs+smem) K48 | 124/4 (smem) K48 |
| nxg1 cb2 chk16 | 128 | 26640 | 128/16 (regs) S4 K64 | 126/12 (smem) K48 | 126/16 (regs) K48 | 126/16 (regs) K48 | 125/12 (smem) K48 |
| nxg1 cb2 chk32 | 128 | 52880 | 128/12 (smem) S12 K64 | 128/4 (smem) S12 K64 | 128/16 (regs+smem) S12 K64 | 125/16 (regs+smem) K48 | 124/4 (smem) K48 |
| nxg1 cb4 chk16 | 128 | 35088 | 128/16 (regs+smem) S592 K288 | 128/8 (smem) S592 K288 | 128/16 (regs) S588 K288 | 128/16 (regs) S424 K256 | 128/8 (smem) S428 K256 |
| nxg1 cb4 chk32 | 128 | 69776 | 128/8 (smem) S636 K304 | 128/4 (smem) S636 K304 | 128/12 (smem) S632 K304 | 128/12 (smem) S464 K272 | 128/4 (smem) S468 K272 |
| nxg2 cb2 chk16 | 128 | 22160 | 124/16 (regs) K48 | 124/16 (regs+smem) K48 | 124/16 (regs) K48 | 122/16 (regs) K48 | 118/16 (regs+smem) K48 |
| nxg2 cb2 chk32 | 128 | 43920 | 125/12 (smem) K48 | 125/8 (smem) K48 | 124/16 (regs) K48 | 122/16 (regs) K48 | 125/8 (smem) K48 |
| nxg2 cb3 chk16 | 128 | 26384 | 126/16 (regs) K48 | 127/12 (smem) K48 | 127/16 (regs) K48 | 123/16 (regs) K48 | 127/12 (smem) K48 |
| nxg2 cb3 chk32 | 128 | 52368 | 127/12 (smem) K48 | 125/4 (smem) K48 | 128/16 (regs+smem) K48 | 126/16 (regs+smem) K48 | 124/4 (smem) K48 |
| nxg2 cb4 chk16 | 128 | 26640 | 128/16 (regs) S8 K64 | 128/12 (smem) K48 | 128/16 (regs) K48 | 126/16 (regs) K48 | 125/12 (smem) K48 |
| nxg2 cb4 chk32 | 128 | 52880 | 128/12 (smem) S16 K64 | 128/4 (smem) S16 K64 | 128/16 (regs+smem) S16 K64 | 125/16 (regs+smem) K48 | 124/4 (smem) K48 |
| nxg2 cb7 chk16 | 128 | 35088 | 128/16 (regs+smem) S456 K240 | 128/8 (smem) S456 K240 | 128/16 (regs) S452 K256 | 128/16 (regs) S348 K208 | 128/8 (smem) S348 K208 |
| nxg2 cb7 chk32 | 128 | 69776 | 128/8 (smem) S500 K256 | 128/4 (smem) S500 K256 | 128/12 (smem) S496 K256 | 128/12 (smem) S388 K224 | 128/4 (smem) S388 K224 |
| nxg4 cb13 chk16 | 128 | 33040 | 128/16 (regs+smem) S392 K240 | 128/12 (smem) S392 K240 | 128/16 (regs) S400 K256 | 128/16 (regs) S332 K224 | 128/12 (smem) S332 K224 |
| nxg4 cb13 chk32 | 128 | 65680 | 128/8 (smem) S436 K256 | 128/4 (smem) S436 K256 | 128/12 (smem) S444 K272 | 128/12 (smem) S372 K240 | 128/4 (smem) S372 K240 |
| nxg4 cb4 chk16 | 128 | 22160 | 124/16 (regs) K48 | 124/16 (regs+smem) K48 | 124/16 (regs) K48 | 121/16 (regs) K48 | 118/16 (regs+smem) K48 |
| nxg4 cb4 chk32 | 128 | 43920 | 125/12 (smem) K48 | 125/8 (smem) K48 | 124/16 (regs) K48 | 122/16 (regs) K48 | 121/8 (smem) K48 |
| nxg4 cb5 chk16 | 128 | 24208 | 128/16 (regs) K48 | 126/16 (regs+smem) K48 | 124/16 (regs) K48 | 122/16 (regs) K48 | 124/16 (regs+smem) K48 |
| nxg4 cb5 chk32 | 128 | 48016 | 126/12 (smem) K48 | 126/8 (smem) K48 | 125/16 (regs+smem) K48 | 121/16 (regs+smem) K48 | 122/8 (smem) K48 |
| nxg4 cb7 chk16 | 128 | 26512 | 128/16 (regs) S8 K64 | 125/12 (smem) K48 | 127/16 (regs) K48 | 123/16 (regs) K48 | 124/12 (smem) K48 |
| nxg4 cb7 chk32 | 128 | 52624 | 128/12 (smem) S24 K80 | 128/4 (smem) K48 | 126/16 (regs+smem) K48 | 122/16 (regs+smem) K48 | 124/4 (smem) K48 |

### vel2d_o3_cb10 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 29632 | 122/16 (regs) K32 | 124/12 (smem) K32 | 122/16 (regs) K32 | 121/16 (regs) K32 | 128/12 (smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 55168 | 124/8 (smem) K32 | 126/4 (smem) K32 | 125/16 (regs+smem) K32 | 123/16 (regs+smem) K32 | 122/4 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 30016 | 128/16 (regs) K32 | 127/12 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 126/12 (smem) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 55936 | 128/8 (smem) S12 K48 | 127/4 (smem) K32 | 128/16 (regs+smem) K32 | 122/16 (regs+smem) K32 | 124/4 (smem) K32 |
| nxg1 cb4 chk16 listmax1024 | 128 | 38464 | 128/16 (regs+smem) S488 K264 | 128/8 (smem) S488 K264 | 128/16 (regs) S492 K264 | 128/16 (regs) S328 K208 | 128/8 (smem) S332 K216 |
| nxg1 cb4 chk32 listmax1024 | 128 | 72832 | 128/8 (smem) S516 K272 | 128/4 (smem) S516 K272 | 128/12 (smem) S520 K280 | 128/12 (smem) S352 K232 | 128/4 (smem) S356 K232 |
| nxg2 cb2 chk16 listmax1024 | 128 | 25536 | 121/16 (regs) K32 | 124/12 (smem) K32 | 124/16 (regs) K32 | 121/16 (regs) K32 | 128/12 (smem) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 46976 | 125/12 (smem) K32 | 126/8 (smem) K32 | 125/16 (regs+smem) K32 | 123/16 (regs+smem) K32 | 122/8 (smem) K32 |
| nxg2 cb3 chk16 listmax1024 | 128 | 29760 | 128/16 (regs) K32 | 128/12 (smem) K32 | 125/16 (regs) K32 | 121/16 (regs) K32 | 124/12 (smem) K32 |
| nxg2 cb3 chk32 listmax1024 | 128 | 55424 | 127/8 (smem) K32 | 128/4 (smem) K32 | 126/16 (regs+smem) K32 | 122/16 (regs+smem) K32 | 120/4 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 30016 | 128/16 (regs) K32 | 127/12 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 126/12 (smem) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 55936 | 128/8 (smem) S12 K48 | 128/4 (smem) K32 | 127/16 (regs+smem) K32 | 125/16 (regs+smem) K32 | 126/4 (smem) K32 |
| nxg2 cb7 chk16 listmax1024 | 128 | 38464 | 128/16 (regs+smem) S364 K224 | 128/8 (smem) S364 K224 | 128/16 (regs) S360 K224 | 128/16 (regs) S248 K176 | 128/8 (smem) S248 K176 |
| nxg2 cb7 chk32 listmax1024 | 128 | 72832 | 128/8 (smem) S392 K240 | 128/4 (smem) S392 K240 | 128/12 (smem) S388 K232 | 128/12 (smem) S272 K192 | 128/4 (smem) S272 K192 |
| nxg4 cb13 chk16 listmax1024 | 128 | 36416 | 128/16 (regs+smem) S316 K216 | 128/8 (smem) S316 K216 | 128/16 (regs) S296 K216 | 128/16 (regs) S224 K184 | 128/8 (smem) S224 K184 |
| nxg4 cb13 chk32 listmax1024 | 128 | 68736 | 128/8 (smem) S344 K232 | 128/4 (smem) S344 K232 | 128/12 (smem) S324 K232 | 128/12 (smem) S248 K200 | 128/4 (smem) S248 K200 |
| nxg4 cb4 chk16 listmax1024 | 128 | 25536 | 121/16 (regs) K32 | 124/12 (smem) K32 | 124/16 (regs) K32 | 121/16 (regs) K32 | 128/12 (smem) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 46976 | 125/12 (smem) K32 | 126/8 (smem) K32 | 126/16 (regs+smem) K32 | 123/16 (regs+smem) K32 | 122/8 (smem) K32 |
| nxg4 cb5 chk16 listmax1024 | 128 | 27584 | 127/16 (regs) K32 | 126/12 (smem) K32 | 125/16 (regs) K32 | 121/16 (regs) K32 | 123/12 (smem) K32 |
| nxg4 cb5 chk32 listmax1024 | 128 | 51072 | 128/12 (smem) K32 | 127/4 (smem) K32 | 125/16 (regs+smem) K32 | 121/16 (regs+smem) K32 | 124/4 (smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 29888 | 128/16 (regs) S4 K40 | 128/12 (smem) K32 | 126/16 (regs) K32 | 122/16 (regs) K32 | 123/12 (smem) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 55680 | 128/8 (smem) S20 K56 | 128/4 (smem) K32 | 127/16 (regs+smem) K32 | 125/16 (regs+smem) K32 | 126/4 (smem) K32 |

### vel2d_o3_cb5 / kl_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 | 128 | 26256 | 126/16 (regs) K48 | 124/12 (smem) K48 | 124/16 (regs) K48 | 121/16 (regs) K48 | 124/12 (smem) K48 |
| nxg1 cb1 chk32 | 128 | 52112 | 126/12 (smem) K48 | 127/4 (smem) K48 | 127/16 (regs+smem) K48 | 123/16 (regs+smem) K48 | 124/4 (smem) K48 |
| nxg1 cb2 chk16 | 128 | 26256 | 128/16 (regs) S4 K64 | 126/12 (smem) K48 | 126/16 (regs) K48 | 126/16 (regs) K48 | 125/12 (smem) K48 |
| nxg1 cb2 chk32 | 128 | 52112 | 128/12 (smem) S12 K64 | 128/4 (smem) S12 K64 | 128/16 (regs+smem) S12 K64 | 125/16 (regs+smem) K48 | 124/4 (smem) K48 |
| nxg2 cb1 chk16 | 128 | 21904 | 120/16 (regs) K48 | 122/16 (regs+smem) K48 | 121/16 (regs) K48 | 118/16 (regs) K48 | 118/16 (regs+smem) K48 |
| nxg2 cb1 chk32 | 128 | 43408 | 120/12 (smem) K48 | 122/8 (smem) K48 | 121/16 (regs) K48 | 118/16 (regs) K48 | 118/8 (smem) K48 |
| nxg2 cb2 chk16 | 128 | 22160 | 124/16 (regs) K48 | 124/16 (regs+smem) K48 | 124/16 (regs) K48 | 122/16 (regs) K48 | 118/16 (regs+smem) K48 |
| nxg2 cb2 chk32 | 128 | 43920 | 125/12 (smem) K48 | 125/8 (smem) K48 | 124/16 (regs) K48 | 122/16 (regs) K48 | 125/8 (smem) K48 |
| nxg2 cb4 chk16 | 128 | 26256 | 128/16 (regs) S8 K64 | 128/12 (smem) K48 | 128/16 (regs) K48 | 126/16 (regs) K48 | 125/12 (smem) K48 |
| nxg2 cb4 chk32 | 128 | 52112 | 128/12 (smem) S16 K64 | 128/4 (smem) S16 K64 | 128/16 (regs+smem) S16 K64 | 125/16 (regs+smem) K48 | 124/4 (smem) K48 |
| nxg4 cb2 chk16 | 128 | 19856 | 120/16 (regs) K48 | 122/16 (regs+smem) K48 | 121/16 (regs) K48 | 117/16 (regs) K48 | 117/16 (regs+smem) K48 |
| nxg4 cb2 chk32 | 128 | 39312 | 120/16 (regs+smem) K48 | 122/8 (smem) K48 | 121/16 (regs) K48 | 117/16 (regs) K48 | 117/8 (smem) K48 |
| nxg4 cb3 chk16 | 128 | 22032 | 122/16 (regs) K48 | 124/16 (regs+smem) K48 | 123/16 (regs) K48 | 122/16 (regs) K48 | 119/16 (regs+smem) K48 |
| nxg4 cb3 chk32 | 128 | 43664 | 122/12 (smem) K48 | 124/8 (smem) K48 | 123/16 (regs) K48 | 122/16 (regs) K48 | 119/8 (smem) K48 |
| nxg4 cb4 chk16 | 128 | 22160 | 124/16 (regs) K48 | 124/16 (regs+smem) K48 | 124/16 (regs) K48 | 121/16 (regs) K48 | 118/16 (regs+smem) K48 |
| nxg4 cb4 chk32 | 128 | 43920 | 125/12 (smem) K48 | 125/8 (smem) K48 | 124/16 (regs) K48 | 122/16 (regs) K48 | 121/8 (smem) K48 |
| nxg4 cb7 chk16 | 128 | 26256 | 128/16 (regs) K48 | 125/12 (smem) K48 | 127/16 (regs) K48 | 123/16 (regs) K48 | 124/12 (smem) K48 |
| nxg4 cb7 chk32 | 128 | 52112 | 128/12 (smem) S4 K64 | 128/4 (smem) K48 | 126/16 (regs+smem) K48 | 122/16 (regs+smem) K48 | 124/4 (smem) K48 |

### vel2d_o3_cb5 / ra_adj

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| nxg1 cb1 chk16 listmax1024 | 128 | 29632 | 122/16 (regs) K32 | 124/12 (smem) K32 | 122/16 (regs) K32 | 121/16 (regs) K32 | 128/12 (smem) K32 |
| nxg1 cb1 chk32 listmax1024 | 128 | 55168 | 124/8 (smem) K32 | 126/4 (smem) K32 | 125/16 (regs+smem) K32 | 123/16 (regs+smem) K32 | 122/4 (smem) K32 |
| nxg1 cb2 chk16 listmax1024 | 128 | 29632 | 128/16 (regs) K32 | 127/12 (smem) K32 | 125/16 (regs) K32 | 123/16 (regs) K32 | 128/12 (smem) K32 |
| nxg1 cb2 chk32 listmax1024 | 128 | 55168 | 128/8 (smem) S12 K48 | 127/4 (smem) K32 | 128/16 (regs+smem) K32 | 124/16 (regs+smem) K32 | 126/4 (smem) K32 |
| nxg2 cb1 chk16 listmax1024 | 128 | 25280 | 120/16 (regs) K32 | 120/12 (smem) K32 | 120/16 (regs) K32 | 116/16 (regs) K32 | 112/12 (smem) K32 |
| nxg2 cb1 chk32 listmax1024 | 128 | 46464 | 120/12 (smem) K32 | 120/8 (smem) K32 | 120/16 (regs+smem) K32 | 116/16 (regs+smem) K32 | 112/8 (smem) K32 |
| nxg2 cb2 chk16 listmax1024 | 128 | 25536 | 121/16 (regs) K32 | 124/12 (smem) K32 | 124/16 (regs) K32 | 121/16 (regs) K32 | 128/12 (smem) K32 |
| nxg2 cb2 chk32 listmax1024 | 128 | 46976 | 125/12 (smem) K32 | 126/8 (smem) K32 | 125/16 (regs+smem) K32 | 123/16 (regs+smem) K32 | 122/8 (smem) K32 |
| nxg2 cb4 chk16 listmax1024 | 128 | 29632 | 128/16 (regs) K32 | 127/12 (smem) K32 | 125/16 (regs) K32 | 122/16 (regs) K32 | 125/12 (smem) K32 |
| nxg2 cb4 chk32 listmax1024 | 128 | 55168 | 128/8 (smem) S12 K48 | 128/4 (smem) K32 | 127/16 (regs+smem) K32 | 124/16 (regs+smem) K32 | 126/4 (smem) K32 |
| nxg4 cb2 chk16 listmax1024 | 128 | 23232 | 118/16 (regs) K32 | 120/16 (regs+smem) K32 | 123/16 (regs) K32 | 118/16 (regs) K32 | 111/16 (regs+smem) K32 |
| nxg4 cb2 chk32 listmax1024 | 128 | 42368 | 118/12 (smem) K32 | 120/8 (smem) K32 | 123/16 (regs) K32 | 118/16 (regs) K32 | 111/8 (smem) K32 |
| nxg4 cb3 chk16 listmax1024 | 128 | 25408 | 120/16 (regs) K32 | 120/12 (smem) K32 | 122/16 (regs) K32 | 118/16 (regs) K32 | 118/12 (smem) K32 |
| nxg4 cb3 chk32 listmax1024 | 128 | 46720 | 120/12 (smem) K32 | 120/8 (smem) K32 | 122/16 (regs+smem) K32 | 118/16 (regs+smem) K32 | 118/8 (smem) K32 |
| nxg4 cb4 chk16 listmax1024 | 128 | 25536 | 121/16 (regs) K32 | 124/12 (smem) K32 | 124/16 (regs) K32 | 121/16 (regs) K32 | 128/12 (smem) K32 |
| nxg4 cb4 chk32 listmax1024 | 128 | 46976 | 125/12 (smem) K32 | 126/8 (smem) K32 | 126/16 (regs+smem) K32 | 123/16 (regs+smem) K32 | 122/8 (smem) K32 |
| nxg4 cb7 chk16 listmax1024 | 128 | 29632 | 128/16 (regs) K32 | 127/12 (smem) K32 | 125/16 (regs) K32 | 122/16 (regs) K32 | 123/12 (smem) K32 |
| nxg4 cb7 chk32 listmax1024 | 128 | 55168 | 128/8 (smem) K32 | 126/4 (smem) K32 | 126/16 (regs+smem) K32 | 125/16 (regs+smem) K32 | 126/4 (smem) K32 |

## Flags (spills, stack, no fit, below 8 warps / SM, compile errors)

| arch | case | family | cfg | regs | flags |
|---|---|---|---|---|---|
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 119 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 113 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 119 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 121 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk16 | 122 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 123 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 119 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 125 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 123 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg1 cb1 chk32 | 124 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 127 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 126 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk16 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 128 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 126 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg1 cb2 chk32 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 111 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 107 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 102 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 105 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk16 | 104 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 111 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 107 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 102 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 105 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb1 chk32 | 104 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 119 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 117 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 117 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 119 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk16 | 120 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 123 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 117 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 125 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 123 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb2 chk32 | 124 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 127 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 126 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk16 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 128 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 126 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 127 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg2 cb4 chk32 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 116 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 109 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 99 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 99 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk16 | 101 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 116 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 109 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 99 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 99 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb2 chk32 | 101 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 112 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 111 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 114 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 113 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk16 | 115 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 112 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 111 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 114 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 113 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb3 chk32 | 115 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 117 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 112 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 117 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 119 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk16 | 121 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 122 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 117 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 125 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 123 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb4 chk32 | 122 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 126 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 125 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk16 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 124 | STACK 48 |
| sm_120 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 124 | STACK 48 |
| sm_80 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 128 | STACK 48 |
| sm_89 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 128 | STACK 48 |
| sm_90 | knot1d_o2_cb5 | kl_adj | nxg4 cb7 chk32 | 128 | STACK 48 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 118 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 114 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 119 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 119 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 120 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 120 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 119 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 123 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 124 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 122 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 127 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 127 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 112 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 112 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 101 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 102 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 103 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 112 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 112 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 101 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 102 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 103 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 120 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 119 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 114 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 115 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 118 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 120 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 118 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 122 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 124 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 123 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 127 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 127 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32 |
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
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 116 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 114 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 112 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 112 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 112 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 116 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 114 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 112 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 112 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 112 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 119 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 116 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 114 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 115 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 117 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 120 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 119 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 122 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 124 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 127 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 124 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 124 | STACK 32 |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | STACK 32 |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | STACK 32 |
| sm_100 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 125 | STACK 32 |
| sm_120 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 125 | STACK 32, LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | knot1d_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 108 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 102 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 104 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 105 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 107 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 108 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 104 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 109 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 110 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 112 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 110 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 126 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 119 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 119 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 111 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 111 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 119 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 120 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 128 | SPILL st384 ld392, STACK 216 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 128 | SPILL st384 ld392, STACK 216 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 128 | SPILL st328 ld320, STACK 224 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 128 | SPILL st328 ld304, STACK 224 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 128 | SPILL st432 ld436, STACK 248 |
| sm_100 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 128 | SPILL st432 ld436, STACK 240 |
| sm_120 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 128 | SPILL st432 ld440, STACK 240 |
| sm_80 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 128 | SPILL st344 ld328, STACK 224 |
| sm_89 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 128 | SPILL st344 ld312, STACK 224 |
| sm_90 | static_o0_10x10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 128 | SPILL st452 ld448, STACK 264 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 108 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 100 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 100 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 101 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 104 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 107 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 102 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 107 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 108 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 110 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 112 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 108 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 122 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 123 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 121 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 114 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 110 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 126 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 125 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 123 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 116 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 115 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 120 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 120 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 116 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 115 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 120 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 121 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 128 | SPILL st232 ld224, STACK 176 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 128 | SPILL st228 ld224, STACK 176 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 128 | SPILL st212 ld200, STACK 176 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 128 | SPILL st212 ld184, STACK 176 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 128 | SPILL st220 ld208, STACK 176 |
| sm_100 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 128 | SPILL st260 ld244, STACK 192 |
| sm_120 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 128 | SPILL st256 ld248, STACK 192 |
| sm_80 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 128 | SPILL st244 ld224, STACK 200 |
| sm_89 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 128 | SPILL st244 ld208, STACK 200 |
| sm_90 | static_o0_10x10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 128 | SPILL st292 ld272, STACK 232 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 128 | SPILL st276 ld280, STACK 208 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 128 | SPILL st272 ld280, STACK 208 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 128 | SPILL st260 ld260, STACK 208 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 128 | SPILL st260 ld244, STACK 208 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 128 | SPILL st316 ld312, STACK 240 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 128 | SPILL st300 ld296, STACK 224 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 128 | SPILL st296 ld296, STACK 224 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 128 | SPILL st300 ld300, STACK 240 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 128 | SPILL st300 ld284, STACK 240 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 128 | SPILL st348 ld336, STACK 264 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 107 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 100 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 100 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 101 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 104 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 108 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 104 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 107 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 108 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 110 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 109 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 106 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 116 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 117 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 114 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 110 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 107 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 120 | STACK 32 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 120 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 119 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 119 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 116 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | SPILL st8 ld8, STACK 40 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 125 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 122 | STACK 32 |
| sm_100 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 119 | STACK 32 |
| sm_120 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 117 | STACK 32 |
| sm_80 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | SPILL st12 ld12, STACK 48 |
| sm_89 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 124 | STACK 32 |
| sm_90 | static_o0_10x10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 123 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 112 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 110 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 122 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 122 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg1 cb1 chk16 listmax1024 | 121 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 117 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 117 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 121 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg1 cb1 chk32 listmax1024 | 122 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 122 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 123 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | SPILL st8 ld8, STACK 40 |
| sm_100 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | SPILL st40 ld44, STACK 80 |
| sm_89 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | SPILL st48 ld48, STACK 88 |
| sm_90 | static_o0_48x1 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | SPILL st72 ld72, STACK 112 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 100 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 97 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 97 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 96 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb1 chk16 listmax1024 | 99 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 101 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 98 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 107 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 107 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb1 chk32 listmax1024 | 109 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 110 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 107 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 119 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 119 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb2 chk16 listmax1024 | 118 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 121 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 117 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 121 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb2 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 118 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 116 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 126 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb3 chk16 listmax1024 | 127 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 128 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 128 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 128 | SPILL st20 ld20, STACK 56 |
| sm_89 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 128 | SPILL st20 ld20, STACK 56 |
| sm_90 | static_o0_48x1 | ra_adj | nxg2 cb3 chk32 listmax1024 | 128 | SPILL st36 ld36, STACK 72 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 100 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 97 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 95 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 95 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb2 chk16 listmax1024 | 98 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 101 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 97 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 106 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 106 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb2 chk32 listmax1024 | 108 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 106 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 103 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 114 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 114 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb3 chk16 listmax1024 | 116 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 111 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 109 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 124 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 121 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb3 chk32 listmax1024 | 119 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 119 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 124 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 120 | STACK 32 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb6 chk16 listmax1024 | 120 | STACK 32 |
| sm_100 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 124 | STACK 32 |
| sm_120 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 123 | STACK 32 |
| sm_80 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 128 | SPILL st12 ld12, STACK 48 |
| sm_89 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 128 | SPILL st12 ld12, STACK 48 |
| sm_90 | static_o0_48x1 | ra_adj | nxg4 cb6 chk32 listmax1024 | 128 | SPILL st12 ld12, STACK 48 |
| sm_100 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 118 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 112 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 117 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 118 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 121 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 120 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 114 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 123 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 124 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 124 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 122 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 125 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 125 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 125 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 116 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 107 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 102 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 102 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 104 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 116 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 107 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 102 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 102 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 104 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 120 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 116 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 114 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 115 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 117 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 122 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 117 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 123 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 122 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 123 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 123 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 127 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32 |
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
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 120 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 114 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 112 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 112 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 113 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 120 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 114 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 112 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 112 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 113 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 120 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 116 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 114 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 115 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 119 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 120 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 116 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 123 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 122 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 123 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 123 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 123 | STACK 32 |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | STACK 32 |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | STACK 32 |
| sm_100 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 125 | STACK 32 |
| sm_120 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | static_o2_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk16 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 124 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 127 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb1 chk32 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 128 | SPILL st4 ld4, STACK 64 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk16 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 124 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 128 | SPILL st12 ld12, STACK 64 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 128 | SPILL st12 ld12, STACK 64, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb2 chk32 | 128 | SPILL st12 ld12, STACK 64 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 128 | SPILL st424 ld308, STACK 256 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 128 | SPILL st428 ld312, STACK 256 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 128 | SPILL st592 ld356, STACK 288 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 128 | SPILL st592 ld356, STACK 288 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk16 | 128 | SPILL st588 ld352, STACK 288 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 128 | SPILL st464 ld324, STACK 272 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 128 | SPILL st468 ld328, STACK 272, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 128 | SPILL st636 ld376, STACK 304 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 128 | SPILL st636 ld376, STACK 304, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg1 cb4 chk32 | 128 | SPILL st632 ld372, STACK 304 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 118 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk16 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 125 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb2 chk32 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 127 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 127 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk16 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 124 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 127 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 125 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb3 chk32 | 128 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 128 | SPILL st8 ld8, STACK 64 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 128 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk16 | 128 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 124 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 128 | SPILL st16 ld16, STACK 64 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 128 | SPILL st16 ld16, STACK 64, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb4 chk32 | 128 | SPILL st16 ld16, STACK 64 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 128 | SPILL st348 ld236, STACK 208 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 128 | SPILL st348 ld236, STACK 208 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 128 | SPILL st456 ld260, STACK 240 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 128 | SPILL st456 ld260, STACK 240 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk16 | 128 | SPILL st452 ld256, STACK 256 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 128 | SPILL st388 ld252, STACK 224 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 128 | SPILL st388 ld252, STACK 224, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 128 | SPILL st500 ld280, STACK 256 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 128 | SPILL st500 ld280, STACK 256, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg2 cb7 chk32 | 128 | SPILL st496 ld276, STACK 256 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 128 | SPILL st332 ld208, STACK 224 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 128 | SPILL st332 ld208, STACK 224 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 128 | SPILL st392 ld228, STACK 240 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 128 | SPILL st392 ld228, STACK 240 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk16 | 128 | SPILL st400 ld244, STACK 256 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 128 | SPILL st372 ld224, STACK 240 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 128 | SPILL st372 ld224, STACK 240, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 128 | SPILL st436 ld248, STACK 256 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 128 | SPILL st436 ld248, STACK 256, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb13 chk32 | 128 | SPILL st444 ld264, STACK 272 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 118 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk16 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 121 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 125 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb4 chk32 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 128 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk16 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 122 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb5 chk32 | 125 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 128 | SPILL st8 ld8, STACK 64 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk16 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 124 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 128 | SPILL st24 ld24, STACK 80 |
| sm_89 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 128 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | kl_adj | nxg4 cb7 chk32 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 121 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk16 listmax1024 | 122 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb1 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 126 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 127 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 124 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | SPILL st12 ld12, STACK 48 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 128 | SPILL st328 ld312, STACK 208 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 128 | SPILL st332 ld316, STACK 216 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 128 | SPILL st488 ld352, STACK 264 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 128 | SPILL st488 ld336, STACK 264 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk16 listmax1024 | 128 | SPILL st492 ld356, STACK 264 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 128 | SPILL st352 ld328, STACK 232 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 128 | SPILL st356 ld332, STACK 232, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 128 | SPILL st516 ld372, STACK 272 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 128 | SPILL st516 ld356, STACK 272, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg1 cb4 chk32 listmax1024 | 128 | SPILL st520 ld376, STACK 280 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 121 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 121 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk16 listmax1024 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 122 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 125 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb2 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 121 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 124 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 120 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 127 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb3 chk32 listmax1024 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 126 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 127 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 125 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | SPILL st12 ld12, STACK 48 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb4 chk32 listmax1024 | 127 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 128 | SPILL st248 ld236, STACK 176 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 128 | SPILL st248 ld236, STACK 176 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 128 | SPILL st364 ld268, STACK 224 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 128 | SPILL st364 ld252, STACK 224 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk16 listmax1024 | 128 | SPILL st360 ld264, STACK 224 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 128 | SPILL st272 ld252, STACK 192 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 128 | SPILL st272 ld252, STACK 192, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 128 | SPILL st392 ld288, STACK 240 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 128 | SPILL st392 ld272, STACK 240, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg2 cb7 chk32 listmax1024 | 128 | SPILL st388 ld284, STACK 232 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 128 | SPILL st224 ld204, STACK 184 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 128 | SPILL st224 ld204, STACK 184 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 128 | SPILL st316 ld244, STACK 216 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 128 | SPILL st316 ld228, STACK 216 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk16 listmax1024 | 128 | SPILL st296 ld240, STACK 216 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 128 | SPILL st248 ld220, STACK 200 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 128 | SPILL st248 ld220, STACK 200, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 128 | SPILL st344 ld264, STACK 232 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 128 | SPILL st344 ld248, STACK 232, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb13 chk32 listmax1024 | 128 | SPILL st324 ld260, STACK 232 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 121 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 121 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk16 listmax1024 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 122 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 125 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb4 chk32 listmax1024 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 121 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 123 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 127 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 121 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 124 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb5 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 123 | STACK 32 |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | SPILL st4 ld4, STACK 40 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | STACK 32 |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk16 listmax1024 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 125 | STACK 32 |
| sm_120 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | SPILL st20 ld20, STACK 56 |
| sm_89 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb10 | ra_adj | nxg4 cb7 chk32 listmax1024 | 127 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk16 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 124 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 126 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 127 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg1 cb1 chk32 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 128 | SPILL st4 ld4, STACK 64 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 126 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk16 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 124 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 128 | SPILL st12 ld12, STACK 64 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 128 | SPILL st12 ld12, STACK 64, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg1 cb2 chk32 | 128 | SPILL st12 ld12, STACK 64 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 118 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 118 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 120 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 122 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk16 | 121 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 118 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 118 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 120 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 122 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb1 chk32 | 121 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 118 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk16 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 125 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb2 chk32 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 126 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 125 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 128 | SPILL st8 ld8, STACK 64 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 128 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk16 | 128 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 125 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 124 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 128 | SPILL st16 ld16, STACK 64 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 128 | SPILL st16 ld16, STACK 64, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg2 cb4 chk32 | 128 | SPILL st16 ld16, STACK 64 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 117 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 117 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 120 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 122 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk16 | 121 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 117 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 117 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 120 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 122 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb2 chk32 | 121 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 119 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 122 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk16 | 123 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 119 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 122 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb3 chk32 | 123 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 121 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 118 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 124 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 124 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk16 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 121 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 125 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb4 chk32 | 124 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 123 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 124 | STACK 48 |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 128 | STACK 48 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 125 | STACK 48 |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk16 | 127 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 122 | STACK 48 |
| sm_120 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 124 | STACK 48, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 128 | SPILL st4 ld4, STACK 64 |
| sm_89 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 128 | STACK 48, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | kl_adj | nxg4 cb7 chk32 | 126 | STACK 48 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 121 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 122 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk16 listmax1024 | 122 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 122 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 124 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg1 cb1 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 127 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 124 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | SPILL st12 ld12, STACK 48 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 127 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg1 cb2 chk32 listmax1024 | 128 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 116 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 112 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 120 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 120 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk16 listmax1024 | 120 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 116 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 112 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 120 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 120 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb1 chk32 listmax1024 | 120 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 121 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 121 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk16 listmax1024 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 122 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 125 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb2 chk32 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 125 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 127 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 124 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | SPILL st12 ld12, STACK 48 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 128 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg2 cb4 chk32 listmax1024 | 127 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 118 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 111 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 118 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 120 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk16 listmax1024 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 118 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 111 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 118 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 120 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb2 chk32 listmax1024 | 123 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 118 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 118 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 120 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 120 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk16 listmax1024 | 122 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 118 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 118 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 120 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 120 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb3 chk32 listmax1024 | 122 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 121 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 128 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 121 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 124 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk16 listmax1024 | 124 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 123 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 122 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 125 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 126 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb4 chk32 listmax1024 | 126 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 122 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 123 | STACK 32 |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 127 | STACK 32 |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk16 listmax1024 | 125 | STACK 32 |
| sm_100 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 125 | STACK 32 |
| sm_120 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 128 | STACK 32 |
| sm_89 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 126 | STACK 32, LOW_OCC 4w |
| sm_90 | vel2d_o3_cb5 | ra_adj | nxg4 cb7 chk32 listmax1024 | 126 | STACK 32 |
