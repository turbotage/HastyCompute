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
- `static_o0_48x1`: static image, order 0, 48 coils x L 1 in one batch (ra nxg 2 cb 3 spill case)

## Per configuration

Cell = registers / warps per SM (limiter); `S` = spills, `K` = stack frame, `X` = does not fit, `-` = not a candidate on that arch (shared memory).

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

## Flags (spills, stack, no fit, below 8 warps / SM, compile errors)

| arch | case | family | cfg | regs | flags |
|---|---|---|---|---|---|
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
