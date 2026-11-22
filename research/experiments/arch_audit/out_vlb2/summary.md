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

### knot1d_o2_cb5 / vt_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 2x4x16 smax32 nwc4 nws1 vxpad0 | 128 | 49184 | 244/8 (regs) | 246/8 (regs+smem) | 254/8 (regs) | 214/8 (regs) | 222/8 (regs+smem) |
| 2x4x16 smax32 nwc4 nws2 vxpad0 | 256 | 55584 | 128/16 (regs+smem) S184 K88 | 128/8 (smem) S184 K88 | 128/16 (regs) S324 K104 | 128/16 (regs) S288 K104 | 128/8 (smem) S328 K104 |
| 2x4x8 smax32 nwc4 nws1 vxpad0 | 128 | 48928 | 244/8 (regs) | 246/8 (regs+smem) | 254/8 (regs) | 214/8 (regs) | 222/8 (regs+smem) |
| 2x4x8 smax32 nwc4 nws2 vxpad0 | 256 | 55328 | 128/16 (regs+smem) S184 K88 | 128/8 (smem) S184 K88 | 128/16 (regs) S324 K104 | 128/16 (regs) S288 K104 | 128/8 (smem) S328 K104 |
| 2x4x8 smax48 nwc4 nws1 vxpad0 | 128 | 69664 | 244/8 (regs+smem) | 246/4 (smem) | 254/8 (regs) | 214/8 (regs) | 222/4 (smem) |
| 2x4x8 smax48 nwc4 nws2 vxpad0 | 256 | 76064 | 128/16 (regs+smem) S184 K88 | 128/8 (smem) S184 K88 | 128/16 (regs) S324 K104 | 128/16 (regs) S288 K104 | 128/8 (smem) S328 K104 |
| 4x4x16 smax48 nwc4 nws1 vxpad0 | 128 | 70720 | 250/8 (regs+smem) | 252/4 (smem) | 254/8 (regs) | 238/8 (regs) | 254/4 (smem) |
| 4x4x16 smax48 nwc4 nws2 vxpad0 | 256 | 77120 | 128/16 (regs+smem) S548 K240 | 128/8 (smem) S568 K256 | 128/16 (regs+smem) S696 K232 | 128/16 (regs+smem) S956 K400 | 128/8 (smem) S916 K248 |
| 4x4x16 smax64 nwc4 nws1 vxpad0 | 128 | 91456 | 250/4 (smem) | 252/4 (smem) | 254/8 (regs+smem) | 238/8 (regs+smem) | 254/4 (smem) |
| 4x4x16 smax64 nwc4 nws2 vxpad0 | 256 | 97856 | 128/8 (smem) S548 K240 | 128/8 (smem) S568 K256 | 128/16 (regs+smem) S696 K232 | 128/16 (regs+smem) S956 K400 | 128/8 (smem) S916 K248 |

### static_o0_10x10 / vs_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 2x4x16 smax32 nwc7 nws1 vxpad0 | 224 | 37536 | 128/14 (regs) S124 K80 | 128/14 (regs+smem) S180 K120 | 128/14 (regs) S196 K88 | 128/14 (regs) S168 K88 | 128/14 (regs+smem) S168 K88 |
| 2x4x16 smax32 nwc7 nws1 vxpad1 | 224 | 38560 | 128/14 (regs) S136 K72 | 128/14 (regs+smem) S168 K88 | 128/14 (regs) S208 K80 | 128/14 (regs) S204 K104 | 128/14 (regs+smem) S196 K104 |
| 2x4x8 smax32 nwc7 nws1 vxpad0 | 224 | 37280 | 128/14 (regs) S124 K80 | 128/14 (regs+smem) S180 K120 | 128/14 (regs) S196 K88 | 128/14 (regs) S168 K88 | 128/14 (regs+smem) S168 K88 |
| 2x4x8 smax32 nwc7 nws1 vxpad1 | 224 | 38304 | 128/14 (regs) S136 K72 | 128/14 (regs+smem) S168 K88 | 128/14 (regs) S208 K80 | 128/14 (regs) S204 K104 | 128/14 (regs+smem) S196 K104 |
| 2x4x8 smax48 nwc7 nws1 vxpad0 | 224 | 55200 | 128/14 (regs+smem) S124 K80 | 128/7 (smem) S180 K120 | 128/14 (regs) S196 K88 | 128/14 (regs) S168 K88 | 128/7 (smem) S168 K88 |
| 2x4x8 smax48 nwc7 nws1 vxpad1 | 224 | 56736 | 128/14 (regs+smem) S136 K72 | 128/7 (smem) S168 K88 | 128/14 (regs) S208 K80 | 128/14 (regs) S204 K104 | 128/7 (smem) S196 K104 |
| 4x4x16 smax64 nwc7 nws1 vxpad0 | 224 | 74176 | 128/14 (regs+smem) S496 K264 | 128/7 (smem) S508 K272 | 128/14 (regs) S616 K368 | 128/14 (regs) S648 K304 | 128/7 (smem) S572 K240 |
| 4x4x16 smax64 nwc7 nws1 vxpad1 | 224 | 76224 | 128/14 (regs+smem) S736 K232 | 128/7 (smem) S804 K304 | 128/14 (regs) S1012 K272 | 128/14 (regs) S856 K256 | 128/7 (smem) S800 K232 |

### static_o0_48x1 / vs_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 2x4x16 smax32 nwc1 nws1 vxpad0 | 32 | 12192 | 254/8 (regs) | 254/7 (smem) | 225/8 (regs) | 248/8 (regs) | 254/7 (smem) |
| 2x4x16 smax32 nwc1 nws1 vxpad1 | 32 | 13216 | 248/8 (regs) | 245/7 (smem) | 250/8 (regs) | 242/8 (regs) | 242/7 (smem) |
| 2x4x16 smax32 nwc1 nws2 vxpad0 | 64 | 12192 | 188/8 (regs) | 200/8 (regs) | 198/8 (regs) | 204/8 (regs) | 252/8 (regs) |
| 2x4x16 smax32 nwc1 nws2 vxpad1 | 64 | 13216 | 188/8 (regs) | 198/8 (regs) | 196/8 (regs) | 204/8 (regs) | 252/8 (regs) |
| 2x4x16 smax32 nwc2 nws1 vxpad0 | 64 | 16416 | 194/8 (regs) | 194/8 (regs) | 210/8 (regs) | 202/8 (regs) | 252/8 (regs) |
| 2x4x16 smax32 nwc2 nws1 vxpad1 | 64 | 17440 | 238/8 (regs) | 238/8 (regs) | 252/8 (regs) | 210/8 (regs) | 254/8 (regs) |
| 2x4x16 smax32 nwc2 nws2 vxpad0 | 128 | 16416 | 192/8 (regs) | 202/8 (regs) | 200/8 (regs) | 208/8 (regs) | 254/8 (regs) |
| 2x4x16 smax32 nwc2 nws2 vxpad1 | 128 | 17440 | 192/8 (regs) | 194/8 (regs) | 200/8 (regs) | 208/8 (regs) | 254/8 (regs) |
| 2x4x16 smax32 nwc3 nws1 vxpad0 | 96 | 20640 | 194/6 (regs) | 194/6 (regs) | 210/6 (regs) | 206/6 (regs) | 252/6 (regs) |
| 2x4x16 smax32 nwc3 nws1 vxpad1 | 96 | 21664 | 240/6 (regs) | 238/6 (regs) | 254/6 (regs) | 202/6 (regs) | 254/6 (regs) |
| 2x4x16 smax32 nwc3 nws2 vxpad0 | 192 | 20640 | 164/12 (regs) | 168/12 (regs) | 162/12 (regs) | 168/12 (regs) | 168/12 (regs) S44 K40 |
| 2x4x16 smax32 nwc3 nws2 vxpad1 | 192 | 21664 | 166/12 (regs) | 168/12 (regs) | 164/12 (regs) | 168/12 (regs) | 168/12 (regs) S40 K40 |
| 2x4x8 smax32 nwc1 nws1 vxpad0 | 32 | 11936 | 254/8 (regs) | 254/7 (smem) | 225/8 (regs) | 248/8 (regs) | 254/7 (smem) |
| 2x4x8 smax32 nwc1 nws1 vxpad1 | 32 | 12960 | 248/8 (regs) | 245/7 (smem) | 250/8 (regs) | 242/8 (regs) | 242/7 (smem) |
| 2x4x8 smax32 nwc1 nws2 vxpad0 | 64 | 11936 | 188/8 (regs) | 200/8 (regs) | 198/8 (regs) | 204/8 (regs) | 252/8 (regs) |
| 2x4x8 smax32 nwc1 nws2 vxpad1 | 64 | 12960 | 188/8 (regs) | 198/8 (regs) | 196/8 (regs) | 204/8 (regs) | 252/8 (regs) |
| 2x4x8 smax32 nwc2 nws1 vxpad0 | 64 | 16160 | 194/8 (regs) | 194/8 (regs) | 210/8 (regs) | 202/8 (regs) | 252/8 (regs) |
| 2x4x8 smax32 nwc2 nws1 vxpad1 | 64 | 17184 | 238/8 (regs) | 238/8 (regs) | 252/8 (regs) | 210/8 (regs) | 254/8 (regs) |
| 2x4x8 smax32 nwc2 nws2 vxpad0 | 128 | 16160 | 192/8 (regs) | 202/8 (regs) | 200/8 (regs) | 208/8 (regs) | 254/8 (regs) |
| 2x4x8 smax32 nwc2 nws2 vxpad1 | 128 | 17184 | 192/8 (regs) | 194/8 (regs) | 200/8 (regs) | 208/8 (regs) | 254/8 (regs) |
| 2x4x8 smax32 nwc3 nws1 vxpad0 | 96 | 20384 | 194/6 (regs) | 194/6 (regs) | 210/6 (regs) | 206/6 (regs) | 252/6 (regs) |
| 2x4x8 smax32 nwc3 nws1 vxpad1 | 96 | 21408 | 240/6 (regs) | 238/6 (regs) | 254/6 (regs) | 202/6 (regs) | 254/6 (regs) |
| 2x4x8 smax32 nwc3 nws2 vxpad0 | 192 | 20384 | 164/12 (regs) | 168/12 (regs) | 162/12 (regs) | 168/12 (regs) | 168/12 (regs) S44 K40 |
| 2x4x8 smax32 nwc3 nws2 vxpad1 | 192 | 21408 | 166/12 (regs) | 168/12 (regs) | 164/12 (regs) | 168/12 (regs) | 168/12 (regs) S40 K40 |
| 2x4x8 smax48 nwc1 nws1 vxpad0 | 32 | 17568 | 254/8 (regs) | 254/5 (smem) | 225/8 (regs) | 248/8 (regs) | 254/5 (smem) |
| 2x4x8 smax48 nwc1 nws1 vxpad1 | 32 | 19104 | 248/8 (regs+smem) | 245/5 (smem) | 250/8 (regs) | 242/8 (regs) | 242/5 (smem) |
| 2x4x8 smax48 nwc1 nws2 vxpad0 | 64 | 17568 | 188/8 (regs) | 200/8 (regs) | 198/8 (regs) | 204/8 (regs) | 252/8 (regs) |
| 2x4x8 smax48 nwc1 nws2 vxpad1 | 64 | 19104 | 188/8 (regs) | 198/8 (regs) | 196/8 (regs) | 204/8 (regs) | 252/8 (regs) |
| 2x4x8 smax48 nwc2 nws1 vxpad0 | 64 | 23840 | 194/8 (regs) | 194/8 (regs+smem) | 210/8 (regs) | 202/8 (regs) | 252/8 (regs+smem) |
| 2x4x8 smax48 nwc2 nws1 vxpad1 | 64 | 25376 | 238/8 (regs) | 238/6 (smem) | 252/8 (regs) | 210/8 (regs) | 254/6 (smem) |
| 2x4x8 smax48 nwc2 nws2 vxpad0 | 128 | 23840 | 192/8 (regs) | 202/8 (regs) | 200/8 (regs) | 208/8 (regs) | 254/8 (regs) |
| 2x4x8 smax48 nwc2 nws2 vxpad1 | 128 | 25376 | 192/8 (regs) | 194/8 (regs) | 200/8 (regs) | 208/8 (regs) | 254/8 (regs) |
| 2x4x8 smax48 nwc3 nws1 vxpad0 | 96 | 30112 | 194/6 (regs) | 194/6 (regs) | 210/6 (regs) | 206/6 (regs) | 252/6 (regs) |
| 2x4x8 smax48 nwc3 nws1 vxpad1 | 96 | 31648 | 240/6 (regs) | 238/6 (regs) | 254/6 (regs) | 202/6 (regs) | 254/6 (regs) |
| 2x4x8 smax48 nwc3 nws2 vxpad0 | 192 | 30112 | 164/12 (regs) | 168/12 (regs) | 162/12 (regs) | 168/12 (regs) | 168/12 (regs) S44 K40 |
| 2x4x8 smax48 nwc3 nws2 vxpad1 | 192 | 31648 | 166/12 (regs) | 168/12 (regs) | 164/12 (regs) | 168/12 (regs) | 168/12 (regs) S40 K40 |
| 4x4x16 smax32 nwc1 nws1 vxpad0 | 32 | 12992 | 254/8 (regs) | 254/7 (smem) | 254/8 (regs) | 250/8 (regs) | 240/7 (smem) |
| 4x4x16 smax32 nwc1 nws1 vxpad1 | 32 | 14016 | 246/8 (regs) | 254/6 (smem) | 254/8 (regs) | 240/8 (regs) | 234/6 (smem) |
| 4x4x16 smax32 nwc1 nws2 vxpad0 | 64 | 12992 | 232/8 (regs) | 228/8 (regs) | 234/8 (regs) | 231/8 (regs) | 232/8 (regs) |
| 4x4x16 smax32 nwc1 nws2 vxpad1 | 64 | 14016 | 230/8 (regs) | 228/8 (regs) | 234/8 (regs) | 231/8 (regs) | 232/8 (regs) |
| 4x4x16 smax32 nwc2 nws1 vxpad0 | 64 | 17216 | 242/8 (regs) | 238/8 (regs) | 254/8 (regs) | 234/8 (regs) | 234/8 (regs) |
| 4x4x16 smax32 nwc2 nws1 vxpad1 | 64 | 18240 | 242/8 (regs) | 242/8 (regs) | 254/8 (regs) | 234/8 (regs) | 234/8 (regs) |
| 4x4x16 smax32 nwc2 nws2 vxpad0 | 128 | 17216 | 232/8 (regs) | 228/8 (regs) | 236/8 (regs) | 236/8 (regs) | 240/8 (regs) |
| 4x4x16 smax32 nwc2 nws2 vxpad1 | 128 | 18240 | 232/8 (regs) | 228/8 (regs) | 236/8 (regs) | 236/8 (regs) | 240/8 (regs) |
| 4x4x16 smax32 nwc3 nws1 vxpad0 | 96 | 21440 | 240/6 (regs) | 240/6 (regs) | 254/6 (regs) | 232/6 (regs) | 234/6 (regs) |
| 4x4x16 smax32 nwc3 nws1 vxpad1 | 96 | 22464 | 242/6 (regs) | 242/6 (regs) | 254/6 (regs) | 232/6 (regs) | 234/6 (regs) |
| 4x4x16 smax32 nwc3 nws2 vxpad0 | 192 | 21440 | 168/12 (regs) S4 K8 | 168/12 (regs) S4 K8 | 168/12 (regs) S4 K8 | 168/12 (regs) S8 K8 | 168/12 (regs) S8 K8 |
| 4x4x16 smax32 nwc3 nws2 vxpad1 | 192 | 22464 | 168/12 (regs) | 168/12 (regs) | 168/12 (regs) S4 K8 | 168/12 (regs) S24 K24 | 168/12 (regs) S32 K32 |
| 4x4x16 smax64 nwc1 nws1 vxpad0 | 32 | 24256 | 254/6 (smem) | 254/4 (smem) | 254/8 (regs) | 250/8 (regs) | 240/4 (smem) |
| 4x4x16 smax64 nwc1 nws1 vxpad1 | 32 | 26304 | 246/6 (smem) | 254/3 (smem) | 254/8 (regs+smem) | 240/8 (regs+smem) | 234/3 (smem) |
| 4x4x16 smax64 nwc1 nws2 vxpad0 | 64 | 24256 | 232/8 (regs) | 228/8 (regs+smem) | 234/8 (regs) | 231/8 (regs) | 232/8 (regs+smem) |
| 4x4x16 smax64 nwc1 nws2 vxpad1 | 64 | 26304 | 230/8 (regs) | 228/6 (smem) | 234/8 (regs) | 231/8 (regs) | 232/6 (smem) |
| 4x4x16 smax64 nwc2 nws1 vxpad0 | 64 | 32576 | 242/8 (regs+smem) | 238/6 (smem) | 254/8 (regs) | 234/8 (regs) | 234/6 (smem) |
| 4x4x16 smax64 nwc2 nws1 vxpad1 | 64 | 34624 | 242/8 (regs+smem) | 242/4 (smem) | 254/8 (regs) | 234/8 (regs) | 234/4 (smem) |
| 4x4x16 smax64 nwc2 nws2 vxpad0 | 128 | 32576 | 232/8 (regs) | 228/8 (regs) | 236/8 (regs) | 236/8 (regs) | 240/8 (regs) |
| 4x4x16 smax64 nwc2 nws2 vxpad1 | 128 | 34624 | 232/8 (regs) | 228/8 (regs+smem) | 236/8 (regs) | 236/8 (regs) | 240/8 (regs+smem) |
| 4x4x16 smax64 nwc3 nws1 vxpad0 | 96 | 40896 | 240/6 (regs) | 240/6 (regs+smem) | 254/6 (regs) | 232/6 (regs) | 234/6 (regs+smem) |
| 4x4x16 smax64 nwc3 nws1 vxpad1 | 96 | 42944 | 242/6 (regs) | 242/6 (regs+smem) | 254/6 (regs) | 232/6 (regs) | 234/6 (regs+smem) |
| 4x4x16 smax64 nwc3 nws2 vxpad0 | 192 | 40896 | 168/12 (regs) S4 K8 | 168/12 (regs+smem) S4 K8 | 168/12 (regs) S4 K8 | 168/12 (regs) S8 K8 | 168/12 (regs+smem) |
| 4x4x16 smax64 nwc3 nws2 vxpad1 | 192 | 42944 | 168/12 (regs) | 168/12 (regs+smem) | 168/12 (regs) S4 K8 | 168/12 (regs) S24 K24 | 168/12 (regs+smem) S32 K32 |

### static_o2_cb5 / vt_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 2x4x16 smax32 nwc4 nws1 vxpad0 | 128 | 49184 | 244/8 (regs) | 246/8 (regs+smem) | 254/8 (regs) | 214/8 (regs) | 222/8 (regs+smem) |
| 2x4x16 smax32 nwc4 nws2 vxpad0 | 256 | 55584 | 128/16 (regs+smem) S184 K88 | 128/8 (smem) S184 K88 | 128/16 (regs) S324 K104 | 128/16 (regs) S288 K104 | 128/8 (smem) S328 K104 |
| 2x4x8 smax32 nwc4 nws1 vxpad0 | 128 | 48928 | 244/8 (regs) | 246/8 (regs+smem) | 254/8 (regs) | 214/8 (regs) | 222/8 (regs+smem) |
| 2x4x8 smax32 nwc4 nws2 vxpad0 | 256 | 55328 | 128/16 (regs+smem) S184 K88 | 128/8 (smem) S184 K88 | 128/16 (regs) S324 K104 | 128/16 (regs) S288 K104 | 128/8 (smem) S328 K104 |
| 2x4x8 smax48 nwc4 nws1 vxpad0 | 128 | 69664 | 244/8 (regs+smem) | 246/4 (smem) | 254/8 (regs) | 214/8 (regs) | 222/4 (smem) |
| 2x4x8 smax48 nwc4 nws2 vxpad0 | 256 | 76064 | 128/16 (regs+smem) S184 K88 | 128/8 (smem) S184 K88 | 128/16 (regs) S324 K104 | 128/16 (regs) S288 K104 | 128/8 (smem) S328 K104 |
| 4x4x16 smax48 nwc4 nws1 vxpad0 | 128 | 70720 | 250/8 (regs+smem) | 252/4 (smem) | 254/8 (regs) | 238/8 (regs) | 254/4 (smem) |
| 4x4x16 smax48 nwc4 nws2 vxpad0 | 256 | 77120 | 128/16 (regs+smem) S548 K240 | 128/8 (smem) S568 K256 | 128/16 (regs+smem) S696 K232 | 128/16 (regs+smem) S956 K400 | 128/8 (smem) S916 K248 |
| 4x4x16 smax64 nwc4 nws1 vxpad0 | 128 | 91456 | 250/4 (smem) | 252/4 (smem) | 254/8 (regs+smem) | 238/8 (regs+smem) | 254/4 (smem) |
| 4x4x16 smax64 nwc4 nws2 vxpad0 | 256 | 97856 | 128/8 (smem) S548 K240 | 128/8 (smem) S568 K256 | 128/16 (regs+smem) S696 K232 | 128/16 (regs+smem) S956 K400 | 128/8 (smem) S916 K248 |

### vel2d_o3_cb10 / vt_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 2x4x16 smax32 nwc7 nws1 vxpad0 | 224 | 70240 | 128/14 (regs+smem) S216 K152 | 128/7 (smem) S260 K176 | 128/14 (regs) S336 K160 | 128/14 (regs) S288 K144 | 128/7 (smem) S276 K128 |
| 2x4x8 smax32 nwc7 nws1 vxpad0 | 224 | 69984 | 128/14 (regs+smem) S216 K152 | 128/7 (smem) S260 K176 | 128/14 (regs) S336 K160 | 128/14 (regs) S288 K144 | 128/7 (smem) S276 K128 |
| 2x4x8 smax48 nwc7 nws1 vxpad0 | 224 | 98656 | 128/7 (smem) S216 K152 | 128/7 (smem) S260 K176 | 128/14 (regs+smem) S336 K160 | 128/14 (regs+smem) S288 K144 | 128/7 (smem) S276 K128 |
| 4x4x16 smax48 nwc7 nws1 vxpad0 | 224 | 99712 | 128/7 (smem) S500 K296 | 128/7 (smem) S572 K400 | 128/14 (regs+smem) S584 K328 | 128/14 (regs+smem) S984 K336 | 128/7 (smem) S972 K352 |
| 4x4x16 smax64 nwc7 nws1 vxpad0 | 224 | 128384 | 128/7 (smem) S500 K296 | - | 128/7 (smem) S584 K328 | 128/7 (smem) S984 K336 | - |

### vel2d_o3_cb5 / vt_fwd

| cfg | threads | smem dyn (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| 2x4x16 smax32 nwc4 nws1 vxpad0 | 128 | 52768 | 244/8 (regs) | 246/4 (smem) | 254/8 (regs) | 212/8 (regs) | 224/4 (smem) |
| 2x4x16 smax32 nwc4 nws2 vxpad0 | 256 | 59168 | 128/16 (regs+smem) S284 K120 | 128/8 (smem) S276 K112 | 128/16 (regs) S428 K136 | 128/16 (regs) S364 K136 | 128/8 (smem) S304 K136 |
| 2x4x8 smax32 nwc4 nws1 vxpad0 | 128 | 52512 | 244/8 (regs) | 246/4 (smem) | 254/8 (regs) | 212/8 (regs) | 224/4 (smem) |
| 2x4x8 smax32 nwc4 nws2 vxpad0 | 256 | 58912 | 128/16 (regs+smem) S284 K120 | 128/8 (smem) S276 K112 | 128/16 (regs) S428 K136 | 128/16 (regs) S364 K136 | 128/8 (smem) S304 K136 |
| 2x4x8 smax48 nwc4 nws1 vxpad0 | 128 | 75040 | 244/8 (regs+smem) | 246/4 (smem) | 254/8 (regs) | 212/8 (regs) | 224/4 (smem) |
| 2x4x8 smax48 nwc4 nws2 vxpad0 | 256 | 81440 | 128/16 (regs+smem) S284 K120 | 128/8 (smem) S276 K112 | 128/16 (regs+smem) S428 K136 | 128/16 (regs+smem) S364 K136 | 128/8 (smem) S304 K136 |
| 4x4x16 smax48 nwc4 nws1 vxpad0 | 128 | 76096 | 250/8 (regs+smem) | 252/4 (smem) | 254/8 (regs) | 238/8 (regs) | 254/4 (smem) |
| 4x4x16 smax48 nwc4 nws2 vxpad0 | 256 | 82496 | 128/16 (regs+smem) S584 K272 | 128/8 (smem) S600 K280 | 128/16 (regs+smem) S824 K288 | 128/16 (regs+smem) S1024 K416 | 128/8 (smem) S720 K304 |
| 4x4x16 smax64 nwc4 nws1 vxpad0 | 128 | 98624 | 250/4 (smem) | 252/4 (smem) | 254/8 (regs+smem) | 238/8 (regs+smem) | 254/4 (smem) |
| 4x4x16 smax64 nwc4 nws2 vxpad0 | 256 | 105024 | 128/8 (smem) S584 K272 | - | 128/16 (regs+smem) S824 K288 | 128/16 (regs+smem) S1024 K416 | - |

## Flags (spills, stack, no fit, below 8 warps / SM, compile errors)

| arch | case | family | cfg | regs | flags |
|---|---|---|---|---|---|
| sm_100 | knot1d_o2_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st288 ld580, STACK 104 |
| sm_120 | knot1d_o2_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st328 ld576, STACK 104 |
| sm_80 | knot1d_o2_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st184 ld412, STACK 88 |
| sm_89 | knot1d_o2_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st184 ld412, STACK 88 |
| sm_90 | knot1d_o2_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st324 ld800, STACK 104 |
| sm_100 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st288 ld580, STACK 104 |
| sm_120 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st328 ld576, STACK 104 |
| sm_80 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st184 ld412, STACK 88 |
| sm_89 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st184 ld412, STACK 88 |
| sm_90 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st324 ld800, STACK 104 |
| sm_120 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws1 vxpad0 | 222 | LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws1 vxpad0 | 246 | LOW_OCC 4w |
| sm_100 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st288 ld580, STACK 104 |
| sm_120 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st328 ld576, STACK 104 |
| sm_80 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st184 ld412, STACK 88 |
| sm_89 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st184 ld412, STACK 88 |
| sm_90 | knot1d_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st324 ld800, STACK 104 |
| sm_120 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 | 252 | LOW_OCC 4w |
| sm_100 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st956 ld2068, STACK 400 |
| sm_120 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st916 ld1900, STACK 248 |
| sm_80 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st548 ld1264, STACK 240 |
| sm_89 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st568 ld1372, STACK 256 |
| sm_90 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st696 ld2132, STACK 232 |
| sm_120 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_80 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 250 | LOW_OCC 4w |
| sm_89 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 252 | LOW_OCC 4w |
| sm_100 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws2 vxpad0 | 128 | SPILL st956 ld2068, STACK 400 |
| sm_120 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws2 vxpad0 | 128 | SPILL st916 ld1900, STACK 248 |
| sm_80 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws2 vxpad0 | 128 | SPILL st548 ld1264, STACK 240 |
| sm_89 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws2 vxpad0 | 128 | SPILL st568 ld1372, STACK 256 |
| sm_90 | knot1d_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws2 vxpad0 | 128 | SPILL st696 ld2132, STACK 232 |
| sm_100 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st168 ld308, STACK 88 |
| sm_120 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st168 ld308, STACK 88 |
| sm_80 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st124 ld264, STACK 80 |
| sm_89 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st180 ld320, STACK 120 |
| sm_90 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st196 ld372, STACK 88 |
| sm_100 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad1 | 128 | SPILL st204 ld396, STACK 104 |
| sm_120 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad1 | 128 | SPILL st196 ld384, STACK 104 |
| sm_80 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad1 | 128 | SPILL st136 ld336, STACK 72 |
| sm_89 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad1 | 128 | SPILL st168 ld376, STACK 88 |
| sm_90 | static_o0_10x10 | vs_fwd | 2x4x16 smax32 nwc7 nws1 vxpad1 | 128 | SPILL st208 ld492, STACK 80 |
| sm_100 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st168 ld308, STACK 88 |
| sm_120 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st168 ld308, STACK 88 |
| sm_80 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st124 ld264, STACK 80 |
| sm_89 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st180 ld320, STACK 120 |
| sm_90 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st196 ld372, STACK 88 |
| sm_100 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad1 | 128 | SPILL st204 ld396, STACK 104 |
| sm_120 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad1 | 128 | SPILL st196 ld384, STACK 104 |
| sm_80 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad1 | 128 | SPILL st136 ld336, STACK 72 |
| sm_89 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad1 | 128 | SPILL st168 ld376, STACK 88 |
| sm_90 | static_o0_10x10 | vs_fwd | 2x4x8 smax32 nwc7 nws1 vxpad1 | 128 | SPILL st208 ld492, STACK 80 |
| sm_100 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st168 ld308, STACK 88 |
| sm_120 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st168 ld308, STACK 88, LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st124 ld264, STACK 80 |
| sm_89 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st180 ld320, STACK 120, LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st196 ld372, STACK 88 |
| sm_100 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad1 | 128 | SPILL st204 ld396, STACK 104 |
| sm_120 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad1 | 128 | SPILL st196 ld384, STACK 104, LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad1 | 128 | SPILL st136 ld336, STACK 72 |
| sm_89 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad1 | 128 | SPILL st168 ld376, STACK 88, LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs_fwd | 2x4x8 smax48 nwc7 nws1 vxpad1 | 128 | SPILL st208 ld492, STACK 80 |
| sm_100 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 128 | SPILL st648 ld1136, STACK 304 |
| sm_120 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 128 | SPILL st572 ld1076, STACK 240, LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 128 | SPILL st496 ld984, STACK 264 |
| sm_89 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 128 | SPILL st508 ld976, STACK 272, LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 128 | SPILL st616 ld1180, STACK 368 |
| sm_100 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 | 128 | SPILL st856 ld2480, STACK 256 |
| sm_120 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 | 128 | SPILL st800 ld2448, STACK 232, LOW_OCC 7w |
| sm_80 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 | 128 | SPILL st736 ld2284, STACK 232 |
| sm_89 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 | 128 | SPILL st804 ld2412, STACK 304, LOW_OCC 7w |
| sm_90 | static_o0_10x10 | vs_fwd | 4x4x16 smax64 nwc7 nws1 vxpad1 | 128 | SPILL st1012 ld2720, STACK 272 |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc1 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc1 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc1 nws1 vxpad1 | 242 | LOW_OCC 7w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc1 nws1 vxpad1 | 245 | LOW_OCC 7w |
| sm_100 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws1 vxpad0 | 206 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws1 vxpad0 | 252 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws1 vxpad0 | 194 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws1 vxpad0 | 194 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws1 vxpad0 | 210 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws1 vxpad1 | 202 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws1 vxpad1 | 240 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws1 vxpad1 | 238 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws2 vxpad0 | 168 | SPILL st44 ld84, STACK 40 |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x16 smax32 nwc3 nws2 vxpad1 | 168 | SPILL st40 ld60, STACK 40 |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc1 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc1 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc1 nws1 vxpad1 | 242 | LOW_OCC 7w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc1 nws1 vxpad1 | 245 | LOW_OCC 7w |
| sm_100 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws1 vxpad0 | 206 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws1 vxpad0 | 252 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws1 vxpad0 | 194 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws1 vxpad0 | 194 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws1 vxpad0 | 210 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws1 vxpad1 | 202 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws1 vxpad1 | 240 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws1 vxpad1 | 238 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws2 vxpad0 | 168 | SPILL st44 ld84, STACK 40 |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax32 nwc3 nws2 vxpad1 | 168 | SPILL st40 ld60, STACK 40 |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc1 nws1 vxpad0 | 254 | LOW_OCC 5w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc1 nws1 vxpad0 | 254 | LOW_OCC 5w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc1 nws1 vxpad1 | 242 | LOW_OCC 5w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc1 nws1 vxpad1 | 245 | LOW_OCC 5w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc2 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc2 nws1 vxpad1 | 238 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws1 vxpad0 | 206 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws1 vxpad0 | 252 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws1 vxpad0 | 194 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws1 vxpad0 | 194 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws1 vxpad0 | 210 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws1 vxpad1 | 202 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws1 vxpad1 | 240 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws1 vxpad1 | 238 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws2 vxpad0 | 168 | SPILL st44 ld84, STACK 40 |
| sm_120 | static_o0_48x1 | vs_fwd | 2x4x8 smax48 nwc3 nws2 vxpad1 | 168 | SPILL st40 ld60, STACK 40 |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 | 240 | LOW_OCC 7w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc1 nws1 vxpad0 | 254 | LOW_OCC 7w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 | 234 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc1 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 | 232 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 | 234 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 | 240 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 | 240 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad0 | 254 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 | 232 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 | 234 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 | 242 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 | 242 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 | 168 | SPILL st8 ld8, STACK 8 |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 | 168 | SPILL st8 ld8, STACK 8 |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 | 168 | SPILL st4 ld4, STACK 8 |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 | 168 | SPILL st4 ld4, STACK 8 |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad0 | 168 | SPILL st4 ld4, STACK 8 |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 | 168 | SPILL st24 ld24, STACK 24 |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 | 168 | SPILL st32 ld68, STACK 32 |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax32 nwc3 nws2 vxpad1 | 168 | SPILL st4 ld4, STACK 8 |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws1 vxpad0 | 240 | LOW_OCC 4w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws1 vxpad0 | 254 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws1 vxpad1 | 234 | LOW_OCC 3w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws1 vxpad1 | 246 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws1 vxpad1 | 254 | LOW_OCC 3w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws2 vxpad1 | 232 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc1 nws2 vxpad1 | 228 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc2 nws1 vxpad0 | 234 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc2 nws1 vxpad0 | 238 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc2 nws1 vxpad1 | 234 | LOW_OCC 4w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc2 nws1 vxpad1 | 242 | LOW_OCC 4w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad0 | 232 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad0 | 234 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad0 | 240 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad0 | 240 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad0 | 254 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad1 | 232 | LOW_OCC 6w |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad1 | 234 | LOW_OCC 6w |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad1 | 242 | LOW_OCC 6w |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad1 | 242 | LOW_OCC 6w |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws1 vxpad1 | 254 | LOW_OCC 6w |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad0 | 168 | SPILL st8 ld8, STACK 8 |
| sm_80 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad0 | 168 | SPILL st4 ld4, STACK 8 |
| sm_89 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad0 | 168 | SPILL st4 ld4, STACK 8 |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad0 | 168 | SPILL st4 ld4, STACK 8 |
| sm_100 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad1 | 168 | SPILL st24 ld24, STACK 24 |
| sm_120 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad1 | 168 | SPILL st32 ld68, STACK 32 |
| sm_90 | static_o0_48x1 | vs_fwd | 4x4x16 smax64 nwc3 nws2 vxpad1 | 168 | SPILL st4 ld4, STACK 8 |
| sm_100 | static_o2_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st288 ld580, STACK 104 |
| sm_120 | static_o2_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st328 ld576, STACK 104 |
| sm_80 | static_o2_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st184 ld412, STACK 88 |
| sm_89 | static_o2_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st184 ld412, STACK 88 |
| sm_90 | static_o2_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st324 ld800, STACK 104 |
| sm_100 | static_o2_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st288 ld580, STACK 104 |
| sm_120 | static_o2_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st328 ld576, STACK 104 |
| sm_80 | static_o2_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st184 ld412, STACK 88 |
| sm_89 | static_o2_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st184 ld412, STACK 88 |
| sm_90 | static_o2_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st324 ld800, STACK 104 |
| sm_120 | static_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws1 vxpad0 | 222 | LOW_OCC 4w |
| sm_89 | static_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws1 vxpad0 | 246 | LOW_OCC 4w |
| sm_100 | static_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st288 ld580, STACK 104 |
| sm_120 | static_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st328 ld576, STACK 104 |
| sm_80 | static_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st184 ld412, STACK 88 |
| sm_89 | static_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st184 ld412, STACK 88 |
| sm_90 | static_o2_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st324 ld800, STACK 104 |
| sm_120 | static_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_89 | static_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 | 252 | LOW_OCC 4w |
| sm_100 | static_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st956 ld2068, STACK 400 |
| sm_120 | static_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st916 ld1900, STACK 248 |
| sm_80 | static_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st548 ld1264, STACK 240 |
| sm_89 | static_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st568 ld1372, STACK 256 |
| sm_90 | static_o2_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st696 ld2132, STACK 232 |
| sm_120 | static_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_80 | static_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 250 | LOW_OCC 4w |
| sm_89 | static_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 252 | LOW_OCC 4w |
| sm_100 | static_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws2 vxpad0 | 128 | SPILL st956 ld2068, STACK 400 |
| sm_120 | static_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws2 vxpad0 | 128 | SPILL st916 ld1900, STACK 248 |
| sm_80 | static_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws2 vxpad0 | 128 | SPILL st548 ld1264, STACK 240 |
| sm_89 | static_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws2 vxpad0 | 128 | SPILL st568 ld1372, STACK 256 |
| sm_90 | static_o2_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws2 vxpad0 | 128 | SPILL st696 ld2132, STACK 232 |
| sm_100 | vel2d_o3_cb10 | vt_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st288 ld884, STACK 144 |
| sm_120 | vel2d_o3_cb10 | vt_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st276 ld880, STACK 128, LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st216 ld648, STACK 152 |
| sm_89 | vel2d_o3_cb10 | vt_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st260 ld688, STACK 176, LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt_fwd | 2x4x16 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st336 ld1056, STACK 160 |
| sm_100 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st288 ld884, STACK 144 |
| sm_120 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st276 ld880, STACK 128, LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st216 ld648, STACK 152 |
| sm_89 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st260 ld688, STACK 176, LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax32 nwc7 nws1 vxpad0 | 128 | SPILL st336 ld1056, STACK 160 |
| sm_100 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st288 ld884, STACK 144 |
| sm_120 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st276 ld880, STACK 128, LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st216 ld648, STACK 152, LOW_OCC 7w |
| sm_89 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st260 ld688, STACK 176, LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt_fwd | 2x4x8 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st336 ld1056, STACK 160 |
| sm_100 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st984 ld2448, STACK 336 |
| sm_120 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st972 ld2428, STACK 352, LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st500 ld1572, STACK 296, LOW_OCC 7w |
| sm_89 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st572 ld1640, STACK 400, LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax48 nwc7 nws1 vxpad0 | 128 | SPILL st584 ld2704, STACK 328 |
| sm_100 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 128 | SPILL st984 ld2448, STACK 336, LOW_OCC 7w |
| sm_80 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 128 | SPILL st500 ld1572, STACK 296, LOW_OCC 7w |
| sm_90 | vel2d_o3_cb10 | vt_fwd | 4x4x16 smax64 nwc7 nws1 vxpad0 | 128 | SPILL st584 ld2704, STACK 328, LOW_OCC 7w |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws1 vxpad0 | 224 | LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws1 vxpad0 | 246 | LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st364 ld836, STACK 136 |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st304 ld844, STACK 136 |
| sm_80 | vel2d_o3_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st284 ld536, STACK 120 |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st276 ld520, STACK 112 |
| sm_90 | vel2d_o3_cb5 | vt_fwd | 2x4x16 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st428 ld1056, STACK 136 |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws1 vxpad0 | 224 | LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws1 vxpad0 | 246 | LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st364 ld836, STACK 136 |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st304 ld844, STACK 136 |
| sm_80 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st284 ld536, STACK 120 |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st276 ld520, STACK 112 |
| sm_90 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax32 nwc4 nws2 vxpad0 | 128 | SPILL st428 ld1056, STACK 136 |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws1 vxpad0 | 224 | LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws1 vxpad0 | 246 | LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st364 ld836, STACK 136 |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st304 ld844, STACK 136 |
| sm_80 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st284 ld536, STACK 120 |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st276 ld520, STACK 112 |
| sm_90 | vel2d_o3_cb5 | vt_fwd | 2x4x8 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st428 ld1056, STACK 136 |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws1 vxpad0 | 252 | LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st1024 ld2528, STACK 416 |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st720 ld2412, STACK 304 |
| sm_80 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st584 ld1400, STACK 272 |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st600 ld1504, STACK 280 |
| sm_90 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax48 nwc4 nws2 vxpad0 | 128 | SPILL st824 ld2712, STACK 288 |
| sm_120 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 254 | LOW_OCC 4w |
| sm_80 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 250 | LOW_OCC 4w |
| sm_89 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws1 vxpad0 | 252 | LOW_OCC 4w |
| sm_100 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws2 vxpad0 | 128 | SPILL st1024 ld2528, STACK 416 |
| sm_80 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws2 vxpad0 | 128 | SPILL st584 ld1400, STACK 272 |
| sm_90 | vel2d_o3_cb5 | vt_fwd | 4x4x16 smax64 nwc4 nws2 vxpad0 | 128 | SPILL st824 ld2712, STACK 288 |
