# Nsight Compute, one knot launch per variant (RTX 4090, 300 W, ncu locked base clocks)

opbench flow_thorax at 3 mm (192 × 192 × 96 grid), 2-D knots 6:1,16:3, order 3, reduced scan (0.44 min), --no-tune,
variant fixed by env (run_ncu.sh; vt with --cb-cap 5 = 50 channels: run_ncu_vt.sh); sections LaunchStats, Occupancy,
SpeedOfLight, WarpStateStats, SchedulerStats; `--launch-skip 6 --launch-count 1`. Details: ncu/<run>.details.txt.

| run | variant | ms | regs | smem / block (KB) | theoretical occupancy % | achieved % | active warps / SM | no eligible % | block limit regs / smem |
|---|---|---|---|---|---|---|---|---|---|
| ra_base | ra 4,7,16 base | 8.05 | 128 | 29.89 | 25 | 24.70 | 11.85 | 53.96 | 4 / 3 |
| ra_ip16 | in place, chunk 16 | 8.17 | 161 | 26.51 | 25 | 24.70 | 11.86 | 54.86 | 3 / 3 |
| ra_ip12 | in place, chunk 12 | 8.43 | 161 | 19.98 | 25 | 24.83 | 11.92 | 55.02 | 3 / 3 |
| ra_ip12l4 | in place, chunk 12, hint 4 | 7.32 | 128 | 19.98 | 33.33 | 32.80 | 15.74 | 47.43 | 4 / 4 |
| ra_cp16 | cp.async, chunk 16 | 8.12 | 125 | 28.50 | 25 | 24.77 | 11.89 | 53.71 | 4 / 3 |
| ra_cp12 | cp.async, chunk 12 | 7.20 | 125 | 21.47 | 33.33 | 33.91 | 16.27 | 46.78 | 4 / 4 |
| ra_cp8 | cp.async, chunk 8 | 7.44 | 125 | 14.45 | 33.33 | 32.88 | 15.78 | 45.85 | 4 / 4 |
| kl_base | kl 4,7,16 base | 8.10 | 158 | 26.51 | 25 | 24.68 | 11.84 | 54.40 | 3 / 3 |
| kl_s12l4 | chunk 12, hint 4 | 7.24 | 125 | 19.98 | 33.33 | 32.96 | 15.82 | 46.06 | 4 / 4 |
| kl_cp12 | cp.async, chunk 12 | 6.98 | 123 | 21.47 | 33.33 | 33.31 | 15.99 | 44.86 | 4 / 4 |
| vt5_base | vt 4 × 4 × 16 SMAX 48, nwc 4 one pass | 5.72 | 254 | 76.10 | 8.33 | 8.30 | 3.98 | 70.96 | |
| vt5_nws2 | nwc 4, nws 2 one pass | 5.31 | 236 | 82.50 | 16.67 | 16.51 | 7.93 | 64.25 | |
| vt5_q24 | nwc 4, sub-chunk 24 | 3.99 | 254 | 42.30 | 16.67 | 16.48 | 7.91 | 57.10 | |
| vt5_q16 | nwc 4, sub-chunk 16 | 4.25 | 254 | 31.04 | 16.67 | 16.43 | 7.89 | 58.45 | |

(The first vt runs, vt_*.log, failed by design of the case: at 3 mm the coil batch is 10, where nwc 4 is not a valid
vt shape; rerun with --cb-cap 5.)
