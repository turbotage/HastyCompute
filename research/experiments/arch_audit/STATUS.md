# arch_audit: status (2026-11-20)

Phase 1 is complete. Results are in research/arch_targets.md §5; the tool is
rustlib/crates/hasty-bench/src/bin/kernaudit.rs.

## Runs (CPU only, NVRTC 13.0, 4 threads, CPUQuota 400 %, MemoryMax 8 G)

| dir | what | rows |
|---|---|---|
| out/ | full audit: 6 engine cases × all autotune candidates (tile, vs, vt, ra, kl, whatif) + FFT axis passes, sm_80/89/90/100/120 | 5611, 0 compile errors |
| out_lb4/ | ra / kl with `__launch_bounds__(NTA, 4)` | 860 |
| out_lb3/ | ra / kl with `__launch_bounds__(NTA, 3)` | 860 |
| out_vlb2/ | vs / vt with `__launch_bounds__(NTV, 2)` | 511 |

Digest: `python3 analyze.py out/results.jsonl out_lb4/results.jsonl out_lb3/results.jsonl out_vlb2/results.jsonl`.

## Rerun (after phase-2 kernel changes)

- `./run.sh` → out/ (tens of minutes at CPUQuota 400 %); `./run.sh <tag> [kernaudit options]` → out_<tag>/, e.g.
  `./run.sh sm90 --arch sm_90`, or a what-if `./run.sh lb4 --subst "__launch_bounds__(NTA)=>__launch_bounds__(NTA, 4)" --families ra,kl`.
- kernaudit mirrors private engine code (source define prefixes, shared-memory formulas, candidate loops) as of
  2026-11-20. If segnufft*.rs / segnufft/*.rs / hasty-fft plan.rs change, re-check the mirrors listed in the tool header.
- NVRTC 13 has its own cubin cache; a cache hit returns no ptxas report, so the tool always passes `--no-cache`.

## Not audited

The fused pad / crop hooked FFT x passes (hook code is private), the multi-image kernels, chirp / plain NUFFT.

## Phase 2 (2026-11-20)

| dir | what | rows |
|---|---|---|
| out_p2/ | families `ra2`, `kl2`, `vs2`: staging variants of ra / kl (in place, cp.async, chunks 8–24, LBMIN 0 / 3 / 4) and vs / vt sub-chunks, all five archs | 2970, 0 compile errors |
| out_p2base/ | base ra / kl / vs re-audited after the source edit (vel2d_o3_cb5, static_o0_48x1): identical to out/ | 608 |

Digest: `python3 analyze_p2.py .` → research/experiments/phase2/audit_tables.md. Rerun:
`kernaudit --families ra2,kl2,vs2 --out out_p2` (≈ 25 min at 4 threads). research/sample_kernels.md §19.2.
