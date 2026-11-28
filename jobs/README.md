# jobs

Run drivers, chains, tests and small helper scripts (tracked in git), so nothing but program output goes to
`data/hasty/`.

| path | what | git |
|---|---|---|
| `jobs/phantom/` | phantom / clearance / LBM chains and campaigns (`make_phantom`, `clearance.py`, `lbm_burst.sh`, …) | tracked |
| `jobs/sim/` | k-space simulation tests and profiles (`simphantom`, Nsight Compute) | tracked |
| `jobs/tools/` | small analysis helpers (Python, run with `rustlib/.venv/bin/python -I`) | tracked |
| `jobs/logs/` | job logs (`<job>.log`) | ignored |
| `jobs/locks/` | the GPU lock (`gpu.lock`, `flock` around every GPU job) | ignored |
| `rustlib/target/snap/` | binary snapshots (e.g. a pre-change reference build) | ignored |

Rules:
- Scripts `cd` to the repo root and use repo-relative paths.
- Big outputs (arrays, run directories) are written by the programs under `data/hasty/…`; logs and scripts never.
- GPU jobs: `flock jobs/locks/gpu.lock systemd-run --user --scope -q -p MemoryMax=… -p MemorySwapMax=0 …`.
