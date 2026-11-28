"""Publish a flow value to the viewer and compute its data-derived surfaces (blood pool + wall shell), then print the URL.

    python -I show_surfaces.py <viewer_flow.hv> --server https://<host>:8443 [--no-wall]

The surfaces come from vel / mag only (flow_bloodpool, flow_wall_shell). Needs a running hasty-serve (started by the user; this script never starts one). --server or HASTY_SERVER is required.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute.store import load_any  # noqa: E402
from hastycompute.v2 import Client  # noqa: E402

A = sys.argv[1:]
server = A[A.index("--server") + 1] if "--server" in A else os.environ.get("HASTY_SERVER")
if not server:
    sys.exit("give --server https://<host>:<port> (or set HASTY_SERVER)")
server = server.rstrip("/")
path = next(a for i, a in enumerate(A) if not a.startswith("--") and (i == 0 or A[i - 1] != "--server"))
c = Client(server)
flow = load_any(path)
name = Path(path).parent.name + " flow"
fid = c.put(flow, name=name, metadata={"kind": "flow4d"})
pool, msg = c.execute("flow_bloodpool", [fid], {}, name + " blood pool")
print("blood pool:", msg[:300])
url = f"{server}/flow/index.html?id={fid}&mesh={pool[0]}"
if "--no-wall" not in A:
    wall, wmsg = c.execute("flow_wall_shell", [fid, pool[0]], {}, name + " wall shell")
    print("wall shell:", wmsg[:300])
    url += f"&wall={wall[0]}"
print("\nopen:", url)
