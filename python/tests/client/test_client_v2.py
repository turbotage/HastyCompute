"""Python client against a running hasty-serve (TLS):
    HASTY_SERVER=https://localhost:18443 HASTY_CA=rustlib/target/test-certs/ca.pem rustlib/.venv/bin/python -I python/tests/client/test_client_v2.py
"""
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from hastycompute.v2 import Client, show  # noqa: E402

c = Client()
events = []
t = threading.Thread(target=lambda: [events.append(e) for e in c.watch()], daemon=True)
t.start()
vol = (np.random.default_rng(0).standard_normal((12, 40, 50)) + 1j).astype(np.complex64)
meta = {"axes": ["z", "y", "x"], "voxel_mm": [2, 2, 2]}
i = c.put({"vol": vol, "info": "python"}, name="py upload", metadata=meta, codec="zstd", block_size=8192)
got = c.get(i, codec="lz4")
assert np.array_equal(got["vol"], vol) and got["info"] == "python"
assert np.array_equal(c.get(i, slice='["vol"][3, ::4, -5:]', codec="zstd"), vol[3, ::4, -5:])
c.write(i, np.zeros((40, 50), np.complex64), slice='["vol"][0]')
assert not c.get(i, slice='["vol"][0]').any()
e = [x for x in c.list() if x["id"] == i][0]
assert e["name"] == "py upload" and e["metadata"] == meta and e["summary"]["type"] == "dict"
v = c.put(vol, name="vol")
(q,), msg = c.execute("comprep_u16", [v], {"mode": "abs"})
assert c.get(q, delta=True).dtype == np.uint16
(f,), _ = c.execute("fft", [v], {"axes": [-1, -2]})
assert np.allclose(c.get(f), np.fft.fftn(vol, axes=(-1, -2)), rtol=1e-4, atol=1e-3)
assert "fft" in c.commands()
url = show(vol[0].real.copy(), name="py show", axes=["y", "x"])
c.delete(i)
import time  # noqa: E402

time.sleep(0.3)
kinds = {x["kind"] for x in events}
assert {"put", "update", "show", "delete"} <= kinds, kinds
print(f"python v2 client: put(zstd)/get(lz4, slices)/write(slice)/list/execute(quantize, fft vs numpy)/show/delete + watch: ok\n  viewer url {url}")
