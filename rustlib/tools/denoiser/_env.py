"""Import first (before torch): keeps torch / inductor / triton caches inside the repository (rustlib/target)."""
import os
from pathlib import Path

_T = Path(__file__).resolve().parents[2] / "target"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_T / "torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_T / "triton"))
os.environ.setdefault("TORCH_HOME", str(_T / "torch-home"))
# inductor writes precompiled headers (and AOTI extracts packages) under tempfile.gettempdir(): keep that inside too
(_T / "tmp").mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_T / "tmp")
