"""Export the v2-validation readout layout for the Rust generator (girfqa --vrot): the girf_sensitivity budget readouts
(48, offset 7) with their 10 previous TRs (encode, rotation), from the cached scanrot_48_7_10.npz of that study."""
import os, numpy as np
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute.store import load_any, exists_any, glob_any  # noqa: E402,F401  (.npy or the .hv written instead)
REPO = os.environ.get('HASTY_REPO', '/home/turbotage/Documents/GitHub/HastyCompute')
z = load_any(os.path.join(REPO, 'data/hasty/seq/studies/girf_sensitivity_2026-11-11/scanrot_48_7_10.npz'))
out = os.path.join(REPO, 'data/hasty/seq/girf_qa/inputs')
os.makedirs(out, exist_ok=True)
np.save(os.path.join(out, 'vrot_R.npy'), np.ascontiguousarray(z['R'], dtype=np.float64))
np.save(os.path.join(out, 'vrot_enc.npy'), np.ascontiguousarray(z['enc'], dtype=np.int64))
np.save(os.path.join(out, 'vrot_idx.npy'), np.ascontiguousarray(z['idx'], dtype=np.int64))
print('R', z['R'].shape, 'enc', z['enc'].shape, '->', out)
