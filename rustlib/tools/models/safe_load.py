"""Load a torch checkpoint from an untrusted source without executing code from the pickle.

torch.load(weights_only=False) runs arbitrary pickled callables. `load_state(path, key)` instead unpickles with an
allowlist (tensor rebuild helpers, storages, OrderedDict, numpy scalar / dtype used by nnU-Net's bookkeeping fields);
any other global aborts. Returns the object stored under `key` (default: nnU-Net's "network_weights").
"""
import collections
import pickle
import types

import numpy as np
import torch

_ALLOW = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("numpy", "dtype"): np.dtype,
    ("torch", "device"): torch.device,
    ("_codecs", "encode"): __import__("_codecs").encode,  # str -> bytes (numpy scalar payloads)
}
for _n in ("Float", "Double", "Half", "BFloat16", "Long", "Int", "Short", "Char", "Byte", "Bool"):
    _ALLOW[("torch", _n + "Storage")] = getattr(torch, _n + "Storage")


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _ALLOW[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"checkpoint references disallowed global {module}.{name}") from None


_shim = types.ModuleType("safe_pickle")
_shim.Unpickler = _Unpickler
_shim.load = pickle.load
_shim.UnpicklingError = pickle.UnpicklingError


def load_state(path, key="network_weights"):
    ck = torch.load(path, map_location="cpu", pickle_module=_shim, weights_only=False)
    return ck[key] if key else ck
