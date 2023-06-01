
import torch
import numpy as np

import ctypes as ct

dll_path = "D:\\Documents\\GitHub\\HastyCompute\\out\\install\\x64-release-cuda\\bin\\HastyPyInterface.dll"
loaded = ct.CDLL(dll_path)

torch.ops.load_library(dll_path)
hasty_py = torch.ops.HastyPyInterface



