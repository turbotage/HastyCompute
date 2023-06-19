
import torch

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_svt = torch.ops.HastySVT

streams = [[torch.cuda.default_stream(), torch.cuda.Stream()], [torch.cuda.Stream()]]

hasty_svt.print_test(4.63, True, None)