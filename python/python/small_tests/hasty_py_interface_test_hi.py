import torch

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
#torch.ops.load_library(dll_path)

torch.classes.load_library(dll_path)

print(torch.classes.loaded_libraries)


hi_mod = torch.classes.HastyInterface

FunctionLambda = hi_mod.FunctionLambda


script = """
def apply(a: Tensor, b: List[Tensor]):
	for t in b:
		a += t
"""

tlist = []
for i in range(3):
    tlist.append(torch.ones(3,3))

fl = FunctionLambda(script, "apply", tlist)

a = torch.rand(3,3)

print(a)
print(tlist)

fl.apply(a)

print(a)
print(tlist)

