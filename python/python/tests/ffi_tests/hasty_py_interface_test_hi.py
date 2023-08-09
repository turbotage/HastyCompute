import torch

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
#torch.ops.load_library(dll_path)

torch.classes.load_library(dll_path)

print(torch.classes.loaded_libraries)


hi_mod = torch.classes.HastyInterface

FunctionLambda = hi_mod.FunctionLambda


script = """
def apply(a: List[Tensor], b: List[Tensor]):
	for i in range(len(b)):
		a[i] += b[i]
	#print('Numpy sqrt: ', np.sqrt(a[0].numpy()))
"""

capture_list = []
input_list = []
for i in range(3):
	capture_list.append(torch.ones(3,3))
	input_list.append(torch.ones(3,3))

fl = FunctionLambda(script, "apply", capture_list)


print(capture_list)
print(input_list)

fl.apply(input_list)

print(capture_list)
print(input_list)

