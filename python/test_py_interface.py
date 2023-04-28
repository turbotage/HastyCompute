import torch

import sys
import os

print(os.getpid())

#class add_path():§
# 	def __init__(self, path):
# 		self.path = path

# 	def __enter__(self):
# 		sys.path.insert(0, self.path)

# 	def __exit__(self, exc_type, exc_value, traceback):
# 		try:
# 			sys.path.remove(self.path)
# 		except ValueError:
# 			pass


#with add_path("D:\\Documents\\GitHub\\HastyCompute\\out\install\\x64-debug-cuda\\bin"):
# 	try:
# 		torch.ops.load_library("D:\\Documents\\GitHub\\HastyCompute\\out\install\\x64-debug-cuda\\bin\\HastyPyInterface.dll")
# 	except Exception as e:
# 		 print(e)

torch.ops.load_library("D:\\GitHub\\HastyCompute\\out\\install\\x64-release-cuda\\bin\\HastyPyInterface.dll")

#import ctypes as ct
#cdll = ct.CDLL("D:\\GitHub\\HastyCompute\\out\\install\\x64-release-cuda\\bin\\HastyPyInterface.dll")

shape = (3,3,3)
a = torch.randint(0,10,shape, dtype=torch.float32)
print(a)

a = torch.ops.HastyPyInterface.add_one(a)

print(a)