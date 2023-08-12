import torch

from hastypy.base.base import Vector, IdentityOp

vec = []
for i in range(2):
    vec.append(torch.rand(3,3,3))
    
idop = IdentityOp([(3,3,3),(3,3,3)])

print(idop(Vector(vec)))

