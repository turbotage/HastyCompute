import torch

import test

from hastypy.base.opalg import Vector, IdentityOp, ScaleOp, CompositeOp, AdditiveOp

vec = []
for i in range(2):
    vec.append(torch.rand(3,3,3))
    
print(vec)
print('Before Op')

idop = IdentityOp([(3,3,3),(3,3,3)])
final_op = idop + ScaleOp(idop, torch.tensor(2.0))

print(final_op(Vector(vec)))
print('After op')


