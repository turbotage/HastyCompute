import torch


A = torch.rand(20, 1, 2, 2, 2)

avec = []
for i in range(10):
	idx = 2*i
	at = A[idx:idx+2,...]
	at = at.flatten()
	avec.append(at)

A1 = torch.stack(avec, dim=0)
A2 = A.view(10, 2, 2, 2, 2).flatten(1)

print(A1)
print(A2)
print(A1 - A2)


