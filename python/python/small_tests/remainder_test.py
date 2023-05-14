import torch

kernel = torch.rand(1,1,3,3,3)

kernel_copy = kernel.clone()

for d in range(2, kernel.ndim):
	rem = torch.remainder(
			-1 * torch.arange(kernel.shape[d]), kernel.shape[d]
		)
	print(rem)

	kernel = kernel.index_select(
		d,
		rem
	)

print(kernel)
print(kernel_copy)



