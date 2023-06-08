import torch

from torch_iterative_alg import TorchIterativeAlg
from torch_linop import TorchLinop

# This is not perfect, should use Rayleigh instead but this way uses less memory
class TorchMaxEig(TorchIterativeAlg):
	def __init__(self, A: TorchLinop, dtype=torch.float32, max_iter=30):
		self.A = A
		self.x = torch.rand(A.ishape, dtype=dtype)
		self.max_eig = torch.inf
		super().__init__(max_iter)
		
	def _update(self):
		y = self.A(self.x)
		self.max_eig = torch.linalg.norm(y)
		self.x = y / self.max_eig

	def _done(self):
		return self.iter >= self.max_iter
	
	def run(self):
		i = 0
		while not self.done():
			print('MaxEig Iter: ', i,'/', self.max_iter, ' MaxEig: ', self.max_eig)
			self.update()
			i += 1
		return self.max_eig