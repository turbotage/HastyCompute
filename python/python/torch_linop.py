import torch

class TorchLinop:
    
	def __init__(self, ishape, oshape):
		self.ishape = ishape
		self.oshape = oshape

	def _apply(self, input: torch.Tensor):
		raise NotImplementedError
	
	def apply(self, input: torch.Tensor):
		if input.shape != self.ishape:
			raise RuntimeError("Incompatible input shape with Linop")
		
		output = self._apply(input)

		if output.shape != self.oshape:
			raise RuntimeError("Incompatible output shape with Linop")
		
		return output

	def __call__(self, input: torch.Tensor):
		return self.apply(input)


class TorchMatrixLinop(TorchLinop):
	def __init__(self, matrix: torch.Tensor):
		self.matrix = matrix
		super().__init__((matrix.shape[1],1), (matrix.shape[0],1))

	def _apply(self, input: torch.Tensor):
		return self.matrix @ input