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
	
class TorchScaleLinop(TorchLinop):
	def __init__(self, linop: TorchLinop, scaling: torch.Tensor):
		self.scaling = scaling
		self.linop = linop
		super().__init__(linop.ishape, linop.oshape)

	def _apply(self, input: torch.Tensor):
		return self.linop(input * self.scaling)
	
class TorchL2Reg(TorchLinop):
	def __init__(self, linop: TorchLinop, lamda = 0.01):
		self.lamda = lamda
		self.linop = linop
		if linop.ishape is not linop.oshape:
			raise RuntimeError("L2Reg linop must have equal input and output shapes")
		super().__init__(linop.ishape, linop.oshape)

	def _apply(self, input: torch.Tensor):
		if self.lamda == 0.0:
			return self.linop(input)
		else:
			ret = self.lamda * input
			return self.linop(input) + ret