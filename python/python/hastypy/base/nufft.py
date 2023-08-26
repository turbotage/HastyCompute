import torch
import numpy as np

from hastypy.base.torch_opalg import TorchLinop
from hastypy.base.opalg import Vector, Linop
import hastypy.ffi.hasty_nufft as hasty_nufft
from hastypy.ffi.hasty_interface import FunctionLambda



class NufftT(TorchLinop):
	def __init__(self, coord, im_size):
		self.ndim = coord.shape[0]
		self.nfreq = coord.shape[1]
		self._nufftop = hasty_nufft.Nufft(coord, [1] + list(im_size), hasty_nufft.NufftOptions.type2())

		super().__init__((1,) + im_size, (1,self.nfreq))
		
	def _apply(self, input: torch.Tensor):
		output = torch.empty((1, self.nfreq), dtype=input.dtype, device=input.device)
		self._nufftop.apply(input, output)
		return output
	
class NufftAdjointT(TorchLinop):
	def __init__(self, coord, im_size):
		self.im_size = im_size
		self.nfreq = coord.shape[1]
		self._nufftop = hasty_nufft.Nufft(coord, [1] + list(im_size), hasty_nufft.NufftOptions.type1())

		super().__init__((1,self.nfreq), (1,) + im_size)
		
	def _apply(self, input: torch.Tensor):
		output = torch.empty((1,) + self.im_size, dtype=input.dtype, device=input.device)
		self._nufftop.apply(input, output)
		return output
	

class NufftNormalT(TorchLinop):
	def __init__(self, coord, im_size, func_lamda: FunctionLambda | None = None):
		self.im_size = im_size
		self.nfreq = coord.shape[1]
		self.func_lamda = func_lamda
		self._nufftop = hasty_nufft.NufftNormal(coord, [1] + list(im_size), 
			hasty_nufft.NufftOptions.type2(), hasty_nufft.NufftOptions.type1())
		
		super().__init__((1,) + im_size, (1,) + im_size)
		
	def _apply(self, input: torch.Tensor):
		output = torch.empty_like(input)
		storage = torch.empty((1,self.nfreq), dtype=input.dtype, device=input.device)
		self._nufftop.apply(input, output, storage, self.func_lamda)
		return output






class Nufft(Linop):
	def __init__(self, coord, im_size):
		self.ndim = coord.shape[0]
		self.nfreq = coord.shape[1]
		self._nufftop = hasty_nufft.Nufft(coord, [1] + list(im_size), hasty_nufft.NufftOptions.type2())

		super().__init__((1,) + im_size, (1,self.nfreq))
		
	def _apply(self, input: Vector):
		inp = input.get_tensor()
		output = torch.empty((1, self.nfreq), dtype=inp.dtype, device=inp.device)
		self._nufftop.apply(inp, output)
		return Vector(output)
	
class NufftAdjoint(Linop):
	def __init__(self, coord, im_size):
		self.im_size = im_size
		self.nfreq = coord.shape[1]
		self._nufftop = hasty_nufft.Nufft(coord, [1] + list(im_size), hasty_nufft.NufftOptions.type1())

		super().__init__((1,self.nfreq), (1,) + im_size)
		
	def _apply(self, input: Vector):
		inp = input.get_tensor()
		output = torch.empty((1,) + self.im_size, dtype=inp.dtype, device=inp.device)
		self._nufftop.apply(inp, output)
		return Vector(output)
	

class NufftNormal(Linop):
	def __init__(self, coord, im_size, func_lamda: FunctionLambda | None = None):
		self.im_size = im_size
		self.func_lamda = func_lamda
		self.nfreq = coord.shape[1]
		self._nufftop = hasty_nufft.NufftNormal(coord, [1] + list(im_size), 
			hasty_nufft.NufftOptions.type2(), hasty_nufft.NufftOptions().type1())
		
		super().__init__((1,) + im_size, (1,) + im_size)
		
	def _apply(self, input: Vector):
		inp = input.get_tensor()
		output = torch.empty_like(inp)
		storage = torch.empty((1,) + self.im_size, dtype=inp.dtype, device=inp.device)
		self._nufftop.apply(inp, output, storage, self.func_lamda)
		return Vector(output)