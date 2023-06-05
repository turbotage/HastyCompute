import torch_linop as tl
from torch_linop import TorchLinop, TorchMatrixLinop


class TorchIterativeAlg:
	def __init__(self, max_iter):
		self.max_iter = max_iter
		self.iter = 0
                
	def _update(self):
		raise NotImplementedError

	def _done(self):
		return self.iter >= self.max_iter

	def update(self):
		"""Perform one update step.

		Call the user-defined _update() function and increment iter.
		"""
		self._update()
		self.iter += 1

	def done(self):
		"""Return whether the algorithm is done.

		Call the user-defined _done() function.
		"""
		return self._done()	