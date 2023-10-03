import hastypy.opt.cuda.sigpy.linop as slinop

from hastypy.opt.cuda.sigpy.linop import Linop, Identity, Resize, Multiply

import hastypy.opt.cuda.sigpy.interpolate as interp
import hastypy.opt.cuda.sigpy.gridding as grid

class Interpolate(Linop):
	def __init__(self, ishape, coord, width=2, param=1):
		ndim = coord.shape[-1]
		oshape = list(ishape[:-ndim]) + list(coord.shape[:-1])

		self.coord = coord
		self.width = width
		self.param = param

		super().__init__(oshape, ishape)


	def _apply(self, input):
			return interp._interpolate(
				input,
				self.coord,
				width=self.width,
				param=self.param,
			)

	def _adjoint_linop(self):
		return Gridding(
			self.ishape,
			self.coord,
			width=self.width,
			param=self.param,
		)

class Gridding(Linop):
	def __init__(self, oshape, coord, width=2, param=1):
		ndim = coord.shape[-1]
		ishape = list(oshape[:-ndim]) + list(coord.shape[:-1])

		self.coord = coord
		self.width = width
		self.param = param

		super().__init__(oshape, ishape)


	def _apply(self, input):
		return grid._gridding(
			input,
			self.coord,
			self.oshape,
			width=self.width,
			param=self.param,
		)

	def _adjoint_linop(self):
		return Interpolate(
			self.oshape,
			self.coord,
			width=self.width,
			param=self.param,
		)