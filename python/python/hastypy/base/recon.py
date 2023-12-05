import torch
import numpy as np
import math

from hastypy.base.sense import BatchedSense, BatchedSenseAdjoint, BatchedSenseNormal, BatchedSenseNormalAdjoint, OuterInnerAutomorphism, InnerOuterAutomorphism
from hastypy.base.svt import Random3DBlocksSVT, Normal3DBlocksSVT

import hastypy.base.opalg as opalg
from hastypy.base.opalg import Vector, MaxEig, MultiplyOp
from hastypy.base.solvers import GradientDescent, PrimalDualHybridGradient
import hastypy.base.proximal as prox


class GriddingReconstruction:
	def __init__(self, coord_vec, smaps, kdata_vec, weights_vec):
		self.coord_vec = coord_vec
		self.smaps = smaps
		self.kdata_vec = kdata_vec
		self.weights_vec = weights_vec

	def run(self):

		self.pilsop = BatchedSenseAdjoint(self.coord_vec, self.smaps, None, self.weights_vec)

		return self.pilsop(Vector(self.kdata_vec)) / Vector(torch.sum(self.smaps.conj() * self.smaps, dim=0).unsqueeze(0))


def stack_frame_datas_fivepoint(coord_vec, kdata_vec):

	new_coord_vec = []
	new_kdata_vec = []

	for enc in range(5):
		stackable_coord = []
		stackable_kdata = []
		for frame in range(len(coord_vec) // 5):
			stackable_coord.append(coord_vec[frame*5 + enc])
			stackable_kdata.append(kdata_vec[frame*5 + enc])

		new_coord_vec.append(torch.concatenate(stackable_coord, dim=1))
		new_kdata_vec.append(torch.concatenate(stackable_kdata, dim=2))

	return new_coord_vec, new_kdata_vec
	


class FivePointFULL:
	def __init__(self, smaps, coord_vec, kdata_vec, weights_vec=None, streams=None, solver='GD', lamda=0.0):
		self.smaps = smaps
		self.coord_vec = coord_vec
		self.kdata_vec = kdata_vec
		self.weights_vec = weights_vec

		self.streams = streams
		self.solver = solver

		self.max_eig_run()
		self.stepsize = 1 / self.max_eig
		
		if self.solver == 'GD':
			self.sensenormalop = BatchedSenseNormal(self.coord_vec, self.smaps, self.kdata_vec,
				self.weights_vec, self.streams)
			
		self.lamda = lamda / self.stepsize
		if self.lamda != 0.0:
			self._dctprox = prox.ProximalL1DCT((5,1) + self.smaps.shape[1:], (True,True,True), self.lamda)
		else:
			self._dctprox = None


	def max_eig_run(self, print_info=True):
		maxeigop = BatchedSenseNormal([self.coord_vec[0]], self.smaps, None, None if self.weights_vec is None else [self.weights_vec[0]])
		self.max_eig = MaxEig(maxeigop, torch.complex64, max_iter=5).run(print_info=print_info).to(torch.float32)

	def run(self, image, iter=50, callback=None, accelerate=True):

		image = Vector(image)

		if self.solver == 'GD':
			gd = GradientDescent(self.sensenormalop, image, self._dctprox, 1 / self.max_eig, accelerate=accelerate, max_iter=iter)
			image = gd.run(callback, print_info=True)

		return image.get_tensor()



class FivePointLLR:
	def __init__(self, smaps, coord_vec, kdata_vec, svt_options: prox.SVTOptions | None, 
			weights_vec=None, grad_streams=None, prox_streams=None, solver='GD',
			tau=None, sigma=None):
		
		self.smaps = smaps
		self.coord_vec = coord_vec
		self.kdata_vec = kdata_vec
		self.weights_vec = weights_vec

		self.svt_options = svt_options
		self.nframes = len(coord_vec) // 5

		self.grad_streams = grad_streams
		self.prox_streams = prox_streams

		self.solver = solver


		def init_gd():
			self.max_eig = 0.0
			print('MaxEigs GD: ')

			perm = np.random.permutation(len(self.coord_vec))[:(len(self.coord_vec) // 8)]
			for p in perm:
				maxeigop = BatchedSenseNormal([self.coord_vec[p]], self.smaps, None, [self.weights_vec[p]] if self.weights_vec is not None else None)
				maxeig = MaxEig(maxeigop, torch.complex64, max_iter=8).run().to(torch.float32).item()
				print("\r Frame/Enc: ", p, " MaxEig: ", maxeig, end="")
				if maxeig > self.max_eig:
					self.max_eig = maxeig

			self.stepsize = (0.5 / self.max_eig)
			self.sensenormalop = BatchedSenseNormal(self.coord_vec, self.smaps, self.kdata_vec,
				self.weights_vec, self.grad_streams)


		def init_pdhg():
			datavec = Vector(self.kdata_vec)
			self.proxfc = prox.L2Reg(opalg.get_shape(datavec), a=-datavec, inplace=True)

			self.tau = tau
			self.sigma = sigma

			self.A = BatchedSense(self.coord_vec, self.smaps, streams=grad_streams)
			self.AH = BatchedSenseAdjoint(self.coord_vec, self.smaps, streams=grad_streams)

			if self.tau is None:
				max_eig = 0.0

				if self.sigma is None:
					self.sigma = Vector(1.0)
					AHA = BatchedSenseNormal(self.coord_vec, self.smaps, streams=grad_streams)
					max_eig = MaxEig(AHA, torch.complex64, max_iter=5).run(print_info=True)
				else:
					BS = BatchedSense(self.coord_vec, self.smaps, streams=grad_streams)
					S = MultiplyOp(self.sigma)
					BSH = BatchedSenseAdjoint(self.coord_vec, self.smaps, streams=grad_streams)
					max_eig = MaxEig(BSH * S * BS, torch.complex64, max_iter=5).run(print_info=True)

				max_eig *= 4.0
				max_eig_sqrt = math.sqrt(max_eig)
				self.sigma /= max_eig
				self.tau = Vector(1.0 / max_eig)

			elif self.sigma is None:
				T = MultiplyOp(self.tau)

				max_eig = MaxEig(self.A*T*self.AH, torch.complex64, max_iter=5).run(print_info=True)
				max_eig_sqrt = math.sqrt(max_eig)

				self.tau /= max_eig_sqrt
				self.sigma = Vector(1.0 / max_eig_sqrt)

		# Setup ||Ax-b|| operators
		if self.solver == 'GD':
			init_gd()
		elif self.solver == 'PDHG':
			init_pdhg()
		elif self.solver == 'Gridding':
			pass

		self.svtshape = (self.nframes,5) + self.smaps.shape[1:]
		self.gradshape = (self.nframes*5,) + self.smaps.shape[1:]

		self.tosvtshape = InnerOuterAutomorphism(self.nframes, 5, self.smaps.shape[1:])
		self.togradshape = OuterInnerAutomorphism(self.nframes, 5, self.smaps.shape[1:])

		# Setup LLR proximality operators
		if self.svt_options is not None:
			self.svtprox = prox.UnitaryTransform(
				prox.Proximal3DSVT(self.svtshape, self.svt_options, self.prox_streams),
				self.tosvtshape,
				self.togradshape)
		else:
			self.svtprox = prox.ZeroFunc(self.gradshape)
			

	def run(self, image, iter=50, callback=None, accelerate=True):

		image = self.togradshape(Vector(image))

		if self.solver == 'GD':
			gd = GradientDescent(self.sensenormalop, image, self.svtprox, self.stepsize, accelerate=accelerate, max_iter=iter)
			image = gd.run(callback, print_info=True)
		elif self.solver == 'PDHG':
			dual = opalg.zeros(opalg.get_shape(Vector(self.kdata_vec)), torch.complex64)

			#self.svtprox = prox.ZeroFunc(opalg.get_shape(image))
			pdhg = PrimalDualHybridGradient(self.proxfc, self.svtprox, self.A, self.AH, image, dual, 
				self.tau, self.sigma, acceleration=0.0, max_iter=iter)
			
			def norm_callback(x: Vector, iter):
				print('Data Norm: ', opalg.norm(self.A(x) - Vector(self.kdata_vec)) / opalg.norm(Vector(self.kdata_vec)))

			image = pdhg.run(callback=callback, print_info=True)
		elif self.solver == 'Gridding':
			image = GriddingReconstruction(self.coord_vec, self.smaps, self.kdata_vec, self.weights_vec).run()

		return self.tosvtshape(image).get_tensor()



