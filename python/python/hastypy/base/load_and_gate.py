import torch
import numpy as np
import scipy as sp
from scipy.ndimage import zoom
import scipy.signal as spsig
import gc

import h5py


from hastypy.base.opalg import Linop
from hastypy.ffi.hasty_sense import BatchedSense, BatchedSenseAdjoint, BatchedSenseNormal


"""
class BatchedSenseLinop(Linop):
	def __init__(self, smaps, coord_vec, kdata_vec=None, weights_vec=None, random=(False, None), streams=None, ninner_batches=1):
		self.nvoxels = smaps.shape[0] * smaps.shape[1] * smaps.shape[2]
		
		self.nfreqs = []
		for coord in coord_vec:
			self.nfreqs.append(coord.shape[1])
		self.ninner_batches = ninner_batches
		self.nouter_batches = len(coord_vec)
		self.inshape = tuple([self.nouter_batches, self.ninner_batches] + list(smaps.shape[1:]))
		self.outshape = 

		self.ncoils = smaps.shape[0]
		self.random = random

		self._senseop = BatchedSense(coord_vec, smaps, kdata_vec, weights_vec, streams)

		super().__init__(self.shape, self.shape)

	def coil_list(self):
		if self.random[0]:
			coil_list = []
			for i in range(self.nframes):
				permuted = np.random.permutation(self.ncoils).astype(np.int32)
				coil_list.append(permuted[:self.random[1]].tolist())
			return coil_list
		return None

	def _apply(self, input: torch.Tensor, output: list[torch.Tensor] | None = None):
		if output is None:
			for nfreq in self.nfreqs:
				output.append(torch.empty(self.ninner_batches, self.ncoils, nfreq))

		self._senseop.apply(input, output, self.coil_list())

		return input

class BatchedSenseAdjointLinop(Linop):
	def __init__(self, smaps, coord_vec, kdata_vec=None, weights_vec=None, random=(False, None), streams=None, inner_batches=1):
		self.nvoxels = smaps.shape[0] * smaps.shape[1] * smaps.shape[2]

		self._senseop = BatchedSense(coord_vec, smaps, kdata_vec, weights_vec, streams)

		self.nouter_batches = len(coord_vec)
		self.shape = tuple([self.nframes, inner_batches] + list(smaps.shape[1:]))

		self.ncoils = smaps.shape[0]
		self.random = random

		super().__init__(self.shape, self.shape)

	def coil_list(self):
		if self.random[0]:
			coil_list = []
			for i in range(self.nframes):
				permuted = np.random.permutation(self.ncoils).astype(np.int32)
				coil_list.append(permuted[:self.random[1]].tolist())
			return coil_list
		return None

	def _apply(self, input: torch.Tensor, output: torch.Tensor | None = None):
		if output is None:
			output = torch.empty_like(input)

		self._senseop.apply(input, output, self.coil_list())

		return input

class BatchedSenseNormalLinop(Linop):
	def __init__(self, smaps, coord_vec, kdata_vec=None, weights_vec=None, random=(False, None), streams=None, inner_batches=1):
		self.nvoxels = smaps.shape[0] * smaps.shape[1] * smaps.shape[2]

		self._senseop = BatchedSenseNormal(coord_vec, smaps, kdata_vec, weights_vec, streams)

		self.nouter_batches = len(coord_vec)
		self.shape = tuple([self.nframes, inner_batches] + list(smaps.shape[1:]))

		self.ncoils = smaps.shape[0]
		self.random = random

		super().__init__(self.shape, self.shape)

	def coil_list(self):
		if self.random[0]:
			coil_list = []
			for i in range(self.nframes):
				permuted = np.random.permutation(self.ncoils).astype(np.int32)
				coil_list.append(permuted[:self.random[1]].tolist())
			return coil_list
		return None

	def _apply(self, input: torch.Tensor, output: torch.Tensor | None = None):
		if output is None:
			output = torch.empty_like(input)

		self._senseop.apply(input, output, self.coil_list())

		return input
"""

class FivePointLoader:

	@staticmethod
	def load_smaps(filepath, sp_dims):
		with torch.inference_mode():
			smaps: torch.Tensor

			def get_zoom(dims):
				if dims == sp_dims:
					return None
				zoom = []
				for i in range(len(sp_dims)):
						zoom.append(sp_dims[i] / dims[i])
				return tuple(zoom)

			print("Loading smaps...")
			with h5py.File(filepath) as f:
				smap_vec = []
				smapsdata = f['Maps']
				ncoils = len(smapsdata.keys())
				for i in range(ncoils):
					print("\r", end="")
					smp = smapsdata['SenseMaps_' + str(i)][()]
					smp = smp['real']+1j*smp['imag']
					zooming = get_zoom(smp.shape)
					if zooming is None:
						smap_vec.append(torch.tensor(smp))
						print('Coil: ', i, '/', ncoils, end="")
					else:
						smap_vec.append(torch.tensor(zoom(smp, get_zoom(smp.shape), order=1)))
						print('Coil: ', i, '/', ncoils, ', Rescaling...', end="")
				smaps = torch.stack(smap_vec, axis=0)
				del smap_vec
			print("\nDone.")

		return smaps
	
	@staticmethod
	def load_raw(filepath, load_coords=True, load_kdata=True, load_weights=True, load_gating=True, gating_names=[]):
		with torch.inference_mode():
			ret: tuple[torch.Tensor] = ()

			with h5py.File(filepath) as f:

				# Coords, K-data, Density Compensation/Weights
				if load_coords or load_kdata or load_weights:
					kdataset = f['Kdata']

					# Coords
					if load_coords:
						print('Loading coords:  ', end="")
						kx_vec = []
						ky_vec = []
						kz_vec = []
						for i in range(5):
							kx_vec.append(torch.tensor(kdataset['KX_E' + str(i)][()]))
							ky_vec.append(torch.tensor(kdataset['KY_E' + str(i)][()]))
							kz_vec.append(torch.tensor(kdataset['KZ_E' + str(i)][()]))
						kx = torch.stack(kx_vec, axis=0)
						ky = torch.stack(ky_vec, axis=0)
						kz = torch.stack(kz_vec, axis=0)
						coords = torch.stack([kx,ky,kz], axis=0)
						ret += (coords,)
						print('Done.')

					# K-Data
					if load_kdata:
						print('Loading kdata:  ', end="")
						kdata_vec = []
						for i in range(5):
							kdata_enc = []
							for j in range(100000):
								coilname = 'KData_E'+str(i)+'_C'+str(j)
								if coilname in kdataset:
									kde = kdataset['KData_E'+str(i)+'_C'+str(j)][()]
									kdata_enc.append(torch.tensor(kde['real'] + 1j*kde['imag']))
								else:
									break
							kdata_vec.append(torch.stack(kdata_enc, axis=0))
							gc.collect()
						kdatas = torch.stack(kdata_vec, axis=0)
						del kdata_vec
						gc.collect()
						ret += (kdatas,)
						print('Done.')

					# Density Compensation
					if load_weights:
						print('Loading weights:  ', end="")
						kw_vec = []
						for i in range(5):
							kw_vec.append(torch.tensor(kdataset['KW_E'+str(i)][()]))
						weights = torch.stack(kw_vec, axis=0)
						ret += (weights,)
						print('Done.')

				# Gating
				if load_gating:
					print('Loading gating:  ', end="")
					gatingset = f['Gating']
					
					gating = {}
					for gatename in gating_names:
						gating[gatename] = torch.tensor(gatingset[gatename][()])

					ret += (gating,)
					print('Done.')

			return ret

	@staticmethod
	def load_simulated(filepath_coord_kdata, filepath_smaps, filepath_true_images, 
			load_coords=True, load_kdata=True, load_smaps=True, load_true_images=True):
		with torch.inference_mode():

			coords = None
			kdatas = None
			smaps = None
			true_images = None

			if load_coords or load_kdata:
				with h5py.File(filepath_coord_kdata) as f:
					nframes = len(f.keys()) // 10
					coords = None
					kdatas = None
					if load_coords:
						coords = []
						for i in range(nframes):
							for j in range(5):
								ijstr = str(i)+'_e'+str(j)
								coords.append(torch.tensor(f['coords_f'+ijstr][()]))
					if load_kdata:
						kdatas = []
						for i in range(nframes):
							for j in range(5):
								ijstr = str(i)+'_e'+str(j)
								kdatas.append(torch.tensor(f['kdatas_f'+ijstr][()]))

				gc.collect()
				
			if load_smaps:
				with h5py.File(filepath_smaps) as f:
					smaps = torch.tensor(f['Maps'][()])

			if load_true_images:
				with h5py.File(filepath_true_images) as f:
					true_images = torch.tensor(f['images'][()])

			return coords, kdatas, None, smaps, true_images

	@staticmethod
	def load_as_full(coords, kdatas, weights=None):
		with torch.inference_mode():
			ret: tuple[list[torch.Tensor]] = ()

			coord_vec = []
			kdata_vec = []

			nlast = coords.shape[-1] * coords.shape[-2]
			ncoil = kdatas.shape[1]

			coordf = coords[:,:,0,:,:].reshape((3,5,nlast))
			kdataf = kdatas[:,:,0,:,:].reshape((5,ncoil,nlast))
			
			for j in range(5):
				coord_vec.append(coordf[:,j,:].detach().clone())
				kdata_vec.append(kdataf[j,:,:].detach().clone().unsqueeze(0))

			ret += (coord_vec, kdata_vec)

			if weights is not None:
				weights_vec = []
				weightsf = weights[:,0,:,:].reshape((5,nlast))
				for j in range(5):
					weights_vec.append(weightsf[j,:].detach().clone().unsqueeze(0))
				ret += (weights_vec,)

			return ret

	@staticmethod
	def full_to_spokes(kdata_vec, ncoils, nspokes, samp_per_spoke):
		kdata_vec_temp = [kdata.reshape(ncoils, 1, nspokes, samp_per_spoke) for kdata in kdata_vec]
		ret = torch.stack(kdata_vec_temp, axis=0).contiguous()
		del kdata_vec_temp
		gc.collect()
		return ret

	@staticmethod
	def gate_ecg_method(coord, kdata, weights, gating, nframes):
		with torch.inference_mode():
			mean = torch.mean(gating)
			upper_bound = 2.0*mean # This is heuristic, perhaps base it on percentile instead

			start_offset = upper_bound * 0.05
			length = (upper_bound-start_offset) / nframes

			spokelen = coord.shape[-1]
			nspokes = coord.shape[-2]
			ncoil = kdata.shape[1]	

			coord_vec = []
			kdata_vec = []
			weights_vec = []

			def add_idx_spokes(idx):
				nspokes_idx = torch.count_nonzero(idx)
				nlast = nspokes_idx*spokelen

				coordf = coord[:,:,0,idx,:].reshape((3,5,nlast))
				kdataf = kdata[:,:,0,idx,:].reshape((5,ncoil,nlast))
				if weights is not None:
					weightsf = weights[:,0,idx,:].reshape((5,nlast))

				for j in range(5):
					coord_vec.append(coordf[:,j,:].detach().clone())
					kdata_vec.append(kdataf[j,:,:].unsqueeze(0).detach().clone())
					if weights is not None:
						weights_vec.append(weightsf[j,:].unsqueeze(0).detach().clone())

			len_start = length + start_offset
			gates = [len_start]
			# First Frame
			if True:
				idx = gating < len_start
				add_idx_spokes(idx.squeeze())

			# Mid Frames
			for i in range(1,nframes):
				new_len_start = len_start + length
				idx = torch.logical_and(len_start <= gating, gating < new_len_start)
				len_start = new_len_start
				gates.append(len_start)

				add_idx_spokes(idx.squeeze())

			# Last Frame
			#if True:
			#	idx = len_start <= gating
			#	add_idx_spokes(idx)

			return (coord_vec, kdata_vec, weights_vec if len(weights_vec) != 0 else None, gates)

	@staticmethod
	def gate_resp_method(coord, kdata, weights, gating, nframes):
		with torch.inference_mode():

			def get_angle_gating(gating):
				timegate = gating['TIME_E0'][0,:]
				respgate = gating['RESP_E0'][0,:]

				timeidx = np.argsort(timegate)
				
				time = timegate[timeidx]
				resp = respgate[timeidx]

				Fs = 1.0 / (time[1] - time[0])
				Fn = Fs / 2.0
				Fco = 0.01
				Fsb = 1.0
				Rp = 0.05
				Rs = 50.0

				def torad(x):
					return x

				N, Wn = spsig.buttord(torad(Fco / Fn), torad(Fsb / Fn), Rp, Rs, analog=True)
				b, a = spsig.butter(N, Wn)
				filtered = spsig.filtfilt(b, a, resp)

				hfilt = spsig.hilbert(resp - filtered)
				angles = np.angle(hfilt)

				invtimeidx = np.argsort(timeidx)
				return torch.tensor(angles[invtimeidx]), torch.tensor(angles)


			angles, sorted_angles = get_angle_gating(gating)

			spokelen = coord.shape[-1]
			nspokes = coord.shape[-2]
			ncoil = kdata.shape[1]	

			coord_vec = []
			kdata_vec = []
			weights_vec = []

			def add_idx_spokes(idx):
				nspokes_idx = torch.count_nonzero(idx)
				nlast = nspokes_idx*spokelen

				coordf = coord[:,:,0,idx,:].reshape((3,5,nlast))
				kdataf = kdata[:,:,0,idx,:].reshape((5,ncoil,nlast))
				if weights is not None:
					weightsf = weights[:,0,idx,:].reshape((5,nlast))

				for j in range(5):
					coord_vec.append(coordf[:,j,:].detach().clone())
					kdata_vec.append(kdataf[j,:,:].unsqueeze(0).detach().clone())
					if weights is not None:
						weights_vec.append(weightsf[j,:].unsqueeze(0).detach().clone())


			gatethreshs = np.linspace(-3.141592, 3.141592, nframes+1)

			gates = []
			for i in range(nframes):
				idx = torch.logical_and(gatethreshs[i] <= angles, angles <= gatethreshs[i+1])
				gates.append(gatethreshs[i+1])

				add_idx_spokes(idx)


			return (coord_vec, kdata_vec, weights_vec if len(weights_vec) != 0 else None, gates, angles, sorted_angles)



def crop_kspace(coords, kdatas, weights, im_size, crop_factors=(1.0,1.0,1.0), prefovkmuls=(1.0,1.0,1.0), postfovkmuls=(1.0,1.0,1.0)):
	with torch.inference_mode():
		if isinstance(coords, list):
			max = 0.0
			for i in range(len(coords)):
				maxi = coords[i].abs().max().item()
				if maxi > max:
					max = maxi

			kim_size = tuple(0.5*im_size[i]*crop_factors[i] for i in range(3))

			upp_bound = torch.tensor(0.99999*torch.pi)
			for i in range(len(coords)):
				coord = coords[i]
				coord[0,:] *= prefovkmuls[0]
				coord[1,:] *= prefovkmuls[1]
				coord[2,:] *= prefovkmuls[2]

				idxx = torch.abs(coord[0,:]) < kim_size[0]
				idxy = torch.abs(coord[1,:]) < kim_size[1]
				idxz = torch.abs(coord[2,:]) < kim_size[2]

				idx = torch.logical_and(idxx, torch.logical_and(idxy, idxz))

				coords[i] = coord[:,idx]
				coords[i][0,:] *= postfovkmuls[0] * torch.pi / kim_size[0]
				coords[i][1,:] *= postfovkmuls[1] * torch.pi / kim_size[1]
				coords[i][2,:] *= postfovkmuls[2] * torch.pi / kim_size[2]

				coords[i] = torch.maximum(torch.minimum(upp_bound, coords[i]), -upp_bound)

				kdatas[i] = kdatas[i][:,:,idx]
				if weights is not None:
					weights[i] = weights[i][:,idx]
			
			return (coords, kdatas, weights)
		else:
			raise NotImplementedError("Not implemented")
		

def translate(coord_vec, kdata_vec, translation):
	with torch.inference_mode():
		cudev = torch.device('cuda:0')
		mult = torch.tensor(list(translation)).unsqueeze(-1).to(cudev)
		for i in range(len(coord_vec)):
			coord = coord_vec[i].to(cudev, non_blocking=True)
			kdata = kdata_vec[i].to(cudev, non_blocking=True)

			kdata *= torch.exp(1j*torch.linalg.vecdot(mult, coord, dim=0)).unsqueeze(0)

			kdata_vec[i] = kdata.cpu()

		return kdata_vec

def center_weights(im_size, width, coord, weights):
	with torch.inference_mode():
		scaled_coords = coord * torch.tensor(list(im_size))[:,None] / (2*torch.pi)
		idx = torch.abs(scaled_coords) < width
		idx = torch.logical_and(idx[0,:], torch.logical_and(idx[1,:], idx[2,:]))
		return weights[0,idx]



