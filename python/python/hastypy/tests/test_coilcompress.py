import h5py
import torch
import gc

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
						kdatas = torch.stack(kdata_vec, axis=0)
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
						gating[gatename] = gatingset[gatename][()]

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
				
			if load_smaps:
				with h5py.File(filepath_smaps) as f:
					smaps = torch.tensor(f['Maps'][()])

			if load_true_images:
				with h5py.File(filepath_true_images) as f:
					true_images = torch.tensor(f['images'][()])

			return coords, kdatas, smaps, true_images

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
				add_idx_spokes(idx)

			# Mid Frames
			for i in range(1,nframes):
				new_len_start = len_start + length
				idx = torch.logical_and(len_start <= gating, gating < new_len_start)
				len_start = new_len_start
				gates.append(len_start)

				add_idx_spokes(idx)

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

	@staticmethod
	def coil_compress(kdata=None, axis=0, target_channels=None):
		kdata_cc = torch.squeeze(kdata[0]) 

		kdata_cc = torch.moveaxis(kdata_cc, axis, -1)

		old_channels = kdata_cc[-1].shape

		# Pick out only 5% of the data for SVD
		mask_shape = kdata_cc.shape

		mask = torch.rand(mask_shape[:-1])<0.3

		kcc = torch.zeros((old_channels[0], torch.sum(mask).item()), dtype=kdata_cc.dtype)

		for c in range(old_channels[0]):
			ktemp = kdata_cc[...,c]
			kcc[c, :] = ktemp[mask]

		kdata_cc = torch.moveaxis(kdata_cc, -1, axis)

		cudev = torch.device('cuda:0')
		#cudev = torch.device('cpu')
		kcc = kcc.to(cudev)

		# SVD
		U, S, Vh = torch.linalg.svd(kcc, full_matrices=False, driver='gesvd')

		del S
		del Vh
		gc.collect()
		torch.cuda.empty_cache()

		for e in range(len(kdata)):
			kdatae = kdata[e].transpose(0,2).to(cudev)

			kdatae = torch.matmul(U.unsqueeze(0), kdatae).squeeze(-1)
			kdata[e] = kdatae[..., :target_channels].cpu()

			torch.cuda.empty_cache()

		return kdata




import time

if __name__ == "__main__":


	c = 44

	Np = 25000000

	start = time.time()

	kdata = [torch.rand(1, c, Np), torch.rand(1, c, Np), torch.rand(1, c, Np), torch.rand(1, c, Np), torch.rand(1, c, Np)]

	end = time.time()
	print('It Took: ', end - start)

	print(1)
	start = time.time()

	compressed = FivePointLoader.coil_compress(kdata, axis=0, target_channels=10)

	end = time.time()

	print('It Took: ', end - start)