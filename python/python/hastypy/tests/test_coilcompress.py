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
	def coil_compress(kdata=None, axis=0, target_channels=None, cudev=torch.device('cuda:0'), thresh=0.25):
		
		kdata_cc = torch.squeeze(kdata[0]) 
		kdata_cc = torch.moveaxis(kdata_cc, axis, -1)
		old_channels = kdata_cc[-1].shape
		mask_shape = kdata_cc.shape

		mask = torch.rand(mask_shape[:-1], )

		def calc_U(mask):
			# Pick out only 5% of the data for SVD
			kcc = torch.zeros((old_channels[0], torch.sum(mask).item()), dtype=kdata_cc.dtype, device=cudev)
			for c in range(old_channels[0]):
				ktemp = kdata_cc[...,c]
				kcc[c, :] = ktemp[mask].to(cudev, non_blocking=True)
			# SVD
			U, S, Vh = torch.linalg.svd(kcc, full_matrices=False, driver='gesvd')

			del S
			del Vh
			gc.collect()
			torch.cuda.empty_cache()

			return U

		U1 = calc_U(mask < thresh)
		U2 = calc_U(torch.logical_and(thresh <= mask, mask < 2*thresh))
		U3 = calc_U(torch.logical_and(2*thresh <= mask, mask < 3*thresh))
		U4 = calc_U(torch.logical_and(3*thresh <= mask, mask < 4*thresh))

		Z0 = torch.zeros_like(U1)

		C1 = torch.cat([U1, Z0])
		C2 = torch.cat([Z0, U2])
		C3 = torch.cat([U3, Z0])
		C4 = torch.cat([Z0, U4])
		


		X = torch.cat([C1, C2, C3, C4], dim=1)




		for e in range(len(kdata)):
			kdatae = kdata[e].transpose(0,2).to(cudev)

			kdatae = torch.matmul(U.unsqueeze(0), kdatae).squeeze(-1)
			kdata[e] = kdatae[..., :target_channels].cpu()

			torch.cuda.empty_cache()

		return kdata




import time

if __name__ == "__main__":


	def full_svd(C1, C2, C3, C4, dev):
		C_full = torch.cat([C1, C2, C3, C4], dim=1)
		cpudev = torch.device('cpu')

		if dev == 'cpu':
			U, s, Vh = torch.linalg.svd(C_full, full_matrices=False)
		else:
			U, s, Vh = torch.linalg.svd(C_full.to(dev, non_blocking=True), full_matrices=False, driver='gesvd')
		
		#CHECK = (U @ torch.diag(s) @ Vh).to(cpudev, non_blocking=True if cpudev != dev else False)
		CHECK = (U @ torch.diag(s) @ Vh).to(cpudev, non_blocking=False)
		
		print('RelErr: ', torch.norm(CHECK - C_full) / torch.norm(C_full))
		print('MaxErr:', torch.max(torch.abs(CHECK - C_full)))


	def combined_svd(C1, C2, C3, C4, dev):
		cpudev = torch.device('cpu')

		if dev == 'cpu':
			U1, s1, Vh = torch.linalg.svd(C1.to(dev), full_matrices=False)
		else:
			U1, s1, Vh = torch.linalg.svd(C1.to(dev), full_matrices=False, driver='gesvd')
		Vh = torch.conj(Vh).transpose(0,1)
		#RY1 = torch.linalg.qr(Vh1, mode='r')
		QY1, RY1 = torch.linalg.qr(Vh)
		if (dev != cpudev):
			QY1 = QY1.to(cpudev, non_blocking=True)


		if dev == 'cpu':
			U2, s2, Vh = torch.linalg.svd(C2.to(dev), full_matrices=False)
		else:
			U2, s2, Vh = torch.linalg.svd(C2.to(dev), full_matrices=False, driver='gesvd')
		Vh = torch.conj(Vh).transpose(0,1)
		#RY2 = torch.linalg.qr(Vh2, mode='r')
		QY2, RY2 = torch.linalg.qr(Vh)
		if (dev != cpudev):
			QY2 = QY2.to(cpudev, non_blocking=True)

		if dev == 'cpu':
			U3, s3, Vh = torch.linalg.svd(C3.to(dev), full_matrices=False)
		else:
			U3, s3, Vh = torch.linalg.svd(C3.to(dev), full_matrices=False, driver='gesvd')
		Vh = torch.conj(Vh).transpose(0,1)
		#RY3 = torch.linalg.qr(Vh3, mode='r')
		QY3, RY3 = torch.linalg.qr(Vh)
		if (dev != cpudev):
			QY3 = QY3.to(cpudev, non_blocking=True)
		
		
		if dev == 'cpu':
			U4, s4, Vh = torch.linalg.svd(C4.to(dev), full_matrices=False)
		else:
			U4, s4, Vh = torch.linalg.svd(C4.to(dev), full_matrices=False, driver='gesvd')
		Vh = torch.conj(Vh).transpose(0,1)
		#RY4 = torch.linalg.qr(Vh4, mode='r')
		QY4, RY4 = torch.linalg.qr(Vh)
		if (dev != cpudev):
			QY4 = QY4.to(cpudev, non_blocking=True)

		del Vh
		torch.cuda.empty_cache()

		X = torch.cat([U1, U2, U3, U4], dim=1)
		QX, RX = torch.linalg.qr(X)
		#del U1, U2, U3, U4


		start = 0
		end = s1.shape[0]
		RX1 = RX[:,start:end]
		start = end
		end += s2.shape[0]
		RX2 = RX[:,start:end]
		start = end
		end += s3.shape[0]
		RX3 = RX[:,start:end]
		start = end
		end += s4.shape[0]
		RX4 = RX[:,start:end]

		UW1, sW1, VHW1 = torch.linalg.svd(RX1 @ torch.diag(s1) @ torch.conj(RY1).transpose(0,1), full_matrices=False, driver='gesvd')
		UW2, sW2, VHW2 = torch.linalg.svd(RX2 @ torch.diag(s2) @ torch.conj(RY2).transpose(0,1), full_matrices=False, driver='gesvd')
		UW3, sW3, VHW3 = torch.linalg.svd(RX3 @ torch.diag(s3) @ torch.conj(RY3).transpose(0,1), full_matrices=False, driver='gesvd')
		UW4, sW4, VHW4 = torch.linalg.svd(RX4 @ torch.diag(s4) @ torch.conj(RY4).transpose(0,1), full_matrices=False, driver='gesvd')

		s = torch.cat([sW1, sW2, sW3, sW4]).to(cpudev, non_blocking=True)
		UW = torch.cat([UW1, UW2, UW3, UW4], dim=1)
		U_final = (QX @ UW)
		if dev != cpudev:
			U_final = U_final.to(cpudev, non_blocking=True)

		p = torch.argsort(s)
		s = s[p]

		U_final = U_final[:,p]

		Vh_final = torch.block_diag(
						(VHW1 @ torch.conj(QY1.to(dev)).transpose(0,1)).to(
							cpudev, non_blocking=True if cpudev != dev else False), 
						(VHW2 @ torch.conj(QY2.to(dev)).transpose(0,1)).to(
							cpudev, non_blocking=True if cpudev != dev else False), 
						(VHW3 @ torch.conj(QY3.to(dev)).transpose(0,1)).to(
							cpudev, non_blocking=True if cpudev != dev else False), 
						(VHW4 @ torch.conj(QY4.to(dev)).transpose(0,1)).to(
							cpudev, non_blocking=True if cpudev != dev else False)
						)
		
		Vh_final = Vh_final[p,:]

		C_full = torch.cat([C1, C2, C3, C4], dim=1)
		CHECK = U_final @ torch.diag(s) @ Vh_final

		print('RelErr: ', torch.norm(CHECK - C_full) / torch.norm(C_full))
		print('MaxErr:', torch.max(torch.abs(CHECK - C_full)))


	NS = 1000000

	C1 = torch.rand(44, NS)
	C2 = torch.rand(44, NS)
	C3 = torch.rand(44, NS)
	C4 = torch.rand(44, NS)

	start = time.time()
	combined_svd(C1, C2, C3, C4, torch.device('cuda:0'))
	end = time.time()
	print('It Took: ', end - start)

	start = time.time()
	full_svd(C1, C2, C3, C4, torch.device('cuda:0'))
	end = time.time()
	print('It Took: ', end - start)
	









	#s = torch.cat([s1, s2, s3, s4])
	#p = torch.argsort(s)






	



	coil_compress = False
	if coil_compress:
		c = 44

		Np = 500000

		start = time.time()

		kdata = [torch.rand(1, c, Np), torch.rand(1, c, Np), torch.rand(1, c, Np), torch.rand(1, c, Np), torch.rand(1, c, Np)]

		end = time.time()
		print('It Took: ', end - start)

		print(1)
		start = time.time()

		compressed = FivePointLoader.coil_compress(kdata, axis=0, target_channels=10)

		end = time.time()

		print('It Took: ', end - start)