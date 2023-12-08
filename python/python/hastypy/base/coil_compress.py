import torch
import gc

class CoilCompress:

	@staticmethod
	def coil_compress(kdata=None, target_channels=None, thresh=0.25, cudev=torch.device('cuda:0')):
		
		kdata_cc = torch.squeeze(kdata[0]) 
		kdata_cc = torch.moveaxis(kdata_cc, 0, -1)
		old_channels = kdata_cc[-1].shape
		mask_shape = kdata_cc.shape

		if target_channels==None:
			return kdata

		mask = torch.rand(mask_shape[:-1], dtype=torch.float32) < thresh

		def calc_U(mask):
			# Pick out only a fraction of the data for SVD
			kcc = torch.zeros((old_channels[0], torch.sum(mask).item()), dtype=kdata_cc.dtype, device=cudev)
			n = torch.sum(mask).item()

			for c in range(old_channels[0]):
				ktemp = kdata_cc[...,c]
				kcc[c,:] = ktemp[mask].to(cudev, non_blocking=False)
			# SVD
			#U, S, Vh = CoilCompress.combined_svd(kcc[0], kcc[1], kcc[2], kcc[3], torch.device('cuda:0'))
			U, _, _ = torch.linalg.svd(kcc, full_matrices=False, driver='gesvd')

			torch.cuda.empty_cache()

			return U

		U = calc_U(mask).unsqueeze(0).to(cudev)

		for e in range(len(kdata)):
			kdatae = kdata[e].transpose(0,2).to(cudev)

			kdatae = torch.matmul(U, kdatae).squeeze(-1)
			kdata[e] = kdatae[..., :target_channels].transpose(0,1).unsqueeze(0).cpu()

			torch.cuda.empty_cache()

		return kdata
