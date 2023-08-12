import torch


class BatchedSenseLinop(TorchLinop):
	def __init__(self, smaps, coord_vec, kdata_vec=None, weights_vec=None, 
	      randomize=(False, None)):
		self.nvoxels = smaps.shape[0] * smaps.shape[1] * smaps.shape[2]
		self.smaps = smaps
		self.nframes = len(coord_vec)
		self.shape = tuple([self.nframes, 1] + list(smaps.shape[1:]))
		self.ncoils = self.smaps.shape[0]
		self.randomize_coils = randomize_coils
		self.num_rand_coils = num_rand_coils
		self.coord_vec = coord_vec
		self.kdata_vec = kdata_vec
		self.weights_vec = weights_vec
		self.clone = clone
		self.coil_list = self.create_coil_list()

		super().__init__(self.shape, self.shape)

	def create_coil_list(self):
		if self.randomize_coils:
			coil_list = list()
			for i in range(self.nframes):
				permuted = np.random.permutation(self.ncoils).astype(np.int32)
				coil_list.append(permuted[:self.num_rand_coils].tolist())
			return coil_list
		else:
			coil_list = list()
			for i in range(self.nframes):
				coil_list.append(np.arange(self.ncoils).tolist())
			return coil_list

	def _apply(self, input):
		input_copy: torch.Tensor
		if self.clone:
			input_copy = input.detach().clone()
		else:
			input_copy = input

		if self.randomize_coils:
			self.coil_list = self.create_coil_list()

			
				
		return input_copy