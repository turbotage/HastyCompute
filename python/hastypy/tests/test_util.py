import torch

def grid1d_coords(nx: int, ny: int):
	coord = torch.zeros(*(1,nx), dtype=torch.float32)
	l = 0
	for x in range(nx):
		#if l % 2 == 0:
		#	continue
		#if l >= NF:
		#	break

		kx = -torch.pi + x * 2 * torch.pi / nx

		coord[0,l] = kx

		l += 1

	return coord

def grid2d_coords(nx: int, ny: int):
	coord = torch.zeros(*(2,nx*ny), dtype=torch.float32)
	l = 0
	for x in range(nx):
		for y in range(ny):
			#if l % 2 == 0:
			#	continue
			#if l >= NF:
			#	break

			kx = -torch.pi + x * 2 * torch.pi / nx
			ky = -torch.pi + y * 2 * torch.pi / ny

			coord[0,l] = kx
			coord[1,l] = ky

			l += 1

	return coord

def grid3d_coords(nx: int, ny: int, nz: int):
	coord = torch.zeros(*(3,nx*ny*nz), dtype=torch.float32)
	l = 0
	for x in range(nx):
		for y in range(ny):
			for z in range(nz):
				#if l % 2 == 0:
				#	continue
				#if l >= NF:
				#	break

				kx = -torch.pi + x * 2 * torch.pi / nx
				ky = -torch.pi + y * 2 * torch.pi / ny
				kz = -torch.pi + z * 2 * torch.pi / nz

				coord[0,l] = kx
				coord[1,l] = ky
				coord[2,l] = kz

				l += 1

	return coord


