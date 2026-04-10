
import torch
import math
from hastycompute.plot import orthoslicer
import h5py

def real_spherical_harmonics(lmin, lmax, theta, phi):
    """
    Compute real spherical harmonics Y_lm^real(theta, phi)
    for lmin <= l <= lmax, -l <= m <= l.

    Args:
        lmin (int): minimum degree (>=0)
        lmax (int): maximum degree
        theta (torch.Tensor): polar angle [0, pi], shape (...)
        phi (torch.Tensor): azimuthal angle [0, 2pi], shape (...)

    Returns:
        dict[(l,m)] -> torch.Tensor with same shape as theta
    """
    x = torch.cos(theta)
    P = {}

    # Base case
    P[(0,0)] = torch.ones_like(x)

    # Diagonal P_m^m
    for m in range(1, lmax+1):
        P[(m,m)] = (-1.0)**m * torch.prod(
            torch.arange(1, 2*m, 2, device=x.device, dtype=x.dtype)
        ) * (1 - x**2).pow(m/2)

    # Off-diagonal P_{m+1}^m
    for m in range(0, lmax):
        P[(m+1, m)] = (2*m+1) * x * P[(m,m)]

    # General recurrence
    for m in range(0, lmax+1):
        for l in range(m+2, lmax+1):
            P[(l,m)] = ((2*l-1) * x * P[(l-1,m)] - (l+m-1) * P[(l-2,m)]) / (l-m)

    Y = {}

    for l in range(lmin, lmax+1):
        for m in range(0, l+1):
            # Normalization factor
            K = math.sqrt((2*l+1)/(4*math.pi) * math.factorial(l-m)/math.factorial(l+m))
            P_lm = P[(l,m)]

            if m == 0:
                Y[(l,0)] = K * P_lm
            else:
                # Positive m -> cos(m phi)
                Y[(l,m)] = math.sqrt(2) * K * P_lm * torch.cos(m*phi)
                # Negative m -> sin(|m| phi)
                Y[(l,-m)] = math.sqrt(2) * K * P_lm * torch.sin(m*phi)

    del P

    return Y

shape = (320,320,320)

x = torch.ones(shape, dtype=torch.float32).to('cuda')
idx = x.nonzero(as_tuple=True)

x = idx[0].float() - 0.5*(shape[0]-1)
y = idx[1].float() - 0.5*(shape[1]-1)
z = idx[2].float() - 0.5*(shape[2]-1)

r = torch.square(x) + torch.square(y) + torch.square(z)
r.sqrt_()

mask = r < (shape[0]//2)

theta = torch.acos(z/(r + 1e-8))
phi = torch.atan2(y, x)

Y = real_spherical_harmonics(0, 2, theta, phi)

ylist = []
for y in Y.values():
    ylist.append(y.view(shape))
y = torch.stack(ylist, dim=0)

off_resonance_map = 1.0/(0.200*torch.exp(-8*torch.square(r/r.max()))) + 1j*torch.exp(-12*torch.square(r/r.max()))

#with h5py.File('/home/turbotage/Documents/4DRecon/other_data/framed_true.h5', 'r') as f:
#    img = f['image'][:]

nspokes = 8000
nsamp_per_spoke = 2563
nframes = 20

#off_resonance_map = None
#torch.empty((nspokes // nframes, nsamp_per_spoke), dtype=torch.complex64, device='cuda')
#for spoke in range(nspokes // nframes):
#    t = 0
#    for samp in range(nsamp_per_spoke):
   
        
orthoslicer.image_nd(y.cpu().numpy())



print('Hello')