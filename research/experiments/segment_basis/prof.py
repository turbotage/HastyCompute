import numpy as np, sys, os, time
D = sys.argv[1]; S = os.path.join(D, 'sim_flow_10min/scan'); out = sys.argv[2]
s = np.load(os.path.join(S,'samples.npy'), mmap_mode='r')
k0 = np.array(s[0], dtype=np.float64)
tau = k0[:,3]; kap = np.linalg.norm(k0[:,:3], axis=1)
gam = 2*np.pi*42.577478e6
dk = np.gradient(k0[:,:3], tau, axis=0)
G = np.linalg.norm(dk, axis=1)/gam
print('tau', tau[0], tau[-1], 'dt', np.diff(tau).mean(), '|G| max mT/m', G.max()*1e3, 'mean', G.mean()*1e3)
# check rotation invariance of |k| profile across readouts
for r in [1, 1000, 30000]:
    kr = np.linalg.norm(np.array(s[r])[:,:3], axis=1); print('ro', r, 'max |kap diff|', np.abs(kr-kap).max())
d = np.load(os.path.join(S,'data_clean.npy'), mmap_mode='r')
ros = np.linspace(0, d.shape[0]-1, 400).astype(int)
E = np.zeros(d.shape[1]); ss = 0.0
t0 = time.time()
for r in ros:
    a = np.array(d[r]).astype(np.complex128)
    p = (np.abs(a)**2).sum(1); E += p; ss += p.sum()
print('read', time.time()-t0, 's')
rms = np.sqrt(ss/(len(ros)*d.shape[1]*d.shape[2]))
print('rms', rms, 'noise_sigma/rms', 39.97583861630753/rms)
E /= len(ros)  # coil-summed energy per sample, mean over readouts
n = len(E)
print('energy per quarter', np.round([E[q*n//4:(q+1)*n//4].sum()/E.sum() for q in range(4)],3))
print('energy first/last 50 samples frac', E[:50].sum()/E.sum(), E[-50:].sum()/E.sum())
print('E[0]/mean', E[0]/E.mean(), 'E[1990]/mean', E[1990]/E.mean())
np.savez(out, tau=tau, kap=kap, G=G, E=E, rms=rms, k0=k0[:,:3])
