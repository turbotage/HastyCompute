# Toy: where does each PnP/RED/split scheme put its bias? 1-D, Fourier sampling with repeats (no DCW) + null space.
import numpy as np
from scipy.fft import dct, idct
rng=np.random.default_rng(0)
n=256
# truth: sparse in DCT (orthonormal)
c=np.zeros(n); idx=rng.choice(n,20,replace=False); c[idx]=rng.normal(0,1,20)*np.exp(-idx/80)*10
x0=idct(c,norm='ortho')
F=lambda v: np.fft.fft(v,norm='ortho'); Fi=lambda v: np.fft.ifft(v,norm='ortho')
k=np.fft.fftfreq(n)*n
# variable density sample counts: ~1/k^2 near centre, zero beyond some random holes
dens=np.minimum(400,40/np.maximum(np.abs(k),0.5)**1.5)
w=rng.poisson(dens*0.5).astype(float)
w[np.abs(k)>100]=0  # high-k null band
print("sampled freq frac",np.mean(w>0)," max/min(nonzero) w",w.max(),w[w>0].min())
sig=0.3
def soft(v,t): return np.sign(v)*np.maximum(np.abs(v)-t,0)
D=lambda z,t: idct(soft(dct(z.real,norm='ortho'),t),norm='ortho')
def data(seed):
    r=np.random.default_rng(seed)
    # y_k: w_k repeats -> sufficient stat: sum of repeats; A^H y = F^H( w*Fx0 + sqrt(w)*noise )
    nz=(r.normal(size=n)+1j*r.normal(size=n))/np.sqrt(2)*sig
    return np.real(Fi(w*F(x0)+np.sqrt(w)*nz))   # real-part projection (real signal)
H=lambda v: np.real(Fi(w*F(v)))
L=w.max()
Qdet=w>=20   # well-determined frequencies
def proj(v,mask): return np.real(Fi(F(v)*mask))
def run(method,lam,AHy,it=3000):
    x=np.zeros(n)
    if method=='pnpI':
        a=1/L
        for _ in range(it): x=D(x-a*(H(x)-AHy),lam*a)
    elif method=='pnpDCW':
        p=1/np.maximum(w,1)
        for _ in range(it): x=D(x-np.real(Fi(p*F(H(x)-AHy))),lam)
    elif method=='red':  # grad f + lam*(x-D(x,theta)), theta fixed=1/... ; solve by gradient with step 1/(L+lam)
        th=1.0
        for _ in range(it): x=x-(H(x)-AHy+lam*(x-D(x,th)))/(L+lam)
    elif method=='split':  # HQS: x=(H+rho)^-1(AHy+rho D(x)) ; rho small -> DDNM-like
        rho=lam; th=1.0
        for _ in range(it):
            x=np.real(Fi((F(AHy)+rho*F(D(x,th)))/(w+rho)))
    return x
methods={'pnpI':[2,5,10,20,40],'pnpDCW':[0.05,0.1,0.2,0.4,0.8],'red':[1,3,10,30,100],'split':[0.3,1,3,10,30]}
xls_mask=w>0
res={}
seeds=range(12)
AHys=[data(s) for s in seeds]
# LS baseline on sampled freqs (BLUE there), zero on null
for m,lams in methods.items():
    best=None
    for lam in lams:
        xs=np.array([run(m,lam,a,it=1500 if m!='pnpI' else 4000) for a in AHys])
        err=np.mean(np.sum((xs-x0)**2,1))/np.sum(x0**2)
        if best is None or err<best[0]: best=(err,lam,xs)
    err,lam,xs=best
    e=xs-x0; mean_e=e.mean(0)
    bQ=np.sum(proj(mean_e,Qdet)**2)/np.sum(x0**2); vQ=np.mean(np.sum(proj(e-mean_e,Qdet)**2,1))/np.sum(x0**2)
    bN=np.sum(proj(mean_e,~Qdet)**2)/np.sum(x0**2)
    # correction spectral profile at one run: c = D(x)-x direction energy in Qdet
    res[m]=(err,lam,bQ,vQ,bN)
    print(f"{m:7s} best lam {lam:6g} relerr {err:.4f}  bias^2(det) {bQ:.2e} var(det) {vQ:.2e} bias^2(undet) {bN:.2e}")
# LS on det band
xls=np.array([np.real(Fi(np.where(Qdet,F(a)/np.maximum(w,1),0))) for a in AHys])
e=xls-proj(x0,Qdet); print("LS on det band: var(det)",np.mean(np.sum(e**2,1))/np.sum(x0**2))
