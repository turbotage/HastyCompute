# Toy: where does each scheme put its bias? 1-D spikes (incoherent with Fourier), variable-density Fourier with
# repeats (no DCW, ill-conditioned) and unsampled frequencies (null space). Denoiser = soft threshold (prox of l1).
import numpy as np
rng=np.random.default_rng(1)
n=256
x0=np.zeros(n); idx=rng.choice(n,10,replace=False); x0[idx]=rng.choice([-1,1],10)*rng.uniform(1,3,10)
F=lambda v: np.fft.fft(v,norm='ortho'); Fi=lambda v: np.fft.ifft(v,norm='ortho')
k=np.fft.fftfreq(n)*n
dens=60/np.maximum(np.abs(k),1)**1.2
w=rng.poisson(dens).astype(float)
print("sampled frac %.2f  w max %.0f  #w>=20 %d"%(np.mean(w>0),w.max(),np.sum(w>=20)))
sig=0.5
soft=lambda v,t: np.sign(v)*np.maximum(np.abs(v)-t,0)
def data(seed):
    r=np.random.default_rng(100+seed)
    nz=(r.normal(size=n)+1j*r.normal(size=n))/np.sqrt(2)*sig
    yk=w*F(x0)+np.sqrt(w)*nz
    # enforce Hermitian symmetry -> real AHy
    return np.real(Fi(yk))
H=lambda v: np.real(Fi(w*F(v)))
L=w.max()
Qdet=w>=8
proj=lambda v,m: np.real(Fi(F(v)*m))
def run(m,lam,AHy,it):
    x=np.zeros(n)
    if m=='pnpI':
        a=1/L
        for _ in range(it): x=soft(x-a*(H(x)-AHy),lam*a)
    elif m=='pnpDCW':
        p=1/np.maximum(w,1)
        for _ in range(it): x=soft(x-np.real(Fi(p*F(H(x)-AHy))),lam)
    elif m=='red':
        th=0.3
        for _ in range(it): x=x-(H(x)-AHy+lam*(x-soft(x,th)))/(L+lam)
    elif m=='split':
        th=0.3
        for _ in range(it): x=np.real(Fi((F(AHy)+lam*F(soft(x,th)))/(w+lam)))
    return x
grid={'pnpI':np.geomspace(0.1,30,12),'pnpDCW':np.geomspace(0.003,1,12),'red':np.geomspace(0.1,300,12),'split':np.geomspace(0.03,100,12)}
AHys=[data(s) for s in range(10)]
nx=np.sum(x0**2)
for m,lams in grid.items():
    out=[]
    for lam in lams:
        xs=np.array([run(m,lam,a,3000) for a in AHys])
        out.append((np.mean(np.sum((xs-x0)**2,1))/nx,lam,xs))
    err,lam,xs=min(out,key=lambda t:t[0])
    e=xs-x0; me=e.mean(0)
    bQ=np.sum(proj(me,Qdet)**2)/nx; vQ=np.mean(np.sum(proj(e-me,Qdet)**2,1))/nx; bN=np.sum(proj(me,~Qdet)**2)/nx
    # correction c at the limit: PnP: c = x - z ; RED/split: c = D(x)-x ; report fraction of ||c||^2 in Qdet
    a=AHys[0]; x=xs[0]
    if m=='pnpI': z=x-(H(x)-a)/L; cc=x-z
    elif m=='pnpDCW': z=x-np.real(Fi(F(H(x)-a)/np.maximum(w,1))); cc=x-z
    else: cc=soft(x,0.3)-x
    fr=np.sum(proj(cc,Qdet)**2)/np.sum(cc**2)
    edge='  (grid edge!)' if lam in (lams[0],lams[-1]) else ''
    print(f"{m:7s} lam {lam:8.3g} relerr {err:.4f} | det band: bias2 {bQ:.2e} var {vQ:.2e} | undet bias2 {bN:.2e} | corr energy in det {fr:.2f}{edge}")
xls=np.array([np.real(Fi(np.where(Qdet,F(a)/np.maximum(w,1),0))) for a in AHys])
print("LS (BLUE) on det band: var %.2e"%(np.mean(np.sum((xls-proj(x0,Qdet))**2,1))/nx), " det-band energy of x0 %.3f"%(np.sum(proj(x0,Qdet)**2)/nx))
