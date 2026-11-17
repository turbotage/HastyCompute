# Where does each PnP / RED / split scheme put its bias?  1-D spikes (incoherent with Fourier), variable-density
# Fourier with repeats (no DCW, kappa ~ 60) and unsampled frequencies (null space); denoiser = soft threshold.
import numpy as np, itertools
rng=np.random.default_rng(1); n=256
x0=np.zeros(n); idx=rng.choice(n,10,replace=False); x0[idx]=rng.choice([-1,1],10)*rng.uniform(1,3,10)
F=lambda v: np.fft.fft(v,norm='ortho'); Fi=lambda v: np.fft.ifft(v,norm='ortho')
k=np.fft.fftfreq(n)*n
dens=60/np.maximum(np.abs(k),1)**1.2
w=rng.poisson(dens).astype(float); w=np.where(k<0, w[(-np.arange(n))%n], w)   # Hermitian-symmetric sampling
sig=0.5; soft=lambda v,t: np.sign(v)*np.maximum(np.abs(v)-t,0)
H=lambda v: np.real(Fi(w*F(v))); L=w.max()
Qdet=w>=8; proj=lambda v,m: np.real(Fi(F(v)*m))
def data(s):
    r=np.random.default_rng(100+s); nz=(r.normal(size=n)+1j*r.normal(size=n))/np.sqrt(2)*sig
    return np.real(Fi(w*F(x0)+np.sqrt(w)*nz))
def run(m,lam,th,AHy,it=3000):
    x=np.zeros(n); xls=np.real(Fi(np.where(Qdet,F(AHy)/np.maximum(w,1),0)))
    for _ in range(it):
        if m=='pnpI': x=soft(x-(H(x)-AHy)/L,th)
        elif m=='pnpDCW': x=soft(x-np.real(Fi(F(H(x)-AHy)/np.maximum(w,1))),th)
        elif m=='red': x=x-(H(x)-AHy+lam*(x-soft(x,th)))/(L+lam)
        elif m=='split': x=np.real(Fi((F(AHy)+lam*F(soft(x,th)))/(w+lam)))
        elif m=='hard': x=xls+proj(soft(x-(H(x)-AHy)/L,th),~Qdet)
    return x
AHys=[data(s) for s in range(8)]; nx=np.sum(x0**2)
def score(xs):
    e=xs-x0; me=e.mean(0)
    return (np.mean(np.sum(e**2,1))/nx, np.sum(proj(me,Qdet)**2)/nx, np.mean(np.sum(proj(e-me,Qdet)**2,1))/nx, np.sum(proj(me,~Qdet)**2)/nx)
T=np.geomspace(0.002,1,12)
cfg={'pnpI':([0],T),'pnpDCW':([0],T*5),'red':(np.geomspace(0.1,300,8),[0.03,0.1,0.3,1]),'split':(np.geomspace(0.03,100,8),[0.03,0.1,0.3,1]),'hard':([0],T)}
print("frac sampled %.2f, det band %d of %d freqs, w max %.0f; x0 energy in det band %.3f"%(np.mean(w>0),Qdet.sum(),n,L,np.sum(proj(x0,Qdet)**2)/nx))
for m,(lams,ths) in cfg.items():
    best=None
    for lam,th in itertools.product(lams,ths):
        s=score(np.array([run(m,lam,th,a) for a in AHys]))
        if best is None or s[0]<best[0][0]: best=(s,lam,th)
    (err,bQ,vQ,bN),lam,th=best
    print(f"{m:7s} lam {lam:7.3g} th {th:7.3g} relerr {err:.4f} | det: bias2 {bQ:.2e} var {vQ:.2e} | undet bias2 {bN:.2e}")
xls=np.array([np.real(Fi(np.where(Qdet,F(a)/np.maximum(w,1),0))) for a in AHys])
e=xls-proj(x0,Qdet); print("BLUE on det band: bias2 %.1e var %.2e"%(np.sum(e.mean(0)**2)/nx,np.mean(np.sum((e-e.mean(0))**2,1))/nx))
