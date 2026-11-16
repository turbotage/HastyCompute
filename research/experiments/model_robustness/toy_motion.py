import numpy as np
from scipy.special import erf
r=np.arange(-40,40,0.25)            # mm
rho=np.linspace(0,1,201); Amp=15.0  # 15 mm diaphragm-like excursion
def edge(x,w=1.0): return 0.5*(1+erf(x/(w*np.sqrt(2))))   # edge width ~1 mm (PSF)
def hat(K):  # linear B-splines with K knots on [0,1]
    kn=np.linspace(0,1,K); h=kn[1]-kn[0]
    return np.maximum(0,1-abs(rho[:,None]-kn[None,:])/h)
def fit_err(Fm,B):   # Fm: states x space; LS fit Fm ~ B C
    C,*_=np.linalg.lstsq(B,Fm,rcond=None); E=Fm-B@C
    return np.sqrt(np.mean(E**2)/0.25*len(r)*0.25)  # RMS err * sqrt(length) ~ L2 per state in mm^0.5 units
u=Amp*rho
F=np.array([edge(r-uu) for uu in u])
print("L2 error per state (units: edge height * sqrt(mm)); pure static model:",fit_err(F,np.ones((len(rho),1))))
print("knots K | basis only | hybrid warp(1-eta)u + basis, eta=0.05 / 0.1 / 0.3 / 0.5")
for K in [2,3,5,9,17,31]:
    B=hat(K); row=[fit_err(F,B)]
    for eta in [0.05,0.1,0.3,0.5]:
        uh=(1-eta)*u
        H=np.array([edge(r-uu+uh_) for uu,uh_ in zip(u,uh)])   # residual after warping back (translation: norm preserved)
        row.append(fit_err(H,B))
    print(f"K={K:3d} (knot step {Amp/(K-1):5.2f} mm) | "+" | ".join(f"{v:.3f}" for v in row))
# double edge: at mid-knot with K=2 basis
B=hat(2); C,*_=np.linalg.lstsq(B,F,rcond=None); mid=100; prof=(B@C)[mid]
print("K=2 profile at mid state has plateau value", prof[np.argmin(abs(r-7.5))].round(3)," (true edge at 7.5 mm -> double edge)")
