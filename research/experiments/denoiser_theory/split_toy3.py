import numpy as np, itertools
exec(open('split_toy2.py').read().split("grid={")[0])   # reuse setup + run()
def run2(m,lam,th,AHy,it=3000):
    x=np.zeros(n)
    if m=='red':
        for _ in range(it): x=x-(H(x)-AHy+lam*(x-soft(x,th)))/(L+lam)
    elif m=='split':
        for _ in range(it): x=np.real(Fi((F(AHy)+lam*F(soft(x,th)))/(w+lam)))
    elif m=='hardsplit':   # PnP-I, but the well-determined band is replaced by its LS (BLUE) value each step
        a=1/L; xls=np.real(Fi(np.where(Qdet,F(AHy)/np.maximum(w,1),0)))
        for _ in range(it):
            x=soft(x-a*(H(x)-AHy),th)
            x=xls+proj(x,~Qdet)
    return x
AHys=[data(s) for s in range(8)]; nx=np.sum(x0**2)
def score(xs):
    e=xs-x0; me=e.mean(0)
    return (np.mean(np.sum(e**2,1))/nx, np.sum(proj(me,Qdet)**2)/nx, np.mean(np.sum(proj(e-me,Qdet)**2,1))/nx, np.sum(proj(me,~Qdet)**2)/nx)
cfg={'red':(np.geomspace(0.1,300,10),[0.1,0.3,1,3]),'split':(np.geomspace(0.03,100,10),[0.1,0.3,1,3]),'hardsplit':([1.0],np.geomspace(0.001,0.1,10))}
for m,(lams,ths) in cfg.items():
    best=None
    for lam,th in itertools.product(lams,ths):
        xs=np.array([run2(m,lam,th,a) for a in AHys]); s=score(xs)
        if best is None or s[0]<best[0][0]: best=(s,lam,th)
    (err,bQ,vQ,bN),lam,th=best
    print(f"{m:9s} lam {lam:7.3g} th {th:6.3g} relerr {err:.4f} | det: bias2 {bQ:.2e} var {vQ:.2e} sum {bQ+vQ:.2e} | undet bias2 {bN:.2e}")
