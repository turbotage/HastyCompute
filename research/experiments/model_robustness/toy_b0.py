# 1D toy: off-resonance model with an estimated field map. Sim on a 4x finer grid (no inverse crime).
import numpy as np
from scipy.stats import spearmanr
rng0=np.random.default_rng(0)
N=128; F=4; Nf=N*F; dx=1.0                     # mm
r=(np.arange(N)-N/2)*dx; rf=(np.arange(Nf)-Nf/2)*dx/F + (dx/F)*(0.5-F/2)/1
kmax=0.5; T=10e-3; S=96; Rtr=24; Rho=8          # readouts used for recon / held out
def obj(x):  # piecewise constant + smooth, complex
    m=1.0*(abs(x)<45)+0.6*((x>-30)&(x<-10))+0.8*((x>5)&(x<12))+0.4*np.exp(-((x-25)/6)**2)
    return m*np.exp(1j*0.3*np.sin(x/20))
def field(x):  # rad/s: smooth background + sharp susceptibility peak near an "interface"
    return 2*np.pi*(25*np.sin(x/30)+45*np.exp(-((x-38)/5)**2)-20*np.exp(-((x+20)/12)**2))
xf=obj(rf); wf=field(rf)
xtrue=xf.reshape(N,F).mean(1); wtrue=field(r)
def traj(R,rng):
    tau=np.linspace(0,T,S); ks=[];ts=[]
    for i in range(R):
        p=rng.uniform(0.6,1.6); sgn=rng.choice([-1,1]); k0=rng.uniform(0,0.01)
        ks.append(sgn*(k0+(kmax-k0)*(tau/T)**p)); ts.append(tau)
    return np.concatenate(ks),np.concatenate(ts)
def op(k,t,w,grid,damp=None):
    A=np.exp(-2j*np.pi*np.outer(k,grid))*np.exp(-1j*np.outer(t,w))
    if damp is not None: A=A*np.exp(-0.5*np.outer(t**2,damp**2))
    return A
D=np.eye(N)-np.roll(np.eye(N),1,1)
def recon(A,y,lam):
    return np.linalg.solve(A.conj().T@A+lam*D.T@D+1e-6*np.eye(N),A.conj().T@y)
def run(seed,emodel,eta,sigma_noise=0.02):
    rng=np.random.default_rng(seed)
    k,t=traj(Rtr+Rho,rng)
    yf=(op(k,t,wf,rf)@xf)/F
    y=yf+sigma_noise*np.sqrt(N)*(rng.standard_normal(len(k))+1j*rng.standard_normal(len(k)))/np.sqrt(2)*0+0
    noise=sigma_noise*(rng.standard_normal(len(k))+1j*rng.standard_normal(len(k)))/np.sqrt(2)*np.abs(yf).max()
    y=yf+noise
    ntr=Rtr*S; tr=slice(0,ntr); ho=slice(ntr,None)
    # field-map error models (none of them is the generator of wtrue)
    if emodel=='scale': wh=(1+eta)*wtrue
    elif emodel=='smooth':
        c=rng.standard_normal(6); e=sum(c[i]*np.cos((i+1)*np.pi*r/128+rng.uniform(0,6.3)) for i in range(6))
        wh=wtrue+eta*np.sqrt(np.mean(wtrue**2))*e/np.sqrt(np.mean(e**2))
    elif emodel=='shift': wh=field(r-eta*10)       # eta=1 -> 10 mm misregistration
    elif emodel=='blur':
        L=max(1e-3,eta*20); g=np.exp(-0.5*(np.arange(-40,41)/L)**2); g/=g.sum(); wh=np.convolve(wtrue,g,'same')
    err=wh-wtrue; sig=np.sqrt(np.mean(err**2))
    lam=1e-3*np.linalg.norm(y[tr])**2/N
    out={}
    def ev(name,w,damp=None,alpha=1.0):
        A=op(k[tr],t[tr],alpha*w,r,damp); xh=recon(A,y[tr],lam)
        Ah=op(k[ho],t[ho],alpha*w,r,damp)
        out[name]=(np.linalg.norm(xh-xtrue)/np.linalg.norm(xtrue), np.mean(abs(y[ho]-Ah@xh)**2)/np.mean(abs(noise[ho])**2))
    ev('ignore',0*wtrue); ev('plugin',wh); ev('oracle',wtrue)
    ev('expected',wh,damp=np.full(N,sig))
    # alpha chosen by held-out PE
    best=None
    for a in [0,0.25,0.5,0.75,1.0,1.25]:
        ev(f'a{a}',wh,alpha=a)
        if best is None or out[f'a{a}'][1]<out[best][1]: best=f'a{a}'
    out['alphaCV']=out[best]; out['alpha_sel']=float(best[1:])
    out['sig_Hz']=sig/2/np.pi
    return out
print("wtrue range Hz:",wtrue.min()/2/np.pi,wtrue.max()/2/np.pi)
rows=[]
for em,etas in [('scale',[-1,-0.5,-0.25,0.25,0.5,1.0,1.5]),('smooth',[0.1,0.25,0.5,1.0,1.5]),('shift',[0.2,0.5,1.0,2.0]),('blur',[0.25,0.5,1.0,2.0])]:
    for eta in etas:
        R=[run(s,em,eta) for s in range(8)]
        def m(n,i): return np.mean([q[n][i] for q in R])
        sel=np.mean([q['alpha_sel'] for q in R]); sh=np.mean([q['sig_Hz'] for q in R])
        print(f"{em:6s} eta={eta:5.2f} errRMS={sh:5.1f}Hz | NRMSE ign {m('ignore',0):.3f} plug {m('plugin',0):.3f} exp {m('expected',0):.3f} aCV {m('alphaCV',0):.3f} (a={sel:.2f}) orac {m('oracle',0):.3f} | PE/noise ign {m('ignore',1):.2f} plug {m('plugin',1):.2f} exp {m('expected',1):.2f} orac {m('oracle',1):.2f}")
        for q in R:
            for n in ['ignore','plugin','expected','oracle']+[f'a{a}' for a in [0,0.25,0.5,0.75,1.0,1.25]]:
                rows.append((q[n][0],q[n][1]))
rows=np.array(rows); print("Spearman(held-out PE, image NRMSE) over all configs:",spearmanr(rows[:,0],rows[:,1]).correlation)
