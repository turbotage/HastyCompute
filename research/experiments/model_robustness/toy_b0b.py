import numpy as np, importlib.util, sys
spec=importlib.util.spec_from_file_location("tb","toy_b0.py")
src=open("toy_b0.py").read().split("print(\"wtrue range")[0]
exec(src)
def recon_w(A,y,lam,wdiag):
    return np.linalg.solve(A.conj().T@A+lam*D.T@D+np.diag(wdiag)+1e-6*np.eye(N),A.conj().T@y)
def run2(seed,emodel,eta,sigma_noise=0.02):
    rng=np.random.default_rng(seed)
    k,t=traj(Rtr+Rho,rng)
    yf=(op(k,t,wf,rf)@xf)/F
    noise=sigma_noise*(rng.standard_normal(len(k))+1j*rng.standard_normal(len(k)))/np.sqrt(2)*np.abs(yf).max()
    y=yf+noise; ntr=Rtr*S; tr=slice(0,ntr); ho=slice(ntr,None)
    if emodel=='scale': wh=(1+eta)*wtrue
    else:
        c=rng.standard_normal(6); e=sum(c[i]*np.cos((i+1)*np.pi*r/128+rng.uniform(0,6.3)) for i in range(6))
        wh=wtrue+eta*np.sqrt(np.mean(wtrue**2))*e/np.sqrt(np.mean(e**2))
    sig=np.sqrt(np.mean((wh-wtrue)**2)); lam=1e-3*np.linalg.norm(y[tr])**2/N
    res={}
    for name,w,dmp in [('ign',0*wtrue,None),('plug',wh,None),('exp',wh,sig)]:
        A=op(k[tr],t[tr],w,r,None if dmp is None else np.full(N,dmp))
        wd=np.zeros(N) if dmp is None else np.full(N,np.sum(1-np.exp(-(dmp*t[tr])**2)))
        xh=recon_w(A,y[tr],lam,wd)
        Ah=op(k[ho],t[ho],w,r,None if dmp is None else np.full(N,dmp))
        rr=abs(y[ho]-Ah@xh)**2
        res[name]=(np.linalg.norm(xh-xtrue)/np.linalg.norm(xtrue), rr.reshape(Rho,S).mean(1))
    # paired per-readout test ignore vs plugin
    d=res['ign'][1]-res['plug'][1]; z=d.mean()/(d.std(ddof=1)/np.sqrt(Rho))
    return res,z,sig/2/np.pi
for em,etas in [('scale',[0.25,0.5,1.0,1.5]),('smooth',[0.25,0.5,1.0,1.5])]:
    for eta in etas:
        out=[run2(s,em,eta) for s in range(10)]
        nr=lambda n: np.mean([o[0][n][0] for o in out])
        # decision accuracy: holdout prefers plugin (z>0) vs truth prefers plugin
        acc=np.mean([ (o[1]>0)==(o[0]['plug'][0]<o[0]['ign'][0]) for o in out]); zm=np.mean([o[1] for o in out])
        print(f"{em:6s} eta={eta:4.2f} err {np.mean([o[2] for o in out]):5.1f}Hz NRMSE ign {nr('ign'):.3f} plug {nr('plug'):.3f} exp+var {nr('exp'):.3f} | holdout-decision acc {acc:.1f}, mean z {zm:5.1f}")
