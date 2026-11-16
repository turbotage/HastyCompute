exec(open("toy_b0.py").read().split("print(\"wtrue range")[0])
def recon_w(A,y,lam,wdiag):
    return np.linalg.solve(A.conj().T@A+lam*D.T@D+np.diag(wdiag)+1e-6*np.eye(N),A.conj().T@y)
for eta in [0.25,0.5]:
  acc={}
  for seed in range(6):
    rng=np.random.default_rng(seed); k,t=traj(Rtr+Rho,rng)
    yf=(op(k,t,wf,rf)@xf)/F
    y=yf+0.02*(rng.standard_normal(len(k))+1j*rng.standard_normal(len(k)))/np.sqrt(2)*np.abs(yf).max()
    tr=slice(0,Rtr*S); wh=(1+eta)*wtrue; sig=np.sqrt(np.mean((wh-wtrue)**2)); lam=1e-3*np.linalg.norm(y[tr])**2/N
    for name,dmp,var in [('plug',None,False),('exp',sig,False),('exp+var',sig,True)]:
        A=op(k[tr],t[tr],wh,r,None if dmp is None else np.full(N,dmp))
        wd=np.full(N,np.sum(1-np.exp(-(sig*t[tr])**2))) if var else np.zeros(N)
        xh=recon_w(A,y[tr],lam,wd); c=np.vdot(xh,xtrue)/np.vdot(xh,xh)
        e1=np.linalg.norm(xh-xtrue)/np.linalg.norm(xtrue); e2=np.linalg.norm(c*xh-xtrue)/np.linalg.norm(xtrue)
        acc.setdefault(name,[]).append((e1,e2,abs(c)))
  for n,v in acc.items(): v=np.array(v); print(f"eta {eta} {n:8s} NRMSE {v[:,0].mean():.3f}  after best scalar {v[:,1].mean():.3f}  |c| {v[:,2].mean():.2f}")
