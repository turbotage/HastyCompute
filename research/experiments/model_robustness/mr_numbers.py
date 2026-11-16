import numpy as np
from scipy.optimize import brentq
T=11e-3
print("phase at tau=11ms (rad) and damping exp(-s^2 t^2/2) at tau=11ms:")
for f in [2,5,10,15,20,30,50]:
    w=2*np.pi*f
    print(f"  {f:3d} Hz: phase {w*T:5.2f} rad, damping {np.exp(-(w*T)**2/2):.3f}, mean-sq over [0,T] |e-1|^2/ (2): {1-np.sin(w*T)/(w*T):.3f}")
# L2 over readout: g(x)=1-sinc(x), x=delta*T
g=lambda x: 1-np.sin(x)/x
xs=np.linspace(1e-6,40,400001); gv=g(xs)
# first max of g
i=np.argmax(gv[:np.searchsorted(xs,6)]); print("first max of 1-sinc at x=",xs[i],"value",gv[i])
# global inf of g over x>=x1max
j=np.searchsorted(xs,xs[i]); print("min of g beyond first max:",gv[j:].min(),"at",xs[j+np.argmin(gv[j:])])
gmin=gv[j:].min()
xstar=brentq(lambda x:g(x)-gmin,0.1,xs[i]); print("x* with g(x*)=min beyond:",xstar, " -> |dw|*T <= x* guarantees L2 condition for ANY w; in Hz at T=11ms:",xstar/(2*np.pi*T))
# linear interpolation of a sub-voxel shift vs ignoring it
print("\nfrequency xi (cyc/voxel), shift s: |H_lin-ideal| vs |ideal-1| (ignore)")
for xi in [0.1,0.25,0.4,0.5]:
    row=[]
    for s in [0.1,0.25,0.5]:
        ideal=np.exp(2j*np.pi*xi*s); lin=(1-s)+s*np.exp(2j*np.pi*xi)
        row.append(f"s={s}: lin {abs(lin-ideal):.3f} ign {abs(ideal-1):.3f}")
    print(f"  xi={xi}: "+"; ".join(row))
# cubic B-spline (with prefilter) interpolation error
def bspl3(t):
    t=abs(t); return np.where(t<1,2/3-t**2+t**3/2,np.where(t<2,(2-t)**3/6,0))
for xi in [0.25,0.4,0.5]:
    row=[]
    for s in [0.1,0.25,0.5]:
        ideal=np.exp(2j*np.pi*xi*s)
        n=np.arange(-3,4)
        B=np.sum(bspl3(n)*np.exp(-2j*np.pi*xi*n)).real  # prefilter denominator
        H=np.sum(bspl3(s-n)*np.exp(2j*np.pi*xi*n))/B
        row.append(f"s={s}: cub {abs(H-ideal):.3f} ign {abs(ideal-1):.3f}")
    print(f"  cubic xi={xi}: "+"; ".join(row))
