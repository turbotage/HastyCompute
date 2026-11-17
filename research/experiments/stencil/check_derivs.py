import numpy as np
from kern import Kernel
for kind in ["es","kb"]:
    k = Kernel(kind, 8, 1.25, clamp=None)
    t = np.linspace(-3.5, 3.5, 15); h = 1e-3
    d = k.derivs_t(t)
    f = lambda tt: k.derivs_t(tt)[0]
    fd = [(f(t+h)-f(t-h))/(2*h), (f(t+h)-2*f(t)+f(t-h))/h**2]
    d3 = (k.derivs_t(t+h)[2]-k.derivs_t(t-h)[2])/(2*h); d4=(k.derivs_t(t+h)[2]-2*k.derivs_t(t)[2]+k.derivs_t(t-h)[2])/h**2
    print(kind, np.max(np.abs(d[1]-fd[0])), np.max(np.abs(d[2]-fd[1])), np.max(np.abs(d[3]-d3)), np.max(np.abs(d[4]-d4)), "scale", np.abs(d).max(axis=1))
    print(" phihat(0) vs sum", k.phihat(0.0), np.trapezoid(f(np.linspace(-4,4,20001)), dx=8/20000))
