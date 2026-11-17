# Gaussian (stationary) prior, coloured Gaussian error: white-noise MMSE denoiser vs noise completion vs coloured MMSE.
import numpy as np
n=4096; k=np.fft.fftfreq(n)*n
c=1/(1+(k/8)**2)                          # prior power spectrum
for spread in [1,10,100,1000]:
    s=0.05*(1+(spread-1)*(np.abs(k)/np.abs(k).max())**1.5)   # error variance rises with |k| (variable-density back-projection)
    s*=0.05*n/s.sum()                     # same total error energy for every spread
    tot=c.sum()
    exact=np.sum(c*s/(c+s))/tot
    best=min(np.sum(((s2/(c+s2))**2)*c+((c/(c+s2))**2)*s)/tot for s2 in np.geomspace(s.min()/10,s.max()*10,400))
    matched=np.sum(((s.mean()/(c+s.mean()))**2)*c+((c/(c+s.mean()))**2)*s)/tot
    comp=np.sum(c*s.max()/(c+s.max()))/tot
    print(f"spread {spread:5d}: coloured MMSE {exact:.4f} | white D at mean sigma {matched:.4f} | white D best sigma {best:.4f} | noise completion to max {comp:.4f}")
