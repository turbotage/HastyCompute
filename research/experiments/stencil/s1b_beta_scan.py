"""beta scan for the plain (order-0) rms error, sigma 1.25 / 1.5, W 5..8: does a re-tuned ES match KB?"""
import numpy as np
from kern import Kernel, es_beta, kb_beta
from s1_kernel_error import errors
for sigma in [1.25, 1.5]:
    for W in [5, 6, 7, 8]:
        out = []
        for kind in ["es", "kb"]:
            b0 = es_beta(W, sigma) if kind == "es" else kb_beta(W, sigma)
            best = min(((np.sqrt((errors(Kernel(kind, W, sigma, beta=f * b0), sigma, 0)[0] ** 2).mean()), f) for f in np.linspace(0.8, 1.2, 21)))
            base = np.sqrt((errors(Kernel(kind, W, sigma), sigma, 0)[0] ** 2).mean())
            out.append(f"{kind}: rule {base:.2e}, best {best[0]:.2e} at {best[1]:.2f}x")
        print(f"sigma {sigma} W {W}: " + "; ".join(out))
