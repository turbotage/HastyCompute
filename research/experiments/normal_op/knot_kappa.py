"""Fluctuation constant kappa of the knot-pair weights: for a cell fed by n_eff samples with i.i.d. acquisition times
(incoherent ordering), ||T(u) - rho(u) G_mean||_F / ||rho(u) G_mean||_F ~= sqrt(kappa / n_eff),
kappa = E||b b^T||_F^2 / ||E b b^T||_F^2 - 1   (b = 96 knot weights of one readout)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
ro, ph = common.readout_table()
idx = np.arange(0, ro.shape[0], 7)
ki, kw = common.knot_weights_per_readout(idx)
B = np.zeros((len(idx), 96)); np.add.at(B, (np.repeat(np.arange(len(idx)), 8), ki.ravel()), kw.ravel())
G = B.T @ B / len(idx)
e2 = np.mean(np.sum(B ** 2, 1) ** 2)          # ||b b^T||_F^2 = ||b||^4
kappa = e2 / np.sum(G ** 2) - 1
print(f"kappa = {kappa:.1f}; n_eff for 1e-2 / 1e-4 rank-1 error: {kappa/1e-4:.3g} / {kappa/1e-8:.3g}")
# readouts per knot (support counts) and how uniformly the times cover the knots
print("readouts per encode", len(idx), "mean weight per knot", B.sum(0).mean(), "min/max", B.sum(0).min(), B.sum(0).max())
