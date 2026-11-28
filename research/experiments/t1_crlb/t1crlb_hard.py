"""Inversion options, head set, flip schedule optimised (K=8, floor 3 deg), period 6 s:
 adiabatic eta known 0.95 | adiabatic eta free | hard-pulse 180 with eta tied to B1: Mz -> cos(pi*B1) Mz (no extra unknown)"""
import sys, time
import numpy as np
from scipy.optimize import minimize
import t1crlb2 as T
import t1crlb_head as H   # sets head tissues / B1 set on t1crlb2

orig = T.sim_super

def sim_hard(rho, T1, B1, eta, periods, tr=T.TR):
    E = np.exp(-tr / T1); Eg = np.exp(-T.T_GAP / T1)
    def run(m0):
        m = m0; out = []
        for prep, al in periods:
            if prep == "irhard":
                m = np.cos(np.pi * B1) * m * Eg + (1 - Eg)
            else:
                m = T._prep(m, prep, eta, Eg)
            for a in al:
                ba = B1 * a
                out.append(rho * np.sin(ba) * m)
                m = 1.0 + (m * np.cos(ba) - 1.0) * E
        return out, m
    _, b = run(np.zeros_like(rho)); _, ab = run(np.ones_like(rho))
    m0 = b / (1.0 - (ab - b)); out, _ = run(m0)
    dur = sum(len(al) * tr + (T.T_GAP if p.startswith("ir") else 0.0) for p, al in periods)
    return np.array(out), dur

T.sim_super = sim_hard
T.ETA0["irhard"] = 1.0
n = T.n_ir(6.0)
for name, prep, free in [("adiabatic, eta known 0.95", "ir", ("rho", "T1", "B1")),
                         ("adiabatic, eta free per voxel", "ir", ("rho", "T1", "B1", "eta")),
                         ("hard 180, eta = -cos(pi B1) tied", "irhard", ("rho", "T1", "B1"))]:
    k, a = H.opt(lambda x: [(prep, T.piecewise(n, x))], 8, 3.0, starts=4, free=free)
    print(f"{name:36s} kappa {k:.4g} angles {np.round(a,1)}")
    sys.stdout.flush()
