"""figure for research/t1_ivim_dti_sequences.md: kappa by design + the optimised IR schedule's signals"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from t1crlb2 import *

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
# (a) kappa by design (values from run4.log / run5.log, geometric mean over fat/liver/myocardium/blood, B1 in {1, 0.7})
names = ["VFA 3°/15°\n(B1 free)", "VFA optimised\n4 angles (B1 free)", "IR fixed flip\n(B1, η free)", "IR optimised 3 s\n(B1, η free)", "IR optimised 6 s\n(B1, η free)", "IR optimised 3 s\n(B1 free, η known)", "IR optimised 3 s\n(B1, η known)"]
vals = [0.735, 0.343, np.nan, 0.429, 0.346, 0.236, 0.221]
cols = ["#999", "#999", "#c44", "#2a7", "#2a7", "#27a", "#27a"]
ax[0].bar(range(len(vals)), [v if np.isfinite(v) else 0 for v in vals], color=cols)
ax[0].text(2, 0.05, "T1 not\nidentifiable", ha="center", color="#c44")
ax[0].set_xticks(range(len(vals)))
ax[0].set_xticklabels(names, rotation=60, ha="right", fontsize=7)
ax[0].set_ylabel(r"$\kappa=\sigma_{T_1}/T_1\,\times$ SNR$_{TR}$ (8 min)")
ax[0].set_title("T1 precision, same TR / ADC duty")

# (b) schedule
x = np.array([16.5, 9, 17.4, 26, 7.3, 1, 1, 1, 1, 1, 1, 13.4])
n = n_ir(3.0)
al = piecewise(n, x)
t = np.arange(n) * TR
ax[1].plot(t, np.rad2deg(al), "k")
ax[1].set_xlabel("time after inversion (s)")
ax[1].set_ylabel("flip angle (deg)")
ax[1].set_title("optimised IR flip schedule (T_p 3 s)")

# (c) signals
per = [("ir", al)]
for nm, T1, c in [("fat", .29, "#e90"), ("myocardium", 1.2, "#c33"), ("blood", 1.75, "#33c")]:
    s, _ = sim_super(np.array([1.0]), np.array([T1]), np.array([1.0]), np.array([0.95]), per)
    ax[2].plot(t, s[:, 0], label=f"{nm} T1 {T1} s", color=c)
ax[2].axhline(0, color="#888", lw=.5)
ax[2].set_xlabel("time after inversion (s)")
ax[2].set_ylabel(r"signal / $\rho$")
ax[2].legend(fontsize=8)
ax[2].set_title("resulting signal evolution")
fig.savefig("t1_crlb_summary.png", dpi=130)
