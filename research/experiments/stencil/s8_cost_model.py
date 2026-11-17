"""(sigma, W) choice under the measured cost structure (research/temporal_operator.md section 9, 11, 17).
T = T_fft * g(sigma) + T_pad + T_ker * [f_tile + (1 - f_tile) * ((W + TT - 1) / (W0 + TT - 1))^3]
g(sigma) = sigma^3 * prune(1/sigma) / same at 1.25 (pruned FFT traffic, reconstruction_operator_section.tex);
the per-sample kernel work (MACs and staging) both scale as (W + 3)^3 for 4^3 tiles; f_tile = per-tile part
(tile-value loads / stores) that does not scale with W (estimated from the section 9 ablations).
W(sigma) from the 3-D measurements (s2_out.md, KB kernel, white-image rms) at two kernel-error budgets."""
def prune(s):
    h = 1 / s
    return (h * h + h + 1) / 3 * (1 + h) / 2
def g(s):
    return s ** 3 * prune(s) / (1.25 ** 3 * prune(1.25))
regimes = {   # name: (T_fft, T_pad, T_ker, W0, f_tile)
    "1 mm static (167 k samples, 22 coils, L 11, order 2)": (0.356, 0.057, 0.150, 10, 0.5),
    "1 mm knot cubic 10 (3.34 M samples, 4 slots)": (3.56, 0.57, 6.06, 10, 0.25),
    "2 mm 2-D lin6 x cubic10 (30.3 M samples, 8 slots, order 0)": (3.0, 0.7, 9.96, 8, 0.1),
}
Wneed = {3e-4: {1.125: 9, 1.25: 7, 1.5: 6, 2.0: 5}, 1e-3: {1.125: 8, 1.25: 6, 1.5: 5, 2.0: 4}}
for name, (tf, tp, tk, W0, ft) in regimes.items():
    base = tf + tp + tk
    print(f"\n{name}: measured {base:.3f} s (sigma 1.25, W {W0})")
    print("| budget | sigma | W | FFT | kernels | total | speed-up |")
    print("|---|---|---|---|---|---|---|")
    for eps, tab in Wneed.items():
        for s, W in tab.items():
            k = tk * (ft + (1 - ft) * ((W + 3) / (W0 + 3)) ** 3)
            T = tf * g(s) + tp + k
            print(f"| {eps:.0e} | {s} | {W} | {tf*g(s):.3f} | {k:.3f} | {T:.3f} | {base/T:.2f} |")
