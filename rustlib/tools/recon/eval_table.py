"""Collect an oracle-vs-estimate evaluation (tools/recon/eval_matrix.sh): one row per configuration with the final
velocity RMSE (absolute and relative to the zero image), the change against the oracle, and the estimate diagnostics
recorded by recon_p2np (coil compression, field error vs truth, ESPIRiT agreement with the true maps).

    python -I rustlib/tools/recon/eval_table.py <out_root> oracle est_field est_smaps est_both

Writes <out_root>/eval_table.md, eval_table.csv and figs/eval_rmse.png (RMSE per iteration, one line per configuration).
"""
import csv
import glob
import json
import os
import sys


def last_metrics(d):
    files = sorted(glob.glob(os.path.join(d, "metrics_*.csv")))
    if not files:
        return None, []
    rows = list(csv.DictReader(open(files[0])))
    return files[0], rows


def main():
    root, cfgs = sys.argv[1], sys.argv[2:]
    table = []
    curves = {}
    for c in cfgs:
        d = os.path.join(root, c)
        if not os.path.exists(os.path.join(d, "run.json")):
            continue
        info = json.load(open(os.path.join(d, "run.json")))
        f, rows = last_metrics(d)
        if not rows:
            continue
        curves[c] = [(int(r["iter"]), float(r["rmse"])) for r in rows]
        last = rows[-1]
        est = info.get("estimates", {})
        coils = info.get("coils", {})
        table.append({
            "config": c,
            "rmse_m_s": float(last["rmse"]),
            "rel": float(last["rel"]),
            "iters": int(last["iter"]),
            "vcoils": f'{coils.get("kept", info.get("vcoils"))}/{coils.get("of", "")}',
            "coil_energy": coils.get("retained"),
            "field_rmse_hz": est.get("field_err_hz", {}).get("rmse"),
            "espirit_1_ip": est.get("espirit", {}).get("mean_1_minus_ip"),
            "r2s": info.get("r2s"),
        })
    if not table:
        print("no finished configurations")
        return
    base = next((r["rmse_m_s"] for r in table if r["config"] == "oracle"), None)
    for r in table:
        r["vs_oracle"] = (r["rmse_m_s"] / base - 1.0) if base else None
    keys = ["config", "rmse_m_s", "rel", "vs_oracle", "iters", "vcoils", "coil_energy", "field_rmse_hz", "espirit_1_ip", "r2s"]
    with open(os.path.join(root, "eval_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(table)

    def fmt(v, k):
        if v is None:
            return "—"
        if k == "rmse_m_s":
            return f"{100 * v:.2f} cm/s"
        if k == "vs_oracle":
            return f"{100 * v:+.1f} %"
        if k in ("rel", "coil_energy"):
            return f"{v:.4f}"
        if k == "field_rmse_hz":
            return f"{v:.1f} Hz"
        if k == "espirit_1_ip":
            return f"{v:.2e}"
        return str(v)

    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r[k], k) for k in keys) + " |" for r in table]
    open(os.path.join(root, "eval_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.join(root, "figs"), exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for c, pts in curves.items():
            pts = [p for p in pts if p[0] >= 0]
            ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], lw=2, label=c)
        ax.set_xlabel("iteration")
        ax.set_ylabel("velocity RMSE (cm/s)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(root, "figs", "eval_rmse.png"), dpi=130)
        print("figure:", os.path.join(root, "figs", "eval_rmse.png"))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
