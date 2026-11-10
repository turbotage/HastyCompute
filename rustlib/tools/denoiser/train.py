"""Train the equivariant 7p denoiser on RACLETTE tier-1 patches (research/denoiser_design.md §10).

    python -I train.py [--device cuda] [--iters 100000] [--batch 4] [--patch 33] [--frames 12] [--widths 2,4,8]
                       [--group Oh] [--lr 1e-3] [--amp] [--compile] [--workers 8] [--val-every 1000] [--val-n 32]
                       [--ckpt-every 250] [--lambda-vel 1.0] [--snr 5,60] [--name NAME] [--out DIR] [--fresh]
    python -I train.py --smoke        # CPU, tiny model, 3 iterations: plumbing check only

Run directory (default data/hasty/runs/denoiser/<name>): run.json, metrics.csv, ckpt_{last,best}.pt, status.json,
figs/ (validation slices noisy / denoised / clean, loss curves) and figs/index.md.

Interrupt / resume: Ctrl-C, SIGTERM or a file named STOP in the run directory finishes the current step, writes
ckpt_last.pt and exits; running the same command again resumes from it (hyperparameters from the saved run.json,
the sample stream continues where it stopped: sample k depends only on (seed, k)). --fresh starts over.
Loss on the patch interior ∩ acquired FOV, normalised by the noise power: complex MSE over all 7 encodes +
λ_vel · complex MSE of the reference-relative encodes c_e = x_e x̄_ref/|x_ref| in the lumen (the velocity carrier).
Temporal windows shorter than the cycle use zero temporal padding and drop `rf_time` frames at each window edge.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: E402,F401  (caches inside the repo; before torch)
import torch  # noqa: E402
from dataset import REPO, Patches, Tier1, collate, interior  # noqa: E402
from hastycompute.pace import pace  # noqa: E402  (python/lib is on the path through dataset)
from model import Denoiser7p, receptive_field  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda")
ap.add_argument("--iters", type=int, default=100_000)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--patch", type=int, default=33)
ap.add_argument("--frames", type=int, default=12)
ap.add_argument("--widths", default="2,4,8")
ap.add_argument("--group", default="Oh")
ap.add_argument("--t-levels", default="0,1,2")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--amp", action="store_true")
ap.add_argument("--compile", action="store_true")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--val-every", type=int, default=1000)
ap.add_argument("--val-n", type=int, default=32)
ap.add_argument("--lambda-vel", type=float, default=1.0)
ap.add_argument("--snr", default="5,60")
ap.add_argument("--margin", type=int, default=4)
ap.add_argument("--name", default="7p")
ap.add_argument("--out", default=None)
ap.add_argument("--ckpt-every", type=int, default=250)
ap.add_argument("--fresh", action="store_true")
ap.add_argument("--smoke", action="store_true")
# live view: push a validation sample (clean / noisy / denoised flow) to the hasty server at every validation
ap.add_argument("--show", action="store_true")
args = ap.parse_args()
stop_reason = None


def request_stop(signum, _frame):  # installed before anything slow: an early Ctrl-C still exits cleanly
    global stop_reason
    stop_reason = f"signal {signum}"


import signal  # noqa: E402

signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)
if args.smoke:
    args.device, args.iters, args.batch, args.patch, args.frames = "cpu", 3, 1, 17, 9
    args.widths, args.workers, args.val_every, args.val_n, args.name = "1,1,1", 0, 2, 2, "smoke"
    torch.set_num_threads(4)
out = Path(args.out) if args.out else REPO / "data/hasty/runs/denoiser" / args.name
resume = (out / "ckpt_last.pt").exists() and not args.fresh
if resume:  # the run's own hyperparameters; only runtime settings come from this command line
    saved = json.loads((out / "run.json").read_text())
    for k, v in saved.items():
        if k in vars(args) and k not in ("device", "workers", "val_every", "ckpt_every", "fresh", "out", "compile", "show"):
            setattr(args, k, v)
dev = torch.device(args.device)
widths = [int(w) for w in args.widths.split(",")]
assert (args.patch - 1) % 2 ** (len(widths) - 1) == 0, "patch must be ≡ 1 mod 2^(levels−1) (exact flip equivariance)"
(out / "figs").mkdir(parents=True, exist_ok=True)
if args.fresh:
    for f in ["ckpt_last.pt", "ckpt_best.pt", "metrics.csv", "status.json"]:
        (out / f).unlink(missing_ok=True)
(out / "STOP").unlink(missing_ok=True)

t_pad = "zeros" if args.frames else "circular"
model = Denoiser7p(widths=widths, group=args.group, t_levels=[int(t) for t in args.t_levels.split(",")], t_pad=t_pad).to(dev)
rs, rt = receptive_field(model)
n_par = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.02)
it0, best = 0, float("inf")
if resume:
    ck = torch.load(out / "ckpt_last.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    it0, best = ck["iter"], ck.get("best", best)
    print(f"resuming {out.name} at iteration {it0} / {args.iters}", flush=True)
fwd = torch.compile(model) if args.compile else model
snr = tuple(float(s) for s in args.snr.split(","))
info = dict(vars(args), cfg=model.cfg, rf_space=rs, rf_time=rt, n_params=n_par, pad_multiple=model.pad_multiple, cmdline=" ".join(sys.argv), torch=torch.__version__)
if not resume:
    (out / "run.json").write_text(json.dumps(info, indent=1))
print(f"model {model.cfg}: {n_par} parameters, receptive field ±{rs} voxels / ±{rt} frames -> {out}", flush=True)

train_ds = Tier1("train", patch=args.patch, frames=args.frames, length=max(args.iters - it0, 0) * args.batch, snr=snr, seed=1, offset=it0 * args.batch)


def worker_init(_):
    import os
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the main process decides when to stop
    os.environ.setdefault("HASTY_VALUE_THREADS", "2")


loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate, worker_init_fn=worker_init,
                                     persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)


def checkpoint(it, path="ckpt_last.pt"):
    ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "iter": it, "cfg": model.cfg, "best": best, "info": info}
    tmp = out / (path + ".tmp")
    torch.save(ck, tmp)
    tmp.replace(out / path)


def status(it, rate):
    left = (args.iters - it) / rate if rate > 0 else float("nan")
    (out / "status.json").write_text(json.dumps({"iter": it, "target": args.iters, "it_per_s": rate, "eta_h": left / 3600, "stopped": stop_reason, "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
make = Patches(args.patch, device=dev)
T = args.frames or None
mt = rt if args.frames else 0
inner = interior(args.patch, args.frames, margin=(args.margin, mt)).to(dev) if args.frames else None
assert inner is None or inner.any(), f"empty loss window: frames {args.frames} ≤ 2 × temporal receptive field {rt}"


def rel_feat(x):
    """c_e = x_e conj(x_ref)/|x_ref| for e = 1..6 (complex [B, 6, T, P, P, P])"""
    from model import inv_abs

    xc = torch.complex(x[:, :, 0], x[:, :, 1])
    u = xc[:, :1] * inv_abs(x[:, :1, 0], x[:, :1, 1])
    return xc[:, 1:] * u.conj()


def losses(b, y):
    """y real [B, 7, 2, T, P, P, P]; returns (total, image term, velocity term) per batch"""
    x = torch.stack([b["clean"].real, b["clean"].imag], 2)
    w = b["valid"][:, None] & (inner[None] if inner is not None else True)  # [B, T, P, P, P]
    s2 = (b["sigma"] ** 2).flatten(1).mean(1).view(-1, 1, 1, 1, 1)
    e_img = ((y - x) ** 2).sum(dim=(1, 2)) / s2  # [B, T, P, P, P]
    l_img = (e_img * w).sum() / w.sum().clamp_min(1) / 7
    wf = w.expand_as(e_img)
    per_sample = ((e_img * wf).flatten(1).sum(1) / wf.flatten(1).sum(1).clamp_min(1) / 7).detach()  # diagnostics
    wl = w & b["mask"]
    e_vel = (rel_feat(y) - rel_feat(x)).abs().pow(2).sum(1) / s2
    l_vel = (e_vel * wl).sum() / wl.sum().clamp_min(1) / 6
    return l_img + args.lambda_vel * l_vel, l_img, l_vel, per_sample


def as_real(z):
    return torch.stack([z.real, z.imag], 2)


# fixed validation set
val_ds = Tier1("val" if any(c["split"] == "val" for c in json.loads((REPO / "data/hasty/train/raclette/index.json").read_text())["cases"]) else "test",
               patch=args.patch, frames=args.frames, length=args.val_n, snr=snr, seed=12345)
val = [make(collate([val_ds[i]])) for i in range(args.val_n)]


def velocity(x, kv):
    """real [B, 7, 2, ...] -> v [B, 3, ...] from the high-VENC encodes (m/s)"""
    xc = torch.complex(x[:, :, 0], x[:, :, 1])
    return torch.stack([torch.angle(xc[:, 2 + 2 * a] * xc[:, 0].conj()) / kv[:, 2 + 2 * a].view(-1, *[1] * (xc.dim() - 2)) for a in range(3)], 1)


@torch.no_grad()
def validate(it):
    model.eval()
    acc = {"loss": 0.0, "nmse_noisy": 0.0, "nmse_den": 0.0, "vrmse_noisy": 0.0, "vrmse_den": 0.0}
    # NMSE as a ratio of sums over the validation set: a batch of phantom patches in air (clean exactly 0, unlike
    # RACLETTE's in-vivo backgrounds) divided by ≈ 0 and gave 5e11 from iteration 18000 on
    num_n, num_d, den_t = 0.0, 0.0, 0.0
    for b in val:
        xn = as_real(b["noisy"])
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=args.amp):
            y = fwd(xn, b["kv"], b["h"]).float()
        acc["loss"] += float(losses(b, y)[0])
        x = as_real(b["clean"])
        w = (b["valid"][:, None] & (inner[None] if inner is not None else True))[:, None, None]
        den_t += float(((x ** 2) * w).sum())
        num_n += float((((xn - x) ** 2) * w).sum())
        num_d += float((((y - x) ** 2) * w).sum())
        wl = (b["mask"] & b["valid"][:, None])[:, None]
        if wl.any():
            vt = velocity(x, b["kv"])
            acc["vrmse_noisy"] += float(((velocity(xn, b["kv"]) - vt) ** 2 * wl).sum().div(wl.sum() * 3).sqrt())
            acc["vrmse_den"] += float(((velocity(y, b["kv"]) - vt) ** 2 * wl).sum().div(wl.sum() * 3).sqrt())
    model.train()
    out = {k: v / len(val) for k, v in acc.items()}
    out["nmse_noisy"], out["nmse_den"] = num_n / max(den_t, 1e-30), num_d / max(den_t, 1e-30)
    return out


def velocity_dual(x, kv):
    """real [B, 7, 2, ...] -> v [B, 3, ...] (m/s): low-VENC phases unwrapped by the high-VENC estimate"""
    xc = torch.complex(x[:, :, 0], x[:, :, 1])
    out = []
    for a in range(3):
        sh = (-1, *[1] * (xc.dim() - 2))
        klo, khi = kv[:, 1 + 2 * a].view(sh), kv[:, 2 + 2 * a].view(sh)
        vhi = torch.angle(xc[:, 2 + 2 * a] * xc[:, 0].conj()) / khi
        vlo = torch.angle(xc[:, 1 + 2 * a] * xc[:, 0].conj()) / klo
        out.append(vlo + 2 * np.pi * torch.round((vhi - vlo) * klo / (2 * np.pi)) / klo)
    return torch.stack(out, 1)


_live = {}


def show_live(it):
    """validation sample 0 as three flow values (overwritten in place, so open flow viewers reload)"""
    try:
        from hastycompute.v2 import Client, flow_value
        if "client" not in _live:
            _live["client"] = Client()
        c = _live["client"]
        b = val[0]
        with torch.no_grad():
            y = fwd(as_real(b["noisy"]), b["kv"], b["h"]).float()
        h = float(b["h"][0])
        for name, a in (("clean", as_real(b["clean"])), ("noisy", as_real(b["noisy"])), ("denoised", y)):
            v = velocity_dual(a, b["kv"])[0]  # [3, T, P0, P1, P2], component a along patch axis a
            vel = v.flip(0).permute(1, 0, 2, 3, 4).cpu().numpy()  # viewer [F, 3, Z, Y, X]: X = patch axis 2
            mag = torch.sqrt(a[0, 0, 0] ** 2 + a[0, 0, 1] ** 2).cpu().numpy()
            value = flow_value(vel, mag, spacing=(h, h, h), dt=0.04, layout="c")
            key = f"{args.name} val0 {name}"
            meta = {"kind": "flow4d", "run": args.name, "iteration": it, "case": b["case"][0], "snr": float(b["snr"][0])}
            if key in _live:
                c.write(_live[key], value)
                c.set_metadata(_live[key], meta)
            else:
                _live[key] = c.put(value, name=key, metadata=meta)
                if name == "denoised":
                    c.show(_live[key], "flow")
                    print(f"live view: {c.viewer_url(_live[key], 'flow')}", flush=True)
    except Exception as e:  # the viewer is optional: never stop training for it
        print(f"--show: {type(e).__name__}: {e}", flush=True)


def figure(it):
    # the first validation sample, and the first one from a phantom case (data/hasty/train/phantom) if any
    samples = [("val", val[0])]
    ph = next((v for v in val if "train/phantom" in str(v["case"][0])), None)
    if ph is not None:
        samples.append(("val_ph", ph))
    for tag, b in samples:
        figure_one(it, tag, b)


def figure_one(it, tag, b):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        y = fwd(as_real(b["noisy"]), b["kv"], b["h"]).float()
    x, xn = as_real(b["clean"]), as_real(b["noisy"])
    t, z = x.shape[3] // 2, x.shape[-1] // 2
    vt, vn, vd = (velocity(a, b["kv"])[0, :, t, :, :, z].cpu() for a in (x, xn, y))
    mag = lambda a: torch.sqrt(a[0, 0, 0, t, :, :, z] ** 2 + a[0, 0, 1, t, :, :, z] ** 2).cpu()
    lum = b["mask"][0, t, :, :, z].cpu().float()
    venc = float(np.pi / b["kv"][0, 2])
    fig, ax = plt.subplots(3, 5, figsize=(16, 9.5))
    hi = float(mag(x).quantile(0.995))
    for i, (img, ttl) in enumerate([(mag(xn), "|x_ref| noisy"), (mag(y), "|x_ref| denoised"), (mag(x), "|x_ref| clean"), ((mag(y) - mag(x)).abs(), "|den − clean| ×5"), ((mag(xn) - mag(x)).abs(), "|noisy − clean| ×5")]):
        ax[0, i].imshow(img.T, origin="lower", cmap="gray" if i < 3 else "magma", vmin=0, vmax=hi if i < 3 else hi / 5)
        ax[0, i].set_title(ttl, fontsize=8)
    for r, a in [(1, 0), (2, 1)]:
        for i, (img, ttl) in enumerate([(vn[a], "noisy"), (vd[a], "denoised"), (vt[a], "clean"), (vd[a] - vt[a], "den − clean (±0.2·VENC)"), (vn[a] - vt[a], "noisy − clean (±0.2·VENC)")]):
            lim = venc if i < 3 else 0.2 * venc
            ax[r, i].imshow((img * (lum > 0.5) if i >= 3 else img).T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[r, i].contour(lum.T, levels=[0.5], colors="y", linewidths=0.5)
            ax[r, i].set_title(f"v{'xyz'[a]} {ttl}", fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"iter {it}: {b['case'][0]}, SNR {float(b['snr'][0]):.1f}, VENC_hi {venc:.1f} m/s, frame {t}, slice z={z}", fontsize=9)
    fig.tight_layout()
    name = f"{tag}_{it:07d}.png"
    fig.savefig(out / "figs" / name, dpi=80)
    plt.close(fig)
    if tag != "val":
        return
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    if len(rows) > 1:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        k = [int(r["iter"]) for r in rows]
        ax[0].semilogy(k, [float(r["train_loss"]) for r in rows], label="train")
        ax[0].semilogy(k, [float(r["val_loss"]) for r in rows], label="val")
        ax[0].legend()
        ax[0].set_title("loss (noise-power units)")
        ax[1].semilogy(k, [float(r["nmse_den"]) for r in rows], label="denoised")
        ax[1].semilogy(k, [float(r["nmse_noisy"]) for r in rows], label="noisy")
        ax[1].legend()
        ax[1].set_title("val NMSE (all encodes)")
        ax[2].plot(k, [float(r["vrmse_den"]) for r in rows], label="denoised")
        ax[2].plot(k, [float(r["vrmse_noisy"]) for r in rows], label="noisy")
        ax[2].legend()
        ax[2].set_title("val lumen velocity RMSE (m/s)")
        fig.tight_layout()
        fig.savefig(out / "figs" / "curves.png", dpi=80)
        plt.close(fig)
    figs = sorted(p.name for p in (out / "figs").glob("val_*.png"))
    (out / "figs" / "index.md").write_text(f"# {out.name}\n\n`{info['cmdline']}`\n\n![curves](curves.png)\n\n" + "".join(f"## {f}\n![{f}]({f})\n\n" for f in figs[-5:]))


with open(out / "metrics.csv", "a", newline="") as fh:
    wr = csv.writer(fh)
    if it0 == 0:
        wr.writerow(["iter", "lr", "train_loss", "train_img", "train_vel", "val_loss", "nmse_noisy", "nmse_den", "vrmse_noisy", "vrmse_den", "sec_per_iter"])
model.train()
run_l = run_i = run_v = 0.0
nrun, t0 = 0, time.time()
it = it0
t_rate, it_rate = time.time(), it0
bad = 0
win = []
for raw in loader:
    if it >= args.iters:
        break
    if stop_reason is None and (out / "STOP").exists():
        stop_reason = "STOP file"
    if stop_reason is not None:
        break
    b = make(raw)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=args.amp):
        y = fwd(as_real(b["noisy"]), b["kv"], b["h"]).float()
    loss, li, lv, ps = losses(b, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not (torch.isfinite(loss) and torch.isfinite(gn)):  # never let one bad batch poison the weights
        bad += 1
        print(f"it {it}: non-finite loss / gradient ({loss.item():.3g}, {gn.item():.3g}) on {raw['case']}: step skipped", flush=True)
        if bad >= 20:
            stop_reason = "20 consecutive non-finite steps"
            break
        opt.zero_grad(set_to_none=True)
        sched.step()
        it += 1
        continue
    bad = 0
    opt.step()
    pace()  # GPU pacing (HASTY_GPU_DUTY / HASTY_GPU_WATTS; a no-op unless set)
    sched.step()
    it += 1
    run_l, run_i, run_v, nrun = run_l + loss.item(), run_i + li.item(), run_v + lv.item(), nrun + 1
    # per-iteration diagnostics: the σ²-normalised loss is heavy-tailed (high SNR × fine structure); log what drives it
    win.append(loss.item())
    with open(out / "loss_iter.csv", "a") as fh:
        fh.write(f"{it},{loss.item():.5g},{li.item():.5g},{lv.item():.5g},{gn.item():.4g}," + ";".join(f"{v:.4g}" for v in ps.tolist()) + "," + ";".join(f"{v:.3g}" for v in b["snr"].tolist()) + "," + ";".join(raw["case"]) + "\n")
    if it % args.val_every == 0 or it == args.iters:
        m = validate(it)
        dt = (time.time() - t0) / nrun
        with open(out / "metrics.csv", "a", newline="") as fh:
            csv.writer(fh).writerow([it, sched.get_last_lr()[0], run_l / nrun, run_i / nrun, run_v / nrun, m["loss"], m["nmse_noisy"], m["nmse_den"], m["vrmse_noisy"], m["vrmse_den"], dt])
        wq = np.quantile(np.array(win), [0.5, 0.9]) if win else [np.nan, np.nan]
        print(f"it {it}: train {run_l / nrun:.4f} (img {run_i / nrun:.4f}, vel {run_v / nrun:.4f}; median {wq[0]:.3f}, p90 {wq[1]:.3f}, max {max(win, default=np.nan):.3g}) val {m['loss']:.4f}, "
              f"NMSE {m['nmse_noisy']:.2e} -> {m['nmse_den']:.2e}, v RMSE {m['vrmse_noisy']:.3f} -> {m['vrmse_den']:.3f} m/s, {dt:.2f} s/it", flush=True)
        if m["loss"] < best:
            best = m["loss"]
            checkpoint(it, "ckpt_best.pt")
        checkpoint(it)
        figure(it)
        if args.show:
            show_live(it)
        run_l = run_i = run_v = 0.0
        nrun, t0 = 0, time.time()
        win = []
    elif it % args.ckpt_every == 0:
        checkpoint(it)
    if it % 20 == 0:
        status(it, (it - it_rate) / max(time.time() - t_rate, 1e-9))
if stop_reason is not None:
    checkpoint(it)
    status(it, (it - it_rate) / max(time.time() - t_rate, 1e-9))
    print(f"stopped ({stop_reason}) at iteration {it} / {args.iters}: checkpoint saved, rerun the same command to resume", flush=True)
else:
    checkpoint(it)
    status(it, 0.0)
    print(f"done: {out}")
