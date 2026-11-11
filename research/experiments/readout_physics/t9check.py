"""T9 check: why the image redefinition (x' = sinc-filtered x, Dirac model) leaves a larger error than omitting the rect
basis. Hypothesis: the redefinition does not commute with the per-voxel time phase C(v, tau) = e^{-i w_v tau - i w.m}
(the true rect-voxel signal at tau is sinc(k) F[C(tau) x], the redefined model F[C(tau) (sinc * x)]). Evaluated with C
and with C = 1, v2, 24 readouts. Writes <OUT>/t9check.json."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import pb

obj = pb.use_truth(pb.Obj(), 'v2')
des = pb.Design('v2')
nsel = des.sample_select(kc=150.0, stride=8)
ros = pb.scan_readouts(24, offset=7)
m4 = pb.monos(obj.t_pos)
nx, ny, nz = obj.dims
gg = np.zeros(nx * ny * nz, np.complex64); gg[obj.idx] = obj.x_tot
G = np.fft.fftn(gg.reshape((nx, ny, nz), order='F'))
kk = [2 * np.pi * np.fft.fftfreq(m, obj.res) * obj.res / 2 for m in (nx, ny, nz)]
sw = np.sinc(kk[0][:, None, None] / np.pi) * np.sinc(kk[1][None, :, None] / np.pi) * np.sinc(kk[2][None, None, :] / np.pi)
t_xp = torch.tensor(np.fft.ifftn(G * sw).ravel(order='F')[obj.idx].astype(np.complex64), device=pb.DEV)


def sincw(ctx):
    k = ctx.k * (obj.res / 2)
    return torch.prod(torch.where(k.abs() < 1e-6, torch.ones_like(k), torch.sin(k) / k), 1).to(torch.float32)


C = lambda ctx, sl: pb.polar(ctx.tau[:, None] * obj.t_om[sl][None, :] + ctx.w4 @ m4[sl].T)
one = lambda ctx, sl: torch.ones((len(ctx.n), len(sl)), dtype=torch.complex64, device=pb.DEV)
ef = {}
for nm, cf in [('C', C), ('C=1', one)]:
    ef[f'omission|{nm}'] = (lambda cf: lambda ctx, sl: cf(ctx, sl) * obj.t_x[sl][None, :] * (sincw(ctx)[:, None] - 1))(cf)
    ef[f'absorbed|{nm}'] = (lambda cf: lambda ctx, sl: cf(ctx, sl) * (t_xp[sl][None, :] - obj.t_x[sl][None, :] * sincw(ctx)[:, None]))(cf)
E, ns = pb.eval_terms(obj, des, ros, ef, nsel=nsel)
res = {k: pb.R_split(obj, e, ns, des) for k, e in E.items()}
for k, v in res.items():
    print(k, {a: round(b, 4) for a, b in v.items()})
json.dump(res, open(os.path.join(pb.OUT, 't9check.json'), 'w'), indent=1)
