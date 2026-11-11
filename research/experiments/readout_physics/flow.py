"""Blood and tissue motion during the readout: omission errors (T5 displacement, T7 M1-perpendicular remainder,
T6 acceleration, T12 motion through the B0 gradient) at the source level, regions and cardiac phases.

Blood: the simulator's Eulerian blood frames (1 mm sources, own velocity), frame 5 f for truth phase f (20 phases of the
cycle); amplitude PD x steady state at its T1 (inflow not modelled); carrier e^{-i w_s tau - tau / T2*_s} e^{-i w_ro.m}
(source's own field and T2*). Coil maps: the whitened true maps at the source's 2 mm voxel.
Lagrangian acceleration a_s = [v(x_s + v_s D, t + D) - v(x_s - v_s D, t - D)] / (2 D) from blood frames 5f +- 1
(D = period / 100 = 8.5 ms), velocities trilinear on their 1 mm lattice.
Regions per source (1 mm truth labels of phase f): chambers (blood_LA/LV/RA/RV), vessels (artery, vein, pulmonary),
coronary (sources within r + 1 mm of a coronary centreline point of phase f: RCA, LAD, LCx, branches, cardiac veins,
coronary sinus).
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import pb
sys.path.insert(0, os.path.join(pb.REPO, 'python/lib'))
from hastycompute import value_format as vf

TRS = {'current': 0.01129, 'v2': 0.0141}
CORONARY = ['Right coronary artery', 'Right inferolateral branch of right coronary artery', 'Anterior interventricular artery',
            'Circumflex artery of heart', 'Septal branches of anterior interventricular artery', 'Marginal artery',
            'Middle cardiac vein', 'Great cardiac vein', 'Coronary sinus']
_SIM = None


def sim():
    global _SIM
    if _SIM is None:
        _SIM = vf.open_lazy(os.path.join(pb.ROOT, 'sim_flow_10min/sim_input.hv'))
    return _SIM


class PointSet:
    """point sources with positions relative to the prescription centre, whitened coil maps of their 2 mm voxel"""

    def __init__(self, obj, pos_rel, amp, om, r2, vel, acc=None, region=None, extra=None):
        dims = obj.dims
        q = np.rint(pos_rel / obj.res + np.array([d // 2 for d in dims])).astype(np.int64)
        ok = np.all((q >= 0) & (q < np.array(dims)), axis=1)
        qq = q[:, 0] + dims[0] * (q[:, 1] + dims[1] * q[:, 2])
        li = np.searchsorted(obj.idx, np.where(ok, qq, 0))
        ok &= (li < obj.N) & (obj.idx[np.minimum(li, obj.N - 1)] == qq)
        self.keep = ok
        self.vox = li[ok]
        self.N = int(ok.sum()); self.nc = obj.nc; self.sigma = obj.sigma
        self.pos = pos_rel[ok]
        self.t_pos = torch.tensor(pos_rel[ok], dtype=torch.float64, device=pb.DEV)
        self.t_ST = obj.t_ST[torch.tensor(li[ok], device=pb.DEV)]
        self.t_a = torch.tensor(amp[ok], dtype=torch.complex64, device=pb.DEV)
        self.t_om = torch.tensor(om[ok], dtype=torch.float64, device=pb.DEV)
        self.t_r2 = torch.tensor(r2[ok], dtype=torch.float32, device=pb.DEV)
        self.v = vel[ok]
        self.t_v = torch.tensor(vel[ok], dtype=torch.float64, device=pb.DEV)
        self.t_acc = None if acc is None else torch.tensor(acc[ok], dtype=torch.float64, device=pb.DEV)
        self.region = None if region is None else {k: torch.tensor(m[ok], device=pb.DEV) for k, m in region.items()}
        self.region_np = None if region is None else {k: m[ok] for k, m in region.items()}
        self.extra = {} if extra is None else {k: torch.tensor(val[ok], dtype=torch.float64, device=pb.DEV) for k, val in extra.items()}
        self.mono = pb.monos(self.t_pos)


def lattice_grid(pos_rel, vals, h=0.001):
    """values of sources on a regular lattice (Eulerian blood frame) -> dense grid + origin (for trilinear sampling)"""
    off = np.median(pos_rel / h - np.floor(pos_rel / h), axis=0)
    ii = np.rint(pos_rel / h - off).astype(np.int64)
    lo = ii.min(0) - 1
    ii -= lo
    shape = tuple(ii.max(0) + 2)
    G = np.zeros(shape + (vals.shape[1],), np.float32)
    G[ii[:, 0], ii[:, 1], ii[:, 2]] = vals
    return G, (lo + off) * h, h


def trilinear(G, org, h, p):
    u = (p - org) / h
    i0 = np.floor(u).astype(np.int64); fr = u - i0
    out = np.zeros((len(p), G.shape[3]), np.float64)
    sh = np.array(G.shape[:3])
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ii = i0 + np.array([dx, dy, dz])
                ok = np.all((ii >= 0) & (ii < sh), axis=1)
                w = (fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * (fr[:, 2] if dz else 1 - fr[:, 2])
                vv = np.zeros((len(p), G.shape[3]))
                vv[ok] = G[ii[ok, 0], ii[ok, 1], ii[ok, 2]]
                out += w[:, None] * vv
    return out


def coronary_points(f):
    g = json.load(open(os.path.join(pb.ROOT, 'graph.json')))
    vs = vf.open_lazy(os.path.join(pb.ROOT, 'vessels.hv'))
    eo = np.asarray(vs['edge_offsets'][:])
    geo = vf.open_lazy(os.path.join(pb.SCAN, 'truth_fine/frames_geom.hv'))
    P = np.asarray(geo['points'][f]); Rr = np.asarray(geo['radius'][f])
    sel = np.zeros(len(P), bool)
    for e, nm in enumerate(g['edge_names']):
        if nm in CORONARY:
            sel[eo[e]:eo[e + 1]] = True
    return P[sel], Rr[sel]


def blood_frame(obj, b, tr, regions=True, accel=True):
    """PointSet of blood frame b (Eulerian, 100 per cycle)"""
    v = sim()
    presc = v['presc']; c = np.array(presc['center']); slab = presc['slab']
    fr = v['blood'][b]
    pos = np.asarray(fr['pos'][:]).astype(np.float64); vel = np.asarray(fr['vel'][:]).astype(np.float64)
    pd = np.asarray(fr['pd'][:]).astype(np.float64); t1 = np.asarray(fr['t1'][:]).astype(np.float64)
    t2s = np.asarray(fr['t2s'][:]).astype(np.float64); fppm = np.asarray(fr['field_ppm'][:]).astype(np.float64)
    inslab = np.abs((pos - np.array(slab['center'])) @ np.array(slab['normal'])) <= 0.5 * slab['thk']
    e1 = np.exp(-tr / np.maximum(t1, 1e-4)); fl = np.deg2rad(12.0)
    amp = pd * np.sin(fl) * (1 - e1) / (1 - np.cos(fl) * e1) * inslab
    om = 2 * np.pi * pb.GBAR * obj.b0 * 1e-6 * fppm
    pos_rel = pos - c
    acc = None
    if accel:
        D = float(v['period']) / 100.0
        nb = len(v['blood'])
        acc_terms = []
        for sgn in (+1, -1):
            f2 = v['blood'][(b + sgn) % nb]
            p2 = np.asarray(f2['pos'][:]).astype(np.float64) - c; v2 = np.asarray(f2['vel'][:]).astype(np.float64)
            G, org, h = lattice_grid(p2, v2)
            acc_terms.append(trilinear(G, org, h, pos_rel + sgn * vel * D))
        acc = (acc_terms[0] - acc_terms[1]) / (2 * D)
    region = None
    extra = {}
    # field gradient at the source from the frame's own 1 mm source field (central differences where both lattice
    # neighbours are blood, one-sided otherwise, 0 if isolated), rad/s/m
    G, org, h = lattice_grid(pos_rel, np.stack([om, np.ones_like(om)], 1))
    W = G[..., 0].astype(np.float64); M = G[..., 1] > 0
    gr = np.zeros(W.shape + (3,))
    for ax in range(3):
        fwd = np.roll(W, -1, ax); bwd = np.roll(W, 1, ax); mf = np.roll(M, -1, ax); mb = np.roll(M, 1, ax)
        both = M & mf & mb; onlyf = M & mf & ~mb; onlyb = M & mb & ~mf
        gr[..., ax] = np.where(both, (fwd - bwd) / (2 * h), np.where(onlyf, (fwd - W) / h, np.where(onlyb, (W - bwd) / h, 0.0)))
    ii = np.rint((pos_rel - org) / h).astype(np.int64)
    extra['gradw'] = gr[ii[:, 0], ii[:, 1], ii[:, 2]]
    if regions:
        f = b // 5
        tv = vf.open_lazy(os.path.join(pb.SCAN, 'truth_fine/truth4d.hv'))
        L = np.asarray(tv['labels'][f])  # Z, Y, X at 1 mm, origin_m corner
        o1 = np.array(tv['origin_m'])
        ii = np.floor((pos - o1) / 0.001).astype(np.int64)
        sh = np.array([L.shape[2], L.shape[1], L.shape[0]])
        okl = np.all((ii >= 0) & (ii < sh), axis=1)
        lab = np.zeros(len(pos), np.uint8)
        lab[okl] = L[ii[okl, 2], ii[okl, 1], ii[okl, 0]]
        P, Rr = coronary_points(f)
        from scipy.spatial import cKDTree
        tree = cKDTree(P)
        d, j = tree.query(pos, distance_upper_bound=0.01)
        cor = np.isfinite(d) & (d <= Rr[np.minimum(j, len(Rr) - 1)] + 0.001)
        region = dict(chambers=np.isin(lab, [20, 21, 22, 23]) & ~cor, vessels=np.isin(lab, [24, 25, 26, 27]) & ~cor, coronary=cor)
        region['all'] = np.ones(len(pos), bool)
    ps = PointSet(obj, pos_rel, amp, om, 1.0 / np.maximum(t2s, 1e-5), vel, acc=acc, region=region, extra=extra)
    return ps


def carrier(ps, ctx, sl):
    ph = ctx.tau[:, None] * ps.t_om[sl][None, :] + ctx.w4 @ ps.mono[sl].T
    z = pb.polar(ph) * torch.exp(-ps.t_r2[sl][None, :] * ctx.tau.to(torch.float32)[:, None])
    return z * ps.t_a[sl][None, :]


def flow_efuns(ps, regions=('all',)):
    """omission error images; the readout M1 / M2 about the echo"""
    def mk(kind, reg):
        def f(ctx, sl):
            v = ps.t_v[sl]
            if kind == 'm1':
                ph = ctx.m1 @ v.T
            elif kind == 'disp':
                ph = ctx.t1[:, None] * (ctx.k @ v.T)
            elif kind == 'perp':
                ph = ctx.m1perp @ v.T
            elif kind == 'acc':
                ph = ctx.m2h @ ps.t_acc[sl].T
            elif kind == 'acc_par':
                ph = ctx.t2[:, None] * (ctx.k @ ps.t_acc[sl].T)
            elif kind == 'pathb0':
                g = (ps.extra['gradw'][sl] * v).sum(1)
                ph = 0.5 * (ctx.t ** 2)[:, None] * g[None, :]
            elif kind == 'm1+acc':
                ph = ctx.m1 @ v.T + ctx.m2h @ ps.t_acc[sl].T
            z = carrier(ps, ctx, sl) * (pb.polar(ph) - 1)
            if reg != 'all':
                z = z * ps.region[reg][sl][None, :].to(torch.complex64)
            return z
        return f
    out = {}
    for reg in regions:
        for kind in ['m1', 'disp', 'perp', 'acc', 'pathb0']:
            out[f'{kind}|{reg}'] = mk(kind, reg)
    return out
