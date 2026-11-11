"""The v2 flow TR (data/hasty/seq/flow_thorax_v2) as played through a GIRF, per readout of the simulated scan.

Physical TR(e, R) = [zeros][encode e (physical)][R . base readout incl. the rewinder leg][flow-comp tail e (physical)],
base readout = tr_grad[0][i_ro : i_tail + 1] (the frame pb.Design('v2') uses), R = the scan rotations of
pb.scan_readouts (paired with the budget). The played trajectory of readout i is integrated from its excitation with
the two previous TRs (their own encode / rotation, acquisition order) in the chain (girf.rs `played_moments` uses one).
ADC at 2 us from the echo: reconstruction window 0 .. 9.49 ms (recon_mask), navigator (rewinder leg) 9.49 .. 10.79 ms.
"""
import json, os
import numpy as np
import gmodel as gm

REPO = os.environ.get('HASTY_REPO', '/home/turbotage/Documents/GitHub/HastyCompute')
RUN = os.path.join(REPO, 'data/hasty/seq/flow_thorax_v2')
SCAN = os.path.join(REPO, 'data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan')
GAM = gm.GAM


class V2:
    def __init__(self, dwell=2e-6):
        s = json.load(open(RUN + '/summary.json'))
        b = s['best']
        self.dt = s['problem']['dt']
        self.te = b['te']
        self.i_ro, self.i_tail, self.i_adc = b['i_ro'], b['i_tail'], b['i_adc']
        self.n_recon = b['recon'][0][1]
        self.tr_grad = np.load(RUN + '/tr_grad.npy')  # (7, n+1, 3)
        self.n_enc = self.tr_grad.shape[0]
        self.n = self.tr_grad.shape[1] - 1  # TR length in raster steps
        self.tr = self.n * self.dt
        self.t_rf = self.i_ro * self.dt - self.te
        self.base = self.tr_grad[0, self.i_ro:self.i_tail + 1].copy()
        self.dwell = dwell
        na_all = int(round(self.i_adc * self.dt / dwell))
        self.t = np.arange(na_all) * dwell  # from the echo
        self.na = int(round(self.n_recon * self.dt / dwell))  # = pb.Design('v2').na
        self.recon = np.arange(na_all) < self.na
        self.k_base = gm.k_at(self.base, self.dt, 0.0, self.t, 0.0)  # rad/m, base frame
        self.kabs = np.linalg.norm(self.k_base, axis=1)

    def tr_phys(self, e, R):
        g = self.tr_grad[e, :self.n].copy()
        g[self.i_ro:self.i_tail + 1] = self.base @ R.T
        return g

    def sequence(self, items):
        """concatenated physical waveform of the TRs items = [(e, R), ...] (last = the readout's own TR)"""
        return np.concatenate([self.tr_phys(e, R) for e, R in items] + [np.zeros((1, 3))], 0)

    def played(self, girf, items, ntail=None):
        """k (na_all x 3, rad/m, physical, from the last TR's excitation) and the B0-term phase (rad) at the ADC
        samples of the last TR of `items`"""
        g = self.sequence(items)
        y, b = girf.apply(g, self.dt, ntail=0 if ntail is None else ntail, nfft=self._nfft(len(g), girf.tail))
        t0 = -gm.PRE * self.dt
        s_last = (len(items) - 1) * self.n * self.dt
        t_exc = s_last + self.t_rf
        times = s_last + self.i_ro * self.dt + self.t
        k = gm.k_at(y, self.dt, t0, times, t_exc)
        ph = gm.phase_at(b, self.dt, t0, times, t_exc)
        return k, ph

    def _nfft(self, n, tail):
        # zero padding >= the chain's response tail (no wrap-around into the window), at least 40 ms
        return 1 << int(np.ceil(np.log2(n + gm.PRE + max(4000, int(tail / self.dt)))))


def scan_rotations(nr=48, offset=7, nprev=10):
    """the budget's readouts (pb.scan_readouts(nr, offset)) plus their nprev predecessors: idx, encodes and
    rotations, shape (nr, nprev + 1) (column -1 = the readout itself); R by Procrustes onto the current design's base
    readout as in pb (exact to 1e-7). Cached under data/hasty/seq/studies/girf_sensitivity_2026-11-11/."""
    out = os.path.join(REPO, 'data/hasty/seq/studies/girf_sensitivity_2026-11-11', f'scanrot_{nr}_{offset}_{nprev}.npz')
    if os.path.exists(out):
        z = np.load(out)
        return dict(idx=z['idx'], enc=z['enc'], R=z['R'], frame=z['frame'])
    import sys
    sys.path.insert(0, os.path.join(REPO, 'research/experiments/readout_physics'))
    import pb
    ro = np.load(os.path.join(SCAN, 'readouts.npy'), mmap_mode='r')
    s = np.load(os.path.join(SCAN, 'sched_nominal.npy'), mmap_mode='r')
    n = ro.shape[0]
    idx = np.linspace(offset, n - 1, nr).round().astype(int)
    kb = pb.Design('current').k
    allidx = idx[:, None] + np.arange(-nprev, 1)[None, :]
    enc = np.zeros(allidx.shape, int)
    R = np.zeros(allidx.shape + (3, 3))
    frame = np.zeros(allidx.shape, int)
    for i in range(allidx.shape[0]):
        for j in range(allidx.shape[1]):
            r = allidx[i, j]
            tab = np.asarray(ro[r])
            enc[i, j] = int(tab[1]); frame[i, j] = int(tab[3])
            kr = np.asarray(s[r, :, :3]).astype(np.float64)
            U, _, Vt = np.linalg.svd(kr.T @ kb)
            R[i, j] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, idx=allidx, enc=enc, R=R, frame=frame)
    return dict(idx=allidx, enc=enc, R=R, frame=frame)


def trajectories(v, girf, rots, ref=None):
    """played k / phase for every readout (nr x na_all x 3, nr x na_all); ref: a second chain whose prediction is
    subtracted (the reconstruction's model); ref=None -> the nominal trajectory (R k_base, zero phase)"""
    nr = rots['R'].shape[0]
    K = np.zeros((nr, len(v.t), 3)); P = np.zeros((nr, len(v.t)))
    for i in range(nr):
        items = [(rots['enc'][i, j], rots['R'][i, j]) for j in range(rots['R'].shape[1])]
        k, p = v.played(girf, items)
        if ref is None:
            k = k - v.k_base @ rots['R'][i, -1].T
        else:
            k2, p2 = v.played(ref, items)
            k, p = k - k2, p - p2
        K[i], P[i] = k, p
    return K, P
