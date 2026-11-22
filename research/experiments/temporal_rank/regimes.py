"""Regime perturbations of the cardiac truth (research/temporal_rank.md, regime sweep).

The vessel blood is replaced by a controllable overlay on the true moving geometry: every fine vessel-blood voxel of
frame f takes the material velocity of its nearest centreline point (frames_geom wvel, the vessel's own motion) plus
blood fraction x Poiseuille profile 2 (1 - d^2/R^2) x the mean velocity U_p(t) = Q_p(t) / (pi R_p^2) along the
centreline tangent, with Q_p(t) the 1-D network flow at that point (network_flow.npz, 25 harmonics: pulse-wave delays
included). The heart chambers (LBM) and the tissue keep the truth. Knobs act on Q per named vessel (edge name):
time shift, added random harmonics, coronary-like diastolic-dominant replacement, and a velocity scale (VENC ratio).
"""
import numpy as np

from rank_lib import ORG, assign_fine


class Overlay:
    def __init__(self, F, Ve, frames=None):
        self.F, self.Ve = F, Ve
        self.fr = list(range(F.nf)) if frames is None else list(frames)
        self.per = {}
        used = set()
        for f in self.fr:
            sel, pi, d, r = assign_fine(F, Ve, f)
            self.per[f] = (sel.astype(np.int64), pi.astype(np.int64), d.astype(np.float32), r.astype(np.float32))
            used.update(np.unique(pi).tolist())
        self.pts = np.array(sorted(used))  # centreline points used
        self.loc = {p: i for i, p in enumerate(self.pts)}
        self.c = Ve.q[self.pts]  # [P, 26] harmonics of Q (m^3/s)
        self.W = Ve.omega
        self.RR = Ve.period
        self.edge = Ve.eid[self.pts]
        self.name = np.array([Ve.names[e] for e in self.edge])
        self.geo = {}
        for f in self.fr:
            p, r, w = Ve.frame(f)
            t = Ve.tangent(p)
            self.geo[f] = (r[self.pts].astype(np.float32), w[self.pts].astype(np.float32), t[self.pts].astype(np.float32))

    def Q(self, t, shift=None):
        """Q_p(t) [P] at time t (s), optional per-point time shift (s)"""
        tt = t - (np.zeros(len(self.pts)) if shift is None else shift)
        e = np.exp(1j * self.W[None, 1:] * tt[:, None])
        return np.real(self.c[:, 0] + 2 * (self.c[:, 1:] * e).sum(1))

    def waveforms(self, knobs, rng_seed=0):
        """per-point Q [P, T] at the frame times under the knobs:
        tau_ms: per-vessel delays U(0, tau) ms; shape: added harmonics h = 1..6 of relative amplitude shape / h (of the
        vessel's peak-to-peak), random phases per vessel; coronary: dict(ramp_ms, peak_ratio) diastolic-dominant
        coronary arteries (systolic-dominant cardiac veins), mean flow kept"""
        rng = np.random.default_rng(rng_seed)
        names = sorted(set(self.name.tolist()))
        nid = np.array([names.index(n) for n in self.name])
        tf = np.array([f * self.RR / self.F.nf for f in self.fr])
        tau = knobs.get("tau_ms", 0.0) * 1e-3 * rng.uniform(0, 1, len(names))
        shift = tau[nid]
        Q = np.stack([self.Q(t, shift) for t in tf], 1)  # [P, T]
        sh = knobs.get("shape", 0.0)
        if sh:
            H = 6
            amp = rng.standard_normal((len(names), H)) / np.arange(1, H + 1)
            ph = rng.uniform(0, 2 * np.pi, (len(names), H))
            pp = Q.max(1) - Q.min(1)
            ph_t = 2 * np.pi * tf / self.RR
            add = np.zeros((len(names), len(tf)))
            for h in range(H):
                add += amp[:, h:h + 1] * np.cos((h + 1) * ph_t[None] + ph[:, h:h + 1])
            add /= np.sqrt((add ** 2).mean(1, keepdims=True)) + 1e-12  # unit rms per vessel
            Q = Q + sh * pp[:, None] * add[nid]
        cor = knobs.get("coronary")
        if cor:
            phase = tf / self.RR
            Sa = coronary_shape(phase, cor.get("ramp_ms", 30) * 1e-3 / self.RR, cor.get("peak_ratio", 2.5))
            Sv = coronary_shape((phase + 0.5) % 1.0, cor.get("ramp_ms", 30) * 1e-3 / self.RR, cor.get("peak_ratio", 2.5))
            art = np.array([any(s in n for s in ("coronary artery", "interventricular artery", "Circumflex artery", "Left coronary")) for n in self.name])
            vein = np.array([any(s in n for s in ("cardiac vein", "Coronary sinus", "Inferior vein of left ventricle")) for n in self.name])
            mean = Q.mean(1, keepdims=True)
            sc = cor.get("v_scale", 1.0)
            Q = np.where(art[:, None], sc * mean * Sa[None], Q)
            Q = np.where(vein[:, None], sc * mean * Sv[None], Q)
        return Q

    def velocity(self, Q, extra_scale=None):
        """vel(f) for build_X: the truth with the vessel blood replaced by the overlay with waveforms Q [P, T]"""
        F = self.F
        fi = {f: j for j, f in enumerate(self.fr)}

        def vel(f):
            v = F.v[f].astype(np.float32).reshape(3, -1)
            sel, pi, d, r = self.per[f]
            li = np.searchsorted(self.pts, pi)
            R, w, t = self.geo[f]
            b = F.blood[f].reshape(-1)[sel].astype(np.float32) / 255
            rho = np.clip(2 * (1 - (d / np.maximum(r, 1e-4)) ** 2), 0, 2)
            U = Q[li, fi[f]] / (np.pi * np.maximum(R[li], 3e-4) ** 2)
            if extra_scale is not None:
                U = U * extra_scale[li]
            vv = w[li] + (b * rho * U)[:, None] * t[li]
            v[:, sel] = vv.T
            return v.reshape((3,) + F.fs)

        return vel


def coronary_shape(phase, ramp, peak_ratio, t_es=0.38):
    """diastolic-dominant coronary arterial flow over the cardiac phase (mean 1): low systolic level 0.3, a logistic rise
    of width `ramp` (phase units) at end-systole t_es to `peak_ratio` x the mean, exponential decay through diastole;
    a short early-systolic dip (intramyocardial compression)"""
    p = np.asarray(phase)
    rise = 1 / (1 + np.exp(-(p - t_es) / max(ramp / 4, 1e-4)))
    fall = 1 / (1 + np.exp(-(p - 0.02) / max(ramp / 4, 1e-4)))  # end-diastole -> systole drop at the cycle start
    dia = np.exp(-(np.maximum(p - t_es, 0)) / 0.45)
    s = 0.3 + rise * (peak_ratio * dia) - 0.25 * np.exp(-((p - 0.08) / 0.04) ** 2) * fall
    # periodic join: the diastolic tail decays into the next systole through `fall`
    s = s * fall + (1 - fall) * (0.3 + peak_ratio * np.exp(-(1 - t_es) / 0.45))
    return s / s.mean()


class SmoothMagnitude:
    """continuous-motion magnitude: the truth's fine labels are pulled back to the nearest 1 mm label (staircase partial
    volume that jumps when a wall crosses a voxel centre); this replaces it by m(x, t) = m_0(X + U(X, 0)), X the atlas
    point with x = X + U(X, t) (motion/cardiac_motion.hv, 2 mm, periodic in time), m_0 the frame-0 fine magnitude,
    trilinear: the partial volume of a 2 mm voxel then changes continuously as the wall moves"""

    def __init__(self, F, iters=4):
        from scipy import ndimage
        from common import PH
        from hastycompute import value_format as vf
        C = vf.open_lazy(PH / "motion/cardiac_motion.hv")
        U = np.asarray(C["U_mm"].read()).astype(np.float32) * 1e-3  # [48, 3 (x y z), z, y, x] m
        org, vox = np.array(C["origin_m"]), float(C["voxel_m"])
        period = float(C["period"])
        nt = U.shape[0]
        Uh = np.fft.rfft(U, axis=0)
        del U
        z, y, x = np.meshgrid(*[np.arange(s) for s in F.fs], indexing="ij")
        P = ORG + (np.stack([x.ravel(), y.ravel(), z.ravel()], 1) + F.f0[::-1]) * 1e-3  # world x y z of fine voxels
        del x, y, z

        def Ut(t):
            ph = np.exp(2j * np.pi * np.arange(Uh.shape[0]) * t / period)
            w = np.full(Uh.shape[0], 2.0)
            w[0] = 1.0
            if nt % 2 == 0:
                w[-1] = 1.0
            return (np.tensordot(w * ph, Uh, axes=(0, 0)).real / nt).astype(np.float32)

        def samp(Uf, Q):
            c = ((Q - org) / vox)[:, ::-1].T
            return np.stack([ndimage.map_coordinates(Uf[a], c, order=1, mode="constant", cval=0.0) for a in range(3)], 1)

        U0 = Ut(0.0)
        m0 = F.m[0].astype(np.float32)
        self.m = np.empty((F.nf,) + F.fs, np.float16)
        c0 = (F.f0[::-1] * 1e-3 + ORG)
        for f in range(F.nf):
            Uf = Ut(f * period / F.nf)
            X = P.copy()
            for _ in range(iters):
                X = P - samp(Uf, X)
            x0 = X + samp(U0, X)
            ci = ((x0 - c0) / 1e-3)[:, ::-1].T  # fine index z y x in the analysis box
            self.m[f] = ndimage.map_coordinates(m0, ci, order=1, mode="nearest").reshape(F.fs).astype(np.float16)

    def __call__(self, f):
        return self.m[f].astype(np.float32)
