"""Omission errors of the motion terms during the readout (blood sources) for both designs, cycle-averaged and at
systole / diastole. Resumable: one JSON per (design, phase set) in <OUT>/flow_omission_*.json.

    run_flow.py [--nr 96] [--nr-phase 48]
"""
import json, os, sys, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import pb, flow

ap = argparse.ArgumentParser()
ap.add_argument('--nr', type=int, default=96)
ap.add_argument('--nr-phase', type=int, default=48)
ap.add_argument('--designs', default='current,v2')
a = ap.parse_args()
OUT = pb.OUT
REGIONS = ('all', 'chambers', 'vessels', 'coronary')
PHASES = {'systole': 3, 'diastole': 12}


def summarise(obj, des, E, nsel, nvox):
    out = {}
    for name, e in E.items():
        kind, reg = name.split('|')
        r = pb.R_split(obj, e, nsel, des)
        r['R_loc'] = r['R'] * np.sqrt(obj.N / max(nvox[reg], 1))
        out.setdefault(kind, {})[reg] = r
    return out


if __name__ == '__main__':
    torch.set_num_threads(6)
    obj0 = pb.Obj()
    for dname in a.designs.split(','):
        des = pb.Design(dname)
        obj = pb.use_truth(obj0, dname)
        nsel = des.sample_select(kc=150.0, stride=8)
        tr = flow.TRS[dname]
        # cycle average: readouts grouped by their own cardiac frame
        fn = os.path.join(OUT, f'flow_omission_{dname}_cycle.json')
        if not os.path.exists(fn):
            ros = pb.scan_readouts(a.nr, offset=11)
            E_all = None; nvox_acc = {r: [] for r in REGIONS}
            for f in sorted(set(ros['frame'])):
                sel = np.flatnonzero(ros['frame'] == f)
                sub = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in ros.items()}
                ps = flow.blood_frame(obj, 5 * f, tr)
                for r in REGIONS:
                    nvox_acc[r].append(len(np.unique(ps.vox[ps.region_np[r]])))
                E, nsel = pb.eval_terms(ps, des, sub, flow.flow_efuns(ps, REGIONS), nsel=nsel)
                if E_all is None:
                    E_all = {k: np.zeros((a.nr, len(nsel[0]))) for k in E}
                for k in E:
                    E_all[k][sel] = E[k]
                pb.log(f'{dname} cycle: frame {f} ({len(sel)} readouts, {ps.N} blood sources)')
            nvox = {r: float(np.mean(v)) for r, v in nvox_acc.items()}
            res = dict(design=dname, phase='cycle', nr=a.nr, nvox=nvox, terms=summarise(obj, des, E_all, nsel, nvox))
            json.dump(res, open(fn, 'w'), indent=1)
            np.savez(fn.replace('.json', '.npz'), n=nsel[0], w=nsel[1], **{k.replace('|', '__'): v.mean(0) for k, v in E_all.items()})
        for ph, f in PHASES.items():
            fn = os.path.join(OUT, f'flow_omission_{dname}_{ph}.json')
            if os.path.exists(fn):
                continue
            ros = pb.scan_readouts(a.nr_phase, offset=5)
            ps = flow.blood_frame(obj, 5 * f, tr)
            nvox = {r: float(len(np.unique(ps.vox[ps.region_np[r]]))) for r in REGIONS}
            E, nsel = pb.eval_terms(ps, des, ros, flow.flow_efuns(ps, REGIONS), nsel=nsel, frame_override=f)
            vstats = {r: np.percentile(np.linalg.norm(ps.v[ps.region_np[r]], axis=1), [50, 95, 99]).tolist() for r in REGIONS if ps.region_np[r].any()}
            astats = {r: np.percentile(np.linalg.norm(ps.t_acc.cpu().numpy()[ps.region_np[r]], axis=1), [50, 95, 99]).tolist() for r in REGIONS if ps.region_np[r].any()}
            res = dict(design=dname, phase=ph, frame=f, nr=a.nr_phase, nvox=nvox, speed_pct_50_95_99=vstats, acc_pct_50_95_99=astats, terms=summarise(obj, des, E, nsel, nvox))
            json.dump(res, open(fn, 'w'), indent=1)
            np.savez(fn.replace('.json', '.npz'), n=nsel[0], w=nsel[1], **{k.replace('|', '__'): v.mean(0) for k, v in E.items()})
            pb.log(f'{dname} {ph}: done')
    pb.log('all done')
