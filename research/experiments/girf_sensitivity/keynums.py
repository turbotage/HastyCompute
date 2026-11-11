"""Key numbers of the MC runs for the note: R stats per run (nonparametric estimate) and by resonance exposure"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(os.environ.get('HASTY_REPO', '/home/turbotage/Documents/GitHub/HastyCompute'), 'data/hasty/seq/studies/girf_sensitivity_2026-11-11')


def st(x):
    x = np.asarray(x)
    return f'{np.median(x):.2f} [{np.percentile(x, 10):.2f}, {np.percentile(x, 90):.2f}] P≤0.2 {np.mean(x <= 0.2):.2f} P≤0.5 {np.mean(x <= 0.5):.2f}'


def main():
    import truths, figs
    f, sp = figs.readout_spectrum()
    out = {}
    for fn in sorted(glob.glob(os.path.join(OUT, 'mc_R_*.jsonl'))):
        tag = os.path.basename(fn)[3:-6]
        R = [json.loads(l) for l in open(fn)]
        print(f'## {tag} ({len(R)} draws, {R[0]["minutes"]:.0f} min)')
        for est in ('np', 'par', 'np_nocross_ref'):
            for lv in ('raw', 'enc', 'refined', 'loc_refined'):
                print(f'  {est:15s} {lv:12s}', st([r[f"R_{est}"][lv] for r in R]))
        if R[0]['opt']['truth'] == 'rand':
            risk = []
            for r in R:
                t = truths.draw(np.random.default_rng(5000 + r['draw']))
                risk.append(max(g / z * np.interp(f0, f, sp) for a in range(3) for f0, z, g in t.axis[a].parts[0].res))
            risk = np.array(risk)
            enc = np.array([r['R_np']['enc'] for r in R])
            ref = np.array([r['R_np']['refined'] for r in R])
            lo = risk < 1.0
            print(f'  exposure < 1 ({lo.sum()} draws): enc {st(enc[lo])}; refined {st(ref[lo])}')
            print(f'  exposure >= 1 ({(~lo).sum()} draws): enc {st(enc[~lo])}; refined {st(ref[~lo])}')
            rk = np.argsort(np.argsort(risk)); rr = np.argsort(np.argsort(enc))
            print(f'  Spearman(exposure, R_enc) {np.corrcoef(rk, rr)[0, 1]:.2f}')
        out[tag] = len(R)
    for fn in sorted(glob.glob(os.path.join(OUT, 'mcrefit_*.jsonl'))):
        R = [json.loads(l) for l in open(fn)]
        print(f'## {os.path.basename(fn)} ({len(R)} draws)')
        for k in ('refined', 'refined_xb', 'refined_xbr', 'loc_refined_xbr'):
            print(f'  {k:16s}', st([r[k] for r in R]))


if __name__ == '__main__':
    main()
