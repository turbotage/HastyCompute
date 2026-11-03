//! Statistics of the centerline flow model on a prepared subject: |v| percentiles over the lumen per cardiac phase and
//! the flow rate through the root and main-branch cross-sections.
//!   flowcheck <subject_dir>
use hasty_phantom::flow::{AorticWaveform, CenterlineFlow, FlowField};
use hasty_phantom::subject::Subject;

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let subj = Subject::load(&dir).unwrap();
    let heart = subj.label_centroid(&[51]).unwrap();
    let wave = AorticWaveform::healthy(0.9);
    let flow = CenterlineFlow::new(subj.centerline.clone(), subj.sdf.clone(), heart, wave, 30, 3.3e-6, 5.0);
    let sdf = &subj.sdf;
    let d = sdf.dims;
    let mut pts = Vec::new();
    for k in (0..d[2]).step_by(2) {
        for j in (0..d[1]).step_by(2) {
            for i in (0..d[0]).step_by(2) {
                if sdf.get(i, j, k) < 0.0 {
                    pts.push(sdf.to_phys([i as f64, j as f64, k as f64]));
                }
            }
        }
    }
    println!("{} lumen points (1 mm subsample); branches (fraction, radius mm): {:?}", pts.len(), flow.branch_info().iter().map(|(f, r)| (format!("{f:.3}"), format!("{:.1}", r * 1e3))).collect::<Vec<_>>());
    println!("probe at the worst point: {:?}", flow.probe(&[-0.049, 0.026, -0.037]));
    for f in [0usize, 2, 3, 4, 6, 10, 15] {
        let t = (f as f64 + 0.5) / 20.0 * 0.9;
        let mut sp: Vec<(f64, usize)> = pts.iter().enumerate().map(|(i, x)| {
            let (v, _) = flow.eval(x, t);
            ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(), i)
        }).collect();
        sp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let n = sp.len();
        let worst = sp[n - 1];
        println!("t = {:.3} s (Q = {:.0} mL/s): |v| median {:.2}, 99% {:.2}, 99.9% {:.2}, max {:.2} m/s at {:?} mm",
            t, wave.q(t) * 1e6, sp[n / 2].0, sp[99 * n / 100].0, sp[999 * n / 1000].0, worst.0, pts[worst.1].map(|v| (v * 1e3).round()));
    }
}
