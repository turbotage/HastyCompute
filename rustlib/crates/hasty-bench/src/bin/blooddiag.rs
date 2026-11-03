//! Phase-term magnitudes of flowing-blood sources (velocity, acceleration) at the end of a readout.
//!   blooddiag <subject_dir>
use hasty_mrsim::sequence::{Acquisition, FlowSpiralParams, SAMPLE_FLOATS};
use hasty_phantom::flow::{AorticWaveform, CenterlineFlow};
use hasty_phantom::prescription::aorta_slab;
use hasty_phantom::scene::{Scene, SceneParams};
use hasty_phantom::subject::Subject;
use hasty_phantom::tissue::TissueModel;

fn pct(mut v: Vec<f64>, ps: &[f64]) -> Vec<String> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ps.iter().map(|p| format!("{:.3}", v[((v.len() - 1) as f64 * p) as usize])).collect()
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let subj = Subject::load(&dir).unwrap();
    let mut fp = FlowSpiralParams::aorta_3t_default();
    fp.venc = 2.0;
    let heart = subj.label_centroid(&[51]).unwrap();
    let flow = CenterlineFlow::new(subj.centerline.clone(), subj.sdf.clone(), heart, AorticWaveform::healthy(fp.mean_rr), 30, 3.3e-6, 5.0);
    let (mp, mr) = flow.main_path(0.5);
    let presc = aorta_slab(&mp, mr, 0.002, [0.36, 0.36], 0.008);
    fp.rot_presc = presc.rot;
    fp.fov = presc.fov;
    fp.res = presc.res;
    fp.scan_time = 1.0;
    let tr = Acquisition::new(fp.clone()).tr;
    let scene = Scene::new(&subj, TissueModel::default_3t(), SceneParams::default_3t(tr), presc, &flow);
    fp.fov_center = scene.fov_center_scanner();
    let acq = Acquisition::new(fp);
    let pts = scene.blood_points();
    let (src, _) = scene.blood_sources(&pts[..pts.len().min(200000)], 0.16);
    let sp = acq.sample_params(0..1);
    for &n in &[0usize, acq.n_adc() / 2, acq.n_adc() - 1] {
        let s = &sp[n * SAMPLE_FLOATS..(n + 1) * SAMPLE_FLOATS];
        let vel: Vec<f64> = src.iter().map(|q| (s[3] as f64 * q.vt[0] + s[4] as f64 * q.vt[1] + s[5] as f64 * q.vt[2]).abs()).collect();
        let acc: Vec<f64> = src.iter().map(|q| (s[6] as f64 * q.at[0] + s[7] as f64 * q.at[1] + s[8] as f64 * q.at[2]).abs()).collect();
        println!("ADC sample {n} (tau {:.2} ms): gamma*M1 = {:?} rad s/m, gamma*M2/2 = {:?}", s[13] * 1e3, &s[3..6], &s[6..9]);
        println!("   |phase from v| percentiles 50/90/99/max: {:?} rad", pct(vel, &[0.5, 0.9, 0.99, 1.0]));
        println!("   |phase from a| percentiles 50/90/99/max: {:?} rad", pct(acc, &[0.5, 0.9, 0.99, 1.0]));
    }
    let amag: Vec<f64> = src.iter().map(|q| (q.at[0].powi(2) + q.at[1].powi(2) + q.at[2].powi(2)).sqrt()).collect();
    println!("|a| percentiles 50/90/99/max: {:?} m/s^2", pct(amag, &[0.5, 0.9, 0.99, 1.0]));
}
