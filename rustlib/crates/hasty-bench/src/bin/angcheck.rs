//! Angular coverage per kz partition of a stack-of-spirals schedule (largest angular gap per partition, encode 0).
use hasty_mrsim::sequence::{Acquisition, FlowSpiralParams};
fn main() {
    let mut p = FlowSpiralParams::aorta_3t_default();
    p.scan_time = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(12.0);
    let acq = Acquisition::new(p);
    let nz = acq.kz.len();
    let mut angs: Vec<Vec<f64>> = vec![Vec::new(); nz];
    for r in acq.readouts.iter().filter(|r| r.enc == 0) {
        angs[r.part].push(r.rot[1][0].atan2(r.rot[0][0]).rem_euclid(2.0 * std::f64::consts::PI));
    }
    let mut worst: f64 = 0.0;
    let mut gaps = Vec::new();
    for a in angs.iter_mut() {
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut g: f64 = 2.0 * std::f64::consts::PI - a.last().unwrap() + a[0];
        for w in a.windows(2) {
            g = g.max(w[1] - w[0]);
        }
        worst = worst.max(g);
        gaps.push(format!("{}:{:.0}°", a.len(), g.to_degrees()));
    }
    println!("per partition (readouts : largest angular gap): {}", gaps.join(" "));
    println!("worst gap {:.1}°, ideal for 16 interleaves 22.5°", worst.to_degrees());
}
