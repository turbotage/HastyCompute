//! Concomitant structure of 3D-rotated readouts: anisotropy of the base trajectory's M(s) = ∫ G Gᵀ dt (the part that
//! rotates with each readout and cannot go into time-segment diagonals). Compares a planar spiral with yarnball-like
//! 3D base curves (direction sweeping the sphere while the radius grows), designed time-optimally at 80 mT/m, 200 T/m/s.
//!   yarn_study
use hasty_mrsim::trajectory::{spiral_curve, time_optimal};
use hasty_mrsim::{GradLimits, Vec3, GAMMA};

fn yarnball(kmax: f64, turns: f64, wobble: f64, n: usize) -> Vec<Vec3> {
    // radius r(s) grows center-out; direction: polar angle sweeps 0 → π once ("ball"), azimuth winds `turns` times, with
    // an extra polar wobble so the gradient keeps changing direction
    (0..=n)
        .map(|i| {
            let s = i as f64 / n as f64;
            let r = kmax * s;
            let th = std::f64::consts::PI * s + wobble * (2.0 * std::f64::consts::PI * 3.0 * turns * s).sin();
            let ph = 2.0 * std::f64::consts::PI * turns * s;
            [r * th.sin() * ph.cos(), r * th.sin() * ph.sin(), r * th.cos()]
        })
        .collect()
}

fn report(name: &str, curve: &[Vec3], lim: &GradLimits, b0: f64, fov_half: f64) {
    let w = time_optimal(curve, lim, 4e-6, None);
    let m = w.moments();
    let n = m.t.len();
    let t_ro = w.duration();
    let wc = GAMMA / (2.0 * b0);
    let mut out = Vec::new();
    for frac in [0.1, 0.25, 0.5, 0.75, 1.0] {
        let i = ((n - 1) as f64 * frac) as usize;
        let p = m.p[i]; // xx yy zz xy xz yz
        let tr = p[0] + p[1] + p[2];
        let a = [p[0] - tr / 3.0, p[1] - tr / 3.0, p[2] - tr / 3.0];
        let an = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + 2.0 * (p[3] * p[3] + p[4] * p[4] + p[5] * p[5])).sqrt();
        let iso = tr / 3.0 * 3f64.sqrt();
        // phase scale: concomitant quadratic ~ (γ/2B0)·P·r², isotropic and anisotropic parts at the FOV edge
        out.push(format!("{:.0}%: aniso/iso {:.2}, iso phase {:.2} rad, aniso phase {:.2} rad", frac * 100.0, an / iso, wc * tr / 3.0 * fov_half * fov_half, wc * an * fov_half * fov_half));
    }
    println!("{name}: readout {:.2} ms, |G|max {:.1} mT/m\n   {}", t_ro * 1e3, w.max_amplitude() * 1e3, out.join("\n   "));
}

fn main() {
    let lim = GradLimits::clinical_3t().derated(0.95);
    let kmax = 0.5 / 0.002;
    let fov_half = 0.15;
    report("planar spiral (16 interleaves), rotated in 3D", &spiral_curve(kmax, 16, 0.36, 0.25), &lim, 3.0, fov_half);
    for (turns, wob) in [(8.0, 0.0), (8.0, 0.3), (16.0, 0.3)] {
        report(&format!("yarnball-like base curve ({turns} azimuth turns, polar wobble {wob})"), &yarnball(kmax, turns, wob, 200000), &lim, 3.0, fov_half);
    }
}
