//! Velocity-encoding comparison at equal protection: for a peak speed v_max that must not alias in *any* direction,
//! each scheme's VENCs are set accordingly (referenced-type layouts: per-axis range ≥ v_max; Hadamard pair decoding
//! measures v_i + v_j, so its range must be ≥ √2 v_max), and reported are: encodes, velocity noise (per component, and
//! normalised to equal scan time), the SNR at which coprime unwrapping fails in 1e-3 of voxels (exact phase-noise
//! model), and the encode block length / TE contribution on the Prisma (with and without PNS derating).
//!   vencstudy [out.json]
use hasty_seq::encode::{design, DualLayout, EncodeSpec, Scheme};
use hasty_seq::system::System;
use hasty_seq::venc::plan;
use serde_json::json;

fn snr_for(pl: &hasty_seq::venc::Plan, ne: usize, target: f64) -> f64 {
    if pl.crt.is_none() || pl.combos[0].hi.is_none() {
        return f64::NAN;
    }
    let (mut lo, mut hi) = (1.0f64, 200.0f64);
    for _ in 0..40 {
        let mid = (lo * hi).sqrt();
        if pl.unwrap_failure_exact(ne, 1.0 / mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

fn main() {
    let out = std::env::args().nth(1);
    let sys = System::prisma_xr();
    let mut rows = vec![];
    for (site, vmax) in [("thorax", 3.0), ("intracranial", 1.0)] {
        println!("\n{site}: no aliasing up to {vmax} m/s in any direction");
        println!("  scheme                     enc  VENC lo/hi m/s   σv@SNR20 mm/s  time-norm  SNR(fail 1e-3)  encode ms (1.0 / 0.5 slew)");
        let s2 = 2f64.sqrt();
        let mut schemes: Vec<(String, Scheme, f64)> = vec![
            ("referenced".into(), Scheme::Referenced, vmax),
            ("hadamard".into(), Scheme::Hadamard, vmax * s2),
            ("classic dual 7p (1:2)".into(), Scheme::DualVenc { venc_hi: vmax }, vmax / 2.0),
        ];
        for p in [2u32, 3, 4] {
            let pf = p as f64 + 1.0;
            schemes.push((format!("coprime p={p} seven"), Scheme::Coprime { p, layout: DualLayout::Seven }, vmax / pf));
            schemes.push((format!("coprime p={p} eight"), Scheme::Coprime { p, layout: DualLayout::Eight }, vmax / pf));
            schemes.push((format!("coprime p={p} eight-hadamard"), Scheme::Coprime { p, layout: DualLayout::EightHadamard }, vmax * s2 / pf));
        }
        for (name, scheme, venc) in schemes {
            let pl = plan(scheme, venc);
            let ne = hasty_seq::encode::scheme_m1(scheme, venc, true).len();
            let sd = pl.velocity_std(ne, 1.0 / 20.0 / 2f64.sqrt());
            let sdm = (sd.iter().map(|x| x * x).sum::<f64>() / 3.0).sqrt();
            let tn = sdm * (ne as f64 / 4.0).sqrt();
            let snr = snr_for(&pl, ne, 1e-3);
            let enc = |f: f64| {
                let es = EncodeSpec { m0: [0.0; 3], m1_offset: [0.0; 3], scheme, venc, balanced: true, t_start: 0.0, t_ref: 0.0 };
                design(&es, &System { smax: sys.smax * f, ..sys.clone() }).duration * 1e3
            };
            let (e1, e5) = (enc(1.0), enc(0.5));
            let (vl, vh) = scheme.vencs(venc);
            println!("  {name:26} {ne:3}  {vl:5.3} / {vh:5.3}   {:8.1}     {:8.1}   {:>10}      {e1:.2} / {e5:.2}", sdm * 1e3, tn * 1e3, if snr.is_nan() { "-".to_string() } else { format!("{snr:.1}") });
            rows.push(json!({"site": site, "vmax": vmax, "scheme": name, "n_enc": ne, "venc_lo": vl, "venc_hi": vh, "sigma_v_snr20": sdm,
                             "sigma_v_time_norm": tn, "snr_fail_1e-3": if snr.is_nan() { serde_json::Value::Null } else { json!(snr) }, "encode_ms": e1, "encode_ms_half_slew": e5}));
        }
    }
    if let Some(o) = out {
        std::fs::write(&o, serde_json::to_string_pretty(&rows).unwrap()).unwrap();
        println!("\nwrote {o}");
    }
}
