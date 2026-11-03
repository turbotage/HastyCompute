//! Diagnostic: representation error of the chirp stencil spectrum vs stencil half-width S, POD tolerance, w-box.
use hasty_nufft::{ChirpFamily, ChirpTables};
use std::time::Instant;

fn main() {
    let modes = [16usize, 16, 16];
    for (sigma, nf) in [(2.0f64, 32usize), (1.25, 20)] {
        for wbox in [0.25f64, 0.5, 1.0] {
            for s_half in [3usize, 4] {
                for pod in [3e-3, 1e-3, 3e-4] {
                    let fam = ChirpFamily {
                        w_box: [wbox; 4],
                        s_half,
                        pod_tol: pod,
                        l_max: 120,
                        nfov: 13,
                        ridge: 1e-7,
                        n_train: 400,
            train: None,
                    };
                    let t = Instant::now();
                    let tab = ChirpTables::build(modes, [nf, nf, nf], &fam);
                    println!("sigma {sigma} wbox {wbox} S={s_half} pod_tol={pod:e}: L={:>3}  held-out row err (max {:.2e}, mean {:.2e})  build {:.1}s", tab.l, tab.heldout_err.0, tab.heldout_err.1, t.elapsed().as_secs_f32());
                }
            }
        }
    }
}
