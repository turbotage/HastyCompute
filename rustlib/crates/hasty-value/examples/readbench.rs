//! cargo run --release -p hasty-value --example readbench -- <case.hv>
use std::time::Instant;

use hasty_value::LazyFile;

fn main() {
    let p = std::env::args().nth(1).unwrap();
    let lf = LazyFile::open(std::path::Path::new(&p)).unwrap();
    let t = lf.root.get("x").unwrap().tensor().unwrap();
    let z = t.shape[2];
    let (lo, hi) = (vec![30, 20, 0, 5, 0, 0], vec![80, 70, z, 17, t.shape[4], 2]);
    for th in [1, 4, 8, 16] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(th).build().unwrap();
        let _ = pool.install(|| lf.read_box(t, &lo, &hi)).unwrap();
        let t0 = Instant::now();
        for _ in 0..5 {
            let _ = pool.install(|| lf.read_box(t, &lo, &hi)).unwrap();
        }
        let dt = t0.elapsed().as_secs_f64() / 5.0;
        let nb: usize = (0..6).map(|d| hi[d] - lo[d]).product::<usize>() * 2;
        println!("{th:2} threads: box {:.1} MB in {:.1} ms ({:.0} MB/s)", nb as f64 / 1e6, dt * 1e3, nb as f64 / 1e6 / dt);
    }
    let t0 = Instant::now();
    let full = lf.read(t).unwrap();
    println!("whole x {:.0} MB in {:.0} ms (global pool)", full.nbytes() as f64 / 1e6, t0.elapsed().as_secs_f64() * 1e3);
}
