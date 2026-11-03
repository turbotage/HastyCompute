//! k-space coverage of one encode: kz planes used, readouts per plane, in-plane angle spread, k ranges.
use hasty_recon::data::SimData;
fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let sd = SimData::load(&dir).unwrap();
    let ros = sd.readouts_of(0, None);
    let mut parts = vec![0usize; sd.acq.kz.len()];
    for &r in &ros { parts[sd.acq.readouts[r].part] += 1; }
    println!("{} readouts of encode 0; per kz partition: {:?}", ros.len(), parts);
    let ident: Vec<Vec<f64>> = (0..sd.nc).map(|i| (0..sd.nc).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let set = sd.sample_set(&ros[..ros.len().min(60)], &ident);
    for a in 0..3 {
        let v = &set.coords[a];
        let (mn, mx) = v.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        let mut u: Vec<i64> = v.iter().map(|x| (x * 1000.0).round() as i64).collect();
        u.sort(); u.dedup();
        println!("axis {a}: range [{mn:.3}, {mx:.3}] rad/voxel, {} distinct values (rounded 1e-3)", u.len());
    }
    // first readout: k at a few samples
    let na = sd.na;
    for n in [0usize, 10, 100, 500, 1000, na - 1] {
        println!("  readout {} sample {n}: k·Δ = ({:.3}, {:.3}, {:.3})", ros[0], set.coords[0][n], set.coords[1][n], set.coords[2][n]);
    }
    println!("rotation of first 3 readouts: {:?}", ros.iter().take(3).map(|&r| sd.acq.readouts[r].rot[0]).collect::<Vec<_>>());
}
