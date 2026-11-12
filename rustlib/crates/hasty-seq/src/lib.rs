//! Sequence designer for our 4D/5D flow sequence (design: `research/sequence_design.md`).
//!
//! - [`system`]: gradient / RF / ADC limits and rasters.
//! - [`wave`]: 3-axis piecewise-linear gradient waveforms, exact moments M0, M1, M2 and P = ∫GGᵀ, k(t).
//! - [`encode`]: velocity encoding (minimum-time lobes for target (M0, M1), encoding schemes incl. coprime dual VENC,
//!   joint timing).
//! - [`venc`]: velocity decoding of every scheme, coprime (Chinese-remainder) unwrapping, noise and unwrap-failure model.
//! - [`girf`]: LTI gradient chain (delay, low-pass, eddy currents, measured tables) → actual gradients / k(t).
//! - [`pns`]: SAFE peripheral-nerve-stimulation model.
//! - [`traj`]: time-optimal traversal of k-space curves, yarnball-type base spokes, rotation sets.
//! - [`tr`]: one TR (RF, encode, rotated spoke, spoiler) on physical axes; PNS of the played sequence.
//! - [`robust`]: GIRF-robust encode blocks (eddy / low-pass sensitivities E(τ), resonance band; QP by ADMM).
//! - [`multiecho`]: short out-and-back multi-echo train (B0 / R2* / fat mapping), flow moments, PNS-fitted slew.
//! - [`metrics`]: information per spoke (rotation-averaged density, spokes to Nyquist), concomitant anisotropy.
//! - [`optim`]: parametric search over base spokes under G_max / S_max / PNS.
//! - [`pulseq`]: Pulseq `.seq` writer (1.5.0 / 1.4.1, labels, rotations, signature) and reader for round-trip checks.
//! - [`pnsge`]: GE's PNS model (TOPPE `pns.m` / pge2 checks, IEC 60601-2-33 form) with the Premier coil parameters.
//! - [`girfqa`]: the GIRF QA measurement protocol (thin slices, phase encoding, inputs, heating, v2 validation).
//! - [`girfmeas`]: measured-GIRF file (JSON) → [`girf::Girf`] at a thermal state, resonance list for the designer.
//!
//! Units: SI throughout (T/m, T/m/s, s, m); k in cycles/m (k = γ̄ M0).

pub mod encode;
pub mod girf;
pub mod metrics;
pub mod multiecho;
pub mod girfmeas;
pub mod girfqa;
pub mod optim;
pub mod pns;
pub mod pnsge;
pub mod pulseq;
pub mod robust;
pub mod spoil;
pub mod system;
pub mod tr;
pub mod traj;
pub mod venc;
pub mod wave;

/// gyromagnetic ratio of ¹H, rad/s/T
pub const GAMMA: f64 = 2.675_221_874_4e8;
/// γ / 2π, Hz/T
pub const GAMMA_BAR: f64 = GAMMA / (2.0 * std::f64::consts::PI);

pub type Vec3 = [f64; 3];
pub type Mat3 = [[f64; 3]; 3];

pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn norm(v: &Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
pub fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn sub(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn scale(a: &Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn mat_vec(m: &Mat3, v: &Vec3) -> Vec3 {
    [0, 1, 2].map(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}
pub fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}
pub fn transpose(m: &Mat3) -> Mat3 {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| m[j][i]))
}
/// rotation matrix of a unit quaternion (w, x, y, z)
pub fn quat_to_mat(q: [f64; 4]) -> Mat3 {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (w, x, y, z) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}
/// rotation by `angle` about the unit `axis`
static THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// worker threads used inside one design evaluation (PNS chunks, per-rotation tails); 0 = all cores
pub fn set_threads(n: usize) {
    THREADS.store(n, std::sync::atomic::Ordering::SeqCst);
}
pub fn threads() -> usize {
    match THREADS.load(std::sync::atomic::Ordering::SeqCst) {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        n => n,
    }
}

/// map `f` over `items` on [`threads`] scoped threads, preserving order
pub fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let nt = threads().min(items.len()).max(1);
    if nt == 1 {
        return items.iter().map(f).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<R>>> = items.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..nt {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= items.len() {
                    break;
                }
                *slots[i].lock().unwrap() = Some(f(&items[i]));
            });
        }
    });
    slots.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}

pub fn axis_angle(axis: &Vec3, angle: f64) -> Mat3 {
    let n = norm(axis);
    let (s, c) = (0.5 * angle).sin_cos();
    quat_to_mat([c, s * axis[0] / n, s * axis[1] / n, s * axis[2] / n])
}
