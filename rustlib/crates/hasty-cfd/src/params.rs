//! Lattice constants (D3Q19, the ordering of `kernels/lbm_core.h`) and the parameter blocks passed to the kernels.

pub const Q: usize = 19;
pub const CX: [i32; Q] = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0];
pub const CY: [i32; Q] = [0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1];
pub const CZ: [i32; Q] = [0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1];
pub const W: [f32; Q] = [
    1.0 / 3.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];
pub const OPP: [usize; Q] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17];

pub const TILE: usize = 8;
pub const TILE3: usize = 512;
pub const VALVE_STRIDE: usize = 320;
/// leaflet strips per valve (record capacity)
pub const MAX_LEAFLETS: usize = 16;
pub const CUT_STRIDE: usize = 32;
/// region ids < MAX_REGIONS are counted by the diagnostics
pub const MAX_REGIONS: usize = 32;
pub const ACC_STRIDE: usize = 30;

pub const F_FLUID: u8 = 1;
pub const F_WAS: u8 = 2;
pub const F_REFILL: u8 = 4;
pub const F_BC: u8 = 8;
pub const F_PSM: u8 = 16;
pub const F_WALL: u8 = 32;

/// shifted equilibrium g_i = f_i^eq − w_i
pub fn geq(i: usize, rho: f32, u: [f32; 3]) -> f32 {
    let cu = CX[i] as f32 * u[0] + CY[i] as f32 * u[1] + CZ[i] as f32 * u[2];
    let usq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    W[i] * ((rho - 1.0) + rho * (3.0 * cu + 4.5 * cu * cu - 1.5 * usq))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MotionParams {
    pub m: i32,
    pub idx: [i32; 4],
    pub w: [f32; 4],
    pub d: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GeomParams {
    pub o: [f32; 3],
    pub dx: f32,
    pub mo: [f32; 3],
    pub mh: f32,
    pub mn: [i32; 3],
    pub so: [f32; 3],
    pub sh: f32,
    pub sn: [i32; 3],
    pub vscale: f32,
    pub iters: i32,
    pub first: i32,
    pub nvalves: i32,
    pub ncuts: i32,
    pub ntiles: i32,
    /// 1: a region map is sampled at the reference position
    pub regions: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StepParams {
    pub ntiles: i32,
    pub tau0: f32,
    pub cwale: f32,
    pub uclip: f32,
    pub ncuts: i32,
    pub fct: i32,
    pub vel_out: f32,
    /// bulk relaxation time (≤ ½: the shear τ)
    pub tau_b: f32,
    /// lattice stress → Pa (ρ0 (dx/dt)²) for the wall shear stress accumulation
    pub stress_out: f32,
    pub pad: i32,
}

/// spin history (kernels/lbm_core.h SpinParams)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SpinParams {
    pub dt_tr: f32,
    pub e1_step: f32,
    pub o: [f32; 3],
    pub nrm: [f32; 3],
    pub dx: f32,
    pub ntab: i32,
    pub az0: f32,
    pub adz: f32,
    pub enabled: i32,
}

unsafe impl cudarc::driver::DeviceRepr for SpinParams {}
unsafe impl cudarc::driver::DeviceRepr for MotionParams {}
unsafe impl cudarc::driver::DeviceRepr for GeomParams {}
unsafe impl cudarc::driver::DeviceRepr for StepParams {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_is_consistent() {
        // weights sum to 1, isotropy Σ w c c = c_s² I, opposites
        let ws: f32 = W.iter().sum();
        assert!((ws - 1.0).abs() < 1e-6);
        for a in 0..3 {
            for b in 0..3 {
                let s: f32 = (0..Q).map(|i| { let c = [CX[i], CY[i], CZ[i]]; W[i] * (c[a] * c[b]) as f32 }).sum();
                assert!((s - if a == b { 1.0 / 3.0 } else { 0.0 }).abs() < 1e-6, "{a}{b} {s}");
            }
        }
        for i in 0..Q {
            assert_eq!([CX[OPP[i]], CY[OPP[i]], CZ[OPP[i]]], [-CX[i], -CY[i], -CZ[i]]);
        }
        // equilibrium moments: Σ g = ρ − 1, Σ c g = ρ u
        let (rho, u) = (1.03f32, [0.05f32, -0.02, 0.01]);
        let s0: f32 = (0..Q).map(|i| geq(i, rho, u)).sum();
        assert!((s0 - (rho - 1.0)).abs() < 1e-6);
        let sx: f32 = (0..Q).map(|i| CX[i] as f32 * geq(i, rho, u)).sum();
        assert!((sx - rho * u[0]).abs() < 1e-6);
    }
}
