//! Hardware limits and rasters.

/// Gradient / RF / ADC system. Gradient limits are per physical axis; base readouts that are rotated are designed with
/// the same numbers as *vector* limits (‖G‖, ‖dG/dt‖), which then hold on every axis for every rotation.
#[derive(Clone, Debug)]
pub struct System {
    /// T/m
    pub gmax: f64,
    /// T/m/s
    pub smax: f64,
    /// gradient raster, s
    pub grad_raster: f64,
    /// ADC dwell raster, s
    pub adc_raster: f64,
    /// RF raster, s
    pub rf_raster: f64,
    /// main field, T
    pub b0: f64,
    /// RF dead time before / ring-down after a pulse, s
    pub rf_dead: f64,
    pub rf_ringdown: f64,
    /// ADC dead time, s
    pub adc_dead: f64,
}

impl System {
    /// Siemens Prisma (XR gradients): 80 mT/m, 200 T/m/s per axis, 10 µs gradient raster, 2.89 T.
    pub fn prisma_xr() -> Self {
        Self { gmax: 80e-3, smax: 200.0, grad_raster: 10e-6, adc_raster: 100e-9, rf_raster: 1e-6, b0: 2.89, rf_dead: 100e-6, rf_ringdown: 30e-6, adc_dead: 10e-6 }
    }
    /// the same system with both gradient limits scaled (safety margin, PNS headroom)
    pub fn derated(&self, f: f64) -> Self {
        Self { gmax: self.gmax * f, smax: self.smax * f, ..self.clone() }
    }
    /// round a duration up to the gradient raster
    pub fn ceil_raster(&self, t: f64) -> f64 {
        (t / self.grad_raster - 1e-9).ceil() * self.grad_raster
    }
}
