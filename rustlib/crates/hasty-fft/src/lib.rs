//! Runtime-generated GPU FFT (CUDA C++ via NVRTC). See `rustlib/PLAN.md`, track F.

mod bluestein;
mod chanfft;
mod codegen;
mod conv;
mod factor;
mod mixz;
mod pack;
mod padfft;
mod plan;
mod radix;

pub use codegen::{generate as generate_axis_source, AxisCfg, DiagOp, KernelFormat, PolyTerm};
pub use conv::{ConvPlan, ConvSpec};
pub use factor::{factorize, RADICES};
pub use mixz::{MixEntry, MixZPlan};
pub use pack::{f32_to_bf16_bits, f32_to_f16_bits, pack_bf16, pack_half};
pub use padfft::{PadFft3, PadFftOps, PolyTable};
pub use plan::{AxisPlan, AxisSpec, Direction, FftError, FftPlan, LaunchPtrs, LineAddr, LineSplit, Norm};
pub use chanfft::PrunedChanFft3;
