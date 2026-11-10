//! Synthetic subjects for the 4D-flow benchmark: anatomy (from `tools/synth/prep_subject.py`), tissue properties,
//! off-resonance from susceptibility, receive coils, flow fields and the conversion into direct-sum simulator sources.
//! Design notes: `research/synthetic_4dflow_pipeline.md`.

pub mod b0;
pub mod coils;
pub mod flow;
pub mod prescription;
pub mod scene;
pub mod shimfit;
pub mod subject;
pub mod tissue;
pub mod volume;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("format error: {0}")]
    Format(String),
}
