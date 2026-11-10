//! Shared helpers for benchmark binaries.

/// Median of a slice (sorts a copy).
pub fn median(v: &[f32]) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

pub mod beats;
pub mod seqsched;
pub mod b0data;
pub mod physio;
