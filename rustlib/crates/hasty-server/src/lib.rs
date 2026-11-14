//! hasty server: a value bank with named commands behind gRPC (`proto/hasty.proto`), gRPC-web and the plotting
//! website, on one TLS port reachable from the LAN. Values travel in the block-compressed hasty value format.
//!
//! - [`server::spawn`]: start a server (own runtime thread); `hasty-serve` is the standalone binary.
//! - [`viz::show`]: publish a value from any binary and get a viewer URL (embedded server if none is running).
//! - [`client::Client`]: async client.

pub mod bank;
pub mod certs;
pub mod client;
pub mod commands;
pub mod flow;
pub mod models;
pub mod server;
mod service;
pub mod viz;

pub mod pb {
    tonic::include_proto!("hasty.v2");
}

pub use bank::Bank;
pub use commands::Registry;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no value {0}")]
    NotFound(uuid::Uuid),
    #[error("{0}")]
    Invalid(String),
    #[error("value: {0}")]
    Value(#[from] hasty_value::Error),
    #[error("tls: {0}")]
    Tls(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// path relative to the HastyCompute repository root
pub fn repo_path(rel: &str) -> std::path::PathBuf {
    hasty_site::repo_path(rel)
}
