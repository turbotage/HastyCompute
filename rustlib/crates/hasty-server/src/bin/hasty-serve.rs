//! Standalone hasty server (bank + commands + website, TLS on all interfaces).
//!
//!   hasty-serve [--bind 0.0.0.0:8443] [--no-tls] [--names a,b]   (env: HASTY_BIND, HASTY_TLS=0, HASTY_TLS_NAMES,
//!                                                                  HASTY_CERT_DIR, HASTY_WEB_DIR, RUST_LOG)
use std::sync::Arc;

use hasty_server::server::{spawn, Config};
use hasty_server::{Bank, Registry};

fn main() {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into())).init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let mut cfg = Config::default();
    if let Some(b) = opt("--bind") {
        cfg.bind = b.parse().expect("--bind host:port");
    }
    if args.iter().any(|a| a == "--no-tls") {
        cfg.tls = false;
    }
    if let Some(n) = opt("--names") {
        cfg.extra_names = n.split(',').map(|s| s.trim().to_string()).collect();
    }
    let web = cfg.web_dir.clone();
    let h = spawn(cfg, Arc::new(Bank::new()), Arc::new(Registry::base())).unwrap_or_else(|e| panic!("hasty-serve: {e}"));
    tracing::info!("serving {} on {}", web.display(), h.urls.join("  "));
    if let Some(ca) = &h.ca_path {
        tracing::info!("viewers: trust {} once (also at <url>/hasty-ca.pem)", ca.display());
    }
    h.wait();
}
