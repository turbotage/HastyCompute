//! Visualise from any binary:
//!
//! ```no_run
//! let url = hasty_server::viz::show("recon", hasty_value::Tensor::from_fortran(&[64, 64, 32], &vec![0f32; 64 * 64 * 32]).unwrap()).unwrap();
//! hasty_server::viz::wait(); // keep an embedded server alive until Ctrl-C
//! ```
//!
//! Target, in order: `HASTY_SERVER` (e.g. `https://gpu-box:8443`); a hasty server already running on this machine
//! (`https://localhost:8443`); otherwise an embedded server in this process (TLS, all interfaces) whose URLs are
//! printed once. Open viewers switch to every new `show` (Watch events).

use std::sync::{Arc, Mutex};

use hasty_value::{Element, Options, Tensor, Value};
use uuid::Uuid;

use crate::bank::Bank;
use crate::client::Client;
use crate::commands::Registry;
use crate::server::{self, Config, Handle};
use crate::Error;

enum Target {
    Remote { rt: tokio::runtime::Runtime, client: Client, base: String },
    Embedded { handle: Handle },
}

static VIZ: Mutex<Option<Target>> = Mutex::new(None);

fn ca_pem(cfg: &Config) -> Option<String> {
    let p = std::env::var("HASTY_CA").map(std::path::PathBuf::from).unwrap_or_else(|_| cfg.cert_dir.join("ca.pem"));
    std::fs::read_to_string(p).ok()
}

fn connect() -> Result<Target, Error> {
    let cfg = Config::default();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let ca = ca_pem(&cfg);
    if let Ok(url) = std::env::var("HASTY_SERVER") {
        let client = rt.block_on(Client::connect(&url, ca.as_deref()))?;
        return Ok(Target::Remote { rt, client, base: url.trim_end_matches('/').to_string() });
    }
    let port = cfg.bind.port();
    let scheme = if cfg.tls { "https" } else { "http" };
    if let Ok(client) = rt.block_on(Client::connect(&format!("{scheme}://localhost:{port}"), ca.as_deref())) {
        let host = crate::certs::lan_addresses().into_iter().find(|i| i.is_ipv4()).map(|i| i.to_string()).unwrap_or_else(|| "localhost".into());
        return Ok(Target::Remote { rt, client, base: format!("{scheme}://{host}:{port}") });
    }
    let handle = server::spawn(cfg, Arc::new(Bank::new()), Arc::new(Registry::base()))?;
    eprintln!("hasty viz: serving on {}", handle.urls.join("  "));
    if let Some(ca) = &handle.ca_path {
        eprintln!("hasty viz: trust {} (or <url>/hasty-ca.pem) once in the browser / OS to avoid certificate warnings", ca.display());
    }
    Ok(Target::Embedded { handle })
}

fn viewer_url(base: &str, id: &Uuid, view: &str) -> String {
    format!("{base}/{view}/index.html?id={}", id.simple())
}

/// Publish `value` under `name` and ask open viewers to show it; returns the viewer URL.
pub fn show(name: &str, value: impl Into<Value>) -> Result<String, Error> {
    show_with(name, value.into(), serde_json::json!({}), "orthoslicer")
}

/// [`show`] with metadata (e.g. `{"axes": ["z", "y", "x"], "voxel_mm": [2, 2, 2]}`) and a viewer name.
pub fn show_with(name: &str, value: Value, metadata: serde_json::Value, view: &str) -> Result<String, Error> {
    let mut g = VIZ.lock().unwrap();
    if g.is_none() {
        *g = Some(connect()?);
    }
    let meta = metadata.to_string();
    match g.as_mut().unwrap() {
        Target::Embedded { handle } => {
            let id = handle.bank.put(name, value, &meta);
            handle.bank.notify("show", Some(id), name, view);
            Ok(viewer_url(handle.url(), &id, view))
        }
        Target::Remote { rt, client, base } => {
            let id = rt.block_on(async {
                let id = client.put(name, &value, &meta, &Options::default()).await?;
                client.show(&id, view).await?;
                Ok::<_, Error>(id)
            })?;
            Ok(viewer_url(base, &id, view))
        }
    }
}

/// Publish a 4D flow value (`crate::flow::Flow`: velocity, magnitude, spacing, dt) to the flow viewer.
pub fn show_flow(name: &str, flow: &crate::flow::Flow) -> Result<String, Error> {
    show_with(name, flow.to_value(), serde_json::json!({ "kind": "flow4d", "dims": flow.dims, "frames": flow.nf }), "flow")
}

/// A hasty x-fastest volume (`dims` = [x, y, z, ...]) as a C-order tensor with shape `[..., z, y, x]`.
pub fn volume<T: Element>(name: &str, dims: &[usize], data: &[T]) -> Result<String, Error> {
    let axes: Vec<String> = (0..dims.len()).rev().map(|a| ["x", "y", "z", "t", "e"].get(a).map_or(format!("d{a}"), |s| s.to_string())).collect();
    show_with(name, Value::Tensor(Tensor::from_fortran(dims, data)?), serde_json::json!({ "axes": axes }), "orthoslicer")
}

/// Keep an embedded server running until Ctrl-C (no-op when publishing to another server).
pub fn wait() {
    let t = VIZ.lock().unwrap().take();
    if let Some(Target::Embedded { handle }) = t {
        eprintln!("hasty viz: Ctrl-C to stop the server");
        let (tx, rx) = std::sync::mpsc::channel();
        let _ = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            rt.block_on(async {
                let _ = tokio::signal::ctrl_c().await;
            });
            let _ = tx.send(());
        });
        let _ = rx.recv();
        handle.shutdown();
    }
}
