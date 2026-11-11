//! One port for everything: native gRPC (HTTP/2), gRPC-web (browsers, HTTP/1.1 or 2), the plotting website and the
//! local CA certificate (`/hasty-ca.pem`), over TLS by default so the viewer is reachable from the LAN.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::routing::get;
use tower_http::services::ServeDir;

use crate::bank::Bank;
use crate::commands::Registry;
use crate::pb::hasty_server::HastyServer;
use crate::service::Service;
use crate::{certs, repo_path, Error};

#[derive(Clone, Debug)]
pub struct Config {
    /// default 0.0.0.0:8443 (env HASTY_BIND)
    pub bind: SocketAddr,
    pub tls: bool,
    /// CA + key directory (env HASTY_CERT_DIR, default data/hasty/certs)
    pub cert_dir: PathBuf,
    /// website root (env HASTY_WEB_DIR, default plotting_website)
    pub web_dir: PathBuf,
    /// extra certificate names (env HASTY_TLS_NAMES, comma separated)
    pub extra_names: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        let env = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());
        Self {
            bind: env("HASTY_BIND").and_then(|s| s.parse().ok()).unwrap_or_else(|| "0.0.0.0:8443".parse().unwrap()),
            tls: env("HASTY_TLS").map(|s| s != "0").unwrap_or(true),
            cert_dir: env("HASTY_CERT_DIR").map(PathBuf::from).unwrap_or_else(|| hasty_site::data().join("certs")),
            web_dir: env("HASTY_WEB_DIR").map(PathBuf::from).unwrap_or_else(|| repo_path("plotting_website")),
            extra_names: env("HASTY_TLS_NAMES").map(|s| s.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default(),
        }
    }
}

/// A running server (on its own tokio runtime thread when started with [`spawn`]).
pub struct Handle {
    pub addr: SocketAddr,
    pub bank: Arc<Bank>,
    pub tls: bool,
    /// https://localhost:port, then one URL per LAN address
    pub urls: Vec<String>,
    pub ca_path: Option<PathBuf>,
    shutdown: Option<axum_server::Handle<SocketAddr>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Handle {
    /// LAN URL if there is one, else the local one
    pub fn url(&self) -> &str {
        self.urls.get(1).unwrap_or(&self.urls[0])
    }

    pub fn shutdown(mut self) {
        self.stop();
    }

    fn stop(&mut self) {
        if let Some(h) = self.shutdown.take() {
            h.shutdown();
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }

    /// block until the server stops (Ctrl-C in `hasty-serve`)
    pub fn wait(mut self) {
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.stop();
    }
}

fn install_crypto() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// Start the server on a background thread; returns once it listens.
pub fn spawn(cfg: Config, bank: Arc<Bank>, registry: Arc<Registry>) -> Result<Handle, Error> {
    install_crypto();
    let tls = if cfg.tls { Some(certs::ensure(&cfg.cert_dir, &cfg.extra_names)?) } else { None };
    let std_listener = std::net::TcpListener::bind(cfg.bind)?;
    std_listener.set_nonblocking(true)?;
    let addr = std_listener.local_addr()?;
    let scheme = if tls.is_some() { "https" } else { "http" };
    let mut urls = vec![format!("{scheme}://localhost:{}", addr.port())];
    if addr.ip().is_unspecified() {
        urls.extend(certs::lan_addresses().into_iter().filter(|ip| ip.is_ipv4()).map(|ip| format!("{scheme}://{ip}:{}", addr.port())));
    }
    let ca_pem = tls.as_ref().map(|t| t.ca_pem.clone()).unwrap_or_default();
    let ca_path = tls.as_ref().map(|t| t.ca_path.clone());
    let svc = Service { bank: bank.clone(), registry };
    let grpc = HastyServer::new(svc).max_decoding_message_size(1 << 30).max_encoding_message_size(1 << 30);
    let web_dir = cfg.web_dir.clone();
    let shutdown = axum_server::Handle::new();
    let sh = shutdown.clone();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    let thread = std::thread::Builder::new()
        .name("hasty-server".into())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_multi_thread().worker_threads(4).thread_name("hasty-io").enable_all().build() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                    return;
                }
            };
            rt.block_on(async move {
                let app = tonic::service::Routes::new(grpc)
                    .prepare()
                    .into_axum_router()
                    .layer(tonic_web::GrpcWebLayer::new())
                    .route("/hasty-ca.pem", get(move || async move { ([("content-type", "application/x-pem-file")], ca_pem.clone()) }))
                    .fallback_service(ServeDir::new(web_dir));
                let served = match tls {
                    Some(t) => {
                        let rc = match axum_server::tls_rustls::RustlsConfig::from_pem(t.cert_chain_pem.into_bytes(), t.key_pem.into_bytes()).await {
                            Ok(c) => c,
                            Err(e) => {
                                let _ = ready_tx.send(Err(e.to_string()));
                                return;
                            }
                        };
                        match axum_server::from_tcp_rustls(std_listener, rc) {
                            Ok(s) => {
                                let _ = ready_tx.send(Ok(()));
                                s.handle(sh).serve(app.into_make_service()).await
                            }
                            Err(e) => {
                                let _ = ready_tx.send(Err(e.to_string()));
                                return;
                            }
                        }
                    }
                    None => match axum_server::from_tcp(std_listener) {
                        Ok(s) => {
                            let _ = ready_tx.send(Ok(()));
                            s.handle(sh).serve(app.into_make_service()).await
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e.to_string()));
                            return;
                        }
                    },
                };
                if let Err(e) = served {
                    tracing::error!("hasty server: {e}");
                }
            });
        })?;
    match ready_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(Error::Tls(e)),
        Err(_) => return Err(Error::Invalid("server thread exited".into())),
    }
    Ok(Handle { addr, bank, tls: cfg.tls, urls, ca_path, shutdown: Some(shutdown), thread: Some(thread) })
}
