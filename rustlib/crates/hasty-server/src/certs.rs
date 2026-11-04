//! TLS for LAN access. A persistent local certificate authority (`ca.pem` + `ca.key`, created once) signs a fresh
//! server certificate at every start, covering localhost, the host name and every current interface address. Import
//! `ca.pem` once into the browser / OS trust store of each viewing machine (the server also serves it at
//! `/hasty-ca.pem`); after that `https://<lan-ip>:<port>` is trusted even when addresses change.

use std::net::IpAddr;
use std::path::{Path, PathBuf};

use rcgen::{BasicConstraints, CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair, KeyUsagePurpose, SanType};

use crate::Error;

pub struct TlsMaterial {
    /// server certificate followed by the CA certificate
    pub cert_chain_pem: String,
    pub key_pem: String,
    pub ca_pem: String,
    pub ca_path: PathBuf,
    /// host names and addresses in the certificate
    pub names: Vec<String>,
}

fn e(x: impl std::fmt::Display) -> Error {
    Error::Tls(x.to_string())
}

pub fn host_name() -> String {
    std::fs::read_to_string("/etc/hostname").ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).or_else(|| std::env::var("HOSTNAME").ok()).unwrap_or_else(|| "localhost".into())
}

/// non-loopback interface addresses (IPv4 first)
pub fn lan_addresses() -> Vec<IpAddr> {
    let mut v: Vec<IpAddr> = if_addrs::get_if_addrs().unwrap_or_default().into_iter().filter(|i| !i.is_loopback()).map(|i| i.ip()).filter(|ip| !matches!(ip, IpAddr::V6(v6) if (v6.segments()[0] & 0xffc0) == 0xfe80)).collect();
    v.sort_by_key(|ip| (ip.is_ipv6(), *ip));
    v.dedup();
    v
}

fn write_private(p: &Path, s: &str) -> Result<(), Error> {
    std::fs::write(p, s)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

/// Load or create the CA in `dir`, then issue a server certificate for this host.
pub fn ensure(dir: &Path, extra_names: &[String]) -> Result<TlsMaterial, Error> {
    std::fs::create_dir_all(dir)?;
    let (ca_cert, ca_key) = (dir.join("ca.pem"), dir.join("ca.key"));
    let host = host_name();
    if !ca_cert.exists() || !ca_key.exists() {
        let key = KeyPair::generate().map_err(e)?;
        let mut p = CertificateParams::default();
        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, format!("hasty local CA ({host})"));
        dn.push(DnType::OrganizationName, "HastyCompute");
        p.distinguished_name = dn;
        p.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        p.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign, KeyUsagePurpose::DigitalSignature];
        p.not_before = rcgen::date_time_ymd(2026, 1, 1);
        p.not_after = rcgen::date_time_ymd(2036, 1, 1);
        let cert = p.self_signed(&key).map_err(e)?;
        std::fs::write(&ca_cert, cert.pem())?;
        write_private(&ca_key, &key.serialize_pem())?;
        tracing::info!("created local CA {} — import it once into the browser / OS trust store", ca_cert.display());
    }
    let ca_pem = std::fs::read_to_string(&ca_cert)?;
    let issuer = Issuer::from_ca_cert_pem(&ca_pem, KeyPair::from_pem(&std::fs::read_to_string(&ca_key)?).map_err(e)?).map_err(e)?;

    let mut names: Vec<String> = vec!["localhost".into(), host.clone(), format!("{host}.local")];
    names.extend(extra_names.iter().cloned());
    let mut ips: Vec<IpAddr> = vec!["127.0.0.1".parse().unwrap(), "::1".parse().unwrap()];
    ips.extend(lan_addresses());
    let mut p = CertificateParams::default();
    let mut dn = DistinguishedName::new();
    dn.push(DnType::CommonName, format!("hasty server ({host})"));
    p.distinguished_name = dn;
    for n in &names {
        match n.parse::<IpAddr>() {
            Ok(ip) => ips.push(ip),
            Err(_) => p.subject_alt_names.push(SanType::DnsName(n.as_str().try_into().map_err(e)?)),
        }
    }
    ips.sort();
    ips.dedup();
    p.subject_alt_names.extend(ips.iter().map(|ip| SanType::IpAddress(*ip)));
    p.key_usages = vec![KeyUsagePurpose::DigitalSignature, KeyUsagePurpose::KeyEncipherment];
    p.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    let now = std::time::SystemTime::now();
    p.not_before = (now - std::time::Duration::from_secs(86400)).into();
    p.not_after = (now + std::time::Duration::from_secs(365 * 86400)).into();
    let key = KeyPair::generate().map_err(e)?;
    let cert = p.signed_by(&key, &issuer).map_err(e)?;
    names.extend(ips.iter().map(|i| i.to_string()));
    Ok(TlsMaterial { cert_chain_pem: format!("{}{}", cert.pem(), ca_pem), key_pem: key.serialize_pem(), ca_pem, ca_path: ca_cert, names })
}
