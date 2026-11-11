//! Runtime locations for portable deployment (local checkout, cluster, container, rented node).
//!
//! Every location is resolved at run time, in this order:
//! 1. an environment variable (set by a site profile, a job script or the container image);
//! 2. the repository the running binary lives in (`<repo>/rustlib/target/<profile>/<bin>`);
//! 3. the repository this crate was compiled in (the old behaviour; only right on the build machine).
//!
//! | function | env | default |
//! |---|---|---|
//! | [`repo`] | `HASTY_REPO` | checkout containing the binary |
//! | [`data`] | `HASTY_DATA` | `<repo>/data/hasty` |
//! | [`runs`] | `HASTY_RUNS` | `<data>/runs` |
//! | [`tools`] | `HASTY_TOOLS` | `<repo>/rustlib/tools` |
//! | [`python`] | `HASTY_PYTHON` | `<repo>/rustlib/.venv/bin/python`, else `python3` on `PATH` |
//! | [`scratch`] | `HASTY_SCRATCH` | none (node-local fast disk on clusters) |
//! | [`kernel_cache`] | `HASTY_KERNEL_CACHE` | `<scratch>/hasty-kernels`, else `<repo>/rustlib/target/hasty-kernels` |
//! | [`target`] | `HASTY_TARGET` | `<repo>/rustlib/target` |
//!
//! A container mirrors the repository layout under one prefix (`/opt/hasty/rustlib/{tools,.venv,target}`) and sets
//! `HASTY_REPO=/opt/hasty`; data lives elsewhere (`HASTY_DATA`).

use std::path::{Path, PathBuf};

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// Repository root of the binary: `<repo>/rustlib/target/<profile>[/deps]/<bin>` → `<repo>`.
fn repo_of_exe() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
    let mut p: &Path = exe.parent()?;
    loop {
        if p.file_name()? == "target" && p.parent()?.file_name()? == "rustlib" {
            return Some(p.parent()?.parent()?.to_path_buf());
        }
        p = p.parent()?;
    }
}

fn repo_of_build() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    root.canonicalize().unwrap_or(root)
}

/// The HastyCompute checkout (or the container prefix mirroring its layout).
pub fn repo() -> PathBuf {
    env_path("HASTY_REPO").or_else(repo_of_exe).unwrap_or_else(repo_of_build)
}

/// Path relative to [`repo`].
pub fn repo_path(rel: impl AsRef<Path>) -> PathBuf {
    repo().join(rel)
}

/// Data root (phantoms, simulations, runs, model weights).
pub fn data() -> PathBuf {
    env_path("HASTY_DATA").unwrap_or_else(|| repo().join("data/hasty"))
}

/// Default parent directory of run output directories.
pub fn runs() -> PathBuf {
    env_path("HASTY_RUNS").unwrap_or_else(|| data().join("runs"))
}

/// The rustlib Python tools (figures, exports).
pub fn tools() -> PathBuf {
    env_path("HASTY_TOOLS").unwrap_or_else(|| repo().join("rustlib/tools"))
}

/// A tool script, `tools()/rel`.
pub fn tool(rel: impl AsRef<Path>) -> PathBuf {
    tools().join(rel)
}

/// Python interpreter for the tools.
pub fn python() -> PathBuf {
    if let Some(p) = env_path("HASTY_PYTHON") {
        return p;
    }
    let venv = repo().join("rustlib/.venv/bin/python");
    if venv.exists() {
        venv
    } else {
        PathBuf::from("python3")
    }
}

/// Node-local fast scratch, if the site has one.
pub fn scratch() -> Option<PathBuf> {
    env_path("HASTY_SCRATCH")
}

/// Cargo target directory of the checkout (build outputs, plugins, local caches).
pub fn target() -> PathBuf {
    env_path("HASTY_TARGET").unwrap_or_else(|| repo().join("rustlib/target"))
}

/// On-disk cache of NVRTC cubins (keyed by source, arch, options and NVRTC version).
pub fn kernel_cache() -> PathBuf {
    env_path("HASTY_KERNEL_CACHE")
        .or_else(|| scratch().map(|s| s.join("hasty-kernels")))
        .unwrap_or_else(|| target().join("hasty-kernels"))
}

/// Bytes of RAM this process can still use: `MemAvailable`, capped by every enclosing memory cgroup's limit minus its
/// current use (systemd-run scopes, Slurm job / step cgroups, containers). cgroup v2 and v1.
pub fn available_memory() -> usize {
    let mut avail = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<usize>().ok()))
        .map_or(usize::MAX, |kb| kb * 1024);
    if let Some(cg) = cgroup_headroom() {
        avail = avail.min(cg);
    }
    avail
}

/// min over the process's memory cgroup and its ancestors of (limit − usage); None without a limit
pub fn cgroup_headroom() -> Option<usize> {
    let groups = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let num = |p: &Path| std::fs::read_to_string(p).ok().and_then(|v| v.trim().parse::<usize>().ok());
    let mut best: Option<usize> = None;
    let mut walk = |root: &Path, rel: &str, limit: &str, usage: &str| {
        let mut dir = root.join(rel.trim_start_matches('/'));
        loop {
            // v2 writes "max" for no limit (no parse); v1 writes a huge number (≥ 2^62)
            if let (Some(l), Some(u)) = (num(&dir.join(limit)), num(&dir.join(usage))) {
                if l < (1usize << 62) {
                    let h = l.saturating_sub(u);
                    best = Some(best.map_or(h, |b| b.min(h)));
                }
            }
            if dir == root || !dir.pop() || !dir.starts_with(root) {
                break;
            }
        }
    };
    for line in groups.lines() {
        let mut f = line.splitn(3, ':');
        let (_, ctrl, path) = (f.next()?, f.next()?, f.next()?);
        if ctrl.is_empty() {
            walk(Path::new("/sys/fs/cgroup"), path, "memory.max", "memory.current");
        } else if ctrl.split(',').any(|c| c == "memory") {
            walk(Path::new("/sys/fs/cgroup/memory"), path, "memory.limit_in_bytes", "memory.usage_in_bytes");
        }
    }
    best
}

#[cfg(test)]
mod tests {
    #[test]
    fn memory_probe() {
        let m = super::available_memory();
        eprintln!("available memory {:.2} GiB (cgroup headroom {:?})", m as f64 / (1u64 << 30) as f64, super::cgroup_headroom());
        assert!(m > 1 << 20 && m < usize::MAX, "{m}");
    }

    #[test]
    fn repo_has_rustlib() {
        // the test binary lives in <repo>/rustlib/target/<profile>/deps
        assert!(super::repo().join("rustlib/Cargo.toml").exists(), "{:?}", super::repo());
    }
}
