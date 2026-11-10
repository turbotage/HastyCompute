//! Process memory: resident set size, and returning freed heap memory to the OS.
//!
//! glibc serves allocations below its mmap threshold (≤ 32 MiB) from heap arenas; freeing them does not shrink the
//! resident set unless the arena top is free. After a setup phase that frees many mid-sized buffers (per-set sample
//! records: coordinates, τ, concomitant coefficients) the memory stays resident until `malloc_trim`.

/// resident set size (bytes) from /proc/self/status (0 where unavailable)
pub fn rss_bytes() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<usize>().ok()))
        .map_or(0, |kb| kb * 1024)
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
extern "C" {
    fn malloc_trim(pad: usize) -> i32;
}

/// return free heap memory to the OS (glibc `malloc_trim(0)`; no-op elsewhere); (RSS before, RSS after) in bytes
pub fn trim() -> (usize, usize) {
    let before = rss_bytes();
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: malloc_trim only releases free memory
    unsafe {
        malloc_trim(0);
    }
    (before, rss_bytes())
}
