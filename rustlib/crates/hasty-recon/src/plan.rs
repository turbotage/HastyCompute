//! Memory planner: before a reconstruction, measure the device (free VRAM, host↔device bandwidth) and choose where
//! every large buffer lives — device, or pinned host memory streamed through chunk-sized device scratch — so that the
//! run fits the budget at the lowest predicted time per iteration. Reconstructions then run on any GPU: small cards
//! swap the P²nP state and the k-space data out, large ones keep everything resident. `--vram-budget` emulates a
//! smaller card (testing); every placement must reproduce the all-device result (tests in `vobj.rs`).
//!
//! Placed buffers (bytes from the problem, see [`Need`]):
//! - k-space data y_t (read once per gradient): device, or host with a per-set staging copy;
//! - ZMSHR1 secant pair x_prev / g_prev (read twice, written once per iteration): device or host;
//! - rank-one direction w (written once, read once): device, host, or recomputed from the secant pair.
//!
//! Always on the device: x, ∇f and the step (three parameter stacks), the operator, coil maps and model scratch.

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream};

use crate::p2np::{StateOpts, WMode};
use crate::Error;

#[derive(Clone, Debug)]
pub struct Probe {
    pub free: usize,
    pub total: usize,
    /// bytes / s
    pub h2d: f64,
    pub d2h: f64,
}

/// free / total VRAM and pinned transfer bandwidth (64 MiB, best of 3)
pub fn probe(ctx: &Arc<CudaContext>, st: &Arc<CudaStream>) -> Result<Probe, Error> {
    let (free, total) = ctx.mem_get_info()?;
    let n = (64usize << 20) / 8;
    let mut h = unsafe { ctx.alloc_pinned_with_flags::<f64>(n, 0)? };
    let mut d = st.alloc_zeros::<f64>(n)?;
    let (mut h2d, mut d2h) = (0f64, 0f64);
    for _ in 0..3 {
        st.synchronize()?;
        let t = Instant::now();
        st.memcpy_htod(&h, &mut d)?;
        st.synchronize()?;
        h2d = h2d.max((n * 8) as f64 / t.elapsed().as_secs_f64());
        let t = Instant::now();
        st.memcpy_dtoh(&d, &mut h)?;
        st.synchronize()?;
        d2h = d2h.max((n * 8) as f64 / t.elapsed().as_secs_f64());
    }
    Ok(Probe { free, total, h2d, d2h })
}

/// what a reconstruction needs (bytes; time of one gradient if measured)
#[derive(Clone, Debug)]
pub struct Need {
    /// already resident and not placeable (operator, maps, model scratch, …), excluding `data` if it is resident now
    pub fixed: usize,
    /// one parameter stack
    pub stack: usize,
    /// largest chunk of a stack
    pub chunk_max: usize,
    /// all k-space data
    pub data: usize,
    /// largest set's data
    pub set_max: usize,
    /// ZMSHR1 / ZMSHR1-M: the secant pair and w exist
    pub quasi: bool,
    /// one gradient (s); 0 when not measured yet
    pub gradient_s: f64,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub data_on_host: bool,
    pub prev_on_host: bool,
    pub w_mode: WMode,
    pub device_bytes: usize,
    pub budget: usize,
    pub transfer_s: f64,
    pub predicted_s: f64,
    /// every candidate: (description, device bytes, predicted s / iteration, fits)
    pub table: Vec<(String, usize, f64, bool)>,
}

impl Plan {
    pub fn state_opts(&self, ctx: &Arc<CudaContext>) -> StateOpts {
        StateOpts { ctx: Some(ctx.clone()), prev_on_host: self.prev_on_host, w_mode: self.w_mode }
    }

    pub fn describe(&self) -> String {
        let gb = |b: usize| b as f64 / (1u64 << 30) as f64;
        let mut s = format!(
            "memory plan: data {}, secant pair {}, w {:?}; device {:.2} / {:.2} GiB budget, transfers {:.3} s, predicted {:.3} s / iteration\n",
            if self.data_on_host { "host (streamed)" } else { "device" },
            if self.prev_on_host { "host (streamed)" } else { "device" },
            self.w_mode,
            gb(self.device_bytes),
            gb(self.budget),
            self.transfer_s,
            self.predicted_s
        );
        for (d, b, t, ok) in &self.table {
            s += &format!("    {} {d:<44} {:>7.2} GiB  {t:.3} s\n", if *ok { "·" } else { "✗" }, gb(*b));
        }
        s
    }
}

/// bytes per gigabyte string ("8G", "512M", "24000000000")
pub fn parse_bytes(s: &str) -> Option<usize> {
    let s = s.trim();
    let (num, mul) = match s.chars().last()? {
        'G' | 'g' => (&s[..s.len() - 1], 1usize << 30),
        'M' | 'm' => (&s[..s.len() - 1], 1 << 20),
        'K' | 'k' => (&s[..s.len() - 1], 1 << 10),
        _ => (s, 1),
    };
    num.parse::<f64>().ok().map(|v| (v * mul as f64) as usize)
}

/// Choose the fastest placement that fits `budget` (bytes available on the device for everything in [`Need`]).
/// `force_data`: keep a previous decision on the data (it can only move once).
pub fn choose(need: &Need, probe: &Probe, budget: usize, force_data: Option<bool>) -> Result<Plan, Error> {
    let s = need.stack;
    let scratch = 6 * need.chunk_max;
    let mut table = vec![];
    let mut best: Option<Plan> = None;
    let datas: Vec<bool> = force_data.map(|d| vec![d]).unwrap_or_else(|| vec![false, true]);
    let prevs: Vec<bool> = if need.quasi { vec![false, true] } else { vec![false] };
    let ws: Vec<WMode> = if need.quasi { vec![WMode::Device, WMode::Host, WMode::Recompute] } else { vec![WMode::Recompute] };
    for &dh in &datas {
        for &ph in &prevs {
            for &w in &ws {
                let mut dev = need.fixed + 3 * s + scratch;
                dev += if dh { 2 * need.set_max } else { need.data };
                if need.quasi && !ph {
                    dev += 2 * s;
                }
                if need.quasi && w == WMode::Device {
                    dev += s;
                }
                // host↔device bytes per iteration
                let (mut up, mut down) = (0f64, 0f64);
                if dh {
                    up += need.data as f64;
                }
                if need.quasi && ph {
                    up += 4.0 * s as f64; // passes 1 and 2 read x_prev, g_prev
                    down += 2.0 * s as f64; // updates
                    if w == WMode::Recompute {
                        up += 2.0 * s as f64; // pass 3 re-reads them
                    }
                }
                if need.quasi && w == WMode::Host {
                    up += s as f64;
                    down += s as f64;
                }
                let tr = up / probe.h2d + down / probe.d2h;
                // recomputing w costs a pointwise pass even on the device: charge a small amount so that storing it
                // wins when memory allows
                let extra = if need.quasi && w == WMode::Recompute { 1e-3 + 0.02 * need.gradient_s } else { 0.0 };
                let t = need.gradient_s + tr + extra;
                let fits = dev <= budget;
                let desc = format!("data {}, secant {}, w {w:?}", if dh { "host" } else { "dev" }, if ph { "host" } else { "dev" });
                table.push((desc, dev, t, fits));
                if fits && best.as_ref().is_none_or(|b| t < b.predicted_s - 1e-12) {
                    best = Some(Plan { data_on_host: dh, prev_on_host: ph, w_mode: w, device_bytes: dev, budget, transfer_s: tr, predicted_s: t, table: vec![] });
                }
            }
        }
    }
    let mut p = best.ok_or_else(|| {
        let min = table.iter().map(|t| t.1).min().unwrap_or(0);
        Error::Format(format!(
            "memory plan: nothing fits — smallest placement needs {:.2} GiB of device memory, budget {:.2} GiB. Reduce the grid (--grid-scale / ROI), nodes (--vmodel), or coil batch",
            min as f64 / (1u64 << 30) as f64,
            budget as f64 / (1u64 << 30) as f64
        ))
    })?;
    p.table = table;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn need() -> Need {
        Need { fixed: 4 << 30, stack: 1 << 30, chunk_max: 64 << 20, data: 3 << 30, set_max: 30 << 20, quasi: true, gradient_s: 2.0 }
    }

    fn probe() -> Probe {
        Probe { free: 20 << 30, total: 24 << 30, h2d: 20e9, d2h: 20e9 }
    }

    #[test]
    fn everything_resident_when_it_fits() {
        let p = choose(&need(), &probe(), 20 << 30, None).unwrap();
        assert!(!p.data_on_host && !p.prev_on_host && p.w_mode == WMode::Device, "{}", p.describe());
    }

    #[test]
    fn small_card_swaps_the_cheapest_buffers_first() {
        // 4 fixed + 3 stacks + scratch = 7.4 GiB; data 3 GiB, secant pair 2 GiB, w 1 GiB
        let n = need();
        let p = choose(&n, &probe(), (10.5 * (1u64 << 30) as f64) as usize, None).unwrap();
        assert!(p.device_bytes <= p.budget, "{}", p.describe());
        // 9 GiB: everything but some state must move; the plan must fit and be the fastest fitting one
        let p = choose(&n, &probe(), 9 << 30, None).unwrap();
        let fastest_fit = p.table.iter().filter(|t| t.3).map(|t| t.2).fold(f64::INFINITY, f64::min);
        assert!((p.predicted_s - fastest_fit).abs() < 1e-12);
        assert!(p.data_on_host || p.prev_on_host, "{}", p.describe());
        // too small: a clear error
        assert!(choose(&n, &probe(), 6 << 30, None).is_err());
        assert_eq!(parse_bytes("8G"), Some(8 << 30));
        assert_eq!(parse_bytes("512M"), Some(512 << 20));
    }
}
