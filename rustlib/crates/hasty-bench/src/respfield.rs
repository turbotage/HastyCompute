//! Breathing- and cardiac-dependent off-resonance of simphantom (`--resp-field`, tools/phantom/resp_field.py,
//! research/resp_field_sim.md).
//!
//! States: breathing amplitudes r_k (sorted, r_0 = 0 end-expiration) × cardiac template times c_q = q T / Q (periodic;
//! Q = 1: breathing only). Per tissue source a table Δ_qk(p) of the change of its off-resonance (the field of state
//! (r_k, c_q) at the source's position in that state minus the reference field at its reference position); per blood
//! frame source Δ_k(p) at the frame's own cardiac time. In a bin with breathing state r and cardiac time c:
//!     ω(p, r, c) = ω_static(p) + Σ w_i Δ_i(p)   (bilinear in (r, c), clamped in r, periodic in c)
//! With K = Q = 1 the table is zero and ω is the static ω bit for bit.

/// The segment j and weight a of r among the sorted states rk (clamped to [r_0, r_{K−1}]); K = 1 → (0, 0).
pub fn segment(rk: &[f64], r: f64) -> (usize, f32) {
    let k = rk.len();
    assert!(k >= 1, "resp field: no states");
    if k == 1 || r <= rk[0] {
        return (0, 0.0);
    }
    if r >= rk[k - 1] {
        return (k - 2, 1.0);
    }
    let j = rk.windows(2).position(|w| r < w[1]).unwrap_or(k - 2);
    (j, ((r - rk[j]) / (rk[j + 1] - rk[j])) as f32)
}

/// The kernel's formula on the host (reference / tests): base[p] + (1 − a) T[p K + j] + a T[p K + j1], j1 = min(j + 1, K − 1).
pub fn lerp(base: &[f32], tab: &[f32], k: usize, j: usize, a: f32) -> Vec<f32> {
    assert_eq!(tab.len(), base.len() * k);
    let j1 = (j + 1).min(k - 1);
    base.iter().enumerate().map(|(p, &b)| b + ((1.0 - a) * tab[p * k + j] + a * tab[p * k + j1])).collect()
}

/// The table of the simulated sources from the input sources' table (ppm, [n_in][K]): source s takes the row of its
/// input source `src_of[s]` (fat peaks, --tissue-sub boxes), scaled by `gam` (rad/s per ppm).
pub fn expand(tab_in: &[f32], k: usize, src_of: &[usize], gam: f64) -> Vec<f32> {
    src_of.iter().flat_map(|&i| tab_in[i * k..(i + 1) * k].iter().map(move |&d| (gam * d as f64) as f32)).collect()
}

/// The cardiac segment of template time c: (q, q + 1 mod Q, weight b) on the periodic grid c_q = q period / Q.
pub fn segment_periodic(nq: usize, period: f64, c: f64) -> (usize, usize, f32) {
    assert!(nq >= 1);
    if nq == 1 {
        return (0, 0, 0.0);
    }
    let g = c.rem_euclid(period) / period * nq as f64;
    let q = (g.floor() as usize).min(nq - 1);
    (q, (q + 1) % nq, (g - q as f64).clamp(0.0, 1.0) as f32)
}

/// The four (state, weight) pairs of (r, c), state s = q K + k (the table's row order [Q][K]).
pub fn bilinear(rk: &[f64], nq: usize, period: f64, r: f64, c: f64) -> [(usize, f32); 4] {
    let nk = rk.len();
    let (j, a) = segment(rk, r);
    let j1 = (j + 1).min(nk - 1);
    let (q, q1, b) = segment_periodic(nq, period, c);
    [(q * nk + j, (1.0 - a) * (1.0 - b)), (q * nk + j1, a * (1.0 - b)), (q1 * nk + j, (1.0 - a) * b), (q1 * nk + j1, a * b)]
}

/// The kernel's bilinear formula on the host (reference / tests): base + Σ w_i col_i.
pub fn bilerp(base: &[f32], cols: [&[f32]; 4], w: [f32; 4]) -> Vec<f32> {
    base.iter().enumerate().map(|(p, &b)| b + (w[0] * cols[0][p] + w[1] * cols[1][p] + w[2] * cols[2][p] + w[3] * cols[3][p])).collect()
}

/// IEEE half → f32 (the tables are stored as f16).
pub fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = (((h >> 15) & 1) as u32, ((h >> 10) & 31) as u32, (h & 1023) as u32);
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let (mut m, mut ee) = (m, 113u32);
            while m & 1024 == 0 {
                m <<= 1;
                ee -= 1;
            }
            (s << 31) | (ee << 23) | ((m & 1023) << 13)
        }
    } else if e == 31 {
        (s << 31) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 31) | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// One state's column of the simulated tissue sources from the f16 table row of the input sources (ppm): source s
/// takes input source `src_of[s]` (fat peaks, --tissue-sub boxes), scaled by `gam` (rad/s per ppm).
pub fn column(row_f16: &[u8], src_of: &[usize], gam: f64) -> Vec<f32> {
    src_of.iter().map(|&i| (gam * f16_to_f32(u16::from_le_bytes([row_f16[2 * i], row_f16[2 * i + 1]])) as f64) as f32).collect()
}

/// CUDA: ω of the bin (one thread per source): rlerp (blood, [n][K] in r), bilerp (tissue, four state columns).
pub const KERNEL: &str = r#"
extern "C" __global__ void bilerp(long long n, const float* base, const float* t0, const float* t1, const float* t2, const float* t3,
                                  float w0, float w1, float w2, float w3, float* out)
{
    long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (p >= n) return;
    out[p] = base[p] + (w0 * t0[p] + w1 * t1[p] + w2 * t2[p] + w3 * t3[p]);
}

extern "C" __global__ void rlerp(long long n, const float* base, const float* tab, int K, int j, float a, float* out)
{
    long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (p >= n) return;
    int j1 = j + 1 < K ? j + 1 : K - 1;
    out[p] = base[p] + ((1.f - a) * tab[p * K + j] + a * tab[p * K + j1]);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k1_is_the_static_field_exactly() {
        let base: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.731).sin() * 2.0e3 - 7.0).collect();
        let tab = vec![0f32; base.len()];
        for r in [0.0, 0.3, 1.0, 1.4, -0.1] {
            let (j, a) = segment(&[0.0], r);
            assert_eq!((j, a), (0, 0.0));
            let w = lerp(&base, &tab, 1, j, a);
            assert!(w.iter().zip(&base).all(|(x, y)| x.to_bits() == y.to_bits()), "K = 1 must reproduce ω bit for bit");
        }
        // r = 0 with K > 1: Δ_0 ≡ 0, so the static field too
        let k = 6;
        let tab: Vec<f32> = (0..base.len() * k).map(|q| if q % k == 0 { 0.0 } else { (q as f32 * 0.01).cos() }).collect();
        let (j, a) = segment(&[0.0, 0.28, 0.56, 0.84, 1.12, 1.4], 0.0);
        let w = lerp(&base, &tab, k, j, a);
        assert!(w.iter().zip(&base).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn linear_between_states_and_exact_at_nodes() {
        let rk = [0.0, 0.28, 0.56, 0.84, 1.12, 1.4];
        let k = rk.len();
        // a table linear in r per source is reproduced at every r (to f32 rounding), the nodes exactly
        let n = 50;
        let slope: Vec<f32> = (0..n).map(|p| (p as f32 - 20.0) * 3.7).collect();
        let tab: Vec<f32> = (0..n).flat_map(|p| { let sp = slope[p] as f64; rk.iter().map(move |&r| (r * sp) as f32) }).collect();
        let base = vec![100f32; n];
        for (i, &r) in rk.iter().enumerate() {
            let (j, a) = segment(&rk, r);
            let w = lerp(&base, &tab, k, j, a);
            for p in 0..n {
                assert_eq!(w[p], base[p] + tab[p * k + i], "node {i}");
            }
        }
        for q in 0..=100 {
            let r = 1.4 * q as f64 / 100.0;
            let (j, a) = segment(&rk, r);
            assert!(j + 1 < k && (0.0..=1.0).contains(&a));
            let w = lerp(&base, &tab, k, j, a);
            for p in 0..n {
                let want = 100.0 + r * slope[p] as f64;
                assert!((w[p] as f64 - want).abs() < 1e-4 * (1.0 + want.abs()), "r {r}: {} vs {want}", w[p]);
            }
        }
        // clamped outside the states
        assert_eq!(segment(&rk, -0.5), (0, 0.0));
        assert_eq!(segment(&rk, 2.0), (k - 2, 1.0));
        // non-uniform states
        let (j, a) = segment(&[0.0, 0.1, 1.0], 0.55);
        assert_eq!(j, 1);
        assert!((a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bilinear_k1_q1_and_periodic() {
        let base: Vec<f32> = (0..300).map(|i| ((i as f32) * 0.37).cos() * 900.0 + 3.0).collect();
        let z = vec![0f32; base.len()];
        // K = Q = 1: one zero column, weight 1 → the static ω bit for bit
        for (r, c) in [(0.0, 0.0), (0.7, 0.31), (1.3, 5.0)] {
            let w = bilinear(&[0.0], 1, 0.85, r, c);
            assert!(w.iter().all(|e| e.0 == 0));
            let o = bilerp(&base, [&z, &z, &z, &z], w.map(|e| e.1));
            assert!(o.iter().zip(&base).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
        // a field bilinear in (r, c) on the nodes (periodic hat in c) is reproduced between them
        let (rk, nq, per) = ([0.0, 0.5, 1.0], 4usize, 0.8);
        let f = |r: f64, q: usize| -> f32 { (10.0 * r + [0.0, 3.0, -2.0, 1.0][q] + r * q as f64) as f32 };
        let cols: Vec<Vec<f32>> = (0..nq).flat_map(|q| rk.iter().map(move |&r| vec![f(r, q); 1])).collect();
        for (r, c) in [(0.25, 0.1), (0.75, 0.7), (1.0, 0.79), (0.0, 0.6)] {
            let w = bilinear(&rk, nq, per, r, c);
            let o = bilerp(&[0.0], [&cols[w[0].0], &cols[w[1].0], &cols[w[2].0], &cols[w[3].0]], w.map(|e| e.1))[0] as f64;
            let g = c / per * nq as f64;
            let (q, b) = (g.floor() as usize % nq, g - g.floor());
            let lin = |q: usize| -> f64 { let (j, a) = segment(&rk, r); (1.0 - a as f64) * f(rk[j], q) as f64 + a as f64 * f(rk[j + 1], q) as f64 };
            let want = (1.0 - b) * lin(q) + b * lin((q + 1) % nq);
            assert!((o - want).abs() < 1e-5, "({r}, {c}): {o} vs {want}");
        }
        // weights sum to 1, the wrap-around segment
        let w = bilinear(&rk, nq, per, 0.3, 0.75);
        assert!((w.iter().map(|e| e.1).sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(segment_periodic(nq, per, 0.75).1, 0);
        assert_eq!(segment_periodic(nq, per, -0.05).0, 3);
    }

    #[test]
    fn f16_decode() {
        for (h, v) in [(0x3c00u16, 1.0f32), (0xc000, -2.0), (0x0000, 0.0), (0x3555, 0.333_251_95), (0x0001, 5.960_464_5e-8), (0x7bff, 65504.0)] {
            assert_eq!(f16_to_f32(h), v, "{h:#x}");
        }
        let row: Vec<u8> = [0x3c00u16, 0xc000].iter().flat_map(|h| h.to_le_bytes()).collect();
        assert_eq!(column(&row, &[1, 0, 1], 2.0), vec![-4.0, 2.0, -4.0]);
    }

    #[test]
    fn expand_follows_the_source_map() {
        let tab_in = vec![0.0f32, 0.1, 0.2, 0.0, -0.3, -0.6];
        let t = expand(&tab_in, 3, &[1, 0, 1], 2.0);
        assert_eq!(t, vec![0.0, -0.6, -1.2, 0.0, 0.2, 0.4, 0.0, -0.6, -1.2]);
    }
}
