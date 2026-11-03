//! Choice of mixed-radix factorisation for a single-pass shared-memory Stockham FFT.

/// Radices with an emitter in `radix.rs`.
pub const RADICES: &[usize] = &[2, 3, 4, 5, 7, 8, 11, 13];

/// Factor `n` into radices from [`RADICES`], fewest passes first (large power-of-two radices), then odd primes.
/// Returns `None` if `n` has a prime factor > 13 (needs Bluestein, not implemented yet).
pub fn factorize(mut n: usize) -> Option<Vec<usize>> {
    if n < 2 {
        return None;
    }
    let mut out = Vec::new();
    let mut a = 0usize;
    while n % 2 == 0 {
        n /= 2;
        a += 1;
    }
    match a % 3 {
        0 => out.extend(std::iter::repeat(8).take(a / 3)),
        1 if a >= 4 => {
            out.extend(std::iter::repeat(8).take(a / 3 - 1));
            out.extend([4, 4]);
        }
        1 => out.push(2),
        _ => {
            out.extend(std::iter::repeat(8).take(a / 3));
            out.push(4);
        }
    }
    for p in [3usize, 5, 7, 11, 13] {
        while n % p == 0 {
            n /= p;
            out.push(p);
        }
    }
    if n != 1 {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn products_match() {
        for n in 2..5000usize {
            if let Some(f) = factorize(n) {
                assert_eq!(f.iter().product::<usize>(), n, "n={n} f={f:?}");
                assert!(f.iter().all(|r| RADICES.contains(r)));
            }
        }
        assert_eq!(factorize(1024).unwrap(), vec![8, 8, 4, 4]);
        assert!(factorize(17).is_none());
    }
}
