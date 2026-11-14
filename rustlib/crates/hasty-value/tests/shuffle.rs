use hasty_value::codec::{shuffle, unshuffle};

#[test]
fn shuffle_matches_reference_for_all_small_lengths_and_words() {
    let mut s = 5u64;
    for word in [1usize, 2, 3, 4, 8, 16] {
        for len in (0..300).chain([1 << 12, (1 << 12) + 7, 1 << 20]) {
            let src: Vec<u8> = (0..len)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (s >> 33) as u8
                })
                .collect();
            let n = len / word;
            let mut want = vec![0u8; len];
            for i in 0..n {
                for b in 0..word {
                    want[b * n + i] = src[i * word + b];
                }
            }
            want[n * word..].copy_from_slice(&src[n * word..]);
            let mut got = vec![0xAAu8; len];
            shuffle(&src, word, &mut got);
            assert_eq!(got, want, "shuffle word {word} len {len}");
            let mut back = vec![0x55u8; len];
            unshuffle(&got, word, &mut back);
            assert_eq!(back, src, "unshuffle word {word} len {len}");
        }
    }
}
