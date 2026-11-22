//! GPU: the transfer kernels (`hasty_bfp::xfer`) equal the host codec bit for bit — device pack of rows = host
//! `encode_serial` per row, device decode = host decode — for every width, odd and even row lengths, tails, one row and
//! many rows (k-space coil rows), edge-case blocks (zeros, NaN, Inf, f32::MAX), and a large buffer (grid-stride loop).
//!   cargo test --release -p hasty-bfp --test xfer_gpu -- --test-threads=1
use hasty_bfp::xfer::{kernels, rows_ex, rows_words};
use hasty_bfp::{Bfp, VALS};
use hasty_cuda::Device;

fn xorshift(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut r = xorshift(seed);
    let mut scale = 1.0;
    let mut x: Vec<f32> = (0..n)
        .map(|i| {
            if i % 37 == 0 {
                scale = (2.0f64).powf(60.0 * r() - 30.0);
            }
            ((r() - 0.5) * 2.0 * (2.0f64).powf(-12.0 * r()) * scale) as f32
        })
        .collect();
    for (i, v) in x.iter_mut().enumerate() {
        match i % 1009 {
            3 => *v = f32::NAN,
            500 => *v = f32::INFINITY,
            700..=740 => *v = 0.0,
            800 => *v = f32::MAX,
            801 => *v = -f32::MAX,
            _ => {}
        }
    }
    x
}

/// host reference: each row its own block sequence, rows back to back
fn host_pack(fmt: Bfp, x: &[f32], n: usize, rows: usize) -> (Vec<u32>, Vec<u8>) {
    let (rw, rb) = (fmt.mant_words(n), Bfp::blocks(n));
    let mut mant = vec![0u32; rows_words(fmt, n, rows)];
    let mut ex = vec![0u8; rows_ex(n, rows)];
    for r in 0..rows {
        fmt.encode_serial(&x[r * n..(r + 1) * n], &mut mant[r * rw..(r + 1) * rw], &mut ex[r * rb..(r + 1) * rb]);
    }
    (mant, ex)
}

#[test]
fn xfer_equals_host_codec_bit_for_bit() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let mut cases = 0;
    for bits in 8..=24u32 {
        let fmt = Bfp::new(bits);
        let k = kernels(&dev.ctx, fmt).unwrap();
        for &(n, rows) in &[(1usize, 1usize), (2, 3), (31, 2), (32, 1), (33, 5), (998, 10), (3990 * 2, 10), (VALS * 4096 + 6, 1), (1 << 21, 1)] {
            let x = data(n * rows, bits as u64 * 7919 + n as u64);
            let (hm, he) = host_pack(fmt, &x, n, rows);
            let xd = st.clone_htod(&x).unwrap();
            // mantissa words and exponent bytes start poisoned: every word and byte of the rows must be written
            let mut dm = st.clone_htod(&vec![0xA5A5_A5A5u32; hm.len().max(1)]).unwrap();
            let mut de = st.clone_htod(&vec![0xA5u8; he.len().max(4)]).unwrap();
            k.pack(&st, &xd, n, rows, &mut dm, &mut de).unwrap();
            let gm = st.clone_dtoh(&dm).unwrap();
            let ge = st.clone_dtoh(&de).unwrap();
            let nbx = rows * Bfp::blocks(n);
            assert_eq!(&gm[..hm.len()], &hm[..], "bits {bits} n {n} rows {rows}: mantissa words differ");
            assert_eq!(&ge[..nbx], &he[..nbx], "bits {bits} n {n} rows {rows}: exponent bytes differ");
            // decode the host-packed bytes on the device = host decode
            let hmd = st.clone_htod(&hm).unwrap();
            let hed = st.clone_htod(&he).unwrap();
            let mut out = st.clone_htod(&vec![f32::from_bits(0x7fc0_dead); n * rows]).unwrap();
            k.unpack(&st, &hmd, &hed, n, rows, &mut out).unwrap();
            let go = st.clone_dtoh(&out).unwrap();
            let (rw, rb) = (fmt.mant_words(n), Bfp::blocks(n));
            let mut ho = vec![0f32; n * rows];
            for r in 0..rows {
                fmt.decode_serial(&hm[r * rw..(r + 1) * rw], &he[r * rb..(r + 1) * rb], &mut ho[r * n..(r + 1) * n]);
            }
            for i in 0..n * rows {
                let (a, b) = (go[i], ho[i]);
                assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan() && a.to_bits() != 0x7fc0_dead), "bits {bits} n {n} rows {rows} value {i}: device {a} host {b}");
            }
            cases += 1;
        }
    }
    println!("{cases} cases: device pack / decode = host codec bit for bit");
}
