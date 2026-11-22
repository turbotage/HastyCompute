//! GPU codec = CPU codec, bit for bit, for every width 8..=24 and every thread layout (needs a CUDA device).
//! Run under `compute-sanitizer --tool memcheck` for the memcheck result in research/bfp_storage.md.
use hasty_bfp::gpu::{BfpKernels, Layout};
use hasty_bfp::{Bfp, BfpBuf, VALS};
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

/// mixed-scale data plus every edge-case block; length not a multiple of 32 (tail block), even (complex layout)
fn data(bits: u32, nblk: usize) -> Vec<f32> {
    let mut r = xorshift(0x9E3779B97F4A7C15 ^ bits as u64);
    let mut x = Vec::new();
    for k in 0..nblk {
        let scale = (2.0f64).powf(80.0 * r() - 40.0);
        for _ in 0..VALS {
            let m = (2.0f64).powf(-12.0 * r());
            x.push(((r() - 0.5) * 2.0 * m * scale) as f32);
        }
        // overwrite some blocks with edge cases
        let blk = &mut x[k * VALS..];
        match k % 13 {
            1 => blk[..VALS].fill(0.0),
            2 => blk[..VALS].iter_mut().enumerate().for_each(|(i, v)| *v = f32::from_bits(i as u32 * 977) * if i % 2 == 0 { 1.0 } else { -1.0 }),
            3 => blk[..VALS].fill(-0.3),
            4 => blk[..VALS].fill(1.0 - (2.0f32).powi(-(bits as i32) - 1)),
            5 => blk[..VALS].iter_mut().enumerate().for_each(|(i, v)| *v = if i % 3 == 0 { f32::MAX } else { -f32::MAX / (1 + i) as f32 }),
            6 => blk[7] = f32::INFINITY,
            7 => blk[0] = f32::NAN,
            8 => {
                // ties: half steps at e = 1
                let step = (2.0f32).powi(2 - bits as i32);
                blk[..VALS].iter_mut().enumerate().for_each(|(i, v)| *v = if i == 0 { 1.0 } else { (i as f32 - 15.5) * step });
            }
            9 => blk[3] = -f32::NAN,
            _ => {}
        }
    }
    x.truncate(nblk * VALS - 6); // tail
    x
}

fn same_bits(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter().zip(b).position(|(p, q)| !(p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan())))
}

#[test]
fn gpu_matches_cpu_bit_for_bit() {
    let dev = Device::new(0).expect("device");
    let st = dev.default_stream();
    for bits in 8..=24u32 {
        let fmt = Bfp::new(bits);
        let mut k = BfpKernels::new(&dev, bits).expect("compile");
        // few blocks: every grid-stride loop runs several iterations
        k.set_blocks(3);
        let x = data(bits, 1300);
        let n = x.len();
        let cpu = BfpBuf::encode(fmt, &x);
        let cpu_dec = cpu.decode();
        let dx = st.clone_htod(&x).unwrap();
        let dm_cpu = st.clone_htod(&cpu.mant).unwrap();
        let de_cpu = st.clone_htod(&cpu.ex).unwrap();
        for l in Layout::ALL {
            let mut dm = st.alloc_zeros::<u32>(cpu.mant.len()).unwrap();
            let mut de = st.alloc_zeros::<u8>(cpu.ex.len()).unwrap();
            k.pack(&st, l, &dx, n, &mut dm, &mut de).unwrap();
            let gm = st.clone_dtoh(&dm).unwrap();
            let ge = st.clone_dtoh(&de).unwrap();
            let nb = Bfp::blocks(n);
            if let Some(i) = ge[..nb].iter().zip(&cpu.ex[..nb]).position(|(a, b)| a != b) {
                panic!("bits {bits} {}: exponent of block {i}: gpu {} cpu {}", l.name(), ge[i], cpu.ex[i]);
            }
            if let Some(i) = gm.iter().zip(&cpu.mant).position(|(a, b)| a != b) {
                panic!("bits {bits} {}: mantissa word {i} (block {}): gpu {:08x} cpu {:08x}", l.name(), i / bits as usize, gm[i], cpu.mant[i]);
            }
            let mut dy = st.alloc_zeros::<f32>(n).unwrap();
            k.unpack(&st, l, &dm_cpu, &de_cpu, n, &mut dy).unwrap();
            let gy = st.clone_dtoh(&dy).unwrap();
            if let Some(i) = same_bits(&gy, &cpu_dec) {
                panic!("bits {bits} {}: decoded value {i}: gpu {} cpu {}", l.name(), gy[i], cpu_dec[i]);
            }
        }
        for complex in [true, false] {
            let mut dy = st.alloc_zeros::<f32>(n).unwrap();
            k.unpack_random(&st, complex, &dm_cpu, &de_cpu, n, &mut dy).unwrap();
            let gy = st.clone_dtoh(&dy).unwrap();
            if let Some(i) = same_bits(&gy, &cpu_dec) {
                panic!("bits {bits} random access (complex {complex}): decoded value {i}: gpu {} cpu {}", gy[i], cpu_dec[i]);
            }
        }
    }
}

#[test]
fn gpu_round_trip_large() {
    // default grid on a buffer larger than one wave: pack then unpack on the GPU equals the CPU round trip
    let dev = Device::new(0).expect("device");
    let st = dev.default_stream();
    for bits in [12u32, 16, 20, 22, 24] {
        let fmt = Bfp::new(bits);
        let k = BfpKernels::new(&dev, bits).unwrap();
        let x = data(bits, 1 << 15);
        let n = x.len();
        let want = fmt.round_trip(&x);
        let dx = st.clone_htod(&x).unwrap();
        for l in Layout::ALL {
            let mut dm = st.alloc_zeros::<u32>(fmt.mant_words(n)).unwrap();
            let mut de = st.alloc_zeros::<u8>(Bfp::blocks(n).div_ceil(4) * 4).unwrap();
            let mut dy = st.alloc_zeros::<f32>(n).unwrap();
            k.pack(&st, l, &dx, n, &mut dm, &mut de).unwrap();
            k.unpack(&st, l, &dm, &de, n, &mut dy).unwrap();
            let gy = st.clone_dtoh(&dy).unwrap();
            assert_eq!(same_bits(&gy, &want), None, "bits {bits} {}", l.name());
        }
    }
}
