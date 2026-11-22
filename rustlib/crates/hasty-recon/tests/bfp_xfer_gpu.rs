//! GPU (short): the BFP transfer paths of the host state and the k-space staging.
//! - `SetStreamer::for_each_sink` with a BFP sink: the gradient packed on the device before the download equals the
//!   host encode of the f32 gradient bit for bit, in any set order; the f32 path still delivers exact values; the
//!   download bytes fall to the BFP size.
//! - `Stager::with_packing`: sets uploaded packed and decoded on the device equal the host decode (a `bfp` store) or
//!   the host BFP round trip of the decoded data (a zstd + q store) bit for bit; coil subsets; `upload`;
//!   `make_resident`; the upload bytes.
//!   cargo test --release -p hasty-recon --test bfp_xfer_gpu -- --nocapture --test-threads=1
use std::path::Path;
use std::sync::Arc;

use hasty_bfp::{Bfp, BfpBuf};
use hasty_cuda::{Device, C32};
use hasty_recon::bfpstate::{Buf, StackSink};
use hasty_recon::hoststack::{HostStack, SetStreamer};
use hasty_recon::kstage::Stager;
use hasty_recon::kstore::{BfpRule, KStore, StoreSpec};

fn data(n: usize, seed: u64, peak: f32) -> Vec<C32> {
    let mut s = seed | 1;
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
    };
    (0..n).map(|i| C32::new(u() + if i % 977 < 4 { peak } else { 0.0 }, u())).collect()
}

fn bits_eq(a: &[C32], b: &[C32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(p, q)| p.re.to_bits() == q.re.to_bits() && p.im.to_bits() == q.im.to_bits())
}

#[test]
fn streamer_packs_gradients_on_the_device() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let (nset, len) = (5usize, 70001usize);
    let lens = vec![len; nset];
    let mut x = HostStack::zeros(&lens);
    for t in 0..nset {
        x.chunk_mut(t).copy_from_slice(&data(len, 11 + t as u64, 300.0 * (t + 1) as f32));
        // a NaN block in one set (the policy: the block decodes as NaN, its neighbours are untouched)
        if t == 3 {
            x.chunk_mut(t)[40_000] = C32::new(f32::NAN, 0.0);
        }
    }
    let order = [3usize, 0, 4, 1, 2];
    let mut sr = SetStreamer::new(&dev.ctx, &st, len).unwrap();
    // g_t = x_t on the device (a copy), so the reference is the host encode of x_t
    let mut f = |_t: usize, xd: &cudarc::driver::CudaSlice<C32>, gd: &mut cudarc::driver::CudaSlice<C32>| -> Result<(), hasty_recon::Error> {
        st.memcpy_dtod(xd, gd)?;
        Ok(())
    };
    for bits in [12u32, 16, 20, 24] {
        let fmt = Bfp::new(bits);
        let mut g = Buf::zeros(Some(fmt), &lens);
        let before = sr.stats.2;
        sr.for_each_sink(&st, &x, &mut g, &order, &mut f).unwrap();
        let down = sr.stats.2 - before;
        let Buf::B(gs) = &g else { unreachable!() };
        for t in 0..nset {
            let h = BfpBuf::encode_c(fmt, x.chunk(t));
            assert_eq!(gs[t].mant, h.mant, "bits {bits} set {t}: mantissas differ from the host encode");
            assert_eq!(gs[t].ex[..gs[t].nblocks()], h.ex[..h.nblocks()], "bits {bits} set {t}: exponents differ");
        }
        let f32b = nset * len * 8;
        println!("bfp{bits}: device-packed gradient download = host encode (5 sets, order {order:?}); {down} bytes down vs {f32b} as f32 ({:.3})", down as f64 / f32b as f64);
        assert!(down < f32b * (bits as usize + 1) / 32 + nset * 64);
    }
    // the f32 path (StackSink) still delivers the exact values
    let mut gh = HostStack::zeros(&lens);
    sr.for_each_sink(&st, &x, &mut StackSink(&mut gh), &order, &mut f).unwrap();
    assert!(bits_eq(&gh.data, &x.data) || gh.data.iter().zip(&x.data).all(|(a, b)| a == b || (a.re.is_nan() && b.re.is_nan())));
    println!("packed slots on the device: {} bytes", sr.packed_device_bytes());
}

#[test]
fn stager_uploads_packed_and_decodes_on_the_device() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let (nvc, lens) = (5usize, vec![30001usize, 0, 4097, 20000]);
    let sigma = 0.2887f64;
    let sets: Vec<Vec<C32>> = lens.iter().enumerate().map(|(t, &m)| data(nvc * m, 77 + t as u64, 2000.0)).collect();
    let rule = BfpRule::Sigma { q: 0.25 };
    for spec in ["bfp", "zstd1+q0.25", "lz4"] {
        let ks = Arc::new(KStore::create(StoreSpec::parse(spec, Path::new("/unused")).unwrap(), nvc, lens.clone(), Some(sigma), "t").unwrap());
        for (t, y) in sets.iter().enumerate() {
            ks.put(t, y).unwrap();
        }
        // reference per set: the host decode, then (non-BFP stores) the BFP round trip of each coil row at the rule's width
        let expect = |t: usize, coils: &[usize]| -> Vec<C32> {
            let m = lens[t];
            let mut h = vec![C32::new(0.0, 0.0); coils.len() * m];
            ks.read(t, Some(coils), &mut h).unwrap();
            if ks.bfp_fmt(t).is_some() || m == 0 {
                return h;
            }
            let peak = h.iter().map(|z| z.re.abs().max(z.im.abs())).fold(0.0f32, f32::max);
            let fmt = rule.bits(peak, Some(sigma)).unwrap();
            h.chunks(m).flat_map(|r| BfpBuf::encode_c(fmt, r).decode_c()).collect()
        };
        let mut sg = Stager::new(&dev.ctx, ks.clone()).unwrap().with_packing(&dev.ctx, rule).unwrap();
        let order = [3usize, 1, 0, 2];
        let all: Vec<usize> = (0..nvc).collect();
        sg.for_each(&st, &order, None, &mut |t, d| {
            let got = st.clone_dtoh(d).unwrap();
            assert!(bits_eq(&got[..nvc * lens[t]], &expect(t, &all)), "{spec}: set {t} differs");
            Ok(())
        })
        .unwrap();
        let s = sg.stats;
        println!("{spec:>12}: packed staging = host reference; {} bytes up for {} f32 bytes ({:.3}), decode {:.3}s", s.bytes, s.f32_bytes, s.bytes as f64 / s.f32_bytes as f64, s.decode_s);
        // coil subset and single upload
        let cs = [3usize, 1];
        sg.for_each(&st, &[0, 2], Some(&cs), &mut |t, d| {
            assert!(bits_eq(&st.clone_dtoh(d).unwrap()[..2 * lens[t]], &expect(t, &cs)), "{spec}: subset set {t}");
            Ok(())
        })
        .unwrap();
        let u = sg.upload(&st, 3, None).unwrap();
        assert!(bits_eq(&st.clone_dtoh(&u).unwrap(), &expect(3, &all)), "{spec}: upload");
        // all sets resident on the device, packed
        let held = sg.make_resident(&st).unwrap();
        sg.for_each(&st, &order, None, &mut |t, d| {
            assert!(bits_eq(&st.clone_dtoh(d).unwrap()[..nvc * lens[t]], &expect(t, &all)), "{spec}: resident set {t}");
            Ok(())
        })
        .unwrap();
        let f32_all: usize = lens.iter().map(|m| nvc * m * 8).sum();
        println!("{spec:>12}: resident packed: {held} device bytes for {f32_all} f32 ({:.3})", held as f64 / f32_all as f64);
        assert!(sg.for_each(&st, &[0], Some(&cs), &mut |_, _| Ok(())).is_err(), "resident: coil subsets refused");
    }
}
