use cudarc::driver::{LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device};

const SRC: &str = r#"
extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"#;

#[test]
fn nvrtc_cubin_launch_roundtrip() {
    let dev = Device::new(0).expect("device");
    let f = dev
        .load_function("saxpy", SRC, "saxpy", &CompileSettings::default())
        .expect("compile");
    let stream = dev.default_stream();
    let n = 1usize << 16;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y: Vec<f32> = vec![1.0; n];
    let dx = stream.clone_htod(&x).unwrap();
    let mut dy = stream.clone_htod(&y).unwrap();
    let a = 2.0f32;
    let ni = n as i32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&a)
            .arg(&dx)
            .arg(&mut dy)
            .arg(&ni)
            .launch(LaunchConfig::for_num_elems(n as u32))
            .unwrap();
    }
    let out = stream.clone_dtoh(&dy).unwrap();
    for i in 0..n {
        assert_eq!(out[i], 2.0 * x[i] + 1.0);
    }
}

#[test]
fn nvrtc_error_reports_log() {
    let dev = Device::new(0).unwrap();
    let e = dev
        .load_module(
            "bad",
            "extern \"C\" __global__ void k( { }",
            &CompileSettings {
                no_cache: true,
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(format!("{e}").contains("bad"), "{e}");
}
