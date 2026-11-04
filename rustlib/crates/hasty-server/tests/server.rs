use std::sync::Arc;

use hasty_server::client::Client;
use hasty_server::server::{spawn, Config};
use hasty_server::{repo_path, Bank, Registry};
use hasty_value::{slice, Options, Tensor, Value};
use num_complex::Complex32;

fn volume() -> Tensor {
    let (nz, ny, nx) = (9, 20, 31);
    let v: Vec<Complex32> = (0..nz * ny * nx).map(|i| Complex32::from_polar(1.0 + (i % nx) as f32 / nx as f32, i as f32 * 0.01)).collect();
    Tensor::from_slice(&[nz, ny, nx], &v).unwrap()
}

#[test]
fn tls_grpc_round_trip_commands_and_events() {
    let cert_dir = repo_path("rustlib/target/test-certs");
    let cfg = Config { bind: "127.0.0.1:0".parse().unwrap(), tls: true, cert_dir: cert_dir.clone(), web_dir: repo_path("plotting_website"), extra_names: vec![] };
    let bank = Arc::new(Bank::new());
    let h = spawn(cfg, bank.clone(), Arc::new(Registry::base())).unwrap();
    let mut events = bank.subscribe();
    let ca = std::fs::read_to_string(cert_dir.join("ca.pem")).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let mut c = Client::connect(&format!("https://localhost:{}", h.addr.port()), Some(&ca)).await.unwrap();
        let vol = volume();
        let v = Value::Dict(vec![("vol".into(), Value::Tensor(vol.clone())), ("note".into(), "hi".into())]);
        let id = c.put("test", &v, r#"{"axes":["z","y","x"]}"#, &Options { block_size: 4096, ..Options::default() }).await.unwrap();
        for codec in ["lz4", "zstd", "none"] {
            assert_eq!(c.get(&id, "", codec).await.unwrap(), v, "{codec}");
        }
        let s = c.get(&id, r#"["vol"][4, ::2, -3:]"#, "lz4").await.unwrap();
        assert_eq!(s, slice::get(&v, r#"["vol"][4, ::2, -3:]"#).unwrap());
        let list = c.list().await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "test");
        assert!(list[0].summary.contains("complex64"));

        let vid = c.put("vol", &Value::Tensor(vol.clone()), "", &Options::default()).await.unwrap();
        let (q, msg) = c.execute("comprep_u16", r#"{"mode":"abs","focus":0.9,"right_mode":"log","right_logc":20}"#, &[vid]).await.unwrap();
        let qv = c.get(&q[0], "", "lz4").await.unwrap();
        assert_eq!(qv.as_tensor().unwrap().dtype(), hasty_value::DType::U16);
        let cfg: serde_json::Value = serde_json::from_str(&msg).unwrap();
        assert_eq!(cfg["focus"].as_f64(), Some(0.9));
        assert_eq!(cfg["right_mode"].as_str(), Some("log"));
        let (f, _) = c.execute("fft", r#"{"axes":[-1,-2]}"#, &[vid]).await.unwrap();
        let (b, _) = c.execute("ifft", r#"{"axes":[-1,-2]}"#, &[f[0]]).await.unwrap();
        let back = c.get(&b[0], "", "lz4").await.unwrap();
        let (x, y) = (vol.as_slice::<Complex32>().unwrap(), back.as_tensor().unwrap().as_slice::<Complex32>().unwrap().to_vec());
        let err = x.iter().zip(&y).map(|(a, b)| (a - b).norm()).fold(0f32, f32::max);
        assert!(err < 1e-5, "fft round trip {err}");
        let (d, _) = c.execute("sub", "", &[vid, b[0]]).await.unwrap();
        let dz = c.get(&d[0], "", "lz4").await.unwrap();
        assert!(dz.as_tensor().unwrap().as_slice::<Complex32>().unwrap().iter().all(|z| z.norm() < 1e-5));
        assert!(c.execute("nope", "", &[]).await.is_err());

        // partial write through the bank, then delete
        bank.write(&vid, "[0, 0, :]", Value::Tensor(Tensor::scalar(Complex32::new(7.0, 0.0)))).unwrap();
        let row = c.get(&vid, "[0, 0, :]", "lz4").await.unwrap();
        assert!(row.as_tensor().unwrap().as_slice::<Complex32>().unwrap().iter().all(|z| z.re == 7.0));
        c.show(&vid, "orthoslicer").await.unwrap();
        c.delete(&id).await.unwrap();
        assert!(c.get(&id, "", "lz4").await.is_err());
    });
    let mut kinds = vec![];
    while let Ok(e) = events.try_recv() {
        kinds.push(e.kind);
    }
    for k in ["put", "update", "show", "delete"] {
        assert!(kinds.iter().any(|x| x == k), "missing {k} event in {kinds:?}");
    }
    eprintln!("server on {:?}: urls {:?}", h.addr, h.urls);
    h.shutdown();
}
