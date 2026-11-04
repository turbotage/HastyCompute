//! Show an array file in the browser viewer.
//!
//!   hasty-show <file.npy | file.hv> [--slice '[..., 0]'] [--name name]
//!
//! Publishes to the running hasty server (HASTY_SERVER, or one on this machine), or starts an embedded one and keeps
//! it alive until Ctrl-C. Fortran-ordered npy files (hasty's x-fastest layout) are shown with axes (…, z, y, x).
use std::path::Path;

use hasty_value::{npy, slice, Value};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let Some(file) = args.first().filter(|a| !a.starts_with("--")) else {
        eprintln!("usage: hasty-show <file.npy|file.hv> [--slice path] [--name name]");
        std::process::exit(2);
    };
    let p = Path::new(file);
    let (mut v, fortran) = if file.ends_with(".npy") {
        let (t, f) = npy::read(p).unwrap_or_else(|e| panic!("{file}: {e}"));
        (Value::Tensor(t), f)
    } else {
        (hasty_value::read(&mut std::io::BufReader::new(std::fs::File::open(p).unwrap_or_else(|e| panic!("{file}: {e}")))).unwrap_or_else(|e| panic!("{file}: {e}")), false)
    };
    if let Some(s) = opt("--slice") {
        v = slice::get(&v, &s).unwrap_or_else(|e| panic!("--slice {s}: {e}"));
    }
    let name = opt("--name").unwrap_or_else(|| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
    let meta = serde_json::json!({ "source": p.canonicalize().map(|c| c.display().to_string()).unwrap_or_default(), "fortran_file": fortran });
    let url = hasty_server::viz::show_with(&name, v, meta, "orthoslicer").unwrap_or_else(|e| panic!("show: {e}"));
    println!("{url}");
    hasty_server::viz::wait();
}
