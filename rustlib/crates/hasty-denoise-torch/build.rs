//! Compile csrc/shim.cpp against libtorch. libtorch is located through `LIBTORCH` (a libtorch / torch package dir
//! with include/ and lib/) or, by default, the torch package of `HASTY_PYTHON` (default: rustlib/.venv/bin/python).
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-changed=csrc/shim.cpp");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=HASTY_PYTHON");
    let (root, abi) = match std::env::var_os("LIBTORCH") {
        Some(p) => (PathBuf::from(p), "1".to_string()),
        None => {
            let py = std::env::var_os("HASTY_PYTHON").map(PathBuf::from).unwrap_or_else(|| manifest.join("../../.venv/bin/python"));
            let o = Command::new(&py)
                .args(["-I", "-c", "import torch, os; print(os.path.dirname(torch.__file__)); print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"])
                .output()
                .unwrap_or_else(|e| panic!("running {}: {e} (set LIBTORCH or HASTY_PYTHON)", py.display()));
            let s = String::from_utf8(o.stdout).unwrap();
            let mut l = s.lines();
            (PathBuf::from(l.next().expect("torch not importable")), l.next().unwrap_or("1").to_string())
        }
    };
    let lib = root.join("lib");
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .file("csrc/shim.cpp")
        .include(root.join("include"))
        .include(root.join("include/torch/csrc/api/include"))
        .define("_GLIBCXX_USE_CXX11_ABI", abi.as_str())
        .flag("-w")
        .opt_level(2)
        .compile("hasty_torch_shim");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    // CUDA backend libraries are loaded for their registrations even though no symbol is referenced directly
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    for l in ["torch_cuda", "c10_cuda"] {
        if lib.join(format!("lib{l}.so")).exists() {
            println!("cargo:rustc-link-lib=dylib={l}");
        }
    }
    println!("cargo:rustc-link-arg=-Wl,--as-needed");
    for l in ["torch", "torch_cpu", "c10"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
