// CPU backend: the same per-node C core as the CUDA kernels (kernels/lbm_core.h), compiled with the system C compiler.
fn main() {
    println!("cargo:rerun-if-changed=kernels/lbm_core.h");
    println!("cargo:rerun-if-changed=kernels/lbm_cpu.c");
    cc::Build::new().file("kernels/lbm_cpu.c").include("kernels").opt_level(3).flag_if_supported("-fno-math-errno").warnings(false).compile("hasty_lbm_cpu");
}
