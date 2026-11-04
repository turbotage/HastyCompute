//! gRPC code generation without protoc: protox parses the proto into a descriptor set.
fn main() {
    println!("cargo:rerun-if-changed=proto/hasty.proto");
    let fds = protox::compile(["proto/hasty.proto"], ["proto"]).expect("parse proto/hasty.proto");
    tonic_prost_build::configure().build_server(true).build_client(true).compile_fds(fds).expect("generate gRPC code");
}
