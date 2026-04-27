module;

export module hasty_server_mod:grpc_server;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_generic_value_mod;
import hasty_threading_mod;

import :cmd_registry;
import :generic_value_bank;


namespace hasty {

export class GrpcServerHandle {
public:
    GrpcServerHandle() = default;
    ~GrpcServerHandle();
    GrpcServerHandle(GrpcServerHandle&&) noexcept;
    GrpcServerHandle& operator=(GrpcServerHandle&&) noexcept;

    void wait();
    void shutdown();

    // Address the server is listening on (e.g. "0.0.0.0:50051").
    // Used by HttpServer to create a loopback channel.
    const std::string& address() const;

    struct Impl;
private:
    std::unique_ptr<Impl> _impl;

    friend GrpcServerHandle start_grpc_server(
        GenericValueBank&, CommandRegistry&, const std::string&);
};


export GrpcServerHandle start_grpc_server(
    GenericValueBank& bank,
    CommandRegistry& registry,
    const std::string& address = "0.0.0.0:50051");


}