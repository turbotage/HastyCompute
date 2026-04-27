module hasty_server_mod;


namespace hasty {
namespace server {

SPtr<GrpcServerHandle>  default_grpc_server_handle;
UPtr<HttpServer>        default_http_server;

void start_default_servers(
    Opt<Vec<Pair<std::string, CommandFn>>> optional_extra_commands,
    u16 grpc_port = 50051,
    u16 http_port = 8080
) {
    if (default_grpc_server_handle || default_http_server)
        throw std::runtime_error("Default servers already running");

    if (optional_extra_commands) {
        global_command_registry.register_commands(*optional_extra_commands);
    }

    default_grpc_server_handle = std::make_shared<GrpcServerHandle>(
        start_grpc_server(global_generic_value_bank, global_command_registry, "0.0.0.0:" + std::to_string(grpc_port)));

    default_http_server = std::make_unique<HttpServer>(
        http_port,
        "/home/turbotage/Documents/GitHub/HastyCompute/plotting_website",
        default_grpc_server_handle);
    default_http_server->start();
}

}
}