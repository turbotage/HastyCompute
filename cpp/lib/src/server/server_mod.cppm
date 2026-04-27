module;

export module hasty_server_mod;

export import :cmd_registry;
export import :generic_value_bank;
export import :grpc_server;
export import :http_server;


namespace hasty {
namespace server {

export extern SPtr<GrpcServerHandle>  default_grpc_server_handle;
export extern UPtr<HttpServer>        default_http_server;    

export void start_default_servers(std::optional<std::pair<std::string, CommandFn>> optional_extra_commands = std::nullopt);

}
}