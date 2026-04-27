#include <configure_file_settings.hpp>

import std;
import hasty_util_mod;
import hasty_server_mod;
import hasty_generic_value_mod;


int main() {

    auto grpc_server_handle = hasty::start_grpc_server(
        hasty::global_generic_value_bank,
        hasty::global_command_registry,
        "0.0.0.0:50051"
    );

    auto http_server = hasty::HttpServer(
        8080,
        "/home/turbotage/Documents/GitHub/HastyCompute/plotting_website",
        grpc_server_handle
    );

    http_server.start();

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cin >> std::ws;
        if (std::cin.peek() == 'q') {
            http_server.stop();
            grpc_server_handle.shutdown();
            break;
        }
    }

    return 0;
}