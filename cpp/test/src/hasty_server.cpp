#include <configure_file_settings.hpp>

import std;
import hasty_util_mod;
import hasty_server_mod;
import hasty_generic_value_mod;


int main() {
    hasty::server::start_default_servers();

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cin >> std::ws;
        if (std::cin.peek() == 'q') {
            hasty::server::default_http_server->stop();
            hasty::server::default_grpc_server_handle->shutdown();
            break;
        }
    }

    return 0;
}