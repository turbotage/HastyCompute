module;

#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>

module hasty_viz_mod;

import std;
import hasty_tensor_mod;
import hasty_server_mod;

namespace hasty {
namespace viz {

void orthoslicer(Tensor volume, const OrthoslicerOptions& options, bool halt_for_input, bool plot_locally)
{
    if (!hasty::server::default_grpc_server_handle || !hasty::server::default_http_server) {
        hasty::server::start_default_servers();
    }

    auto uuid = hasty::server::global_generic_value_bank.push_value(GenericValue(std::move(volume)));

    // Store metadata for potential retrieval
    nlohmann::json opts_json;
    opts_json["volumename"] = options.volumename;
    if (options.comprep_config.has_value()) {
        opts_json["comprep_config"] = comprep::config_to_string(options.comprep_config.value());
    }
    std::string key(reinterpret_cast<const char*>(uuid.data()), 16);
    hasty::server::global_generic_value_bank.write_metadata(key, opts_json.dump());

    // Build 32-char hex UUID string
    static constexpr char hex_chars[] = "0123456789abcdef";
    std::string uuid_hex(32, '0');
    for (int i = 0; i < 16; ++i) {
        uuid_hex[i * 2]     = hex_chars[(uuid[i] >> 4) & 0xF];
        uuid_hex[i * 2 + 1] = hex_chars[uuid[i] & 0xF];
    }

    int http_port = hasty::server::default_http_server->port();

    std::string url = "https://localhost:" + std::to_string(http_port)
                    + "/orthoslicer/index.html?uuid=" + uuid_hex;

    if (options.comprep_config.has_value()) {
        std::string cfg_str = comprep::config_to_string(options.comprep_config.value());
        std::string encoded;
        encoded.reserve(cfg_str.size() * 3);
        for (unsigned char c : cfg_str) {
            if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
                encoded += static_cast<char>(c);
            } else {
                char buf[4];
                std::snprintf(buf, sizeof(buf), "%%%02X", c);
                encoded += buf;
            }
        }
        url += "&config=" + encoded;
    }

    if (plot_locally) {
        std::system(("xdg-open \"" + url + "\" &").c_str());
    } else {
        std::cout << "Orthoslicer URL: " << url << std::endl;
    }
    if (halt_for_input) {
        std::cout << "Press Enter to continue..." << std::flush;
        std::cin.get();
    }
}



}
}
