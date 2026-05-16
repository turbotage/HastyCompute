module;

#include "configure_file_settings.hpp"
#include <nlohmann/json.hpp>
#include <unistd.h>   // readlink — no C++ standard equivalent for /proc/self/exe

export module hasty_python_mod;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_generic_value_mod;
import hasty_server_mod;
import hasty_io_mod;

export import :nifti_and_registration;

namespace hasty {
namespace python {

// ─── Types ───────────────────────────────────────────────────────────────────

export struct ScriptResult {
    int                      exit_code;
    std::vector<std::string> output_uuids;  // hex strings — pre-registered output slot keys
    std::string              stderr_text;
};

// ─── Internal helpers ─────────────────────────────────────────────────────────

namespace {

std::string exe_dir() {
    char buf[4096] = {};
    ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return ".";
    std::string path(buf, static_cast<size_t>(len));
    auto pos = path.rfind('/');
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

std::string resolve_path(const char* p) {
    if (p[0] == '/') return p;
    return exe_dir() + "/" + p;
}

int parse_port(const std::string& addr) {
    auto pos = addr.rfind(':');
    if (pos == std::string::npos) return 50051;
    return std::stoi(addr.substr(pos + 1));
}

std::string trim(std::string s) {
    auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Unique temp path base using process id + monotonic counter.
std::filesystem::path tmp_base() {
    static std::atomic<int> n{0};
    auto name = "hasty_" + std::to_string(::getpid())
              + "_" + std::to_string(n.fetch_add(1, std::memory_order_relaxed));
    return std::filesystem::temp_directory_path() / name;
}

} // namespace

// ─── Exported path helpers ────────────────────────────────────────────────────

export std::string venv_python() { return resolve_path(HASTY_VENV_PYTHON); }
export std::string scripts_dir() { return resolve_path(HASTY_SCRIPTS_DIR); }

// ─── UUID helpers ─────────────────────────────────────────────────────────────

export std::string uuid_to_hex(const std::array<u8, 16>& uuid) {
    static constexpr char hex_chars[] = "0123456789abcdef";
    std::string out(32, '0');
    for (int i = 0; i < 16; ++i) {
        out[i * 2]     = hex_chars[(uuid[i] >> 4) & 0xF];
        out[i * 2 + 1] = hex_chars[uuid[i] & 0xF];
    }
    return out;
}

export std::array<u8, 16> hex_to_uuid_array(const std::string& hex) {
    if (hex.size() != 32)
        throw std::runtime_error("[python] UUID hex string must be 32 chars");
    std::array<u8, 16> out{};
    for (int i = 0; i < 16; ++i) {
        auto nibble = [](char c) -> u8 {
            if (c >= '0' && c <= '9') return static_cast<u8>(c - '0');
            if (c >= 'a' && c <= 'f') return static_cast<u8>(c - 'a' + 10);
            if (c >= 'A' && c <= 'F') return static_cast<u8>(c - 'A' + 10);
            throw std::runtime_error("[python] Invalid hex char");
        };
        out[i] = static_cast<u8>((nibble(hex[i*2]) << 4) | nibble(hex[i*2+1]));
    }
    return out;
}

// ─── run_script ──────────────────────────────────────────────────────────────

// num_outputs: number of output values the script will produce.
//   For each, a UUID slot is pre-registered in the global bank and passed as
//   --output0=HEX (or --output=HEX for exactly one) so the script can write
//   results via gRPC write_value rather than printing UUIDs to stdout.
//   result.output_uuids is populated with those pre-known hex strings.
export ScriptResult run_script(
    const std::string&              script_path,
    const std::vector<std::string>& args        = {},
    int                             num_outputs = 0,
    bool                            debug       = false,
    int                             debug_port  = 5678)
{
    if (!server::default_grpc_server_handle || !server::default_http_server)
        server::start_default_servers();

    int grpc_port = parse_port(server::default_grpc_server_handle->address());

    // Pre-register output slots so Python can write results via gRPC.
    std::vector<std::array<u8, 16>> output_uuids_raw;
    std::vector<std::string>        output_hex;
    output_uuids_raw.reserve(num_outputs);
    output_hex.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
        auto uuid = hasty::generate_uuid();
        hasty::server::global_generic_value_bank.push_value_with_key(uuid, hasty::GenericValue{});
        output_uuids_raw.push_back(uuid);
        output_hex.push_back(uuid_to_hex(uuid));
    }

    auto base      = tmp_base();
    auto err_path  = base.string() + "_err.txt";
    auto exit_path = base.string() + "_exit.txt";

    std::string cmd = "\"" + venv_python() + "\""
                    + " \"" + script_path + "\""
                    + " --grpc-port=" + std::to_string(grpc_port);
    if (debug)
        cmd += " --debug-port=" + std::to_string(debug_port);
    for (const auto& a : args)
        cmd += " " + a;
    if (num_outputs == 1) {
        cmd += " --output=" + output_hex[0];
    } else {
        for (int i = 0; i < num_outputs; ++i)
            cmd += " --output" + std::to_string(i) + "=" + output_hex[i];
    }
    // Discard stdout; capture stderr and exit code.
    cmd += " >/dev/null 2>\"" + err_path + "\""
         + "; echo $? >\"" + exit_path + "\"";

    if (debug) {
        std::cout << "[python::run_script] Launching with debugpy on port " << debug_port
                  << "\n  Make sure 'Python: Listen for subprocess (debugpy)' is running in VS Code.\n"
                  << std::flush;
    }

    std::system(cmd.c_str());

    ScriptResult result;

    // Read exit code written by shell.
    if (std::ifstream ef{exit_path}; ef)
        ef >> result.exit_code;
    else
        result.exit_code = -1;

    // Read stderr.
    if (std::ifstream ef{err_path}; ef)
        result.stderr_text.assign(std::istreambuf_iterator<char>(ef), {});

    // Cleanup temp files.
    for (auto& p : {err_path, exit_path})
        std::filesystem::remove(p);

    if (result.exit_code != 0) {
        std::cerr << "[python::run_script] Script exited " << result.exit_code
                  << "\n--- stderr ---\n" << result.stderr_text << "--- end ---\n";
        // Clean up pre-registered slots that Python never filled.
        for (const auto& uuid : output_uuids_raw) {
            std::string key(reinterpret_cast<const char*>(uuid.data()), 16);
            hasty::server::global_generic_value_bank.delete_value(key);
        }
    } else {
        result.output_uuids = std::move(output_hex);
    }

    return result;
}

} // namespace python
} // namespace hasty
