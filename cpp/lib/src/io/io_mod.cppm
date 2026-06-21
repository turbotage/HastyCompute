module;

#include <configure_file_settings.hpp>

export module hasty_io_mod;

import std;
import hasty_util_mod;

namespace hasty {
namespace io {

    export extern std::filesystem::path module_cache_dir;
    export extern std::filesystem::path tensor_cache_dir;
    export extern std::filesystem::path data_relative_dir;
    export extern std::filesystem::path hasty_data_dir;
    export extern std::filesystem::path hasty_scripts_dir;
    export extern std::filesystem::path hasty_log_dir;
    export extern std::unordered_map<std::string, std::filesystem::path> hasty_venv_python_dirs;

    export void setup_default_dirs() {
        module_cache_dir  = hasty::resolve_exe_relative_path(MODULE_CACHE_RELATIVE_PATH);
        tensor_cache_dir  = hasty::resolve_exe_relative_path(TENSOR_CACHE_RELATIVE_PATH);
        data_relative_dir = hasty::resolve_exe_relative_path(DATA_RELATIVE_PATH);
        hasty_data_dir    = std::filesystem::path(HASTY_DATA_DIR);
        hasty_scripts_dir = hasty::resolve_exe_relative_path(HASTY_SCRIPTS_DIR);
        hasty_log_dir     = hasty::resolve_exe_relative_path(HASTY_LOG_DIR);
        hasty_venv_python_dirs["default"] = hasty::resolve_exe_relative_path(HASTY_VENV_PYTHON);

        std::filesystem::create_directories(module_cache_dir);
        std::filesystem::create_directories(tensor_cache_dir);
        std::filesystem::create_directories(hasty_log_dir);

        // Tensor cache files are transient — valid only within one run.
        // Remove any stale files left by previous crashes or kills.
        std::error_code ec;
        for (auto& entry : std::filesystem::directory_iterator(tensor_cache_dir, ec))
            std::filesystem::remove(entry.path(), ec);
    }

}
}