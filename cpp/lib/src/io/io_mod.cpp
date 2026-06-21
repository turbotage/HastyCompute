module;

module hasty_io_mod;

import std;

namespace hasty {
namespace io {

std::filesystem::path module_cache_dir;
std::filesystem::path tensor_cache_dir;
std::filesystem::path data_relative_dir;
std::filesystem::path hasty_data_dir;
std::filesystem::path hasty_scripts_dir;
std::filesystem::path hasty_log_dir;
std::unordered_map<std::string, std::filesystem::path> hasty_venv_python_dirs;




}
}