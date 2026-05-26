module;

#include <unistd.h>

export module hasty_util_mod:io;

import std;

namespace hasty {

// Resolve p relative to the directory containing the running executable.
// Absolute paths (starting with '/') are returned unchanged.
// Falls back to CWD-relative if /proc/self/exe is unreadable.
export std::filesystem::path resolve_exe_relative_path(const char* p)
{
    if (p[0] == '/') return std::filesystem::path(p);
    char buf[4096] = {};
    ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return std::filesystem::path(p);
    return std::filesystem::path(std::string(buf, static_cast<std::size_t>(len)))
               .parent_path() / p;
}

export bool remove_file_if_exists(const std::filesystem::path& path)
{
    std::error_code ec;
    // Check existence first (optional but explicit)
    if (!std::filesystem::exists(path, ec)) {
        return false; // file does not exist
    }
    // Try to remove it
    bool removed = std::filesystem::remove(path, ec);
    if (ec) {
        // error occurred during removal
        std::cerr << "Failed to remove file: " << ec.message() << "\n";
        return false;
    }
    return removed; // true if file was removed, false if it didn't exist
}

}