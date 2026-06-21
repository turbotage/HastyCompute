module;

#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

export module hasty_util_mod:io;

import std;

namespace hasty {

// No portable stdlib facility exists for "path to running executable", so
// this dispatches to the platform API and falls back to CWD-relative on failure.
std::filesystem::path exe_path()
{
#if defined(__linux__)
    std::error_code ec;
    auto path = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::filesystem::path() : path;
#elif defined(__APPLE__)
    char buf[4096];
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) != 0) return {};
    std::error_code ec;
    auto path = std::filesystem::canonical(buf, ec);
    return ec ? std::filesystem::path(buf) : path;
#elif defined(_WIN32)
    wchar_t buf[4096];
    DWORD len = ::GetModuleFileNameW(nullptr, buf, sizeof(buf) / sizeof(wchar_t));
    if (len == 0 || len == sizeof(buf) / sizeof(wchar_t)) return {};
    return std::filesystem::path(std::wstring(buf, len));
#else
    return {};
#endif
}

// Resolve p relative to the directory containing the running executable.
// Absolute paths are returned unchanged.
// Falls back to CWD-relative if the executable path can't be determined.
export std::filesystem::path resolve_exe_relative_path(const char* p)
{
    std::filesystem::path rel(p);
    if (rel.is_absolute()) return rel;
    auto exe = exe_path();
    if (exe.empty()) return rel;
    return exe.parent_path() / rel;
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