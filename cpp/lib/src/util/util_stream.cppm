module;

#include "configure_file_settings.hpp"

export module hasty_util_mod:stream;

import std;
import :io;

namespace hasty {

export class OStreamInterface {
public:
    virtual ~OStreamInterface() = default;
    virtual OStreamInterface& operator<<(const std::string& str) = 0;
    virtual OStreamInterface& operator<<(std::string&& str) = 0;
    virtual OStreamInterface& operator<<(std::string_view str_view) = 0;
    virtual OStreamInterface& operator<<(const std::vector<std::uint8_t>& data) = 0;
    virtual OStreamInterface& operator<<(std::vector<std::uint8_t>&& data) = 0;
};

export class FlushingFileStream : public OStreamInterface {
public:

    explicit FlushingFileStream(std::ofstream&& stream)
        : _stream(std::move(stream))
    {
        if (!_stream) {
            throw std::runtime_error("Failed to move-construct FlushingFileStream");
        }
    }

    explicit FlushingFileStream(std::filesystem::path filepath, bool append = false)
        : _stream(filepath, append ? std::ios::app : std::ios::trunc)
    {
        if (!_stream) {
            throw std::runtime_error("Failed to open file: " + filepath.string());
        }
    }

    void write(const std::vector<std::uint8_t>& data) {
        std::scoped_lock lock(_mtx);
        _stream.write(reinterpret_cast<const char*>(data.data()), data.size());
        if (!_stream) {
            throw std::runtime_error("Failed to write to stream");
        }
        _stream.flush();
    }

    void write(std::vector<std::uint8_t>&& data) {
        auto captured_data = std::move(data);
        write(captured_data);
    }

    void write(const std::string& str) {
        write(std::vector<std::uint8_t>(str.begin(), str.end()));
    }

    void write(std::string&& str) {
        auto captured_str = std::move(str);
        write(captured_str);
    }

    void write(std::string_view str_view) {
        write(std::vector<std::uint8_t>(str_view.begin(), str_view.end()));
    }

    operator std::ofstream&() { return _stream; }
    operator const std::ofstream&() const { return _stream; }

    FlushingFileStream& operator<<(const std::string& str) override {
        write(str);
        return *this;
    }
    FlushingFileStream& operator<<(std::string&& str) override {
        write(std::move(str));
        return *this;
    }
    FlushingFileStream& operator<<(std::string_view str_view) override {
        write(str_view);
        return *this;
    }
    FlushingFileStream& operator<<(const std::vector<std::uint8_t>& data) override {
        write(data);
        return *this;
    }
    FlushingFileStream& operator<<(std::vector<std::uint8_t>&& data) override {
        write(std::move(data));
        return *this;
    }

private:
    std::ofstream _stream;
    std::mutex _mtx;
};

export using LogStream = FlushingFileStream;

export std::string log_dir() { return hasty::resolve_exe_relative_path(HASTY_LOG_DIR).string(); }

}
