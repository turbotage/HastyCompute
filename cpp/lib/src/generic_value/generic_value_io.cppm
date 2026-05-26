module;

#include <nlohmann/json.hpp>

export module hasty_generic_value_mod:io;

import std;
import hasty_util_mod;
import hasty_threading_mod;
import hasty_io_mod_blosc;
import :generic_value;

namespace hasty {
namespace io {

// ── Blosc-backed GenericValue serialisation ───────────────────────────────────
//
// Wire format: same as io_blosc compress_stream / decompress_stream.
// GenericValue::serialize / deserialize handle the payload.

namespace blosc {

constexpr i64 GVFILE_CHUNK_SIZE = 64 * 1024 * 1024; // 64MB


export void write_generic_value(
    const GenericValue&      value,
    std::ostream&            os,
    i32                      compression_level    = 5,
    BloscCompressionStrategy compression_strategy = BloscCompressionStrategy::LZ4,
    BloscShuffleFilter       shuffle_filter       = BloscShuffleFilter::BitShuffle)
{
    threadsafe_stream stream;
    std::exception_ptr serializer_exc;

    std::thread serializer([&]() {
        try {
            GenericValue::serialize(value, stream, GVFILE_CHUNK_SIZE);
            stream.set_finished();
        } catch (...) {
            serializer_exc = std::current_exception();
            stream.set_finished();
        }
    });

    compress_stream(stream, os, compression_level, GVFILE_CHUNK_SIZE, compression_strategy, shuffle_filter);
    serializer.join();

    if (serializer_exc) std::rethrow_exception(serializer_exc);
}

export void write_generic_value(
    const GenericValue&      value,
    const std::filesystem::path& path,
    i32                      compression_level    = 5,
    BloscCompressionStrategy compression_strategy = BloscCompressionStrategy::LZ4,
    BloscShuffleFilter       shuffle_filter       = BloscShuffleFilter::BitShuffle)
{
    if (auto parent = path.parent_path(); !parent.empty())
        std::filesystem::create_directories(parent);
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs)
        throw std::runtime_error("[gv blosc write] cannot open file: " + path.string());
    write_generic_value(value, ofs, compression_level, compression_strategy, shuffle_filter);
}

export void write_generic_value(
    const GenericValue&      value,
    const std::string&       filename,
    i32                      compression_level    = 5,
    BloscCompressionStrategy compression_strategy = BloscCompressionStrategy::LZ4,
    BloscShuffleFilter       shuffle_filter       = BloscShuffleFilter::BitShuffle)
{
    write_generic_value(value, std::filesystem::path(filename),
                        compression_level, compression_strategy, shuffle_filter);
}

export GenericValue read_generic_value(std::istream& is)
{
    threadsafe_stream stream;
    std::exception_ptr producer_exc;

    std::thread producer([&]() {
        try {
            decompress_stream(is, stream);
        } catch (...) {
            producer_exc = std::current_exception();
            stream.set_finished();
        }
    });

    GenericValue result = [&]() {
        try {
            return GenericValue::deserialize(stream, GVFILE_CHUNK_SIZE);
        } catch (...) {
            producer.join();
            throw;
        }
    }();

    producer.join();
    if (producer_exc) std::rethrow_exception(producer_exc);
    return result;
}

export GenericValue read_generic_value(const std::filesystem::path& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("[gv blosc read] cannot open file: " + path.string());
    return read_generic_value(ifs);
}

export GenericValue read_generic_value(const std::string& filename)
{
    return read_generic_value(std::filesystem::path(filename));
}

} // namespace blosc


// ── HDF5-backed GenericValue serialisation ────────────────────────────────────
//
// Implementations live in io_hdf5_impl.cpp (module hasty_generic_value_io_mod).

namespace hdf5 {

export GenericValue read_generic_value(const std::string& filename,
                                        bool ignore_nongv_entries = true);

export GenericValue read_generic_value_entry(const std::string& filename,
                                              const std::string& entry_name,
                                              bool ignore_nongv_entries = true);

export void write_generic_value(const GenericValue& value,
                                 const std::string& filename);

export void write_generic_value_entry(const GenericValue& value,
                                       const std::string& filename,
                                       const std::string& entry_name);

// Full parse: returns (GenericValue tree, json metadata, unparseable paths).
export auto parse_hdf5_as_generic_value(const std::string& filename,
                                          bool ignore_nongv_entries = true)
    -> Tup<GenericValue, nlohmann::json, std::vector<std::string>>;

} // namespace hdf5

} // namespace io
} // namespace hasty
