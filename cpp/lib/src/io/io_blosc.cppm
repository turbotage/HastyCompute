module;

#include <blosc2.h>

export module hasty_io_mod:blosc;

import std;
import hasty_util_mod;
import hasty_threading_mod;
import hasty_generic_value_mod;

namespace hasty {
namespace io {
namespace blosc {

constexpr i64 GVFILE_CHUNK_SIZE = 64 * 1024 * 1024; // 64MB

export enum struct BloscCompressionStrategy : u8 {
    LZ4 = 0,
    LZ4_HC = 1,
    ZSTD = 2
};

export enum struct BloscShuffleFilter : u8 {
    NoShuffle  = BLOSC_NOFILTER,
    Shuffle    = BLOSC_SHUFFLE,
    BitShuffle = BLOSC_BITSHUFFLE,
};

// blosc2_init/destroy mutate a global thread pool and must be balanced.
// We use a reference-counted guard: the first user calls blosc2_init(),
// the last one out calls blosc2_destroy().  All per-call compression work
// uses the thread-safe context API, so concurrent operations are safe while
// the guard is held.
namespace {

    std::mutex  g_blosc_mutex;
    int         g_blosc_refcount = 0;

} // namespace

// This guard is used internally by the read/write functions, it makes sure blosc2_init is 
// called before any compression/decompression and that blosc2_destroy is called at the end.
// It also server concurrency purpuses, allowing multiple read/write operations to run concurrently
// without multiple initializations of the blosc library.
// To disable unecessary initialization/destruction pairs, users can create a single BloscGuard and keep it alive
// for the duration of multiple read/write calls.
export struct BloscGuard {
    BloscGuard() {
        std::lock_guard lock(g_blosc_mutex);
        g_blosc_refcount += 1;
        if (g_blosc_refcount == 1)
            blosc2_init();
    }
    ~BloscGuard() {
        std::lock_guard lock(g_blosc_mutex);
        g_blosc_refcount -= 1;
        if (g_blosc_refcount == 0)
            blosc2_destroy();
    }
    BloscGuard(const BloscGuard&) = delete;
    BloscGuard& operator=(const BloscGuard&) = delete;
};


// ---------------------------------------------------------------------------
// Core stream-based write.
// os must support seekp (std::ofstream, std::fstream, std::stringstream all do).
// ---------------------------------------------------------------------------
export void write_generic_value(
    const GenericValue& value,
    std::ostream& os,
    i32 compression_level = 5,
    BloscCompressionStrategy compression_strategy = BloscCompressionStrategy::LZ4,
    BloscShuffleFilter shuffle_filter = BloscShuffleFilter::BitShuffle)
{
    BloscGuard blosc_guard;

    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    switch (compression_strategy) {
    case BloscCompressionStrategy::LZ4:    cparams.compcode = BLOSC_LZ4;   break;
    case BloscCompressionStrategy::LZ4_HC: cparams.compcode = BLOSC_LZ4HC; break;
    case BloscCompressionStrategy::ZSTD:   cparams.compcode = BLOSC_ZSTD;  break;
    }
    cparams.clevel   = compression_level;
    cparams.typesize = 1;
    // The shuffle filter occupies the last slot of the filters pipeline.
    // BLOSC2_CPARAMS_DEFAULTS already sets filters[5]=BLOSC_SHUFFLE; override it.
    cparams.filters[BLOSC2_MAX_FILTERS - 1] = static_cast<uint8_t>(shuffle_filter);
    // BLOSC2_CPARAMS_DEFAULTS sets nthreads=1; override to use all cores.
    cparams.nthreads = static_cast<int16_t>(std::thread::hardware_concurrency());

    blosc2_context* cctx = blosc2_create_cctx(cparams);
    if (!cctx)
        throw std::runtime_error("[blosc write] Failed to create compression context");

    // Write placeholder nchunks; patched via seekp at the end.
    int64_t nchunks = 0;
    auto header_pos = os.tellp();
    os.write(reinterpret_cast<const char*>(&nchunks), sizeof(nchunks));

    // Serializer runs in a background thread; main thread compresses + writes.
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

    auto flush_accum = [&](std::vector<uint8_t>& accum, size_t src_size) {
        int32_t sz32    = static_cast<int32_t>(src_size);
        int32_t max_dst = sz32 + BLOSC2_MAX_OVERHEAD;
        std::vector<uint8_t> compressed(max_dst);

        int cmp_size = blosc2_compress_ctx(cctx, accum.data(), sz32, compressed.data(), max_dst);
        if (cmp_size <= 0) {
            blosc2_free_ctx(cctx);
            throw std::runtime_error("[blosc write] Compression failed (err=" + std::to_string(cmp_size) + ")");
        }

        int64_t unc64 = sz32, cmp64 = cmp_size;
        os.write(reinterpret_cast<const char*>(&unc64), sizeof(unc64));
        os.write(reinterpret_cast<const char*>(&cmp64), sizeof(cmp64));
        os.write(reinterpret_cast<const char*>(compressed.data()), cmp_size);

        ++nchunks;
        accum.erase(accum.begin(), accum.begin() + src_size);
    };

    std::vector<uint8_t> accum;
    accum.reserve(static_cast<size_t>(GVFILE_CHUNK_SIZE) * 2);

    try {
        while (!stream.is_finished()) {
            auto [chunk, split_type] = stream.read_max_nbytes_blocking(GVFILE_CHUNK_SIZE);
            if (!chunk.empty())
                accum.insert(accum.end(), chunk.begin(), chunk.end());

            while (static_cast<i64>(accum.size()) >= GVFILE_CHUNK_SIZE)
                flush_accum(accum, static_cast<size_t>(GVFILE_CHUNK_SIZE));
        }
        if (!accum.empty())
            flush_accum(accum, accum.size());
    } catch (...) {
        serializer.join();
        blosc2_free_ctx(cctx);
        throw;
    }

    serializer.join();
    blosc2_free_ctx(cctx);

    if (serializer_exc) std::rethrow_exception(serializer_exc);

    // Patch the nchunks header now that we know the final count.
    os.seekp(header_pos);
    os.write(reinterpret_cast<const char*>(&nchunks), sizeof(nchunks));
}

export void write_generic_value(
    const GenericValue& value,
    const std::filesystem::path& path,
    i32 compression_level = 5,
    BloscCompressionStrategy compression_strategy = BloscCompressionStrategy::LZ4,
    BloscShuffleFilter shuffle_filter = BloscShuffleFilter::BitShuffle)
{
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs)
        throw std::runtime_error("[blosc write] Cannot open file: " + path.string());
    write_generic_value(value, ofs, compression_level, compression_strategy, shuffle_filter);
}

export void write_generic_value(
    const GenericValue& value,
    const std::string& filename,
    i32 compression_level = 5,
    BloscCompressionStrategy compression_strategy = BloscCompressionStrategy::LZ4,
    BloscShuffleFilter shuffle_filter = BloscShuffleFilter::BitShuffle)
{
    write_generic_value(value, std::filesystem::path(filename), compression_level, compression_strategy, shuffle_filter);
}

// ---------------------------------------------------------------------------
// Core stream-based read.
// Reads from the current position of is.
// ---------------------------------------------------------------------------
export GenericValue read_generic_value(std::istream& is)
{
    BloscGuard blosc_guard;

    int64_t nchunks = 0;
    is.read(reinterpret_cast<char*>(&nchunks), sizeof(nchunks));
    if (!is)
        throw std::runtime_error("[blosc read] Failed to read chunk count from stream");

    blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
    // BLOSC2_DPARAMS_DEFAULTS sets nthreads=1; override to use all cores.
    dparams.nthreads = static_cast<int16_t>(std::thread::hardware_concurrency());
    blosc2_context* dctx = blosc2_create_dctx(dparams);
    if (!dctx)
        throw std::runtime_error("[blosc read] Failed to create decompression context");

    threadsafe_stream stream;
    std::exception_ptr producer_exc;

    std::thread producer([&, nchunks]() {
        try {
            for (int64_t i = 0; i < nchunks; ++i) {
                int64_t unc64 = 0, cmp64 = 0;
                is.read(reinterpret_cast<char*>(&unc64), sizeof(unc64));
                is.read(reinterpret_cast<char*>(&cmp64), sizeof(cmp64));

                std::vector<uint8_t> compressed(cmp64);
                is.read(reinterpret_cast<char*>(compressed.data()), cmp64);

                std::vector<uint8_t> decompressed(unc64);
                int result = blosc2_decompress_ctx(
                    dctx, compressed.data(), static_cast<int32_t>(cmp64),
                    decompressed.data(), static_cast<int32_t>(unc64));
                if (result < 0) {
                    throw std::runtime_error(
                        "[blosc read] Decompression failed on chunk " + std::to_string(i) +
                        " (err=" + std::to_string(result) + ")");
                }
                stream.write(std::move(decompressed));
            }
            stream.set_finished();
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
            blosc2_free_ctx(dctx);
            throw;
        }
    }();

    producer.join();
    blosc2_free_ctx(dctx);

    if (producer_exc) std::rethrow_exception(producer_exc);

    return result;
}

export GenericValue read_generic_value(const std::filesystem::path& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("[blosc read] Cannot open file: " + path.string());
    return read_generic_value(ifs);
}

export GenericValue read_generic_value(const std::string& filename)
{
    return read_generic_value(std::filesystem::path(filename));
}

}
}
}