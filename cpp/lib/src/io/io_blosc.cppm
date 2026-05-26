module;

#include <blosc2.h>

export module hasty_io_mod_blosc;

import std;
import hasty_util_mod;
import hasty_threading_mod;

namespace hasty {
namespace io {
namespace blosc {

export constexpr i64 BLOSC_DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024; // 64MB

export enum struct BloscCompressionStrategy : u8 {
    LZ4    = 0,
    LZ4_HC = 1,
    ZSTD   = 2
};

export enum struct BloscShuffleFilter : u8 {
    NoShuffle  = BLOSC_NOFILTER,
    Shuffle    = BLOSC_SHUFFLE,
    BitShuffle = BLOSC_BITSHUFFLE,
};

namespace {
    std::mutex g_blosc_mutex;
    int        g_blosc_refcount = 0;
}

// Reference-counted blosc2_init / blosc2_destroy guard.
// Construct once per logical operation (or once globally) to avoid
// repeated init/destroy pairs across concurrent read/write calls.
export struct BloscGuard {
    BloscGuard() {
        std::lock_guard lock(g_blosc_mutex);
        if (++g_blosc_refcount == 1) blosc2_init();
    }
    ~BloscGuard() {
        std::lock_guard lock(g_blosc_mutex);
        if (--g_blosc_refcount == 0) blosc2_destroy();
    }
    BloscGuard(const BloscGuard&) = delete;
    BloscGuard& operator=(const BloscGuard&) = delete;
};

// ── Raw stream compress / decompress ─────────────────────────────────────────
//
// These functions operate on raw byte streams (threadsafe_stream / std::ostream)
// with no knowledge of GenericValue.  Callers responsible for serialisation.
//
// Wire format (written by compress_stream, read by decompress_stream):
//   int64_t  nchunks
//   for each chunk:
//     int64_t  uncompressed_size
//     int64_t  compressed_size
//     uint8_t  compressed_data[compressed_size]

export void compress_stream(
    threadsafe_stream&       source,
    std::ostream&            os,
    i32                      compression_level  = 5,
    i64                      chunk_size         = BLOSC_DEFAULT_CHUNK_SIZE,
    BloscCompressionStrategy compression_strategy = BloscCompressionStrategy::LZ4,
    BloscShuffleFilter       shuffle_filter     = BloscShuffleFilter::BitShuffle)
{
    BloscGuard guard;

    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    switch (compression_strategy) {
        case BloscCompressionStrategy::LZ4:    cparams.compcode = BLOSC_LZ4;   break;
        case BloscCompressionStrategy::LZ4_HC: cparams.compcode = BLOSC_LZ4HC; break;
        case BloscCompressionStrategy::ZSTD:   cparams.compcode = BLOSC_ZSTD;  break;
    }
    cparams.clevel   = compression_level;
    cparams.typesize = 1;
    cparams.filters[BLOSC2_MAX_FILTERS - 1] = static_cast<uint8_t>(shuffle_filter);
    cparams.nthreads = static_cast<int16_t>(std::thread::hardware_concurrency());

    blosc2_context* cctx = blosc2_create_cctx(cparams);
    if (!cctx)
        throw std::runtime_error("[blosc compress_stream] failed to create compression context");

    int64_t nchunks   = 0;
    auto    hdr_pos   = os.tellp();
    os.write(reinterpret_cast<const char*>(&nchunks), sizeof(nchunks));

    auto flush = [&](std::vector<uint8_t>& accum, std::size_t src_size) {
        int32_t sz32    = static_cast<int32_t>(src_size);
        int32_t max_dst = sz32 + BLOSC2_MAX_OVERHEAD;
        std::vector<uint8_t> compressed(max_dst);
        int cmp = blosc2_compress_ctx(cctx, accum.data(), sz32,
                                       compressed.data(), max_dst);
        if (cmp <= 0) {
            blosc2_free_ctx(cctx);
            throw std::runtime_error("[blosc compress_stream] compression failed (err="
                                     + std::to_string(cmp) + ")");
        }
        int64_t unc64 = sz32, cmp64 = cmp;
        os.write(reinterpret_cast<const char*>(&unc64), sizeof(unc64));
        os.write(reinterpret_cast<const char*>(&cmp64), sizeof(cmp64));
        os.write(reinterpret_cast<const char*>(compressed.data()), cmp);
        ++nchunks;
        accum.erase(accum.begin(), accum.begin() + src_size);
    };

    std::vector<uint8_t> accum;
    accum.reserve(static_cast<std::size_t>(chunk_size) * 2);

    try {
        while (!source.is_finished()) {
            auto [chunk, _] = source.read_max_nbytes_blocking(chunk_size);
            if (!chunk.empty())
                accum.insert(accum.end(), chunk.begin(), chunk.end());
            while (static_cast<i64>(accum.size()) >= chunk_size)
                flush(accum, static_cast<std::size_t>(chunk_size));
        }
        if (!accum.empty()) flush(accum, accum.size());
    } catch (...) {
        blosc2_free_ctx(cctx);
        throw;
    }
    blosc2_free_ctx(cctx);

    os.seekp(hdr_pos);
    os.write(reinterpret_cast<const char*>(&nchunks), sizeof(nchunks));
}

export void decompress_stream(
    std::istream&      is,
    threadsafe_stream& sink)
{
    BloscGuard guard;

    int64_t nchunks = 0;
    is.read(reinterpret_cast<char*>(&nchunks), sizeof(nchunks));
    if (!is)
        throw std::runtime_error("[blosc decompress_stream] failed to read chunk count");

    blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
    dparams.nthreads = static_cast<int16_t>(std::thread::hardware_concurrency());
    blosc2_context* dctx = blosc2_create_dctx(dparams);
    if (!dctx)
        throw std::runtime_error("[blosc decompress_stream] failed to create decompression context");

    try {
        for (int64_t i = 0; i < nchunks; ++i) {
            int64_t unc64 = 0, cmp64 = 0;
            is.read(reinterpret_cast<char*>(&unc64), sizeof(unc64));
            is.read(reinterpret_cast<char*>(&cmp64), sizeof(cmp64));

            std::vector<uint8_t> compressed(cmp64);
            is.read(reinterpret_cast<char*>(compressed.data()), cmp64);

            std::vector<uint8_t> decompressed(unc64);
            int r = blosc2_decompress_ctx(dctx, compressed.data(),
                                           static_cast<int32_t>(cmp64),
                                           decompressed.data(),
                                           static_cast<int32_t>(unc64));
            if (r < 0) {
                throw std::runtime_error(
                    "[blosc decompress_stream] chunk " + std::to_string(i) +
                    " failed (err=" + std::to_string(r) + ")");
            }
            sink.write(std::move(decompressed));
        }
        sink.set_finished();
    } catch (...) {
        sink.set_finished();
        blosc2_free_ctx(dctx);
        throw;
    }
    blosc2_free_ctx(dctx);
}

// Convenience: decompress in background thread, return the sink stream.
// Caller owns the returned stream and must consume it before the thread joins.
export std::pair<threadsafe_stream, std::thread>
decompress_stream_async(std::istream& is)
{
    threadsafe_stream sink;
    std::thread t([&is, &sink]() {
        try {
            decompress_stream(is, sink);
        } catch (...) {
            sink.set_finished();
        }
    });
    return {std::move(sink), std::move(t)};
}


}
}
}
