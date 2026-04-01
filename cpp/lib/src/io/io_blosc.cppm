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

void write_generic_value(const GenericValue& value, const std::string& filename, i32 compression_level = 5, GVFCompressionStrategy compression_strategy = GVFCompressionStrategy::LZ4)
{
    blosc2_init();

    // Options for Blosc2 super-chunk
    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    switch (compression_strategy) {
    case GVFCompressionStrategy::LZ4:
        cparams.compcode = BLOSC_LZ4;
        break;
    case GVFCompressionStrategy::LZ4_HC:
        cparams.compcode = BLOSC_LZ4HC;
        break;
    case GVFCompressionStrategy::ZSTD:
        cparams.compcode = BLOSC_ZSTD;
        break;
    }
    cparams.typesize = 1;                   // store bytes
    cparams.clevel = compression_level;     // compression level 0-9

    blosc2_storage storage = BLOSC2_STORAGE_DEFAULTS;
    storage.cparams = &cparams;
    blosc2_schunk* schunk = blosc2_schunk_new(&storage);

    // Create threadsafe stream to feed data
    threadsafe_stream stream;

    // Serialize in a separate thread
    std::thread serializer([&]() {
        GenericValue::serialize(value, stream, GVFILE_CHUNK_SIZE);
        stream.set_finished();
    });

    // Consume the stream and append to Blosc2 super-chunk
    while (!stream.is_finished()) {
        auto [chunk, split_type] = stream.read_max_nbytes_blocking(GVFILE_CHUNK_SIZE); // 64MB chunks
        if (!chunk.empty()) {
            blosc2_schunk_append_buffer(schunk, chunk.data(), chunk.size());
        }
    }

    serializer.join();

    // Save the compressed frame to file
    blosc2_schunk_to_file(schunk, filename.c_str());
    blosc2_schunk_free(schunk);
    blosc2_destroy();
}

GenericValue read_generic_value(const std::string& filename)
{
    blosc2_init();

    blosc2_schunk* schunk = blosc2_schunk_open(filename.c_str());
    if (!schunk) {
        blosc2_destroy();
        throw std::runtime_error("Failed to open Blosc2 frame: " + filename);
    }

    threadsafe_stream stream;

    // Producer: decompress chunks into the stream
    std::thread producer([&]() {
        for (size_t i = 0; i < schunk->nchunks; ++i) {
            std::vector<uint8_t> buffer(schunk->chunksize);
            int decompressed_bytes = blosc2_schunk_decompress_chunk(schunk, i, buffer.data(), buffer.size());
            if (decompressed_bytes < 0) {
                throw std::runtime_error("Failed to decompress chunk " + std::to_string(i));
            }
            buffer.resize(decompressed_bytes);
            stream.write(std::move(buffer));
        }
        stream.set_finished();
    });

    // Deserialize from the stream
    GenericValue result = GenericValue::deserialize(stream, GVFILE_CHUNK_SIZE);

    producer.join();
    blosc2_schunk_free(schunk);
    blosc2_destroy();

    return result;
}



}
}
}