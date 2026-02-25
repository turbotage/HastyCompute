module;

#include "pch.hpp"

export module thread_stream;

import std;
import util_mod;

namespace hasty {
namespace threading {

export class threadsafe_stream {
public:

    enum class chunksplit_type : u8 {
        NONE = 0,
        SPLIT = 1,
        MERGE = 2,
        MERGE_AND_SPLIT = 3
    };

    threadsafe_stream() = default;

    threadsafe_stream(threadsafe_stream&& other) noexcept
        : _chunks(std::move(other._chunks)),
          _finished(other._finished)
    {}

    threadsafe_stream& operator=(threadsafe_stream&& other) noexcept {
        if (this != &other) {
            std::lock_guard<std::mutex> lock(_mutex);
            std::lock_guard<std::mutex> other_lock(other._mutex);
            _chunks = std::move(other._chunks);
            _finished = other._finished;
        }
        return *this;
    }

    void write(std::vector<u8> chunk) {
        std::lock_guard<std::mutex> lock(_mutex);
        _chunks.emplace_back(std::move(chunk));
        _cv.notify_one();
    }

    // Read a chunk if available, else return empty vector
    std::vector<u8> read_chunk() {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_chunks.empty()) {
            std::vector<u8> chunk = std::move(_chunks.front());
            _chunks.pop_front();
            return chunk;
        } else {
            return {};
        }
    }

    // Read a chunk, blocking until one is available
    std::vector<u8> read_chunk_blocking(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(_mutex);
        if (_cv.wait_for(lock, timeout, [this] { return !_chunks.empty() || _finished; })) {
            if (!_chunks.empty()) {
                std::vector<u8> chunk = std::move(_chunks.front());
                _chunks.pop_front();
                return chunk;
            }
        }
        throw std::runtime_error("Timeout while waiting for chunk");
    }

    // Read a chunk of at most nbytes, if no chunk is available return empty vector
    std::pair<std::vector<u8>,u8> read_max_nbytes(i64 nbytes) {
        if (nbytes <= 0) {
            throw std::runtime_error("nbytes must be positive");
        }
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_chunks.empty()) {
            std::vector<u8> chunk = std::move(_chunks.front());
            _chunks.pop_front();

            if (nbytes >= chunk.size()) {
                return {chunk, (u8)chunksplit_type::NONE};
            } else {
                std::vector<u8> result(chunk.begin(), chunk.begin() + nbytes);
                std::vector<u8> remaining(chunk.begin() + nbytes, chunk.end());
                _chunks.push_front(std::move(remaining));
                return {result, (u8)chunksplit_type::SPLIT};
            }
        } else {
            return {};
        }
    }

    std::pair<std::vector<u8>,u8> read_max_nbytes_blocking(i64 nbytes, std::chrono::milliseconds timeout) {
        if (nbytes <= 0) {
            throw std::runtime_error("nbytes must be positive");
        }
        std::unique_lock<std::mutex> lock(_mutex);
        if (_cv.wait_for(lock, timeout, [this] { return !_chunks.empty() || _finished; })) {
            if (!_chunks.empty()) {
                std::vector<u8> chunk = std::move(_chunks.front());
                _chunks.pop_front();

                if (nbytes >= chunk.size()) {
                    return {chunk, (u8)chunksplit_type::NONE};
                } else {
                    std::vector<u8> result(chunk.begin(), chunk.begin() + nbytes);
                    std::vector<u8> remaining(chunk.begin() + nbytes, chunk.end());
                    _chunks.push_front(std::move(remaining));
                    return {result, (u8)chunksplit_type::SPLIT};
                }
            }
        }
        throw std::runtime_error("Timeout while waiting for chunk");
    }

    std::pair<std::vector<u8>,u8> read_exact_nbytes_blocking(i64 nbytes, std::chrono::milliseconds timeout_per_chunk) {
        if (nbytes <= 0) {
            throw std::runtime_error("nbytes must be positive");
        }
        std::vector<u8> result;
        result.reserve(nbytes);
        i64 bytes_read = 0;
        u8 split_type = (u8)chunksplit_type::NONE;

        i32 nchunks = 0;
        std::unique_lock<std::mutex> lock(_mutex);
        while (bytes_read < nbytes) {
            if (_cv.wait_for(lock, timeout_per_chunk, [this] { return !_chunks.empty() || _finished; })) {
                if (!_chunks.empty()) {
                    std::vector<u8> chunk = std::move(_chunks.front());
                    _chunks.pop_front();

                    i64 to_copy = std::min(static_cast<i64>(chunk.size()), nbytes - bytes_read);
                    result.insert(result.end(), chunk.begin(), chunk.begin() + to_copy);
                    bytes_read += to_copy;

                    if (to_copy < static_cast<i64>(chunk.size())) {
                        // We have found enough bytes over all chunks
                        std::vector<u8> remaining(chunk.begin() + to_copy, chunk.end());
                        _chunks.push_front(std::move(remaining));
                        if (nchunks == 0) {
                            // We only used part of one chunk
                            split_type = (u8)chunksplit_type::SPLIT;
                        } else {
                            // We used multiple chunks and the last were split
                            split_type = (u8)chunksplit_type::MERGE_AND_SPLIT;
                        }
                        break;
                    }
                }
            } else {
                throw std::runtime_error("Timeout while waiting for chunk");
            }
            nchunks++;
        }

        if (nchunks > 1 && split_type == (u8)chunksplit_type::NONE) {
            // We used multiple chunks, and none were split
            split_type = (u8)chunksplit_type::MERGE;
        } else {
            // We used precisely one chunk, no merging or splitting
        }

        return {result, split_type};
    }



    void set_finished() {
        std::lock_guard<std::mutex> lock(_mutex);
        _finished = true;
        _cv.notify_all();
    }

    bool is_finished() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _finished && _chunks.empty();
    }

    bool is_set_finished() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _finished;
    }

    std::size_t pending_chunks() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _chunks.size();
    }

    // Reset the stream, it can now be used again
    void reset() {
        std::lock_guard<std::mutex> lock(_mutex);
        _chunks.clear();
        _finished = false;
    }

private:
    mutable std::mutex _mutex;
    std::condition_variable _cv;

    std::deque<std::vector<u8>> _chunks;
    bool _finished = false;
};

}
}