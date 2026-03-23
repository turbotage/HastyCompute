export module viz;

import std;
import hasty_threading_mod;


namespace hasty::viz {

// ---------------------------------------------------------------------------
// PlotRequest
// Holds a single plot command and its data as a pre-serialised JSON string.
// Using std-only types keeps this module dependency-free so that consumers
// (e.g. HastyServer) do not need to pull in tensor / LibTorch headers.
// ---------------------------------------------------------------------------

export struct PlotRequest {
    std::string command;    // "line" | "scatter" | "heatmap"
    std::string title;
    hasty::threadsafe_stream data; // Pre-serialised JSON string containing the plot data
};

// ---------------------------------------------------------------------------
// PlotQueue  -  process-global singleton, thread-safe
// ---------------------------------------------------------------------------

export class PlotQueue {
public:
    static PlotQueue& instance() {
        static PlotQueue inst;
        return inst;
    }

    PlotQueue(const PlotQueue&) = delete;
    PlotQueue& operator=(const PlotQueue&) = delete;

    void push(PlotRequest req) {
        {
            std::unique_lock lock(m_mutex);
            m_queue.push(std::move(req));
        }
        m_cv.notify_one();
    }

    // Non-blocking pop -- returns nullopt if the queue is empty.
    std::optional<PlotRequest> try_pop() {
        std::unique_lock lock(m_mutex);
        if (m_queue.empty()) return std::nullopt;
        auto req = std::move(m_queue.front());
        m_queue.pop();
        return req;
    }

    // Blocking pop with timeout.
    std::optional<PlotRequest> pop_wait_for(std::chrono::milliseconds timeout) {
        std::unique_lock lock(m_mutex);
        if (m_cv.wait_for(lock, timeout, [this] { return !m_queue.empty(); })) {
            auto req = std::move(m_queue.front());
            m_queue.pop();
            return req;
        }
        return std::nullopt;
    }

private:
    PlotQueue() = default;

    std::mutex              m_mutex;
    std::condition_variable m_cv;
    std::queue<PlotRequest> m_queue;
};

}
