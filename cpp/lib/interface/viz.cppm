export module viz;

import std;

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
    std::string json_data;  // JSON payload, ready to forward to the frontend
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

// ---------------------------------------------------------------------------
// JSON serialisation helpers (std-only, no external library needed)
// ---------------------------------------------------------------------------

namespace detail {

    std::string to_json_array(const std::vector<double>& v) {
        std::string s;
        s.reserve(v.size() * 12 + 2);
        s += '[';
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (i) s += ',';
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.17g", v[i]);
            s += buf;
        }
        s += ']';
        return s;
    }

    std::string to_json_array2d(const std::vector<std::vector<double>>& m) {
        std::string s;
        s += '[';
        for (std::size_t i = 0; i < m.size(); ++i) {
            if (i) s += ',';
            s += to_json_array(m[i]);
        }
        s += ']';
        return s;
    }

    std::string make_payload(
        std::string_view command,
        std::string_view title,
        std::string_view data_json)
    {
        std::string s;
        s.reserve(64 + data_json.size());
        s += "{\"command\":\"";
        s += command;
        s += "\",\"title\":\"";
        s += title;
        s += "\",\"data\":";
        s += data_json;
        s += '}';
        return s;
    }

} // namespace detail

// ---------------------------------------------------------------------------
// Convenience free functions
// ---------------------------------------------------------------------------

// Line plot: single y-vector (x is implicit 0..n-1)
export void plot_line(std::vector<double> y, std::string title = {}) {
    std::string data = "{\"y\":" + detail::to_json_array(y) + '}';
    PlotQueue::instance().push({
        .command   = "line",
        .title     = title,
        .json_data = detail::make_payload("line", title, data)
    });
}

// Line plot: explicit x and y vectors
export void plot_line(std::vector<double> x, std::vector<double> y, std::string title = {}) {
    std::string data = "{\"x\":" + detail::to_json_array(x)
                     + ",\"y\":" + detail::to_json_array(y) + '}';
    PlotQueue::instance().push({
        .command   = "line",
        .title     = title,
        .json_data = detail::make_payload("line", title, data)
    });
}

// Scatter plot
export void plot_scatter(std::vector<double> x, std::vector<double> y, std::string title = {}) {
    std::string data = "{\"x\":" + detail::to_json_array(x)
                     + ",\"y\":" + detail::to_json_array(y) + '}';
    PlotQueue::instance().push({
        .command   = "scatter",
        .title     = title,
        .json_data = detail::make_payload("scatter", title, data)
    });
}

// Heatmap (row-major 2-D data)
export void plot_heatmap(std::vector<std::vector<double>> z, std::string title = {}) {
    std::string data = "{\"z\":" + detail::to_json_array2d(z) + '}';
    PlotQueue::instance().push({
        .command   = "heatmap",
        .title     = title,
        .json_data = detail::make_payload("heatmap", title, data)
    });
}

} // namespace hasty::viz
