module;

#include <pch.hpp>

export module hasty_threading_mod:dep_threadpool;

import std;

namespace hasty {

// Type-erased completion state. Shared between DepFuture<T> and downstream task nodes.
export struct TaskDepState {
    std::atomic<bool> _done{false};
    std::mutex _mutex;
    std::vector<std::function<void()>> _on_complete;

    void complete() {
        _done.store(true, std::memory_order_release);
        std::vector<std::function<void()>> cbs;
        {
            std::lock_guard lock(_mutex);
            cbs = std::move(_on_complete);
        }
        for (auto& cb : cbs) cb();
    }

    // Returns false if already done (callback will NOT be called — caller must handle that).
    bool try_register(std::function<void()> cb) {
        std::lock_guard lock(_mutex);
        if (_done.load(std::memory_order_acquire)) return false;
        _on_complete.push_back(std::move(cb));
        return true;
    }

    bool is_done() const { return _done.load(std::memory_order_acquire); }
};

export template<typename T>
class DepFuture {
public:
    T get() { return _future.get(); }
    bool ready() const { return _dep_state->is_done(); }
    std::shared_ptr<TaskDepState> dep() const { return _dep_state; }

private:
    friend class DepThreadPool;

    DepFuture(std::shared_future<T> future, std::shared_ptr<TaskDepState> dep_state)
        : _future(std::move(future)), _dep_state(std::move(dep_state)) {}

    std::shared_future<T> _future;
    std::shared_ptr<TaskDepState> _dep_state;
};

export template<>
class DepFuture<void> {
public:
    void get() { _future.get(); }
    bool ready() const { return _dep_state->is_done(); }
    std::shared_ptr<TaskDepState> dep() const { return _dep_state; }

private:
    friend class DepThreadPool;

    DepFuture(std::shared_future<void> future, std::shared_ptr<TaskDepState> dep_state)
        : _future(std::move(future)), _dep_state(std::move(dep_state)) {}

    std::shared_future<void> _future;
    std::shared_ptr<TaskDepState> _dep_state;
};

export class DepThreadPool {
    struct TaskNode {
        std::function<void()> func;
        std::atomic<int> remaining_deps{0};
    };

public:
    explicit DepThreadPool(int num_workers = std::thread::hardware_concurrency())
        : _stop(false)
    {
        _threads.reserve(num_workers);
        try {
            for (int i = 0; i < num_workers; ++i)
                _threads.emplace_back(&DepThreadPool::worker_loop, this);
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~DepThreadPool() { shutdown(); }

    DepThreadPool(DepThreadPool&&) = delete;
    DepThreadPool(const DepThreadPool&) = delete;
    DepThreadPool& operator=(DepThreadPool&&) = delete;
    DepThreadPool& operator=(const DepThreadPool&) = delete;

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> DepFuture<std::invoke_result_t<F, Args...>>
    {
        return enqueue_after({}, std::forward<F>(f), std::forward<Args>(args)...);
    }

    // deps: list of TaskDepState handles from prior DepFuture::dep() calls.
    // Returned DepFuture becomes ready only after all deps complete.
    // No thread blocks waiting — deps fire callbacks that push to ready queue.
    template<class F, class... Args>
    auto enqueue_after(
        std::vector<std::shared_ptr<TaskDepState>> deps,
        F&& f, Args&&... args)
        -> DepFuture<std::invoke_result_t<F, Args...>>
    {
        using R = std::invoke_result_t<F, Args...>;

        auto dep_state = std::make_shared<TaskDepState>();
        auto promise   = std::make_shared<std::promise<R>>();
        auto future    = promise->get_future().share();

        auto node = std::make_shared<TaskNode>();
        // +1 setup fence: prevents premature ready before all deps are registered.
        node->remaining_deps.store(static_cast<int>(deps.size()) + 1, std::memory_order_relaxed);

        node->func = [
            bound  = std::bind(std::forward<F>(f), std::forward<Args>(args)...),
            p      = promise,
            ds     = dep_state
        ]() mutable {
            try {
                if constexpr (std::is_void_v<R>) {
                    bound();
                    p->set_value();
                } else {
                    p->set_value(bound());
                }
            } catch (...) {
                p->set_exception(std::current_exception());
            }
            ds->complete();
        };

        // Capture pool by raw pointer — pool must outlive all submitted tasks.
        auto try_ready = [this, n = node]() {
            if (n->remaining_deps.fetch_sub(1, std::memory_order_acq_rel) == 1)
                push_ready(n);
        };

        for (auto& dep : deps) {
            if (!dep->try_register(try_ready))
                try_ready(); // dep already done — count down immediately
        }

        try_ready(); // release setup fence

        return DepFuture<R>(std::move(future), std::move(dep_state));
    }

private:
    void push_ready(std::shared_ptr<TaskNode> node) {
        {
            std::lock_guard lock(_queue_mutex);
            if (_stop) return;
            _ready_queue.push(std::move(node));
        }
        _cv.notify_one();
    }

    void worker_loop() {
        while (true) {
            std::shared_ptr<TaskNode> task;
            {
                std::unique_lock lock(_queue_mutex);
                _cv.wait(lock, [this] { return _stop || !_ready_queue.empty(); });
                if (_stop && _ready_queue.empty()) return;
                task = std::move(_ready_queue.front());
                _ready_queue.pop();
            }
            task->func();
        }
    }

    void shutdown() {
        {
            std::lock_guard lock(_queue_mutex);
            _stop = true;
        }
        _cv.notify_all();
        for (auto& t : _threads)
            if (t.joinable()) t.join();
    }

    std::queue<std::shared_ptr<TaskNode>> _ready_queue;
    std::mutex _queue_mutex;
    std::condition_variable _cv;
    std::vector<std::thread> _threads;
    bool _stop;
};



export extern DepThreadPool global_dep_thread_pool;



}
