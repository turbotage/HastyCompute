module;

#include <pch.hpp>

export module hastay_threading_mod:threadpool;

import std;
import hasty_util_mod;

namespace hasty {

export class ThreadPool {
public:

    explicit ThreadPool(int num_workers = std::thread::hardware_concurrency())
        : _stop(false), _work_length(0)
    {
        _threads.reserve(num_workers);
        try {
            for (auto i = 0; i < num_workers; i++) {
                _threads.emplace_back(&ThreadPool::work, this);
            }
        }
        catch (...) {
            {
                std::lock_guard<std::mutex> lock(_queue_mutex);
                _stop = true;
            }
            _queue_notifier.notify_all();
            for (auto& thread : _threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            throw;
        }
    }

    ~ThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(_queue_mutex);
            _stop = true;
        }
        _queue_notifier.notify_all();
        for (auto& worker : _threads) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template<class F, class... Args>
    std::future<typename std::invoke_result<F, Args...>::type> enqueue(F&& f, Args&&... args)
    {
        using return_type = typename std::invoke_result<F, Args...>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<return_type> res = task->get_future();
        {
            std::lock_guard<std::mutex> lock(_queue_mutex);
            _work.emplace([task]() { (*task)(); });
        }
        _queue_notifier.notify_one();
        _work_length += 1;
        return res;
    }

    int work_length() { return _work_length.load(); }

    ThreadPool(ThreadPool&&) = delete;
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:

    void work()
    {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(_queue_mutex);
                while (_work.empty()) {
                    if (_stop) {
                        return;
                    }
                    _queue_notifier.wait(lock);
                }
                task = std::move(_work.front());
                _work.pop();
            }
            task();
            _work_length -= 1;
        }
    }

private:
    std::queue<std::function<void()>> _work;
    std::condition_variable _queue_notifier;
    std::mutex _queue_mutex;
    std::vector<std::thread> _threads;
    bool _stop;

    std::atomic<int> _work_length;
};


}