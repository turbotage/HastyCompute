#include "thread_pool.hpp"


void hasty::ThreadPool::work() {
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

hasty::ThreadPool::ThreadPool(int num_workers)
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

hasty::ThreadPool::~ThreadPool() {
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
