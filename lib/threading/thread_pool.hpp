#pragma once

#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>


namespace hasty {

	class ThreadPool {
	public:

		explicit ThreadPool(int num_workers = std::thread::hardware_concurrency());
		~ThreadPool();

		template<class F, class... Args>
		std::future<typename std::invoke_result<F, Args...>::type> enqueue(F&& f, Args&&... args);

		int work_length() { return _work_length.load(); }

		ThreadPool(ThreadPool&&) = delete;
		ThreadPool(const ThreadPool&) = delete;
		ThreadPool& operator=(ThreadPool&&) = delete;
		ThreadPool& operator=(const ThreadPool&) = delete;

	private:

		void work();

	private:
		std::queue<std::function<void()>> _work;
		std::condition_variable _queue_notifier;
		std::mutex _queue_mutex;
		std::vector<std::thread> _threads;
		bool _stop;

		std::atomic<int> _work_length;
	};

	template<class F, class ...Args>
	inline std::future<typename std::invoke_result<F,Args...>::type> ThreadPool::enqueue(F&& f, Args && ...args)
	{
		using return_type = typename std::invoke_result<F,Args...>::type;
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

}
