module;

#include <pch.hpp>

export module hasty_threading_mod:storage_threadpool;

//import pch;
import std;
import hasty_util_mod;

namespace hasty {

export class storage {
private:

	struct storage_item {
		std::shared_ptr<void> ptr;
		std::type_index type;
	};

	std::unordered_map<std::string, storage_item> _storage;

public:

	template<typename T>
	auto get_ref(const std::string& name) -> std::optional<std::reference_wrapper<T>> {
		auto it = _storage.find(name);
		if (it != _storage.end()) {
			if (it->second.type != typeid(T)) {
				return std::nullopt;
			}
			return std::optional<std::reference_wrapper<T>>(*(T*)it->second.ptr.get());
		}
		return std::nullopt;
	}

	template<typename T>
	auto get_ref_throw(const std::string& name) -> T& {
		auto it = _storage.find(name);
		if (it != _storage.end()) {
			if (it->second.type != typeid(T)) {
				throw std::runtime_error("Storage does not contain key: " + name + " with type: " + typeid(T).name());
			}
			return *(T*)it->second.ptr.get();
		}
		throw std::runtime_error("Storage does not contain key: " + name);
	}

	template<typename T>
	auto get_ptr(const std::string& name) -> std::shared_ptr<T> {
		auto it = _storage.find(name);
		if (it != _storage.end()) {
			if (it->second.type != typeid(T)) {
				return nullptr;
			}
			return std::static_pointer_cast<T>(it->second.ptr);
		}
		return nullptr;
	}

	template<typename T>
	auto get_ptr_throw(const std::string& name) -> std::shared_ptr<T>
	{
		auto it = _storage.find(name);
		if (it != _storage.end()) {
			if (it->second.type != typeid(T)) {
				throw std::runtime_error("Storage does not contain key: " + name + " with type: " + typeid(T).name());
			}
			return std::static_pointer_cast<T>(it->second.ptr);
		}
		throw std::runtime_error("Storage does not contain key: " + name);
	}

	template<typename T>
	void add(const std::string& name, std::shared_ptr<T> ptr) {
		std::shared_ptr<void> inptr = std::static_pointer_cast<void>(ptr);
		_storage.insert({name, {inptr, typeid(T)}});
	}

	template<typename T>
	void clear(const std::string& name) {
		auto it = _storage.find(name);
		if (it != _storage.end()) {
			if (it->second.type != typeid(T)) {
				throw std::runtime_error("Storage does not contain key: " + name + " with type: " + typeid(T).name());
			}
			_storage.erase(it);
		}
	}

	template<typename T>
	void clear_throw(const std::string& name) {
		auto it = _storage.find(name);
		if (it != _storage.end()) {
			if (it->second.type != typeid(T)) {
				throw std::runtime_error("Storage does not contain key: " + name + " with type: " + typeid(T).name());
			}
			_storage.erase(it);
		} else {
			throw std::runtime_error("Storage does not contain key: " + name);
		}
	}

	void clear(const std::string& name) {
		auto it = _storage.find(name);
		if (it != _storage.end()) {
			_storage.erase(it);
		}
	}

	bool exist(const std::string& name) {
		return _storage.find(name) != _storage.end();
	}

	vset<std::string> names() {
		vset<std::string> result;
		result.reserve(_storage.size());
		for (const auto& [name, str] : _storage) {
			result.insert_without_check(name);
		}
		return result;
	}

};

/**
@brief
A thread pool with storage for each thread. Tasks can require to be run by a thread with certain data available in its storage.
Good usage is for long running tasks on the GPU, where data for the task has already been prepared in the storage on said GPU.
Works also as a simple thread pool, when no requirements on storage are given. However, the pool sorts through tasks after
threads with the optimal storage for the task, so thread-contention and bloking is higher than in a simple thread pool.

@param task_optimality_wait_time
	The time in milliseconds a task waits for a thread with the optimal storage before it is executed by a thread with suboptimal storage.
	Default is 100ms. Note: As of now, if the work item has no storage requirements or good-to-have storage items, 
	the task will be executed immediately, without waiting for the optimal thread. This might change in the future. So that when 
	no requirements are given. This work should be done by a thread with minimal storage items. So that threads with more storage items,
	are available for tasks that require more storage items.
*/
export class storage_thread_pool {
public:

	storage_thread_pool(std::vector<storage>&& storages, i32 task_optimality_wait_time = 100)
		: _stop(false), _work_length(0), _storages(storages), _task_optimality_wait_time(task_optimality_wait_time)
	{
		_nthreads = _storages.size();
		_threads.reserve(_nthreads);
		try {
			for (size_t i = 0; i < _nthreads; i++) {
				_threads.emplace_back([this, i]() { work(i); });
			}    
		}
		catch (...) {
			{
				std::lock_guard<std::mutex> lock(_mutex);
				_stop = true;
			}
			_cv.notify_all();
			for (auto& thread : _threads) {
				if (thread.joinable()) {
					thread.join();
				}
			}
			throw;
		}
	}

	~storage_thread_pool() {
		{
			std::lock_guard<std::mutex> lock(_mutex);
			_stop = true;
		}
		_cv.notify_all();
		for (auto& thread : _threads) {
			if (thread.joinable()) {
				thread.join();
			}
		}
	}

	template<class F, class... Args>
	std::future<typename std::invoke_result<F, storage&, Args...>::type> enqueue(
		F&& f, 
		vset<std::string> must_have_all_names,
		vset<std::string> must_have_any_names,
		vset<std::string> good_to_have_names,
		Args&&... args) 
	{
		using return_type = typename std::invoke_result<F, storage&, Args...>::type;

		auto task = std::make_shared<std::packaged_task<return_type(storage&)>>(
			std::bind(std::forward<F>(f), std::placeholders::_1, std::forward<Args>(args)...)
		);

		std::future<return_type> res = task->get_future();
		
		{
			std::lock_guard<std::mutex> lock(_mutex);
			_work.push_back({
				[task](storage& store) { (*task)(store); },
				std::move(must_have_all_names),
				std::move(must_have_any_names),
				std::move(good_to_have_names),
				std::chrono::steady_clock::now()
			});
		}
		_cv.notify_one();
		_work_length += 1;
		return res;
	}

	const std::vector<storage>& storages() const {
		return _storages;
	}

	void clear_from_storages(vset<std::string> storage_match_all, 
							vset<std::string> storage_match_any,
							vset<std::string> names_to_clear) 
	{
		{
			// Step 0: Pause the pool
			std::lock_guard<std::mutex> lock(_mutex);
			_paused.store(true);
		}

		// Step 1: Wait until all work is done
		{
			std::unique_lock<std::mutex> lock(_mutex);
			_cv.wait(lock, [this] { return _work_length.load() == 0; });
		}

		// Step 2: Clear storages
		{
			std::lock_guard<std::mutex> lock(_mutex);

			for (auto& store : _storages) {
				bool has_all = std::all_of(storage_match_all.begin(), storage_match_all.end(),
										[&](const std::string& s){ return store.exist(s); });

				bool has_any = storage_match_any.empty() ||
							std::any_of(storage_match_any.begin(), storage_match_any.end(),
										[&](const std::string& s){ return store.exist(s); });

				if (has_all && has_any) {
					for (const auto& s : names_to_clear) {
						store.clear(s);
					}
				}
			}
		}

	}

private:

	void work(size_t index) {
		auto& store = _storages[index];

		while (true) {
			
			std::function<void(storage&)> task;
			{
				std::unique_lock<std::mutex> lock(_mutex);

				_cv.wait(lock, [this] { return _stop || !_work.empty(); });
				if (_stop) return;

				// Find best matching task for this thread
				auto now = std::chrono::steady_clock::now();
				
				int best_good_matches = -1;
				bool perfect_good_match = false;
				
				auto best_it = _work.end();
				for (auto it = _work.begin(); it != _work.end(); ++it) {
					// Hard constraints: must_have_all (empty set is vacuously true)
					const auto& must_all = it->must_all;
					if (!std::all_of(must_all.begin(), must_all.end(),
									[&](const std::string& s){ return store.exist(s); }))
						continue;

					// Hard constraints: must_have_any (empty means vacuously true)
					const auto& must_any = it->must_any;
					if (!(must_any.empty() ||
						std::any_of(must_any.begin(), must_any.end(),
									[&](const std::string& s){ return store.exist(s); })))
						continue;

					// Soft constraint scoring: good_to_have
					const auto& gth = it->good_to_have;
					int good_matches = 0;
					for (const auto& s : gth) if (store.exist(s)) ++good_matches;

					if (!gth.empty() && good_matches == static_cast<int>(gth.size())) {
						// Perfect soft match — take immediately.
						perfect_good_match = true;
						best_it = it;
						break;
					}

					if (good_matches > best_good_matches) {
						best_good_matches = good_matches;
						best_it = it;
					}
				}

				// Pick logic: perfect match or waited long enough
				if (best_it != _work.end()) {
					auto wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - best_it->enqueue_time).count();
					if (perfect_good_match || wait_ms > _task_optimality_wait_time) { // configurable wait time
						task = std::move(best_it->func);
						_work.erase(best_it);
					}
				}
			}

			if (task) {
				task(store);
				_work_length--;
			} else {
				std::this_thread::sleep_for(std::chrono::milliseconds(2));
			}
		}
	}

private:
	i32 _nthreads;
	std::vector<storage> _storages;
	std::vector<std::thread> _threads;
	i32 _task_optimality_wait_time;

	std::condition_variable _cv;
	std::mutex _mutex;

	std::atomic<bool> _paused;

	struct WorkItem {
		std::function<void(storage&)> func;
		std::set<std::string> must_all;
		std::set<std::string> must_any;
		std::set<std::string> good_to_have;
		std::chrono::steady_clock::time_point enqueue_time;
	};

	std::list<WorkItem> _work;

	bool _stop;
	std::atomic<int> _work_length;

};


export class storage_thread_pool_interface {
public:

	storage_thread_pool_interface(std::shared_ptr<storage_thread_pool> thread_pool, 
									vset<std::string> must_have_all_names = {},
									vset<std::string> must_have_any_names = {},
									vset<std::string> good_to_have_names = {})
		:   _thread_pool(std::move(thread_pool)), 
			_must_have_all_names(std::move(must_have_all_names)),
			_must_have_any_names(std::move(must_have_any_names)), 
			_good_to_have_names(std::move(good_to_have_names))
			{}

	template<class F, class... Args>
	std::future<typename std::invoke_result<F, storage&, Args...>::type> enqueue(
		F&& f, vset<std::string> good_to_have_names, Args&&... args) {
		return _thread_pool->enqueue(
			std::forward<F>(f), 
			_must_have_all_names, 
			_must_have_any_names, 
			merge(_good_to_have_names, good_to_have_names),
			std::forward<Args>(args)...
		);
	}

	void clear_from_storages(vset<std::string> names_to_clear) {
		_thread_pool->clear_from_storages(
			_must_have_all_names, 
			_must_have_any_names, 
			names_to_clear
		);
	}

private:
	std::shared_ptr<storage_thread_pool> _thread_pool;
	vset<std::string> _must_have_all_names;
	vset<std::string> _must_have_any_names;
	vset<std::string> _good_to_have_names;
};


}
