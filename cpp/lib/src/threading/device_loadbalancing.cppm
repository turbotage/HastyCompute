module;

#include <cuda_runtime.h>

export module hasty_threading_mod:device_loadbalancing;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import :dep_threadpool;

namespace hasty {

// ─── CUDALoadBalancer ────────────────────────────────────────────────────────
//
// Routes CUDA tasks to the global DepThreadPool across available devices.
// Submitted lambdas receive a Device as their first argument.
// Selection: fewest pending tasks, tie-break by free CUDA memory.

export class CUDALoadBalancer {
    struct DeviceQueue {
        Device           dev;
        std::atomic<int> pending{0};
    };

    std::vector<std::unique_ptr<DeviceQueue>> _queues;

    static std::size_t _free_mem(int dev_idx) noexcept {
        std::size_t free_bytes = 0, total = 0;
        cudaSetDevice(dev_idx);
        cudaMemGetInfo(&free_bytes, &total);
        return free_bytes;
    }

    int _select() const {
        int best        = 0;
        int min_pending = _queues[0]->pending.load(std::memory_order_relaxed);
        for (int i = 1; i < (int)_queues.size(); ++i) {
            int p = _queues[i]->pending.load(std::memory_order_relaxed);
            if (p < min_pending) { min_pending = p; best = i; }
        }
        std::size_t best_mem = 0;
        for (int i = 0; i < (int)_queues.size(); ++i) {
            if (_queues[i]->pending.load(std::memory_order_relaxed) == min_pending) {
                auto m = _free_mem((int)_queues[i]->dev.index);
                if (m > best_mem) { best_mem = m; best = i; }
            }
        }
        return best;
    }

    int _index_for(Device dev) const {
        for (int i = 0; i < (int)_queues.size(); ++i)
            if (_queues[i]->dev == dev) return i;
        throw std::invalid_argument("CUDALoadBalancer: device not registered");
    }

    template<class F>
    auto _submit_to(int qi,
                    std::vector<std::shared_ptr<TaskDepState>> deps,
                    F&& f)
        -> DepFuture<std::invoke_result_t<F, Device>>
    {
        auto& q = *_queues[qi];
        q.pending.fetch_add(1, std::memory_order_relaxed);
        Device dev = q.dev;

        auto wrapped = [this, qi, dev, fn = std::forward<F>(f)]() mutable {
            cudaSetDevice((int)dev.index);
            struct Guard {
                CUDALoadBalancer* lb; int qi;
                ~Guard() { lb->_queues[qi]->pending.fetch_sub(1, std::memory_order_relaxed); }
            } guard{this, qi};
            return fn(dev);
        };

        return deps.empty()
            ? global_dep_thread_pool.enqueue(std::move(wrapped))
            : global_dep_thread_pool.enqueue_after(std::move(deps), std::move(wrapped));
    }

public:
    explicit CUDALoadBalancer(std::vector<int> device_indices = {})
    {
        if (device_indices.empty()) {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
                throw std::runtime_error("CUDALoadBalancer: no CUDA devices available");
            device_indices.resize(count);
            std::iota(device_indices.begin(), device_indices.end(), 0);
        }
        _queues.reserve(device_indices.size());
        for (int idx : device_indices) {
            auto q = std::make_unique<DeviceQueue>();
            q->dev = Device{eDeviceType::CUDA, (DeviceIndex)idx};
            _queues.push_back(std::move(q));
        }
    }

    int    num_devices()  const { return (int)_queues.size(); }
    Device device(int i)  const { return _queues[i]->dev; }

    template<class F>
    auto submit(F&& f) -> DepFuture<std::invoke_result_t<F, Device>>
    {
        return _submit_to(_select(), {}, std::forward<F>(f));
    }

    template<class F>
    auto submit_after(std::vector<std::shared_ptr<TaskDepState>> deps, F&& f)
        -> DepFuture<std::invoke_result_t<F, Device>>
    {
        return _submit_to(_select(), std::move(deps), std::forward<F>(f));
    }

    template<class F>
    auto submit_on(Device dev, F&& f) -> DepFuture<std::invoke_result_t<F, Device>>
    {
        return _submit_to(_index_for(dev), {}, std::forward<F>(f));
    }

    template<class F>
    auto submit_on_after(Device dev,
                         std::vector<std::shared_ptr<TaskDepState>> deps,
                         F&& f)
        -> DepFuture<std::invoke_result_t<F, Device>>
    {
        return _submit_to(_index_for(dev), std::move(deps), std::forward<F>(f));
    }

    void sync() {
        std::vector<DepFuture<void>> fences;
        fences.reserve(_queues.size());
        for (int i = 0; i < (int)_queues.size(); ++i)
            fences.push_back(_submit_to(i, {}, [](Device dev) {
                cudaSetDevice((int)dev.index);
                cudaDeviceSynchronize();
            }));
        for (auto& f : fences) f.get();
    }
};

export extern CUDALoadBalancer global_cuda_load_balancer;

}
