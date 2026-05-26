module;

#include <cuda_runtime.h>

module hasty_threading_mod;

import std;

namespace hasty {

// Global dep thread pool — hardware_concurrency workers.
DepThreadPool global_dep_thread_pool{
    static_cast<int>(std::thread::hardware_concurrency())
};

// Global CUDA load balancer — all available devices, discovered at startup
// via cudaGetDeviceCount (no hasty_tensor_mod import needed here).
CUDALoadBalancer global_cuda_load_balancer{[]() -> std::vector<int> {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return {0};
    std::vector<int> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}()};

}
