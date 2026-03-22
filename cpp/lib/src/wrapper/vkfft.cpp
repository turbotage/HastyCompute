module;

#include "pch.hpp"

module vkfft;

namespace hasty {
    namespace fft {

        std::array<VkFFT_Cache, (std::size_t)device_alias::MAX_CUDA_DEVICES> global_vkfft_cache = {};

    }
}