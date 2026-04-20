module;

module hasty_vkfft_mod;

namespace hasty {
namespace fft {

std::array<VkFFT_Cache, (std::size_t)device_alias::MAX_CUDA_DEVICES> global_vkfft_cache = {};

}
}