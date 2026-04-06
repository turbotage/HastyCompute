module;

#include <pch.hpp>
#include <vkFFT.h>

export module hasty_vkfft_mod;

import std;
import hasty_util_mod;
import hasty_tensor_mod;

namespace hasty {
	namespace fft {

		export enum class VkFFT_Type {
			C2C,
			R2C,
			DCT1,
			DCT2,
			DCT3,
			DCT4,
			DST1,
			DST2,
			DST3,
			DST4
		};

		export enum class VkFFT_Convolution_Conjugation {
			NONE,
			CONJUGATE_INPUT,
			CONJUGATE_KERNEL
		};

		export class VkFFT_Cache {
		public:
			class VkFFT_Key {
			private:

				class AppWrapper {
				public:
					AppWrapper() 
						: initialized(false)
					{
						app = {};
						config = {};
					}
					~AppWrapper() {
						if (initialized) {
							deleteVkFFT(&app);
						}
					}

					void initialize() {
						if (!initialized) {
							VkFFTResult res = initializeVkFFT(&app, config);
							if (res != VKFFT_SUCCESS) {
								throw std::runtime_error("Failed to initialize VkFFT: " + std::to_string(res));
							}
							initialized = true;
						}
					}

					bool initialized = false;
					VkFFTConfiguration config;
					VkFFTApplication app;
				};

				u64 bufferSize;
			public:
				int device_idx;
				std::array<u64, VKFFT_MAX_FFT_DIMENSIONS> size{};
				std::array<u64, VKFFT_MAX_FFT_DIMENSIONS> fft_zeropad_left{};
				std::array<u64, VKFFT_MAX_FFT_DIMENSIONS> fft_zeropad_right{};
				std::array<bool, VKFFT_MAX_FFT_DIMENSIONS> performZeropadding{};
				std::array<bool, VKFFT_MAX_FFT_DIMENSIONS> omitDimension{};
				u32 numberBatches = 1;
				u16 FFTdim = 1;
				u16 fft_type = (u16)VkFFT_Type::C2C;
				u16 conjugateConvolution = (u16)VkFFT_Convolution_Conjugation::NONE;
				u16 coalescedMemory = 128;
				bool normalize = false;
				bool makeForwardPlanOnly = false;
				bool makeInversePlanOnly = false;
				bool frequencyZeroPadding = false;
				bool performConvolution = false;
				bool kernelConvolution = false;
				bool useLUT = false;
				bool useLUT_4step = false;

				bool keepShaderCode = false;

				std::shared_ptr<AppWrapper> appwrap;

				VkFFT_Key(int device_idx) 
					: device_idx(device_idx)
				{
				}

				bool operator==(const VkFFT_Key& other) const {
					return 
						device_idx == other.device_idx &&
						FFTdim == other.FFTdim &&
						fft_type == other.fft_type &&
						performConvolution == other.performConvolution &&
						kernelConvolution == other.kernelConvolution &&
						std::equal(std::begin(size), std::end(size), std::begin(other.size)) &&
						std::equal(std::begin(fft_zeropad_left), std::end(fft_zeropad_left), std::begin(other.fft_zeropad_left)) &&
						std::equal(std::begin(fft_zeropad_right), std::end(fft_zeropad_right), std::begin(other.fft_zeropad_right)) &&
						std::equal(std::begin(performZeropadding), std::end(performZeropadding), std::begin(other.performZeropadding)) &&
						numberBatches == other.numberBatches &&
						coalescedMemory == other.coalescedMemory &&
						std::equal(std::begin(omitDimension), std::end(omitDimension), std::begin(other.omitDimension)) &&
						conjugateConvolution == other.conjugateConvolution &&
						normalize == other.normalize &&
						makeForwardPlanOnly == other.makeForwardPlanOnly &&
						makeInversePlanOnly == other.makeInversePlanOnly &&
						frequencyZeroPadding == other.frequencyZeroPadding;
				}

				void build() {
					appwrap = std::make_unique<AppWrapper>();
					appwrap->config.device = &device_idx;
					appwrap->config.FFTdim = FFTdim;
					bufferSize = 1;
					for (int i = 0; i < VKFFT_MAX_FFT_DIMENSIONS; ++i) {
						appwrap->config.size[i] = size[i];
						bufferSize *= size[i] > 0 ? size[i] : 1;
						appwrap->config.fft_zeropad_left[i] = fft_zeropad_left[i];
						appwrap->config.fft_zeropad_right[i] = fft_zeropad_right[i];
						appwrap->config.performZeropadding[i] = performZeropadding[i] ? 1 : 0;
						appwrap->config.omitDimension[i] = omitDimension[i] ? 1 : 0;
					}
					appwrap->config.bufferSize = &bufferSize;
					appwrap->config.numberBatches = numberBatches;
					switch (fft_type) {
					case (int)VkFFT_Type::C2C:
						break;
					case (int)VkFFT_Type::R2C:
						appwrap->config.performR2C = 1;
						break;
					case (int)VkFFT_Type::DCT1:
					case (int)VkFFT_Type::DCT2:
					case (int)VkFFT_Type::DCT3:
					case (int)VkFFT_Type::DCT4:
						appwrap->config.performDCT = fft_type - (int)VkFFT_Type::DCT1 + 1;
						break;
					case (int)VkFFT_Type::DST1:
					case (int)VkFFT_Type::DST2:
					case (int)VkFFT_Type::DST3:
					case (int)VkFFT_Type::DST4:
						appwrap->config.performDST = fft_type - (int)VkFFT_Type::DST1 + 1;
						break;
					}
					appwrap->config.coalescedMemory = coalescedMemory;
					appwrap->config.normalize = normalize ? 1 : 0;
					appwrap->config.conjugateConvolution = (int)conjugateConvolution;
					appwrap->config.makeForwardPlanOnly = makeForwardPlanOnly ? 1 : 0;
					appwrap->config.makeInversePlanOnly = makeInversePlanOnly ? 1 : 0;
					appwrap->config.frequencyZeroPadding = frequencyZeroPadding ? 1 : 0;
					appwrap->config.performConvolution = performConvolution ? 1 : 0;
					appwrap->config.kernelConvolution = kernelConvolution ? 1 : 0;
					appwrap->config.useLUT = useLUT ? 1 : 0;
					appwrap->config.useLUT_4step = useLUT_4step ? 1 : 0;


					if (keepShaderCode) {
						appwrap->config.keepShaderCode = 1;
					}

					appwrap->initialize();
				}

				VkFFTApplication& get_app() {
					if (!appwrap) {
						build();
					} else if (!appwrap->initialized) {
						appwrap->initialize();
					}
					return appwrap->app;
				}
			};

			bool contains(const VkFFT_Key& key) {
				std::unique_lock<std::mutex> lock(_mutex);
				return _cache.contains(key);
			}

			bool erase(const VkFFT_Key& key) {
				std::unique_lock<std::mutex> lock(_mutex);
				return _cache.erase(key) > 0;
			}

			VkFFTApplication& get_or_create(const VkFFT_Key& key) {
				std::unique_lock<std::mutex> lock(_mutex);
				auto [retkey, created] = _cache.insert(key);
				if (created) {
					retkey.get().build();
				}
				return retkey.get().get_app();
			}

		private:
			std::mutex _mutex;
			vset<VkFFT_Key> _cache;
		};

		export extern std::array<VkFFT_Cache, (std::size_t)device_alias::MAX_CUDA_DEVICES> global_vkfft_cache;

	}
}