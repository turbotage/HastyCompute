#pragma once

#include "block.hpp"
#include "../torch_util.hpp"
#include "sense.hpp"
#include "../threading/thread_pool.hpp"

namespace hasty {

	using TensorVec = std::vector<at::Tensor>;
	using TensorVecVec = std::vector<TensorVec>;

		
	// Runs 
	class LLR_4DEncodes {
	public:

		struct Options {
			Options(const c10::Device& device, const c10::Stream& stream)
			{
				devices.emplace_back(device, std::vector{stream});
			}

			void push_back_device(const c10::Device& device, const c10::Stream& stream)
			{
				for (auto& dev : devices) {
					if (dev.first == device) {
						dev.second.push_back(stream);
					}
				}
			}

			std::vector<std::pair<c10::Device,std::vector<c10::Stream>>> devices;
		};

	public:

		LLR_4DEncodes(const LLR_4DEncodes&) = delete;
		LLR_4DEncodes(LLR_4DEncodes&&) = delete;
		LLR_4DEncodes& operator=(const LLR_4DEncodes&) = delete;

		LLR_4DEncodes(
			const Options& options,
			at::Tensor& image,
			const TensorVecVec& coords,
			const at::Tensor& smaps,
			const TensorVecVec& kdata);

		LLR_4DEncodes(
			const Options& options,
			at::Tensor& image,
			const TensorVecVec& coords,
			const at::Tensor& smaps,
			const TensorVecVec& kdata,
			const TensorVecVec& weights);

		LLR_4DEncodes(
			const Options& options,
			at::Tensor& image,
			TensorVecVec&& coords,
			const at::Tensor& smaps,
			TensorVecVec&& kdata);

		LLR_4DEncodes(
			const Options& options,
			at::Tensor& image,
			TensorVecVec&& coords,
			const at::Tensor& smaps,
			TensorVecVec&& kdata,
			TensorVecVec&& weights);

		void step_llr(const std::vector<Block<3>>& blocks, const std::vector<int16_t>& ranks);

		void step_l2_sgd(const std::vector<
			std::pair<int, std::vector<int>>>& encode_coil_indices);

	private:

		void construct();

	private:

		struct DeviceContext {

			DeviceContext(const c10::Device& dev, const c10::Stream& stream) : 
				device(dev), stream(stream) {}
			DeviceContext(const DeviceContext&) = delete;
			DeviceContext& operator=(const DeviceContext&) = delete;
			DeviceContext(DeviceContext&&) = default;

			std::string str();

			c10::Device device;
			c10::Stream stream;

			at::Tensor image;
			at::Tensor kdata;
			std::optional<at::Tensor> weights;
			at::Tensor coords;

			at::Tensor out;

			at::Tensor in_storage; // TensorOptions as image
			//at::Tensor freq_storage; // TensorOptions as kdata

			at::Tensor smaps;

			std::unique_ptr<SenseNormal> sense;

		};

		void coil_encode_step(DeviceContext& dctxt, int frame, int encode, const std::vector<int32_t>& coils);

		void block_svt_step(DeviceContext& dctxt, const Block<3>& block, int16_t rank);

	private:

		std::unique_ptr<ContextThreadPool<DeviceContext>> _tpool;

		Options _options;
		at::Tensor& _image;
		TensorVecVec _coords;
		const at::Tensor& _smaps;
		TensorVecVec _kdata;
		std::optional<TensorVecVec> _weights;

		std::vector<int64_t> _nmodes;

		int _nframe;
		std::mutex _copy_back_mutex;

		std::vector<DeviceContext> _dcontexts;

	};

	class RandomBlocksSVT {
	public:

		struct DeviceContext {

			DeviceContext(const c10::Stream& stream) : 
				stream(stream) {}
			DeviceContext(const DeviceContext&) = delete;
			DeviceContext& operator=(const DeviceContext&) = delete;
			DeviceContext(DeviceContext&&) = default;

			std::string str();

			c10::Stream stream;

		};
	public:

		RandomBlocksSVT(std::vector<DeviceContext>& contexts,
			at::Tensor& image, int32_t nblocks, int32_t block_size, double thresh, bool soft);

	private:

		void block_svt_step(DeviceContext& dctxt, const Block<3>& block, double thresh, bool soft);

	private:
		std::mutex _mutex;
		at::Tensor _image;
		int32_t _nctxt;
	};

}