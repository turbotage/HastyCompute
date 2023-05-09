#pragma once

#include "../fft/nufft.hpp"
#include "../threading/thread_pool.hpp"

namespace hasty {

	using TensorVec = std::vector<at::Tensor>;
	using TensorVecVec = std::vector<TensorVec>;

	class Sense {
	public:

	private:
	};

	class SenseNormal {
	public:

		SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

		void apply(const at::Tensor& in, at::Tensor& out, const std::vector<std::reference_wrapper<const at::Tensor>>& smaps,
			std::optional<at::Tensor> in_storage, std::optional<at::Tensor> freq_storage,
			std::optional<std::function<void(at::Tensor&,int32_t)>> freq_manip);

		void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int32_t>& coils,
			std::optional<at::Tensor> in_storage, std::optional<at::Tensor> freq_storage,
			std::optional<std::function<void(at::Tensor&,int32_t)>> freq_manip);

	private:

		NufftNormal _normal_nufft;

	};

	class BatchedSense {
	public:

		// i: denotes coil number i
		// (Tensor: NUFFT(image[frame] * smap_i), Tensor: kdata[frame]_i)
		typedef std::function<void(at::Tensor&, at::Tensor&)> FreqManipulator;
		// (Tensor: NUFFT(image[frame] * smap_i), Tensor: kdata[frame]_i, Tensor: weights[frame]_i)
		typedef std::function<void(at::Tensor&, at::Tensor&, at::Tensor&)> WeightedFreqManipulator;

	private:

		struct DeviceContext {

			DeviceContext(const c10::Device& dev, const c10::Stream& stream) :
				device(dev), stream(stream) {}

			std::string str();

			c10::Device device;
			c10::Stream stream;

			at::Tensor smaps;
		};

	public:

		BatchedSense(
			std::vector<DeviceContext>&& contexts,
			TensorVec&& coords);

		BatchedSense(
			std::vector<DeviceContext>&& contexts,
			TensorVec&& coords,
			TensorVec&& kdata);

		BatchedSense(
			std::vector<DeviceContext>&& contexts,
			TensorVec&& coords,
			TensorVec&& kdata,
			TensorVec&& weights);

		void apply(at::Tensor& in,
			const std::optional<std::vector<int32_t>>& coils,
			const std::optional<WeightedFreqManipulator>& wmanip,
			const std::optional<FreqManipulator>& manip);

		void apply_toep(at::Tensor& in,
			const std::optional<std::vector<int32_t>>& coils);

	private:

		void construct();

		void apply_outer_batch(DeviceContext& dctxt, int32_t outer_batch, at::Tensor& in,
			const std::vector<int32_t>& coils,
			const std::optional<WeightedFreqManipulator>& wmanip,
			const std::optional<FreqManipulator>& manip);

		void apply_outer_batch_toep(DeviceContext& dctxt, int32_t outer_batch, at::Tensor& in,
			const std::vector<int32_t>& coils);

	private:

		TensorVec					_coords;
		TensorVec					_kdata;
		std::optional<TensorVec>	_weights;

		int32_t _ndim;
		std::vector<int64_t> _nmodes;

		std::mutex _copy_back_mutex;
		std::vector<DeviceContext> _dcontexts;
	};

}


