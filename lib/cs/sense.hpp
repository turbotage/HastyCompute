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

	class SenseNormalToeplitz {
	public:

		SenseNormalToeplitz(const at::Tensor& coords, const std::vector<int64_t>& nmodes, double tol = 1e-5);

		SenseNormalToeplitz(at::Tensor&& diagonal, const std::vector<int64_t>& nmodes);

		void apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage1, at::Tensor& storage2,
			const at::Tensor& smaps, const std::vector<int32_t>& coils) const;

	private:

		NormalNufftToeplitz _normal_nufft;

	};
	

	class BatchedSense {
	public:

		// i: denotes coil number i
		// (Tensor: NUFFT(image[frame] * smap_i), Tensor: kdata[frame]_i)
		typedef std::function<void(at::Tensor&, at::Tensor&)> FreqManipulator;
		// (Tensor: NUFFT(image[frame] * smap_i), Tensor: weights[frame])
		typedef std::function<void(at::Tensor&, at::Tensor&)> WeightedManipulator;
		// (Tensor: NUFFT(image[frame] * smap_i), Tensor: kdata[frame]_i, Tensor: weights[frame])
		typedef std::function<void(at::Tensor&, at::Tensor&, at::Tensor&)> WeightedFreqManipulator;
		
	public:

		struct DeviceContext {

			DeviceContext(const c10::Device& dev, const c10::Stream& stream) :
				device(dev), stream(stream) {}

			//DeviceContext(DeviceContext&&) = default;
			//DeviceContext& operator=(DeviceContext&&) = default;

			std::string str();

			c10::Device device;
			c10::Stream stream;

			at::Tensor smaps;
		};

	public:

		BatchedSense(
			std::vector<DeviceContext>&& contexts,
			at::Tensor&& diagonals);

		BatchedSense(
			std::vector<DeviceContext>&& contexts,
			std::optional<TensorVec> coords,
			std::optional<TensorVec> kdata,
			std::optional<TensorVec> weights);

		void apply(at::Tensor& in,
			const std::optional<std::vector<std::vector<int32_t>>>& coils,
			const std::optional<WeightedManipulator>& wmanip,
			const std::optional<FreqManipulator>& fmanip,
			const std::optional<WeightedFreqManipulator>& wfmanip);

		void apply_toep(at::Tensor& in,
			const std::optional<std::vector<std::vector<int32_t>>>& coils);

	private:

		void construct();

		void apply_outer_batch(DeviceContext& dctxt, int32_t outer_batch, at::Tensor& in,
			const std::vector<int32_t>& coils,
			const std::optional<WeightedManipulator>& wmanip,
			const std::optional<FreqManipulator>& fmanip,
			const std::optional<WeightedFreqManipulator>& wfmanip);

		void apply_outer_batch_toep(DeviceContext& dctxt, int32_t outer_batch, at::Tensor& in,
			const std::vector<int32_t>& coils);

	private:

		at::Tensor					_diagonals;
		TensorVec					_coords;
		TensorVec					_kdata;
		TensorVec					_weights;

		int32_t _ndim;
		std::vector<int64_t> _nmodes;

		std::mutex _copy_back_mutex;
		std::vector<DeviceContext> _dcontexts;
	};

}


