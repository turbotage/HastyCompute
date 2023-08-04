#pragma once

#include "../fft/nufft.hpp"
#include "../threading/thread_pool.hpp"

namespace hasty {

	using TensorVec = std::vector<at::Tensor>;
	using TensorVecVec = std::vector<TensorVec>;

	class Sense {
	public:

		Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes, bool adjoint = false);

		at::Tensor apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
			const std::optional<at::Tensor>& storage,
			const std::optional<std::function<void(at::Tensor&, int32_t)>>& manip, bool sum = false, bool sumnorm=false);

	private:

		Nufft _nufft;
		std::vector<int64_t> _nmodes;
		bool _adjoint;

	};

	class SenseNormal {
	public:

		SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes, bool adjoint=false);

		/*
		void apply(const at::Tensor& in, at::Tensor& out, const std::vector<std::reference_wrapper<const at::Tensor>>& smaps,
			const std::optional<at::Tensor>& in_storage, const std::optional<at::Tensor>& freq_storage,
			const std::optional<std::function<void(at::Tensor&,int32_t)>>& freq_manip);
		*/

		void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
			const std::optional<at::Tensor>& in_storage, const std::optional<at::Tensor>& freq_storage,
			const std::optional<std::function<void(at::Tensor&,int32_t)>>& freq_manip);

	private:

		NufftNormal _normal_nufft;
		std::vector<int64_t> _nmodes;
		bool _adjoint;

	};

	class SenseNormalToeplitz {
	public:

		SenseNormalToeplitz(const at::Tensor& coords, const std::vector<int64_t>& nmodes, double tol);

		SenseNormalToeplitz(at::Tensor&& diagonal, const std::vector<int64_t>& nmodes);

		void apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage1, at::Tensor& storage2,
			const at::Tensor& smaps, const std::vector<int64_t>& coils) const;

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

			DeviceContext(const c10::Stream& stream) :
				stream(stream) {}

			DeviceContext(const c10::Stream& stream, const at::Tensor& smaps) :
				stream(stream), smaps(smaps) {}

			std::string str();

			c10::Stream stream;
			at::Tensor smaps;
		};

	public:

		BatchedSense(
			std::vector<DeviceContext>&& contexts,
			at::Tensor&& diagonals);

		BatchedSense(
			std::vector<DeviceContext>&& contexts,
			const std::optional<TensorVec>& coords,
			const std::optional<TensorVec>& kdata,
			const std::optional<TensorVec>& weights);

		at::Tensor apply_forward(const at::Tensor& in, at::TensorList out,
			bool sum, bool sumnorm,
			const std::optional<std::vector<std::vector<int64_t>>>& coils,
			const std::optional<WeightedManipulator>& wmanip,
			const std::optional<FreqManipulator>& fmanip,
			const std::optional<WeightedFreqManipulator>& wfmanip);

		at::Tensor apply_adjoint(const TensorVec& in, at::Tensor& out,
			bool sum, bool sumnorm,
			const std::optional<std::vector<std::vector<int64_t>>>& coils,
			const std::optional<WeightedManipulator>& wmanip,
			const std::optional<FreqManipulator>& fmanip,
			const std::optional<WeightedFreqManipulator>& wfmanip);

		void apply_normal(at::Tensor& in,
			const std::optional<std::vector<std::vector<int64_t>>>& coils,
			const std::optional<WeightedManipulator>& wmanip,
			const std::optional<FreqManipulator>& fmanip,
			const std::optional<WeightedFreqManipulator>& wfmanip);

		void apply_normal_adjoint(TensorVec& in,
			const std::optional<std::vector<std::vector<int64_t>>>& coils,
			const std::optional<WeightedManipulator>& wmanip,
			const std::optional<FreqManipulator>& fmanip,
			const std::optional<WeightedFreqManipulator>& wfmanip);

		void apply_toep(at::Tensor& in,
			const std::optional<std::vector<std::vector<int64_t>>>& coils);

	private:

		void construct();

		at::Tensor apply_outer_batch_forward(DeviceContext& dctxt, int32_t outer_batch, const at::Tensor& in, at::Tensor out,
			bool sum, bool sumnorm,
			const std::vector<int64_t>& coils,
			const std::optional<WeightedManipulator>& wmanip,
			const std::optional<FreqManipulator>& fmanip,
			const std::optional<WeightedFreqManipulator>& wfmanip);

		at::Tensor apply_outer_batch_adjoint(DeviceContext& dctxt, int32_t outer_batch, const at::Tensor& in, at::Tensor& out,
			bool sum, bool sumnorm,
			const std::vector<int64_t>& coils,
			const std::optional<WeightedManipulator>& wmanip,
			const std::optional<FreqManipulator>& fmanip,
			const std::optional<WeightedFreqManipulator>& wfmanip);

		void apply_outer_batch_normal(DeviceContext& dctxt, int32_t outer_batch, at::Tensor& in,
			const std::vector<int64_t>& coils,
			const std::optional<WeightedManipulator>& wmanip,
			const std::optional<FreqManipulator>& fmanip,
			const std::optional<WeightedFreqManipulator>& wfmanip);

		void apply_outer_batch_normal_adjoint(DeviceContext& dctxt, int32_t outer_batch, at::Tensor& in,
			const std::vector<int64_t>& coils,
			const std::optional<WeightedManipulator>& wmanip,
			const std::optional<FreqManipulator>& fmanip,
			const std::optional<WeightedFreqManipulator>& wfmanip);

		void apply_outer_batch_toep(DeviceContext& dctxt, int32_t outer_batch, at::Tensor& in,
			const std::vector<int64_t>& coils);

	private:

		at::Tensor					_diagonals;
		TensorVec					_coords;
		TensorVec					_kdata;
		TensorVec					_weights;

		int32_t _ndim;
		std::vector<int64_t> _nmodes;

		std::mutex _copy_back_mutex;
		std::vector<DeviceContext> _dcontexts;
		int32_t _nctxt;
	};

}


