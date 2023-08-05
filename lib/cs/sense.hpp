#pragma once

#include "../fft/nufft.hpp"
#include "../threading/thread_pool.hpp"

namespace hasty {

	using TensorVec = std::vector<at::Tensor>;
	using TensorVecVec = std::vector<TensorVec>;

	struct InnerData {
		at::Tensor weights;
		at::Tensor kdata;
	};

	using CoilApplier = std::function<void(at::Tensor&, int32_t)>;
	
	using InnerBatchFetcher = std::function<CoilApplier(int32_t, const InnerData&, at::Stream)>;
	using InnerBatchApplier = std::function<void(at::Tensor&, int32_t, const InnerData&, at::Stream)>;
	struct InnerManipulator {

		InnerManipulator() = default;
		InnerManipulator(const InnerBatchFetcher& fet) : fetcher(fet) {}

		std::optional<InnerBatchApplier> preapplier;
		std::optional<InnerBatchApplier> postapplier;
		std::optional<InnerBatchFetcher> fetcher;

		void setPreApply(const InnerBatchApplier& apply) {
			if (preapplier.has_value())
				throw std::runtime_error("Tried to set non nullopt applier");
			preapplier = std::make_optional(apply);
		}

		void setPostApply(const InnerBatchApplier& apply) {
			if (postapplier.has_value())
				throw std::runtime_error("Tried to set non nullopt applier");
			postapplier = std::make_optional(apply);
		}

		void setFetcher(const InnerBatchFetcher& fetch) {
			if (fetcher.has_value())
				throw std::runtime_error("Tried to set non nullopt fetcher");
			fetcher = std::make_optional(fetch);
		}

		std::optional<CoilApplier> getCoilApplier(int32_t inner_batch, const InnerData& data, at::Stream stream) {
			if (fetcher.has_value()) {
				return std::make_optional((*fetcher)(inner_batch, data, stream));
			}
			return std::nullopt;
		}

	};

	using OuterBatchFetcher = std::function<InnerManipulator(int32_t, at::Stream)>;
	using OuterBatchApplier = std::function<void(at::Tensor&, int32_t, at::Stream)>;
	struct OuterManipulator {

		OuterManipulator() = default;

		std::optional<OuterBatchApplier> preapplier;
		std::optional<OuterBatchApplier> postapplier;
		std::optional<OuterBatchFetcher> fetcher;

		void setPreApply(const OuterBatchApplier& apply) {
			if (preapplier.has_value())
				throw std::runtime_error("Tried to set non nullopt applier");
			preapplier = std::make_optional(apply);
		}
		
		void setPostApply(const OuterBatchApplier& apply) {
			if (postapplier.has_value())
				throw std::runtime_error("Tried to set non nullopt applier");
			postapplier = std::make_optional(apply);
		}
		
		void setFetcher(const OuterBatchFetcher& fetch) {
			if (fetcher.has_value())
				throw std::runtime_error("Tried to set non nullopt fetcher");
			fetcher = std::make_optional(fetch);
		}

		void setStandardFreqManipulator() {
			if (fetcher.has_value())
				throw std::runtime_error("Tried to set non nullopt fetcher");

			fetcher = [](int32_t outer_batch, at::Stream) -> InnerManipulator {

				return InnerManipulator(
					[](int32_t inner_batch, const InnerData& data, at::Stream) -> CoilApplier {
						return [&kdata = std::as_const(data.kdata)](at::Tensor& in, int32_t coil) -> void {
							in.sub_(kdata.select(0, coil).unsqueeze(0));
						};
					}
				);

			};
		}

		void setStandardWeightedManipulator() {
			if (fetcher.has_value())
				throw std::runtime_error("Tried to set non nullopt fetcher");

			fetcher = [](int32_t outer_batch, at::Stream) -> InnerManipulator {
				return InnerManipulator(
					[](int32_t inner_batch, const InnerData& data, at::Stream) -> CoilApplier {
						return [&weights = std::as_const(data.weights)](at::Tensor& in, int32_t coil) -> void {
							in.mul_(weights);
						};
					}
				);

			};
		}

		void setStandardWeightedFreqManipulator() {
			if (fetcher.has_value())
				throw std::runtime_error("Tried to set non nullopt fetcher");

			fetcher = [](int32_t outer_batch, at::Stream) -> InnerManipulator {
				return InnerManipulator(
					[](int32_t inner_batch, const InnerData& data, at::Stream) -> CoilApplier {
						return [&data](at::Tensor& in, int32_t coil) -> void {
							in.sub_(data.kdata.select(0, coil).unsqueeze(0));
							in.mul_(data.weights);
						};
					}
				);
			};
		}

		InnerManipulator getInnerManipulator(int32_t outer_batch, at::Stream stream) const {
			if (fetcher.has_value()) {
				return (*fetcher)(outer_batch, stream);
			}
			return InnerManipulator();
		}

	};





	class Sense {
	public:

		Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

		at::Tensor apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
			const std::optional<at::Tensor>& imspace_storage, const std::optional<at::Tensor>& kspace_storage,
			const std::optional<CoilApplier>& premanip,
			const std::optional<CoilApplier>& postmanip);

	private:
		Nufft _nufft;
		std::vector<int64_t> _nmodes;
	};

	class SenseAdjoint {
	public:

		SenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

		at::Tensor apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
			const std::optional<at::Tensor>& imspace_storage, const std::optional<at::Tensor>& kspace_storage,
			const std::optional<CoilApplier>& premanip,
			const std::optional<CoilApplier>& postmanip);

	private:
		Nufft _nufft;
		std::vector<int64_t> _nmodes;
	};

	class SenseNormal {
	public:

		SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

		void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
			const std::optional<at::Tensor>& imspace_storage, const std::optional<at::Tensor>& kspace_storage,
			const std::optional<CoilApplier>& premanip,
			const std::optional<CoilApplier>& midmanip,
			const std::optional<CoilApplier>& postmanip);

	private:

		NufftNormal _normal_nufft;
		std::vector<int64_t> _nmodes;

	};

	class SenseNormalAdjoint {
	public:

		SenseNormalAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

		void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
			const std::optional<at::Tensor>& imspace_storage,
			const std::optional<CoilApplier>& premanip,
			const std::optional<CoilApplier>& midmanip,
			const std::optional<CoilApplier>& postmanip);

	private:

		NufftNormal _normal_nufft;
		std::vector<int64_t> _nmodes;
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
	

	class BatchedSenseBase {
	public:

		struct DeviceContext {

			DeviceContext(const at::Stream& stream) :
				stream(stream) {}

			DeviceContext(const at::Stream& stream, const at::Tensor& smaps) :
				stream(stream), smaps(smaps) {}

			at::Stream stream;
			at::Tensor smaps;
		};
	};

	class BatchedSense : public BatchedSenseBase {
	public:

		BatchedSense(
			std::vector<DeviceContext>&& contexts,
			const at::TensorList& coords,
			const std::optional<at::TensorList>& kdata,
			const std::optional<at::TensorList>& weights);

		void apply(const at::Tensor& in, at::TensorList out, const std::vector<std::vector<int64_t>>& coils, const OuterManipulator& manips);

	private:

		void construct();

		void apply_outer_batch(const at::Tensor& in, at::Tensor out, const std::vector<int64_t>& coils, 
			DeviceContext& dctxt, int32_t outer_batch, const OuterManipulator& manips);

	private:
		TensorVec					_coords;
		TensorVec					_kdata;
		TensorVec					_weights;
		 
		int32_t _ndim;
		std::vector<int64_t> _nmodes;

		std::mutex _copy_back_mutex;
		std::vector<DeviceContext> _dcontexts;
		int32_t _nctxt;
	};


	class OldBatchedSense {
	public:


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


