module;

#include <torch/torch.h>

export module batch_sense;

import <optional>;
import <functional>;

import torch_util;
import sense;
import thread_pool;

namespace hasty {

	namespace batched_sense {

		export struct InnerData {
			at::Tensor weights;
			at::Tensor kdata;
		};
		export using InnerBatchFetcher = std::function<sense::CoilManipulator(int32_t, const InnerData&, at::Stream)>;
		export using InnerBatchApplier = std::function<void(at::Tensor&, int32_t, const InnerData&, at::Stream)>;
		export struct InnerManipulator {

			InnerManipulator() = default;
			InnerManipulator(const InnerBatchFetcher& fet) : fetcher(fet) {}

			std::optional<InnerBatchApplier> preapplier;
			std::optional<InnerBatchApplier> postapplier;
			std::optional<InnerBatchFetcher> fetcher;

			InnerManipulator& setPreApply(const InnerBatchApplier& apply) {
				if (preapplier.has_value())
					throw std::runtime_error("Tried to set non nullopt applier");
				preapplier = std::make_optional(apply);
				return *this;
			}

			InnerManipulator& setPostApply(const InnerBatchApplier& apply) {
				if (postapplier.has_value())
					throw std::runtime_error("Tried to set non nullopt applier");
				postapplier = std::make_optional(apply);
				return *this;
			}

			InnerManipulator& setFetcher(const InnerBatchFetcher& fetch) {
				if (fetcher.has_value())
					throw std::runtime_error("Tried to set non nullopt fetcher");
				fetcher = std::make_optional(fetch);
				return *this;
			}

			sense::CoilManipulator getCoilManipulator(int32_t inner_batch, const InnerData& data, at::Stream stream) {
				if (fetcher.has_value()) {
					return (*fetcher)(inner_batch, data, stream);
				}
				return sense::CoilManipulator();
			}

		};

		export using OuterBatchFetcher = std::function<InnerManipulator(int32_t, at::Stream)>;
		export using OuterBatchApplier = std::function<void(at::Tensor&, int32_t, at::Stream)>;
		export struct OuterManipulator {

			OuterManipulator() = default;
			OuterManipulator(const OuterBatchFetcher& fet) : fetcher(fet) {}

			std::optional<OuterBatchApplier> preapplier;
			std::optional<OuterBatchApplier> postapplier;
			std::optional<OuterBatchFetcher> fetcher;

			OuterManipulator& setPreApply(const OuterBatchApplier& apply) {
				if (preapplier.has_value())
					throw std::runtime_error("Tried to set non nullopt applier");
				preapplier = std::make_optional(apply);
				return *this;
			}

			OuterManipulator& setPostApply(const OuterBatchApplier& apply) {
				if (postapplier.has_value())
					throw std::runtime_error("Tried to set non nullopt applier");
				postapplier = std::make_optional(apply);
				return *this;
			}

			OuterManipulator& setFetcher(const OuterBatchFetcher& fetch) {
				if (fetcher.has_value())
					throw std::runtime_error("Tried to set non nullopt fetcher");
				fetcher = std::make_optional(fetch);
				return *this;
			}

			OuterManipulator& setStandardFreqManipulator() {
				if (fetcher.has_value())
					throw std::runtime_error("Tried to set non nullopt fetcher");

				fetcher = [](int32_t outer_batch, at::Stream) -> InnerManipulator {
					return InnerManipulator(
						[](int32_t inner_batch, const InnerData& data, at::Stream) -> sense::CoilManipulator {
							return sense::CoilManipulator().setMidApply([&data](at::Tensor& in, int32_t coil) -> void {
								in.sub_(data.kdata.select(0, coil).unsqueeze(0));
								});
						}
					);
					};
				return *this;
			}

			OuterManipulator& setStandardWeightedManipulator() {
				if (fetcher.has_value())
					throw std::runtime_error("Tried to set non nullopt fetcher");

				fetcher = [](int32_t outer_batch, at::Stream) -> InnerManipulator {
					return InnerManipulator(
						[](int32_t inner_batch, const InnerData& data, at::Stream) -> sense::CoilManipulator {
							return sense::CoilManipulator().setMidApply([&data](at::Tensor& in, int32_t coil) -> void {
								in.mul_(data.weights);
								});
						}
					);
					};
				return *this;
			}

			OuterManipulator& setStandardWeightedFreqManipulator() {
				if (fetcher.has_value())
					throw std::runtime_error("Tried to set non nullopt fetcher");

				fetcher = [](int32_t outer_batch, at::Stream) -> InnerManipulator {
					return InnerManipulator(
						[](int32_t inner_batch, const InnerData& data, at::Stream) -> sense::CoilManipulator {
							return sense::CoilManipulator().setMidApply([&data](at::Tensor& in, int32_t coil) -> void {
								in.sub_(data.kdata.select(0, coil).unsqueeze(0));
								in.mul_(data.weights);
								});
						}
					);
					};
				return *this;
			}

			InnerManipulator getInnerManipulator(int32_t outer_batch, at::Stream stream) const {
				if (fetcher.has_value()) {
					return (*fetcher)(outer_batch, stream);
				}
				return InnerManipulator();
			}

		};


		export class BatchedSenseBase {
		public:
			struct DeviceContext {

				DeviceContext(const at::Stream& stream) :
					stream(stream) {}

				DeviceContext(const at::Stream& stream, const at::Tensor& smaps) :
					stream(stream), smaps(smaps) {}

				at::Stream stream;
				at::Tensor smaps;
			};

			BatchedSenseBase(
				const std::vector<DeviceContext>& contexts,
				const at::TensorList& coords,
				const at::optional<at::TensorList>& kdata,
				const at::optional<at::TensorList>& weights);

			int32_t nouter_batches() { return _coords.size(); }
			int32_t ncoils() { return _dcontexts[0].smaps.size(0); }
			int32_t ndim() { return _ndim; }
			bool has_kdata() { return _kdata.size() > 0; }
			bool has_weights() { return _weights.size() > 0; }

		protected:
			TensorVec					_coords;
			TensorVec					_kdata;
			TensorVec					_weights;

			int32_t _ndim;
			std::vector<int64_t> _nmodes;

			std::mutex _copy_back_mutex;
			std::vector<DeviceContext> _dcontexts;
			int32_t _nctxt;
			std::unique_ptr<ContextThreadPool<DeviceContext>> _tpool;
		};

		export class BatchedSense : public BatchedSenseBase {
		public:

			BatchedSense(
				const std::vector<DeviceContext>& contexts,
				const at::TensorList& coords,
				const at::optional<at::TensorList>& kdata,
				const at::optional<at::TensorList>& weights);

			void apply(const at::Tensor& in, at::TensorList out, const std::vector<std::vector<int64_t>>& coils, const OuterManipulator& manips);

			static OuterManipulator standard_kdata_manipulator();

			static OuterManipulator standard_weighted_manipulator();

			static OuterManipulator standard_weighted_kdata_manipulator();

		private:

			void apply_outer_batch(const at::Tensor& in, at::Tensor out, const std::vector<int64_t>& coils,
				DeviceContext& dctxt, int32_t outer_batch, const OuterManipulator& outmanip);

		};

		export class BatchedSenseAdjoint : public BatchedSenseBase {
		public:

			BatchedSenseAdjoint(
				const std::vector<DeviceContext>& contexts,
				const at::TensorList& coords,
				const at::optional<at::TensorList>& kdata,
				const at::optional<at::TensorList>& weights);

			void apply(const at::TensorList& in, at::Tensor out, const std::vector<std::vector<int64_t>>& coils, const OuterManipulator& manips);

			static OuterManipulator standard_kdata_manipulator();

			static OuterManipulator standard_weighted_manipulator();

			static OuterManipulator standard_weighted_kdata_manipulator();

		private:

			void apply_outer_batch(const at::Tensor& in, at::Tensor out, const std::vector<int64_t>& coils,
				DeviceContext& dctxt, int32_t outer_batch, const OuterManipulator& outmanip);

		};

		export class BatchedSenseNormal : public BatchedSenseBase {
		public:

			BatchedSenseNormal(
				const std::vector<DeviceContext>& contexts,
				const at::TensorList& coords,
				const at::optional<at::TensorList>& kdata,
				const at::optional<at::TensorList>& weights);

			void apply(const at::Tensor& in, at::Tensor out, const std::vector<std::vector<int64_t>>& coils, const OuterManipulator& manips);

			/*
			void apply_addinplace(at::Tensor& inout, const at::Tensor& scalar, const std::vector<std::vector<int64_t>>& coils, const OuterManipulator& manips);
			*/

			static OuterManipulator standard_kdata_manipulator();

			static OuterManipulator standard_weighted_manipulator();

			static OuterManipulator standard_weighted_kdata_manipulator();

		private:

			void apply_outer_batch(const at::Tensor& in, at::Tensor out, const std::vector<int64_t>& coils,
				DeviceContext& dctxt, int32_t outer_batch, const OuterManipulator& outmanip);
			/*
			void apply_addinplace_outer_batch(at::Tensor& inout, const at::Tensor& scalar, const std::vector<int64_t>& coils,
				DeviceContext& dctxt, int32_t outer_batch, const OuterManipulator& outmanip);
			*/

		};


	}

}

