module;

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

export module batch_sense;

import <optional>;
import <functional>;

import torch_util;
import sense;
import thread_pool;

import precond;
import opalgs;
import opalgebra;
import mriop;

namespace hasty {

	namespace mri {

		export struct InnerData {
			at::Tensor weights;
			at::Tensor kdata;
		};
		export using InnerBatchFetcher = std::function<mri::CoilManipulator(int32_t, const InnerData&, at::Stream)>;
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

			mri::CoilManipulator getCoilManipulator(int32_t inner_batch, const InnerData& data, at::Stream stream) {
				if (fetcher.has_value()) {
					return (*fetcher)(inner_batch, data, stream);
				}
				return mri::CoilManipulator();
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
						[](int32_t inner_batch, const InnerData& data, at::Stream) -> mri::CoilManipulator {
							return mri::CoilManipulator().setMidApply([&data](at::Tensor& in, int32_t coil) -> void {
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
						[](int32_t inner_batch, const InnerData& data, at::Stream) -> mri::CoilManipulator {
							return mri::CoilManipulator().setMidApply([&data](at::Tensor& in, int32_t coil) -> void {
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
						[](int32_t inner_batch, const InnerData& data, at::Stream) -> mri::CoilManipulator {
							return mri::CoilManipulator().setMidApply([&data](at::Tensor& in, int32_t coil) -> void {
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



		export class SenseDeviceContext {
		public:
			virtual const at::Tensor& smaps() const = 0;
			virtual const at::Stream& stream() const = 0;
		};

		export template<typename T>
		concept SenseDeviceContextConcept = std::derived_from<T, SenseDeviceContext>;

		export class DefaultSenseDeviceContext : public SenseDeviceContext {
		public:
			at::Tensor _smaps;
			at::Stream _stream;

			const at::Tensor& smaps() const override { return _smaps; }

			const at::Stream& stream() const override { return _stream; }
		
		};

		export template<typename SDeviceContext>
		requires SenseDeviceContextConcept<SDeviceContext>
		class SenseBatchConjugateGradientLoader : public op::BatchConjugateGradientLoader<SDeviceContext> {
		public:

			SenseBatchConjugateGradientLoader(
				std::vector<at::Tensor> coords, std::vector<int64_t> nmodes,
				std::vector<at::Tensor> kdata, at::Tensor smaps,
				std::shared_ptr<op::Admm::Context> ctx,
				at::optional<std::vector<at::Tensor>> preconds = at::nullopt)
				: _coords(std::move(coords)), _nmodes(std::move(nmodes)),
				_kdata(std::move(kdata)), _smaps(std::move(smaps)),
				_ctx(std::move(ctx)), _preconds(std::move(preconds))
			{}

			// maybe this should take in another SDeviceContext
			op::ConjugateGradientLoadResult load(SDeviceContext& sdctx, size_t idx) override
			{
				SenseDeviceContext& dctx = dynamic_cast<SenseDeviceContext&>(sdctx);

				c10::InferenceMode im_guard;
				at::Stream strm = dctx.stream();
				c10::cuda::CUDAStreamGuard guard(strm);

				auto device = dctx.stream().device();

				auto coords = _coords[idx].to(device, true);
				auto kdata = _kdata[idx].to(device, true);

				std::vector<int64_t> coils(dctx.smaps().size(0));
				std::generate(coils.begin(), coils.end(), [n = int64_t(0)]() mutable { return n++; });

				std::shared_ptr<op::Operator> CGop;

				std::shared_ptr<op::SenseNOp> SHS = op::SenseNOp::Create(coords, _nmodes, dctx.smaps(), coils);
				std::shared_ptr<op::AdjointableOp> AHA;

				std::shared_ptr<op::AdjointableOp> AH;
				std::shared_ptr<op::AdjointableOp> A;
				std::shared_ptr<op::AdjointableOp> B;

				op::Vector z;
				op::Vector u;
				std::optional<op::Vector> c;

				op::Vector CGvec;

				{
					std::unique_lock<std::mutex> lock(_ctx->ctxmut);

					auto localA = op::downcast<op::AdjointableVStackedOp>(_ctx->A);
					if (!localA)
						throw std::runtime_error("SenseAdmmLoader requires A to be AdjointableVStackedOp");
					A = op::downcast<op::AdjointableOp>(localA->get_slice_ptr(idx)->to_device(dctx.stream));
					AH = A->adjoint();

					auto localB = op::downcast<op::AdjointableVStackedOp>(_ctx->B);
					if (!localB)
						throw std::runtime_error("SenseAdmmLoader requires B to be AdjointableVStackedOp");
					B = op::downcast<op::AdjointableOp>(localB->get_slice_ptr(idx)->to_device(dctx.stream));

					if (_ctx->AHA != nullptr) {
						auto localAHA = op::downcast<op::AdjointableVStackedOp>(_ctx->AHA);
						if (!localAHA)
							throw std::runtime_error("SenseAdmmLoader requires AHA to be AdjointableVStackedOp");

						AHA = op::downcast<op::AdjointableOp>(std::move(localAHA->get_slice_ptr(idx)->to_device(dctx.stream)));
					}
					else {
						AHA = op::upcast<op::AdjointableOp>(std::move(op::mul(AH, A)));
					}

					if (!AHA)
						throw std::runtime_error("AHA could not be cast to an AdjointableOp on this device");

					z = _ctx->z[idx].copy().to_device(dctx.stream());
					u = _ctx->u[idx].copy().to_device(dctx.stream());
					if (_ctx->c.has_value())
						c = (*_ctx->c)[idx].copy().to_device(dctx.stream());
				}

				AHA = op::mul(at::tensor(0.5 * _ctx->rho), AHA);

				CGop = op::staticcast<op::AdjointableOp>(op::add(std::move(SHS), std::move(AHA)));

				if (c.has_value())
					CGvec = _ctx->rho * AH->apply(*c - B->apply(z) - u);
				else
					CGvec = (0.5 * _ctx->rho) * AH->apply(B->apply(z) - u);

				CGvec += SHS->apply_backward(op::Vector(kdata));

				return op::ConjugateGradientLoadResult{ std::move(CGop), std::move(CGvec), _preconds.has_value() ?
					mri::CirculantPreconditionerOp::Create((*_preconds)[idx], true, std::nullopt, false) : nullptr
				};
			}



		private:
			std::vector<at::Tensor> _coords;
			std::vector<int64_t> _nmodes;
			std::vector<at::Tensor> _kdata;
			at::Tensor _smaps;

			std::shared_ptr<op::Admm::Context> _ctx;

			at::optional<std::vector<at::Tensor>> _preconds;
		};

		
		export template<SenseDeviceContextConcept SDeviceContext>
		class BatchSenseAdmmMinimizer : public hasty::op::AdmmMinimizer {
		public:

			BatchSenseAdmmMinimizer(std::shared_ptr<hasty::ContextThreadPool<SDeviceContext>> sensethreadpool,
				std::vector<at::Tensor> coords, std::vector<int64_t> nmodes,
				std::vector<at::Tensor> kdata, at::Tensor smaps,
				std::shared_ptr<op::Admm::Context> ctx,
				at::optional<std::vector<at::Tensor>> preconds = at::nullopt)
			{
				auto senseloader =
					std::make_shared<SenseBatchConjugateGradientLoader<SDeviceContext>>(coords, nmodes, kdata, smaps, ctx, preconds);

				_batchcg = std::make_unique<op::BatchConjugateGradient<SenseDeviceContext>>(senseloader, sensethreadpool);

				//_batchcg = new op::BatchConjugateGradient<SDeviceContext>(senseloader, sensethreadpool);

				_iters = 10;
				_tol = 0.0;
			}

			void set_iters(int iters);

			void set_tol(double tol);

		private:


			std::unique_ptr<op::BatchConjugateGradient<SDeviceContext>> _batchcg;
			int _iters;
			double _tol;

		};


	}

}

