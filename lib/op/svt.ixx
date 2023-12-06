module;

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

export module svt;

import <optional>;
import <random>;
import <future>;
import <queue>;

import block;

import hasty_util;
import torch_util;
import thread_pool;
import device;
import proxop;
import vec;

namespace hasty {

	namespace svt {

		export at::Tensor extract_block(const at::Tensor& tensor, const Block<4>& block, const std::optional<std::vector<int64_t>>& perms, int flatten_split = 1, bool transpose = false);

		export at::Tensor extract_block(const at::Tensor& tensor, const Block<3>& block, const std::optional<std::vector<int64_t>>& perms, int flatten_split = 1, bool transpose = false);

		export at::Tensor svt_hard(const at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

		export void svt_hard_inplace(at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

		export at::Tensor svt_soft(const at::Tensor& in, float lambda, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

		export void svt_soft_inplace(at::Tensor& in, float lambda, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

		export void insert_block(at::Tensor& in, const Block<4>& block, at::Tensor& block_tensor, const std::optional<std::vector<int64_t>>& perms, bool transpose = false);

		export void insert_block(at::Tensor& in, const Block<3>& block, at::Tensor& block_tensor, const std::optional<std::vector<int64_t>>& perms, bool transpose = false);

		export class BlocksSVTBase {
		public:

			struct DeviceContext {

				DeviceContext(const at::Stream& stream) :
					stream(stream) {}

				at::Stream stream;
			};

			BlocksSVTBase(const std::vector<DeviceContext>& contexts);

		protected:
			std::mutex _mutex;
			std::vector<DeviceContext> _dcontexts;
			int32_t _nctxt;
			std::unique_ptr<ContextThreadPool<DeviceContext>> _tpool;
		};

		export class Random3DBlocksSVT : public BlocksSVTBase {
		public:

			Random3DBlocksSVT(const std::vector<DeviceContext>& contexts);

			void apply(at::Tensor in, int64_t nblocks, std::array<int64_t, 3> block_shape, double thresh, bool soft);

		private:

			void block_svt_step(DeviceContext& dctxt, at::Tensor& in, const Block<3>& block, double thresh, bool soft);

		};

		export class OldNormal3DBlocksSVT : public BlocksSVTBase {
		public:

			OldNormal3DBlocksSVT(const std::vector<DeviceContext>& contexts);

			void apply(at::Tensor in, const std::array<int64_t, 3>& block_strides, const std::array<int64_t, 3>& block_shape,
				int64_t block_iter, double thresh, bool soft);

		private:

			void block_svt_step(DeviceContext& dctxt, at::Tensor& in, const Block<3>& block, double thresh, bool soft);

		};


		export template<hasty::DeviceContextConcept DContext>
		class Normal3DBlocksSVT {
		public:

			Normal3DBlocksSVT(sptr<ContextThreadPool<DContext>> threadpool)
				: _threadpool(std::move(threadpool))
			{}

			void apply(at::Tensor& in, const std::array<int64_t, 3>& block_strides, const std::array<int64_t, 3>& block_shape,
				int64_t block_iter, double thresh, bool soft)
			{
				c10::InferenceMode inference_guard;

				int nctxt = _threadpool->nthreads();

				std::deque<std::future<void>> futures;

				if (block_iter < 1)
					throw std::runtime_error("Can't perform NormalBlocksSVT with block_iter < 1");

				std::array<int64_t, 3> bounds{ in.size(2) - block_shape[0], in.size(3) - block_shape[1], in.size(4) - block_shape[2] };

				int Sx = in.size(2);
				int Sy = in.size(3);
				int Sz = in.size(4);

				int bx = Sx / block_strides[0];
				int by = Sy / block_strides[1];
				int bz = Sz / block_strides[2];

				std::function<void(DeviceContext&)> blockrunner;

				std::array<std::vector<int>, 3> shifts;

				std::default_random_engine generator;

				// Randomize shifts
				for (int d = 0; d < 3; ++d) {
					shifts[d].resize(block_iter);
					shifts[d][0] = 0;

					if (block_iter > 1) {
						std::uniform_int_distribution<int> distribution(0, block_shape[d]);
						for (int biter = 1; biter < block_iter; ++biter) {
							shifts[d][biter] = distribution(generator);
						}
					}
				}

				// Create blocks
				for (int iter = 0; iter < block_iter; ++iter) {

					int shiftx = shifts[0][iter];
					int shifty = shifts[1][iter];
					int shiftz = shifts[2][iter];

					for (int nx = 0; nx < bx; ++nx) {
						int sx = nx * block_strides[0] + shiftx;
						int ex = sx + block_shape[0];

						if (ex >= Sx)
							continue;

						for (int ny = 0; ny < by; ++ny) {
							int sy = ny * block_strides[1] + shifty;
							int ey = sy + block_shape[1];

							if (ey >= Sy)
								continue;

							for (int nz = 0; nz < bz; ++nz) {
								int sz = nz * block_strides[2] + shiftz;
								int ez = sz + block_shape[2];

								if (ez >= Sz)
									continue;

								Block<3> block;
								block.first_corner[0] = sx; block.second_corner[0] = ex;
								block.first_corner[1] = sy; block.second_corner[1] = ey;
								block.first_corner[2] = sz; block.second_corner[2] = ez;

								// Actual block push

								blockrunner = [this, &in, block, thresh, soft](DeviceContext& context) {
									block_svt_step(context, in, block, thresh, soft);
									};

								futures.emplace_back(_threadpool->enqueue(blockrunner));

								if (futures.size() > 32 * nctxt) {
									torch_util::future_catcher(futures.front());
									futures.pop_front();
								}
							}
						}
					}
				}

				// we wait for all promises
				while (futures.size() > 0) {
					torch_util::future_catcher(futures.front());
					futures.pop_front();
				}


			}

		private:

			void block_svt_step(DeviceContext& dctxt, at::Tensor& in, const Block<3>& block, double thresh, bool soft)
			{
				c10::InferenceMode inference_guard;
				int nctxt = _threadpool->nthreads();

				auto stream = dctxt.stream();
				c10::cuda::CUDAStreamGuard guard(stream);

				at::Tensor cuda_block;
				if (nctxt > 1)
				{
					at::Tensor tmp_block;
					{
						std::lock_guard<std::mutex> lock(_mutex);
						tmp_block = hasty::svt::extract_block(in, block, std::nullopt, 0, true).detach().clone();
					}
					cuda_block = tmp_block.to(stream.device());
				}
				else {
					cuda_block = hasty::svt::extract_block(in, block, std::nullopt, 0, true).to(stream.device());
				}

				//std::cout << cuda_block << std::endl;

				at::Tensor low_ranked = soft ?
					hasty::svt::svt_soft(cuda_block, (float)(thresh), std::nullopt) :
					hasty::svt::svt_hard(cuda_block, (int)(thresh + 0.5), std::nullopt);

				if (nctxt > 1)
				{
					at::Tensor back_block = low_ranked.cpu();
					{
						std::lock_guard<std::mutex> lock(_mutex);
						hasty::svt::insert_block(in, block, back_block, std::nullopt, true);
					}
				}
				else {
					at::Tensor back_block = low_ranked.cpu();
					hasty::svt::insert_block(in, block, back_block, std::nullopt, true);
				}
			}

		private:
			std::mutex _mutex;

			sptr<ContextThreadPool<DContext>> _threadpool;
		};

		export template<hasty::DeviceContextConcept DContext>
		class Normal3DBlocksSVTProxOp : public op::ProximalOperator {
		public:

			Normal3DBlocksSVTProxOp(sptr<ContextThreadPool<DContext>> threadpool, 
				const std::array<int64_t, 3>& block_strides, const std::array<int64_t, 3>& block_shape, bool soft)
				: _svtop(std::move(threadpool)), _block_strides(block_strides), _block_shape(block_shape), _soft(soft) {}

			op::Vector _apply(const op::Vector& input, double alpha) const override
			{
				if (input.has_children()) {
					throw std::runtime_error("Can't use Normal3DBlocksSVTProxOp on a vector with children");
				}

				return _svtop.apply(input.tensor(), alpha);
			}

		private:
			Normal3DBlocksSVT<DContext> _svtop;
			std::array<int64_t, 3> _block_strides;
			std::array<int64_t, 3> _block_shape;
			double _thresh;
			bool _soft;
		};



	}
}

