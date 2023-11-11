module;

#include "../torch_util.hpp"

export module svt;

import <optional>;
import block;

import thread_pool;

namespace hasty {

	template<typename DeviceContext>
	class ContextThreadPool;

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

		export class Normal3DBlocksSVT : public BlocksSVTBase {
		public:

			Normal3DBlocksSVT(const std::vector<DeviceContext>& contexts);

			void apply(at::Tensor in, const std::array<int64_t, 3>& block_strides, const std::array<int64_t, 3>& block_shape,
				int64_t block_iter, double thresh, bool soft);

		private:

			void block_svt_step(DeviceContext& dctxt, at::Tensor& in, const Block<3>& block, double thresh, bool soft);

		};

	}
}

