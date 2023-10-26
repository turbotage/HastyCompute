#pragma once

#include "../torch_util.hpp"
#include "../op/op.hpp"
#include "../threading/thread_pool.hpp"

namespace hasty {
	namespace mri {

		template<typename F, typename DeviceContext> concept CGLoaderTrait =
			std::invocable<F, DeviceContext, size_t> &&
			std::same_as <
			std::tuple<op::Operator, op::Vector, at::optional<op::Operator>>,
			std::invoke_result_t<F, DeviceContext, size_t>>;


		class ConjugateGradientLoader {

		};

		class BatchConjugateGradient {
		public:

		private:

		};


		template<CGLoaderTrait CGLoader, typename DeviceContext>
		void batch_cg(const CGLoader& loader, ContextThreadPool<DeviceContext>& pool)
		{
			
		}


	}
}

