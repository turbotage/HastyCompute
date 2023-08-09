#pragma once

#include "py_util.hpp"

namespace hasty {
	namespace ffi {

		class LIB_EXPORT FunctionLambda : public torch::CustomClassHolder {
		public:

			FunctionLambda(const std::string& script, const std::string& entry, at::TensorList captures);

			void apply(at::Tensor in) const;

		private:
			std::shared_ptr<at::CompilationUnit> _cunit;
			std::string _entry;
			std::vector<at::Tensor> _captures;
		};

	}
}


namespace hasty {
	namespace dummy {

		void LIB_EXPORT dummy(at::TensorList tensorlist);

		void LIB_EXPORT stream_dummy(const at::optional<at::ArrayRef<at::Stream>>& streams, const torch::Tensor& in);

	}
}