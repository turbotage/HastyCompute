module;

export module lsqnonlin;

import <memory>;
import <stdexcept>;
import <vector>;
import <string>;
import <optional>;

import hasty;
import hasty_compute;
import permute;

import nlsq_symbolic;



namespace hasty {
	namespace cuda {

		export class FGradF {
		public:

			FGradF(uptr<RawCudaFunction> f, uptr<RawCudaFunction> gradf, int32_t nparam, int32_t nconst, const std::string& name)
				: _f(std::move(f)), _gradf(std::move(gradf)), _nparam(nparam), _nconst(nconst), _name(name)
			{

			}


		private:
			int32_t _nparam;
			int32_t _nconst;
			std::string _name;

			uptr<RawCudaFunction> _f;
			uptr<RawCudaFunction> _gradf;

		};



	}
}