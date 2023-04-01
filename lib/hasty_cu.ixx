module;

export module hasty_cu;

import std;

import hasty_util;

namespace hasty {

	export class RawCudaFunction {
	public:

		virtual std::string dfid() const = 0;

		virtual std::string dcode() const = 0;

		virtual std::string kfid() const { return "k_" + dfid(); }

		virtual std::string kcode() const { throw NotImplementedError(); }

		virtual vec<sptr<RawCudaFunction>> deps() const
		{
			return vec<sptr<RawCudaFunction>>();
		}
	};

	export void code_generator(std::string& code, const RawCudaFunction& func) {

		auto deps = func.deps();
		for (auto& dep : deps) {
			code_generator(code, *dep);
		}

		code += func.dcode();
	}

}