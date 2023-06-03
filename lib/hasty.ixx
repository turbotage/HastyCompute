module;

export module hasty;

import <memory>;
import <stdexcept>;
import <vector>;
import <string>;

export import hasty_util;

namespace hasty {
	namespace cuda {

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

	export std::string hash_string(size_t num, size_t len) 
	{
		std::string numstr = std::to_string(num);
		static std::string lookup("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789");
		std::string ret;
		for (int i = 0; i < numstr.size(); i += 2) {
			std::string substr = numstr.substr(i, 2);
			int index = std::atoi(substr.c_str());
			index = index % lookup.size();
			ret += lookup[index];
		}
		return ret.substr(0,len);
	}

}