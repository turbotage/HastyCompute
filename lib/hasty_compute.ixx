module;

export module hasty_compute;

#ifdef STL_AS_MODULES
import std;
#else
import <memory>;
import <stdexcept>;
import <vector>;
import <string>;
#endif

import hasty_util;

namespace hasty {

	export enum class dtype {
		f32,
		c64
	};

	export std::string dtype_to_string(hasty::dtype dtype) {
		switch (dtype) {
		case hasty::dtype::f32:
			return "float";
		case hasty::dtype::c64:
			return "complex<float>";
		}
		throw NotImplementedError();
		//return "";
	}

	export std::string dims_type(const vec<i32>& ndims, hasty::dtype dtype)
	{
		std::string ret = "_";
		for (auto& ndim : ndims) {
			ret += std::to_string(ndim) + "_";
		}

		switch (dtype) {
		case hasty::dtype::f32:
			ret += "f"; break;
		case hasty::dtype::c64:
			ret += "cf"; break;
		}
		return ret;
	}

	export std::string code_replacer(const std::string& code, const vecp<std::string, std::string>& replace_dict)
	{
		std::string ret = code;

		for (auto& pair : replace_dict) {
			while (true) {
				size_t index = ret.find("{{" + pair.first + "}}");
				if (index != std::string::npos) {
					ret.replace(index, pair.first.length() + 4, pair.second);
				}
				else {
					break;
				}
			}
		}

		return ret;
	}

}