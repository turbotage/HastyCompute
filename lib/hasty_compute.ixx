module;

export module hasty_compute;

import std;

//import <iostream>;
//import <sstream>;
//import <string>;
//import <vector>;

import <arrayfire.h>;

import hasty_util;

namespace hasty {

	export std::string dtype_to_string(af::dtype dtype) {
		switch (dtype) {
		case af::dtype::f32:
			return "float";
		case af::dtype::c32:
			return "complex<float>";
		}
		throw std::runtime_error("Not Implemented Yet");
	}

	export std::string dims_type(const vec<i32>& ndims, af::dtype dtype)
	{
		std::string ret = "_";
		for (auto& ndim : ndims) {
			ret += std::to_string(ndim) + "_";
		}

		switch (dtype) {
		case af::dtype::f32:
			ret += "f"; break;
		case af::dtype::c32:
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