module;

export module hasty_compute;

import <iostream>;
import <sstream>;
import <string>;
import <vector>;

using namespace std;

export {
	using i8 = int8_t;
	using i16 = int16_t;
	using i32 = int32_t;
	using i64 = int64_t;

	auto NotImplementedThrow = std::runtime_error("Not Implemented Error");
}


namespace hasty {

	export enum eDType {
		F32,
		CF32
	};

	export string dtype_to_string(eDType dtype) {
		switch (i32(dtype)) {
		case F32:
			return "float";
		case CF32:
			return "complex<float>";
		}
		throw NotImplementedThrow;
	}

	export namespace fidend {

		string dims_type(const std::vector<i32>& ndims, eDType dtype)
		{
			string ret = "_";
			for (auto& ndim : ndims) {
				ret += to_string(ndim) + "_";
			}
			
			switch (i32(dtype)) {
			case F32:
				ret += "f"; break;
			case CF32:
				ret += "cf"; break;
			}
			return ret;
		}

	}


	export class RawCudaFunction {
	public:

		virtual string dfid() const = 0;

		virtual string dcode() const = 0;

		virtual string kfid() const { throw NotImplementedThrow; }

		virtual string kcode() const { throw NotImplementedThrow; }

		virtual vector<shared_ptr<RawCudaFunction>> deps() const
			{ return vector<shared_ptr<RawCudaFunction>>(); }
	};



	export string code_replacer(const string& code, const vector<pair<string, string>>& replace_dict)
	{
		string ret = code;

		for (auto& pair : replace_dict) {
			while (true) {
				size_t index = ret.find("{{" + pair.first + "}}");
				if (index != string::npos) {
					ret.replace(index, pair.first.length() + 4, pair.second);
				}
				else {
					break;
				}
			}
		}

		return ret;
	}

	export void code_generator(std::string& code, const RawCudaFunction& func) {

		auto deps = func.deps();
		for (auto& dep : deps) {
			code_generator(code, *dep);
		}

		code += func.dcode();
	}

}