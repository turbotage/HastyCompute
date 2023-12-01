module;

export module permute;

import <memory>;
import <vector>;
import <string>;

import hasty_util;
import hasty_compute;
import hasty;

namespace hasty {
	namespace cuda {

		export class LID : public RawCudaFunction {
		public:

			static std::string s_dfid() { return "lid"; }

			std::string dfid() const override { return s_dfid(); };

			static std::string s_dcode() {
				return
R"cuda(
__device__
int lid(int i, int j) {
	return i*(i+1)/2 + j;
}
)cuda";
			}

			std::string dcode() const override { return s_dcode(); }

		};


		export class MaxDiagAbs : public RawCudaFunction {
		public:

			MaxDiagAbs(i32 ndim, hasty::dtype dtype)
				: _ndim(ndim), _dtype(dtype)
			{}

			std::string dfid() const override {
				return "max_diag_abs" +
					dims_type({ _ndim }, _dtype);
			}

			static std::string s_dcode() {
				return
R"cuda(
__device__
int {{funcid}}(const {{fp_type}}* mat, int offset) 
{
	{{fp_type}} max_abs = -1.0f;
	int max_index = 0;
	for (int i = offset; i < {{ndim}}; ++i) {
		if ({{abs_fid}}(mat[i*{{ndim}}+i]) > max_abs) {
			max_index = i;
		}
	}
	return max_index; 
}
)cuda";
			}

			std::string dcode() const override {
				vec<std::pair<std::string, std::string>> rep_dict = {
					{"ndim", std::to_string(_ndim)},
					{ "fp_type", dtype_to_string(_dtype)},
					{ "funcid", dfid()}
				};
				std::string ret = hasty::code_replacer(s_dcode(), rep_dict);
				return ret;
			}

		private:

			i32 _ndim;
			hasty::dtype _dtype;

		};


		export class RowInterchangeI : public RawCudaFunction {
		public:

			RowInterchangeI(i32 ndim, hasty::dtype dtype)
				: _ndim(ndim), _dtype(dtype)
			{}

			std::string dfid() const override {
				return "row_interchange_i" +
					dims_type({ _ndim }, _dtype);
			}

			static std::string s_dcode() {
				return
R"cuda(
__device__
void {{funcid}}({{fp_type}}* mat, int ii, int jj) 
{
	for (int k = 0; k < {{ndim}}; ++k) {
		int ikn = ii*{{ndim}}+k;
		int jkn = jj*{{ndim}}+k;
		{{fp_type}} temp;
		temp = mat[ikn];
		mat[ikn] = mat[jkn];
		mat[jkn] = temp;
	}
}
)cuda";
			}

			std::string dcode() const override {
				vec<std::pair<std::string, std::string>> rep_dict = {
					{"ndim", std::to_string(_ndim)},
					{ "fp_type", dtype_to_string(_dtype)},
					{ "funcid", dfid()}
				};
				std::string ret = hasty::code_replacer(s_dcode(), rep_dict);
				return ret;
			}

		private:

			i32 _ndim;
			hasty::dtype _dtype;

		};


		export class ColInterchangeI : public RawCudaFunction {
		public:

			ColInterchangeI(i32 ndim, hasty::dtype dtype)
				: _ndim(ndim), _dtype(dtype)
			{}

			std::string dfid() const override {
				return "col_interchange_i" +
					dims_type({ _ndim }, _dtype);
			}

			static std::string s_dcode() {
				return
R"cuda(
__device__
void {{funcid}}({{fp_type}}* mat, int ii, int jj) 
{
	for (int k = 0; k < {{ndim}}; ++k) {
		int kin = k*{{ndim}}+ii;
		int kjn = k*{{ndim}}+jj;
		{{fp_type}} temp;
		temp = mat[kin];
		mat[kin] = mat[kjn];
		mat[kjn] = temp;
	}
}
)cuda";
			}

			std::string dcode() const override {
				vec<std::pair<std::string, std::string>> rep_dict = {
					{"ndim", std::to_string(_ndim)},
					{ "fp_type", dtype_to_string(_dtype)},
					{ "funcid", dfid()}
				};
				std::string ret = hasty::code_replacer(s_dcode(), rep_dict);
				return ret;
			}

		private:

			i32 _ndim;
			hasty::dtype _dtype;

		};


		export class DiagPivot : public RawCudaFunction {
		public:

			DiagPivot(i32 ndim, hasty::dtype dtype)
				: _ndim(ndim), _dtype(dtype), _deps()
			{
				_deps.emplace_back(std::make_shared<MaxDiagAbs>(_ndim, _dtype));
				_deps.emplace_back(std::make_shared<RowInterchangeI>(_ndim, _dtype));
				_deps.emplace_back(std::make_shared<ColInterchangeI>(_ndim, _dtype));
			}

			std::string dfid() const override {
				return "diag_pivot" +
					dims_type({ _ndim }, _dtype);
			}

			static std::string s_dcode() {
				return
R"cuda(
__device__
void {{funcid}}({{fp_type}}* mat, int* perm) 
{
	for (int i = 0; i < {{ndim}}; ++i) {
		perm[i] = i;
	}
	for (int i = 0; i < {{ndim}}; ++i) {
		int max_abs = {{max_diag_abs_fid}}(mat, i);
		{{row_interchange_fid}}(mat, i, max_abs);
		{{col_interchange_fid}}(mat, i, max_abs);
		int temp = perm[i];
		perm[i] = perm[max_abs];
		perm[max_abs] = temp;
	}
}
)cuda";
			}

			std::string dcode() const override {
				vec<std::pair<std::string, std::string>> rep_dict = {
					{"ndim", std::to_string(_ndim)},
					{"fp_type", dtype_to_string(_dtype)},
					{"funcid", dfid()},
					{"max_diag_abs_fid", _deps[0]->dfid()},
					{"row_interchange_fid", _deps[1]->dfid()},
					{"col_interchange_fid", _deps[2]->dfid()}
				};
				std::string ret = hasty::code_replacer(s_dcode(), rep_dict);
				return ret;
			}

			vec<sptr<RawCudaFunction>> deps() const override {
				return _deps;
			}

		private:

			i32 _ndim;
			hasty::dtype _dtype;

			vec<sptr<RawCudaFunction>> _deps;

		};


		export class PermuteVec : public RawCudaFunction {
		public:

			PermuteVec(i32 ndim, hasty::dtype dtype)
				: _ndim(ndim), _dtype(dtype)
			{}

			std::string dfid() const override {
				return "permute_vec" +
					dims_type({ _ndim }, _dtype);
			}

			static std::string s_dcode() {
				return
R"cuda(
__device__
void {{funcid}}(const {{fp_type}}* vec, const int* perm, {{fp_type}}* ovec) 
{
	for (int i = 0; i < {{ndim}}; ++i) {
		ovec[i] = vec[perm[i]];
	}
}
)cuda";
			}

			std::string dcode() const override {
				vec<std::pair<std::string, std::string>> rep_dict = {
					{"ndim", std::to_string(_ndim)},
					{"fp_type", dtype_to_string(_dtype)},
					{"funcid", dfid()}
				};
				std::string ret = hasty::code_replacer(s_dcode(), rep_dict);
				return ret;
			}

		private:

			i32 _ndim;
			hasty::dtype _dtype;

		};


		export class InvPermuteVec : public RawCudaFunction {
		public:

			InvPermuteVec(i32 ndim, hasty::dtype dtype)
				: _ndim(ndim), _dtype(dtype)
			{}

			std::string dfid() const override {
				return "inv_permute_vec" +
					dims_type({ _ndim }, _dtype);
			}

			static std::string s_dcode() {
				return
R"cuda(
__device__
void {{funcid}}(const {{fp_type}}* vec, const int* perm, {{fp_type}}* ovec) 
{
	for (int i = 0; i < {{ndim}}; ++i) {
		ovec[perm[i]] = vec[i];
	}
}
)cuda";
			}

			std::string dcode() const override {
				vec<std::pair<std::string, std::string>> rep_dict = {
					{"ndim", std::to_string(_ndim)},
					{"fp_type", dtype_to_string(_dtype)},
					{"funcid", dfid()}
				};
				std::string ret = hasty::code_replacer(s_dcode(), rep_dict);
				return ret;
			}

		private:

			i32 _ndim;
			hasty::dtype _dtype;

		};

	}
}