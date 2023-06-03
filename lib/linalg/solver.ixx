module;

export module solver;

import <memory>;
import <stdexcept>;
import <vector>;
import <string>;

import hasty_compute;
import hasty;
import permute;

namespace hasty {

	namespace cuda {

		export class GMW81 : public RawCudaFunction {
		public:

			GMW81(i32 ndim, hasty::dtype dtype)
				: _ndim(ndim), _dtype(dtype)
			{
			}

			std::string dfid() const override {
				return "gmw81" +
					dims_type({ _ndim }, _dtype);
			}

			static std::string s_dcode() {
				return
R"cuda(
__device__
void {{funcid}}({{fp_type}}* mat) 
{
	{{fp_type}} t0;
	{{fp_type}} t1 = 0.0f; // gamma
	{{fp_type}} t2 = 0.0f; // nu
	{{fp_type}} beta2 = {{machine_eps}};
	{{fp_type}} delta = {{machine_eps}};
	for (int i = 0; i < {{ndim}}; ++i) {
		for (int j = 0; j <= i; ++j) {
			t0 = {{abs_func}}(mat[i*{{ndim}}+j]);
			if (i == j) {
				if (t0 > t1)
					t1 = t0;
			} else {
				if (t0 > t2)
					t2 = t0;
			}
		}
	}
	if ({{ndim}} > 1) {
		t2 /= {{sqrt_func}}({{ndim}}*{{ndim}} - 1);
	}
	if (beta2 < t1)
		beta2 = t1;
	if (beta2 < t2)
		beta2 = t2;
	t0 = t1 + t2;
	if (t0 > 1.0f)
		delta *= t0;
	// delta = eps*max(gamma + nu, 1)
	// beta2 = max(gamma, nu/sqrt(n^^2-1), eps)
	for (int j = 0; j < {{ndim}}; ++j) { // compute column j
		
		for (int s = 0; s < j; ++s)
			mat[j*{{ndim}}+s] /= mat[s*{{ndim}}+s];
		for (int i = j + 1; i < {{ndim}}; ++i) {
			t0 = mat[i*{{ndim}}+j];
			for (int s = 0; s < j; ++s)
				t0 -= mat[j*{{ndim}}+s] * mat[i*{{ndim}}+s];
			mat[i*{{ndim}}+j] = t0;
		}
		t1 = 0.0f;
		for (int i = j + 1; i < {{ndim}}; ++i) {
			t0 = {{abs_func}}(mat[i*{{ndim}}+j]);
			if (t1 < t0)
				t1 = t0;
		}
		t1 *= t1;
		t2 = {{abs_func}}(mat[j*{{ndim}}+j]);
		if (t2 < delta)
			t2 = delta;
		t0 = t1 / beta2;
		if (t2 < t0)
			t2 = t0;
		mat[j*{{ndim}}+j] = t2;
		if (j < {{ndim}}) {
			for (int i = j + 1; i < {{ndim}}; ++i) {
				t0 = mat[i*{{ndim}}+j];
				mat[i*{{ndim}}+i] -= t0*t0/t2;
			}
		}
	}
}
)cuda";
			}

			std::string dcode() const override {
				vecp<std::string, std::string> rep_dict = {
					{"ndim", std::to_string(_ndim)},
					{"fp_type", dtype_to_string(_dtype)},
					{"funcid", dfid()}
				};

				switch (_dtype)
				{
				case hasty::dtype::f32:
				{
					rep_dict.emplace_back(std::pair{ "abs_func", "fabsf" });
					rep_dict.emplace_back(std::pair{ "sqrt_func", "sqrtf" });
					rep_dict.emplace_back(std::pair{ "machine_eps", "2e-7" });
				}
				break;
				default:
					throw NotImplementedError();
					break;
				}


				std::string ret = hasty::code_replacer(s_dcode(), rep_dict);
				return ret;
			}

			static std::string s_kcode() {
				return
R"cuda(
extern \"C\" __global__
void {{funcid}}({{fp_type}}* mat, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{fp_type}} mat_copy[{{ndim}}*{{ndim}}];
		int k = 0;
		for (int i = 0; i < {{ndim}}; ++i) {
			for (int j = 0; i <= j; ++j) {
				{{fp_type}} temp = mat[k*N+tid];
				mat_copy[i*{{ndim}}+j] = temp;
				mat_copy[j*{{ndim}}+i] = temp;
				++k;
			}
		}
		{{dfuncid}}(mat_copy);
		k = 0;
		for (int i = 0; i < {{ndim}}; ++i) {
			for (int j = 0; j <= i; ++j) {
				mat[k*{{ndim}}+tid] = mat_copy[i*{{ndim}}+j];
				++k;
			}
		}
	}
}
)cuda";
			}

			std::string kcode() const override {
				vecp<std::string, std::string> rep_dict = {
					{"ndim", std::to_string(_ndim)},
					{"fp_type", dtype_to_string(_dtype)},
					{"dfuncid", dfid()},
					{"funcid", kfid()}
				};

				std::string ret = hasty::code_replacer(s_kcode(), rep_dict);
				return ret;
			}

		private:

			i32 _ndim;
			hasty::dtype _dtype;

		};


		export class LDL : public RawCudaFunction {
		public:

			LDL(i32 ndim, hasty::dtype dtype)
				: _ndim(ndim), _dtype(dtype)
			{
			}

			std::string dfid() const override {
				return "ldl" +
					dims_type({ _ndim }, _dtype);
			}

			static std::string s_dcode() {
				return
R"cuda(
__device__
void {{funcid}}({{fp_type}}* mat) 
{
	{{fp_type}} arr[{{ndim}}];
	for (int i = 0; i < {{ndim}}; ++i) {
		{{fp_type}} d = mat[i*ndim + i];
		for (int j = i + 1; j < {{ndim}}) {
			arr[j] = mat[j*{{ndim}}+i];
			mat[j*{{ndim}}+i] /= d;
		}
		for (int j = i + 1; j < {{ndim}}; ++j) {
			float aj = arr[j];
			for (int k = j; k < {{ndim}}; ++k) {
				mat[k*{{ndim}}+j] -= aj * mat[k*{{ndim}}+i];
			}
		}
	}
}
)cuda";
			}

			std::string dcode() const override {
				vecp<std::string, std::string> rep_dict = {
					{"ndim", std::to_string(_ndim)},
					{"fp_type", dtype_to_string(_dtype)},
					{"funcid", dfid()}
				};

				switch (_dtype)
				{
				case hasty::dtype::f32:
				{
					rep_dict.emplace_back(std::pair{ "abs_func", "fabsf" });
					rep_dict.emplace_back(std::pair{ "sqrt_func", "sqrtf" });
				}
				break;
				default:
					throw NotImplementedError();
					break;
				}


				std::string ret = hasty::code_replacer(s_dcode(), rep_dict);
				return ret;
			}

			static std::string s_kcode() {
				return
R"cuda(
extern \"C\" __global__
void {{funcid}}({{fp_type}}* mat, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{fp_type}} mat_copy[{{ndim}}*{{ndim}}];
		int k = 0;
		for (int i = 0; i < {{ndim}}; ++i) {
			for (int j = 0; i <= j; ++j) {
				{{fp_type}} temp = mat[k*N+tid];
				mat_copy[i*{{ndim}}+j] = temp;
				mat_copy[j*{{ndim}}+i] = temp;
				++k;
			}
		}
		{{dfuncid}}(mat_copy);
		k = 0;
		for (int i = 0; i < {{ndim}}; ++i) {
			for (int j = 0; j <= i; ++j) {
				mat[k*{{ndim}}+tid] = mat_copy[i*{{ndim}}+j];
				++k;
			}
		}
	}
}
)cuda";
			}

			std::string kcode() const override {
				vecp<std::string, std::string> rep_dict = {
					{"ndim", std::to_string(_ndim)},
					{"fp_type", dtype_to_string(_dtype)},
					{"dfuncid", dfid()},
					{"funcid", kfid()}
				};

				std::string ret = hasty::code_replacer(s_kcode(), rep_dict);
				return ret;
			}

		private:

			i32 _ndim;
			hasty::dtype _dtype;

		};


		export class ForwardSubsUnitDiaged : public RawCudaFunction {
		public:

			ForwardSubsUnitDiaged(i32 ndim, hasty::dtype dtype)
				: _ndim(ndim), _dtype(dtype)
			{
			}

			std::string dfid() const override {
				return "forward_subs_unit_diaged" +
					dims_type({ _ndim }, _dtype);
			}

			static std::string s_dcode() {
				return
R"cuda(
__device__
void {{funcid}}(const {{fp_type}}* mat, const {{fp_type}}* rhs, {{fp_type}}* sol) 
{
	for (int i = 0; i < {{ndim}}; ++i) {
		sol[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			sol[i] -= mat[i*{{ndim}}+j] * mat[j*{{ndim}}+j] * sol[j];
		}
		sol[i] /= mat[i*{{ndim}}+i];
	}
}
)cuda";
			}

			std::string dcode() const override {
				vecp<std::string, std::string> rep_dict = {
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


		export class BackwardSubsUnitT : public RawCudaFunction {
		public:

			BackwardSubsUnitT(i32 ndim, hasty::dtype dtype)
				: _ndim(ndim), _dtype(dtype)
			{
			}

			std::string dfid() const override {
				return "backward_subs_unit_t" +
					dims_type({ _ndim }, _dtype);
			}

			static std::string s_dcode() {
				return
R"cuda(
__device__
void {{funcid}}(const {{fp_type}}* mat, const {{fp_type}}* rhs, {{fp_type}}* sol) 
{
	for (int i = {{ndim}} - 1; i >= 0; --i) {
		sol[i] = rhs[i];
		for (int j = i + 1; j < {{ndim}}; ++j) {
			sol[i] -= mat[j*{{ndim}}+i] * sol[j];
		}
	}
}
)cuda";
			}

			std::string dcode() const override {
				vecp<std::string, std::string> rep_dict = {
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


		export class LDLSolve : public RawCudaFunction {
		public:

			LDLSolve(i32 ndim, hasty::dtype dtype)
				: _ndim(ndim), _dtype(dtype)
			{
				_deps.emplace_back(std::make_shared<ForwardSubsUnitDiaged>(_ndim, _dtype));
				_deps.emplace_back(std::make_shared<BackwardSubsUnitT>(_ndim, _dtype));
			}

			std::string dfid() const override 
			{
				return "ldl_solve" +
					dims_type({ _ndim }, _dtype);
			}

			static std::string s_dcode() 
			{
				return
R"cuda(
__device__
void {{funcid}}(const {{fp_type}}* mat, const {{fp_type}}* rhs, {{fp_type}}* sol) 
{
	{{fp_type}} arr[{{ndim}}];
	{{forward_funcid}}(mat, rhs, arr);
	{{backward_funcid}}(mat, arr, sol);
}
)cuda";
			}

			std::string dcode() const override {
				vecp<std::string, std::string> rep_dict = {
					{"ndim", std::to_string(_ndim)},
					{"fp_type", dtype_to_string(_dtype)},
					{"funcid", dfid()},
					{"forward_funcid", _deps[0]->dfid()},
					{"backward_funcid", _deps[1]->dfid()}
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


		export class GMW81Solve : public RawCudaFunction {
		public:

			GMW81Solve(i32 ndim, hasty::dtype dtype)
				: _ndim(ndim), _dtype(dtype)
			{
				_deps.emplace_back(std::make_shared<DiagPivot>(_ndim, _dtype));
				_deps.emplace_back(std::make_shared<PermuteVec>(_ndim, _dtype));
				_deps.emplace_back(std::make_shared<GMW81>(_ndim, _dtype));
				_deps.emplace_back(std::make_shared<LDLSolve>(_ndim, _dtype));
				_deps.emplace_back(std::make_shared<InvPermuteVec>(_ndim, _dtype));
			}

			std::string dfid() const override {
				return "gmw81_solve" +
					dims_type({ _ndim }, _dtype);
			}

			static std::string s_dcode() {
				return
R"cuda(
__device__
void {{funcid}}({{fp_type}}* mat, const {{fp_type}}* rhs, {{fp_type}}* sol) 
{	
	// Diagonal pivoting of matrix and right hand side
	int perm[{{ndim}}];
	{{fp_type}} arr1[{{ndim}}];
	{{fp_type}} arr2[{{ndim}}];
	{{diag_pivot_funcid}}(mat, perm);
	{{permute_vec_funcid}}(rhs, perm, arr1);
	
	// Diagonaly scale the matrix and rhs to improve condition number
	{{fp_type}} scale[{{ndim}}];
	for (int i = 0; i < {{ndim}}; ++i) {
		scale[i] = {{sqrt_func}}({{abs_func}}(mat[i*{{ndim}}+i]));
		arr1[i] /= scale[i];
	}
	for (int i = 0; i < {{ndim}}; ++i) {
		for (int j = 0; j <= i; ++j) {
			mat[i*{{ndim}}+j] /= (scale[i] * scale[j]);
		}
	}
	{{gmw81_funcid}}(mat);
	{{ldl_solve_funcid}}(mat, arr1, arr2);
	// Unscale
	for (int i = 0; i < {{ndim}}; ++i) {
		arr2[i] /= scale[i];
	}
	// Unpivot solution
	{{inv_permute_vec_funcid}}(arr2, perm, sol);
}
)cuda";
			}

			std::string dcode() const override {
				vecp<std::string, std::string> rep_dict = {
					{"ndim", std::to_string(_ndim)},
					{"fp_type", dtype_to_string(_dtype)},
					{"funcid", dfid()},
					{"diag_pivot_funcid", _deps[0]->dfid()},
					{"permute_vec_funcid", _deps[1]->dfid()},
					{"gmw81_funcid", _deps[2]->dfid()},
					{"ldl_solve_funcid", _deps[3]->dfid()},
					{"inv_permute_vec_funcid", _deps[4]->dfid()}
				};

				switch (_dtype)
				{
				case hasty::dtype::f32:
				{
					rep_dict.emplace_back(std::pair{ "abs_func", "fabsf" });
					rep_dict.emplace_back(std::pair{ "sqrt_func", "sqrtf" });
				}
				break;
				default:
					throw NotImplementedError();
					break;
				}

				std::string ret = hasty::code_replacer(s_dcode(), rep_dict);
				return ret;
			}

			static std::string s_kcode() {
				return
R"cuda(
extern "C" __global__
void {{funcid}}(const {{fp_type}}* mat, const {{fp_type}}* rhs, const char* step_type, {{fp_type}}* sol, int Nprobs) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		if (step_type[tid] == 0) {
			return;
		}
		{{fp_type}} mat_copy[{{ndim}}*{{ndim}}];
		{{fp_type}} rhs_copy[{{ndim}}];
		{{fp_type}} sol_copy[{{ndim}}];
		for (int i = 0; i < {{ndim}}; ++i) {
			rhs_copy[i] = rhs[i*Nprobs+tid];
			sol_copy[i] = sol[i*Nprobs+tid];
		}
		int k = 0;
		for (int i = 0; i < {{ndim}}; ++i) {
			for (int j = 0; j <= i; ++j) {
				{{fp_type}} temp = mat[k*Nprobs+tid];
				mat_copy[i*{{ndim}}+j] = temp;
				if (i != j) {
					mat_copy[j*{{ndim}}+i] = temp;
				}
				++k;
			}
		}
		{{dfuncid}}(mat_copy, rhs_copy, sol_copy);
		for (int i = 0; i < {{ndim}}; ++i) {
			sol[i*Nprobs+tid] = sol_copy[i];
		}
	}
}
)cuda";
			}

			std::string kcode() const override {
				vecp<std::string, std::string> rep_dict = {
					{"ndim", std::to_string(_ndim)},
					{"nmat", std::to_string(_ndim * (_ndim + 1) / 2)},
					{"fp_type", dtype_to_string(_dtype)},
					{"dfuncid", dfid()},
					{"funcid", kfid()}
				};

				std::string ret = hasty::code_replacer(s_kcode(), rep_dict);
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

	}

}