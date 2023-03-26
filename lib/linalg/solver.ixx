module;

export module solver;

import <string>;
import <vector>;
import <memory>;

import hasty_compute;

using namespace std;

namespace hasty {

	namespace cuda {

		export class GMW81 : public RawCudaFunction {
		public:

			GMW81(i32 ndim, eDType dtype)
				: _ndim(ndim), _dtype(dtype)
			{
			}

			string dfid() const override {
				return "diag_pivot" +
					fidend::dims_type({ _ndim }, _dtype);
			}

			static string s_dcode() {
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

			string dcode() const override {
				vector<pair<string, string>> rep_dict = {
					{"ndim", to_string(_ndim)},
					{"fp_type", dtype_to_string(_dtype)},
					{"funcid", dfid()}
				};

				switch (i32(_dtype))
				{
				case F32:
				{
					rep_dict.emplace_back(pair{ "abs_func", "fabsf" });
					rep_dict.emplace_back(pair{ "sqrt_func", "sqrtf" });
					rep_dict.emplace_back(pair{ "machine_eps", "2e-7" });
				}
					break;
				default:
					throw NotImplementedThrow;
					break;
				}


				string ret = hasty::code_replacer(s_dcode(), rep_dict);
				return ret;
			}

			string kfid() const override {
				return "k_" + dfid();
			}

			static string s_kcode() {
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

			string kcode() const override {
				vector<pair<string, string>> rep_dict = {
					{"ndim", to_string(_ndim)},
					{"fp_type", dtype_to_string(_dtype)},
					{"dfuncid", dfid()},
					{"funcid", kfid()}
				};

				string ret = hasty::code_replacer(s_kcode(), rep_dict);
				return ret;
			}

		private:

			i32 _ndim;
			eDType _dtype;

		};




	}

}