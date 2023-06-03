module;

export module nlsq_symbolic;

import <memory>;
import <stdexcept>;
import <vector>;
import <string>;

import hasty_compute;
import hasty_util;
import hasty;
import expr;
import symbolic;

namespace hasty {

	namespace nlsq {

		std::pair<vec<std::string>, std::string> get_residuals(
			const expr::Expression& expr, const expr::SymbolicContext& context)
		{
			for (const auto& var : context)
			{
				if (var.sym_type() == expr::SymbolicVariable::Type::eVarying) {

				}
			}

			throw NotImplementedError();
		}

		export class Expr {
		public:

			Expr(const std::string& expr, const std::vector<std::string>& pars, const std::vector<std::string>& consts,
				std::optional<std::vector<std::string>>&& nonlin_terms, std::optional<std::vector<std::string>>&& nonlin_pars)
				:
				_expr(expr),
				_pars(pars),
				_consts(consts),
				_nonlin_terms(std::move(nonlin_terms)),
				_nonlin_pars(std::move(nonlin_pars))
			{

			}

			size_t hash() {
				return std::hash<std::string>(_expr);
			}

			std::pair<vecp<std::string, uptr<expr::Expression>>, uptr<expr::Expression>> get_residuals() {
				
			}

		private:
			std::string _expr;
			std::vector<std::string> _pars;
			std::vector<std::string> _consts;
			std::optional<std::vector<std::string>> _nonlin_terms;
			std::optional<std::vector<std::string>> _nonlin_pars;
		};

	}

	namespace cuda {

		export class ExprF : public RawCudaFunction {
		public:

			ExprF(const nlsq::Expr& expr, int32_t ndata, hasty::dtype dtype)
				: _expr(expr), _ndata(ndata), _dtype(dtype)
			{}

			std::string dfid() const override {
				return "exprf_" + hasty::hash_string(_expr.hash());
			}

			static std::string s_dcode() {
				return
R"cuda(
__device__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, const {{fp_type}}* data,
	{{fp_type}}* f, int tid, int Nprobs) 
{
	{{fp_type}} pars[{{nparam}}];
	for (int i = 0; i < {{ndata}}; ++i) {
		pars[i] = params[i*Nprobs+tid];
	}

	{{fp_type}} res;

	for (int i = 0; i < {{ndata}}; ++i) {
{{sub_expr}}

{{res_expr}}

		f[tid] += res * res;
	}
}
)cuda";
			}

			std::string dcode() const override {
				vecp<std::string, std::string> rep_dict = {
					{"n"}
				};


			}

		private:
			nlsq::Expr _expr;
			int32_t _ndata;
			hasty::dtype _dtype;

		};



	}

}


