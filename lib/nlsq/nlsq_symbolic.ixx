module;

export module nlsq_symbolic;

import <memory>;
import <stdexcept>;
import <vector>;
import <string>;
import <optional>;

import hasty_compute;
import hasty_util;
import hasty;
import expr;
import symbolic;

namespace hasty {

	namespace nlsq {

		export class Expr {
		public:

			Expr(const std::string& expr, const std::vector<std::string>& pars, const std::vector<std::string>& consts,
				std::optional<std::vector<std::string>>&& nonlin_terms, std::optional<std::vector<std::string>>&& nonlin_pars)
				:
				_expr(util::to_lower_case(util::remove_whitespace(expr))),
				_pars(pars),
				_consts(consts),
				_nonlin_terms(std::move(nonlin_terms)),
				_nonlin_pars(std::move(nonlin_pars))
			{
				for (auto& par : _pars) {
					par = util::to_lower_case(par);
				}
				for (auto& con : _consts) {
					con = util::to_lower_case(con);
				}

				if (_nonlin_terms.has_value()) {
					for (auto& nonlinterm : *_nonlin_terms) {
						nonlinterm = util::to_lower_case(nonlinterm);
					}
				}

				if (_nonlin_pars.has_value()) {
					for (auto& nonlinpar : *_nonlin_pars) {
						nonlinpar = util::to_lower_case(nonlinpar);
					}
				}
			}

			size_t hash() const {
				return std::hash<std::string>{}(_expr);
			}

			std::pair<vecp<std::string, uptr<expr::Expression>>, uptr<expr::Expression>> get_funcexprs()
			{
				std::pair<vecp<std::string, uptr<expr::Expression>>,vec<uptr<expr::Expression>>> cse_exprs =
					expr::Expression::cse({ _expr });
				uptr<expr::Expression> func = std::move(cse_exprs.second[0]);
				return std::make_pair(std::move(cse_exprs.first), std::move(func));
			}

			std::pair<vecp<std::string, uptr<expr::Expression>>, vec<uptr<expr::Expression>>> get_first_derivatives()
			{
				std::vector<std::string> vars = util::vec_concat(_pars, _consts);

				expr::Expression expr(_expr, vars);

				vec<std::string> derivatives;
				for (auto& par : _pars) {
					derivatives.push_back(expr.diff(par)->str(std::nullopt));
				}

				std::pair<vecp<std::string, uptr<expr::Expression>>, vec<uptr<expr::Expression>>> ret = 
					expr::Expression::cse(derivatives);

				return ret;
			}

			std::pair<vecp<std::string, uptr<expr::Expression>>, vec<uptr<expr::Expression>>> get_second_derivatives()
			{
				std::vector<std::string> vars = util::vec_concat(_pars, _consts);

				expr::Expression expr(_expr, vars);

				vec<std::string> derivatives;
				for (int i = 0; i < _pars.size(); ++i) {
					for (int j = 0; j < i + 1; ++j) {
						const auto& pari = _pars[i];
						const auto& parj = _pars[j];

						derivatives.push_back(std::move(expr.diff(pari)->diff(parj)->str(std::nullopt)));
					}
				}

				std::pair<vecp<std::string, uptr<expr::Expression>>, vec<uptr<expr::Expression>>> ret =
					expr::Expression::cse(derivatives);

				return ret;
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
				return "exprf_" + util::hash_string(_expr.hash(), 12);
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
{{var_expr}}

{{sub_expr}}

{{res_expr}}

		f[tid] += res * res;
	}
}
)cuda";
			}

			std::string dcode() const override {
				return "";
			}

		private:
			nlsq::Expr _expr;
			int32_t _ndata;
			hasty::dtype _dtype;

		};



	}

}


