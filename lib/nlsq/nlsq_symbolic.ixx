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

		private:
			std::string _expr;
			std::vector<std::string> _pars;
			std::vector<std::string> _consts;
			std::optional<std::vector<std::string>> _nonlin_terms;
			std::optional<std::vector<std::string>> _nonlin_pars;
		};

		/*
		export class ExprF : public RawCudaFunction {
		public:

			ExprF(const Expr& expr, int32_t ndata, ) {}

		private:


		};
		*/



	}

}


