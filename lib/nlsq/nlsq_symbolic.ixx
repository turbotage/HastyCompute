module;

export module nlsq_symbolic;

#ifdef STL_AS_MODULES
import std.compat;
#else
import <memory>;
import <stdexcept>;
import <vector>;
import <string>;
#endif

import hasty_util;
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

	}

}