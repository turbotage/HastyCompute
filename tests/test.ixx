
import hasty_compute;
import solver_cu;
import permute_cu;

import <symengine/expression.h>;
import <symengine/simplify.h>;
import <symengine/parser.h>;

#include <torch/torch.h>

import <chrono>;

void symengine_test() {
	/*
	hasty::cuda::DiagPivot mda(4, hasty::eDType::F32);
	string code = "";
	hasty::code_generator(code, mda);
	std::cout << code;
	*/

	/*
	hasty::cuda::GMW81Solve gmw81s(4, hasty::F32);
	std::string code = "";
	hasty::code_generator(code, gmw81s);
	std::cout << code;
	*/
	using SymbolSym = SymEngine::RCP<const SymEngine::Symbol>;
	using BasicSym = SymEngine::RCP<const SymEngine::Basic>;

	auto expr = SymEngine::simplify(SymEngine::parse("sgn(x_0*exp(-b*x_1))"));

	SymEngine::vec_basic exprs;

	exprs.push_back(expr);

	std::cout << expr->__str__() << "\n";

	SymbolSym x0 = SymEngine::symbol("x_0");
	SymbolSym x1 = SymEngine::symbol("x_1");
	SymbolSym x2 = SymEngine::symbol("x_2");
	SymbolSym x3 = SymEngine::symbol("x_3");

	SymEngine::vec_sym bsyms = { x0, x1, x2, x3 };

	SymEngine::vec_basic d_expr;
	SymEngine::vec_basic dd_expr;

	std::cout << "First derivatives\n";

	for (auto sym : bsyms) {
		auto de = SymEngine::simplify(expr->diff(sym));
		d_expr.push_back(de);
		exprs.push_back(de);
		std::cout << "Diff: " << de->__str__() << std::endl;
	}



	std::cout << "Second derivatives\n";

	for (auto de : d_expr) {
		for (auto sym : bsyms) {
			auto dde = SymEngine::simplify(de->diff(sym));
			dd_expr.push_back(dde);
			exprs.push_back(dde);
			std::cout << "Diff: " << dde->__str__() << std::endl;
		}
		std::cout << "next row\n";
	}

	SymEngine::vec_pair reps;
	SymEngine::vec_basic reduced;

	SymEngine::cse(reps, reduced, exprs);

	std::cout << "\n\nReplacements: \n";

	for (auto& rep : reps) {
		std::cout << "First: " << rep.first->__str__() + "\n";
		std::cout << "Second: " << rep.second->__str__() + "\n";
	}

	std::cout << "\nReduced: \n";

	for (auto red : reduced) {
		std::cout << red->__str__() << "\n";
	}

	std::string a;
	std::cin >> a;
	//game_of_life();
}

int main() {

	torch::Tensor tensor = torch::eye(3);
	std::cout << tensor << std::endl;

	auto tensor2 = tensor.to(torch::Device(torch::kCUDA));

	std::cout << tensor.device() << std::endl;

	std::cout << tensor2.device() << std::endl;

	return 0;
}
