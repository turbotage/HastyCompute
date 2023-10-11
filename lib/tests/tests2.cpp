#include "tests2.hpp"

import hasty_util;
import nlsq_symbolic;

#include <iostream>

import <string>;
import <vector>;


void hasty::tests::test_funcexprs()
{
	std::string exprstr = "S0*(f*exp(-b*D1) + (1-f)*exp(-b*D2))";
	std::vector<std::string> pars{"S0", "f", "D1", "D2"};
	std::vector<std::string> consts{"b"};

	hasty::nlsq::Expr expr(exprstr, pars, consts, std::nullopt, std::nullopt);

	auto funcexprs = expr.get_funcexprs();

	// Subexpressions
	std::cout << "Subexpressions\n";
	for (auto& subexpr : funcexprs.first) {
		std::cout << subexpr.first << ": " << subexpr.second->str(std::nullopt) << std::endl;
	}

	std::cout << "Reduced\n";
	// Reduced
	std::cout << funcexprs.second->str(std::nullopt);
}


void hasty::tests::test_first_derivatives()
{
	std::string exprstr = "S0*(f*exp(-b*D1) + (1-f)*exp(-b*D2))";
	std::vector<std::string> pars{"S0", "f", "D1", "D2"};
	std::vector<std::string> consts{"b"};

	hasty::nlsq::Expr expr(exprstr, pars, consts, std::nullopt, std::nullopt);

	auto funcexprs = expr.get_first_derivatives();

	// Subexpressions
	std::cout << "Subexpressions\n";
	for (auto& subexpr : funcexprs.first) {
		std::cout << subexpr.first << ": " << subexpr.second->str(std::nullopt) << std::endl;
	}

	std::cout << "Reduced\n";
	// Reduced
	for (auto& red : funcexprs.second) {
		std::cout << red->str(std::nullopt) << std::endl;
	}
}

void hasty::tests::test_second_derivatives()
{
	std::string exprstr = "S0*(f*exp(-b*D1) + (1-f)*exp(-b*D2))";
	std::vector<std::string> pars{"S0", "f", "D1", "D2"};
	std::vector<std::string> consts{"b"};

	hasty::nlsq::Expr expr(exprstr, pars, consts, std::nullopt, std::nullopt);

	auto funcexprs = expr.get_second_derivatives();

	// Subexpressions
	std::cout << "Subexpressions\n";
	for (auto& subexpr : funcexprs.first) {
		std::cout << subexpr.first << ": " << subexpr.second->str(std::nullopt) << std::endl;
	}

	std::cout << "Reduced\n";
	// Reduced
	for (auto& red : funcexprs.second) {
		std::cout << red->str(std::nullopt) << std::endl;
	}
}