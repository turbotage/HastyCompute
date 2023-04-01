
import hasty_compute;
import solver_cu;
import permute_cu;

import std;

import <symengine/expression.h>;
import <symengine/simplify.h>;
import <symengine/parser.h>;


#include <chrono>

void game_of_life() {

	using namespace af;

	try {
		static const float h_kernel[] = { 1, 1, 1, 1, 0, 1, 1, 1, 1 };
		static const int reset = 500;
		static const int game_w = 128, game_h = 128;

		af::info();

		std::cout << "This example demonstrates the Conway's Game of Life "
			"using ArrayFire"
			<< std::endl
			<< "There are 4 simple rules of Conways's Game of Life"
			<< std::endl
			<< "1. Any live cell with fewer than two live neighbours "
			"dies, as if caused by under-population."
			<< std::endl
			<< "2. Any live cell with two or three live neighbours lives "
			"on to the next generation."
			<< std::endl
			<< "3. Any live cell with more than three live neighbours "
			"dies, as if by overcrowding."
			<< std::endl
			<< "4. Any dead cell with exactly three live neighbours "
			"becomes a live cell, as if by reproduction."
			<< std::endl
			<< "Each white block in the visualization represents 1 alive "
			"cell, black space represents dead cells"
			<< std::endl
			<< std::endl;

		std::cout
			<< "The conway_pretty example visualizes all the states in Conway"
			<< std::endl
			<< "Red   : Cells that have died due to under population"
			<< std::endl
			<< "Yellow: Cells that continue to live from previous state"
			<< std::endl
			<< "Green : Cells that are new as a result of reproduction"
			<< std::endl
			<< "Blue  : Cells that have died due to over population"
			<< std::endl
			<< std::endl;
			
		std::cout
			<< "This examples is throttled so as to be a better visualization"
			<< std::endl;

		af::Window simpleWindow(512, 512,
			"Conway's Game Of Life - Current State");
		af::Window prettyWindow(512, 512,
			"Conway's Game Of Life - Visualizing States");
		simpleWindow.setPos(32, 32);
		prettyWindow.setPos(512 + 32, 32);

		int frame_count = 0;

		// Initialize the kernel array just once
		const af::array kernel(3, 3, h_kernel, afHost);
		array state;
		state = (af::randu(game_h, game_w, f32) > 0.4).as(f32);

		array display = tile(state, 1, 1, 3, 1);

		while (!simpleWindow.close() && !prettyWindow.close()) {
			af::timer delay = timer::start();

			if (!simpleWindow.close()) simpleWindow.image(state);
			if (!prettyWindow.close()) prettyWindow.image(display);
			frame_count++;

			// Generate a random starting state
			if (frame_count % reset == 0)
				state = (af::randu(game_h, game_w, f32) > 0.5).as(f32);

			// Convolve gets neighbors
			af::array nHood = convolve(state, kernel);

			// Generate conditions for life
			// state == 1 && nHood < 2 ->> state = 0
			// state == 1 && nHood > 3 ->> state = 0
			// else if state == 1 ->> state = 1
			// state == 0 && nHood == 3 ->> state = 1
			af::array C0 = (nHood == 2);
			af::array C1 = (nHood == 3);

			array a0 = (state == 1) && (nHood < 2);  // Die of under population
			array a1 = (state != 0) && (C0 || C1);   // Continue to live
			array a2 = (state == 0) && C1;           // Reproduction
			array a3 = (state == 1) && (nHood > 3);  // Over-population

			display = join(2, a0 + a1, a1 + a2, a3).as(f32);

			// Update state
			state = state * C0 + C1;

			double fps = 30;
			while (timer::stop(delay) < (1 / fps)) {}
		}
	}
	catch (af::exception& e) {
		fprintf(stderr, "%s\n", e.what());
		throw;
	}
}

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

	return 0;
}
