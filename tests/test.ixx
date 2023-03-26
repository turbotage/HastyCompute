
import hasty_compute;
import solver;
import permute;

import <iostream>;
import <vector>;
import <string>;


using namespace std;

int main() {
	
	hasty::cuda::DiagPivot mda(4, hasty::eDType::F32);

	string code = "";

	hasty::code_generator(code, mda);

	std::cout << code;

	return 0;
}
