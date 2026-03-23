
import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_generic_value_mod;

/*
int main() {
    
auto randten = hasty::rand({2, 3, 4}, hasty::nullopt, hasty::eScalarType::ComplexFloat);

std::cout << randten.toString() << std::endl;

return 0;
}
*/

class GenericValueBank {
public:

    

private:
    std::unordered_map<std::string, hasty::GenericValue> _bank;
};