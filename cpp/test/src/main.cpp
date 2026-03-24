import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_generic_value_mod;
import hasty_server_mod;

import hasty_viz_mod;

int main() {

    std::cout << "Generating example plot..." << std::endl;
    
    auto fig = hasty::viz::example_plot();
    
    fig.show();

    std::cout << "Starting server..." << std::endl;

    GenericValueBank bank;
    CommandRegistry registry;

    // function_id 0: element-wise tensor add
    registry.register_command(0, "add",
        [](const std::string&, std::vector<hasty::GenericValue> inputs)
            -> std::vector<hasty::GenericValue>
        {
            if (inputs.size() != 2 || !inputs[0].is_tensor() || !inputs[1].is_tensor())
                throw std::runtime_error("add: requires 2 tensor inputs");
            return { hasty::GenericValue(inputs[0].as_tensor().add(inputs[1].as_tensor())) };
        });

    auto handle = start_server(bank, registry, "0.0.0.0:50051");
    //auto handle = start_server(bank, registry, "unix:///tmp/hasty.sock");

    // Signal readiness to any waiting test runner
    std::cout << "READY" << std::endl;

    handle.wait();
    return 0;
}
