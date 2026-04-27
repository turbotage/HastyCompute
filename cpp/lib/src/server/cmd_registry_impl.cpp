module;

module hasty_server_mod;

namespace hasty {

CommandRegistry::CommandRegistry(
    bool base_arithmetic,
    bool base_creation,
    bool base_reduction,
    bool base_comparison,
    bool base_logical,
    bool base_bitwise,
    bool base_fft
)
{
    i32 command_counter = 0;
    if (base_arithmetic) {
        register_command(command_counter++, "add",
            [](const std::string&, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (inputs.size() != 2 || !inputs[0].is_tensor() || !inputs[1].is_tensor())
                    throw std::runtime_error("add: requires 2 tensor inputs");
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(inputs[0].as_tensor().add(inputs[1].as_tensor()))});
            });

        register_command(command_counter++, "sub",
            [](const std::string&, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (inputs.size() != 2 || !inputs[0].is_tensor() || !inputs[1].is_tensor())
                    throw std::runtime_error("subtract: requires 2 tensor inputs");
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(inputs[0].as_tensor().sub(inputs[1].as_tensor()))});
            });

        register_command(command_counter++, "mult",
            [](const std::string&, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (inputs.size() != 2 || !inputs[0].is_tensor() || !inputs[1].is_tensor())
                    throw std::runtime_error("multiply: requires 2 tensor inputs");
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(inputs[0].as_tensor().mul(inputs[1].as_tensor()))});
            });

        register_command(command_counter++, "div",
            [](const std::string&, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (inputs.size() != 2 || !inputs[0].is_tensor() || !inputs[1].is_tensor())
                    throw std::runtime_error("divide: requires 2 tensor inputs");
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(inputs[0].as_tensor().div(inputs[1].as_tensor()))});
            });

        register_command(command_counter++, "neg",
            [](const std::string&, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (inputs.size() != 1 || !inputs[0].is_tensor())
                    throw std::runtime_error("negate: requires 1 tensor input");
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(inputs[0].as_tensor().neg())});
            });

        register_command(command_counter++, "abs",
            [](const std::string&, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (inputs.size() != 1 || !inputs[0].is_tensor())
                    throw std::runtime_error("abs: requires 1 tensor input");
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(inputs[0].as_tensor().abs())});
            });
    }

    if (base_creation) {
        register_command(command_counter++, "rand",
            [](const std::string& options, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (!inputs.empty())
                    throw std::runtime_error("rand: requires no inputs");
                auto [topts, shape] = hasty::Tensor::from_metadata_string(options);
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(hasty::rand(shape, topts))});
            });

        register_command(command_counter++, "zeros",
            [](const std::string& options, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (!inputs.empty())
                    throw std::runtime_error("zeros: requires no inputs");
                auto [topts, shape] = hasty::Tensor::from_metadata_string(options);
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(hasty::zeros(shape, topts))});
            });

        register_command(command_counter++, "ones",
            [](const std::string& options, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (!inputs.empty())
                    throw std::runtime_error("ones: requires no inputs");
                auto [topts, shape] = hasty::Tensor::from_metadata_string(options);
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(hasty::ones(shape, topts))});
            });
    }

}

}

namespace hasty {

    CommandRegistry global_command_registry(
        true,   // base_arithmetic
        true,   // base_creation
        true,   // base_reduction
        true,   // base_comparison
        true,   // base_logical
        true,   // base_bitwise
        true    // base_fft
    );

}