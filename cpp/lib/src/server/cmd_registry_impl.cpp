module;

module hasty_server_mod;

namespace hasty {
namespace server {

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
    register_command(0, "get_available_commands", 
        [this](const std::string&, std::vector<hasty::GenericValue>)
            -> std::pair<std::string, std::vector<hasty::GenericValue>>
        {
            std::vector<hasty::GenericValue> cmds;
            cmds.reserve(_commands.size());
            for (const auto& [id, cmd] : _commands) {
                cmds.emplace_back(std::to_string(id) + ":" + cmd.name);
            }
            return std::make_pair(std::string(""), std::move(cmds));
        });

    i32 command_counter = 1;

    register_command(command_counter++, "get_gv_metadata_string",
        [](const std::string& options, std::vector<hasty::GenericValue> inputs)
            -> std::pair<std::string, std::vector<hasty::GenericValue>>
        {
            if (inputs.size() != 1)
                throw std::runtime_error("get_gv_metadata_string: requires exactly 1 input");

            return std::make_pair(
                std::string(""), 
                std::vector<hasty::GenericValue>{
                    inputs[0].metadata_string()
                });
        });

    register_command(command_counter++, "get_gv_unfolded_metadata_string",
        [](const std::string& options, std::vector<hasty::GenericValue> inputs)
            -> std::pair<std::string, std::vector<hasty::GenericValue>>
        {
            if (inputs.size() != 1)
                throw std::runtime_error("get_gv_unfolded_metadata_string: requires exactly 1 input");

            return std::make_pair(
                std::string(""), 
                std::vector<hasty::GenericValue>{
                    inputs[0].unfolded_metadata_string()
                });
        });

    register_command(command_counter++, "to", 
        [](const std::string& options, std::vector<hasty::GenericValue> inputs)
            -> std::pair<std::string, std::vector<hasty::GenericValue>>
        {
            if (inputs.size() != 1 || !inputs[0].is_tensor())
                throw std::runtime_error("to: requires exactly 1 tensor input");

            auto t = inputs[0].as_tensor();
            auto [topts, shape] = Tensor::from_metadata_string(options);
            return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(t.to(topts))});
        });

    register_command(command_counter++, "compress_ui16_config",
        [](const std::string& options, std::vector<hasty::GenericValue> inputs)
            -> std::pair<std::string, std::vector<hasty::GenericValue>>
        {
            if (inputs.size() != 1 || !inputs[0].is_tensor())
                throw std::runtime_error("compress_ui16_config: requires exactly 1 tensor input");

            auto t = inputs[0].as_tensor();
            auto cfg = comprep::string_to_config(options);
            auto q = comprep::compress_ui16_config(t, cfg);
            return std::make_pair("", std::vector<hasty::GenericValue>{hasty::GenericValue(std::move(q))});
        });

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

        register_command(command_counter++, "sgn",
            [](const std::string&, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (inputs.size() != 1 || !inputs[0].is_tensor())
                    throw std::runtime_error("sgn: requires 1 tensor input");
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(inputs[0].as_tensor().sgn())});
            });

        register_command(command_counter++, "max",
            [](const std::string&, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (inputs.size() != 1 || !inputs[0].is_tensor())
                    throw std::runtime_error("max: requires 2 tensor inputs");
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(inputs[0].as_tensor().max())});
            });

        register_command(command_counter++, "min",
            [](const std::string&, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (inputs.size() != 1 || !inputs[0].is_tensor())
                    throw std::runtime_error("min: requires 2 tensor inputs");
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(inputs[0].as_tensor().min())});
            });

        register_command(command_counter++, "mean",
            [](const std::string&, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (inputs.size() != 1 || !inputs[0].is_tensor())
                    throw std::runtime_error("mean: requires 2 tensor inputs");
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(inputs[0].as_tensor().mean())});
            });

        register_command(command_counter++, "std",
            [](const std::string&, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (inputs.size() != 1 || !inputs[0].is_tensor())
                    throw std::runtime_error("std: requires 2 tensor inputs");
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(inputs[0].as_tensor().std())});
            });

        register_command(command_counter++, "statistics_string",
            [](const std::string&, std::vector<hasty::GenericValue> inputs)
                -> std::pair<std::string, std::vector<hasty::GenericValue>>
            {
                if (inputs.size() != 1 || !inputs[0].is_tensor())
                    throw std::runtime_error("statistics: requires 1 tensor input");
                return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(inputs[0].as_tensor().statistics_string())});
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
}

namespace hasty {
namespace server {

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
}