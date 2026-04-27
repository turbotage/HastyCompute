export module hasty_server_mod:cmd_registry;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_generic_value_mod;

namespace hasty {
namespace server {

export using CommandFn = std::function<
            std::pair<std::string, std::vector<hasty::GenericValue>>(
        const std::string& options,
        std::vector<hasty::GenericValue> inputs)>;

export class CommandRegistry {
public:

    CommandRegistry() = default;

    CommandRegistry(
        bool base_arithmetic,
        bool base_creation,
        bool base_reduction,
        bool base_comparison,
        bool base_logical,
        bool base_bitwise,
        bool base_fft
    );

    
    std::int32_t highest_command_id() const {
        if (_commands.empty()) return 0;
        return std::max_element(
            _commands.begin(), _commands.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; }
        )->first;
    }

    void register_commands(const Vec<Pair<std::string, CommandFn>>& cmds) {
        std::int32_t next_id = highest_command_id();
        for (const auto& [name, fn] : cmds) {
            register_command(++next_id, name, fn);
        }
    }

    void register_command(std::int32_t id, std::string name, CommandFn fn) {
        _commands.emplace(id, Entry{ std::move(name), std::move(fn) });
    }

    std::pair<std::string, std::vector<hasty::GenericValue>> execute(
        std::int32_t id,
        const std::string& options,
        std::vector<hasty::GenericValue> inputs) const
    {
        auto it = _commands.find(id);
        if (it == _commands.end())
            throw std::runtime_error("Unknown function_id: " + std::to_string(id));
        return it->second.fn(options, std::move(inputs));
    }

    bool has_command(std::int32_t id) const { return _commands.contains(id); }

private:
    struct Entry { std::string name; CommandFn fn; };
    std::unordered_map<std::int32_t, Entry> _commands;
};


export extern CommandRegistry global_command_registry;


}
}