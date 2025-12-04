module;

export module util:alias;

import std;

namespace hasty {

    export template<typename T>
    using vec = std::vector<T>;

    export template<typename T>
    using refw = std::reference_wrapper<T>;

    export template<typename T>
    using crefw = std::reference_wrapper<const T>;

    export template<typename T>
    using opt = std::optional<T>;

    export using nullopt_t = std::nullopt_t;
    export inline constexpr nullopt_t nullopt = std::nullopt;

    export template<typename T>
    using optrefw = std::optional<std::reference_wrapper<T>>;

    export template<typename T>
    using optcrefw = std::optional<std::reference_wrapper<const T>>;

    export template<typename T>
    using sptr = std::shared_ptr<T>;

    export template<typename T>
    using uptr = std::unique_ptr<T>;


}