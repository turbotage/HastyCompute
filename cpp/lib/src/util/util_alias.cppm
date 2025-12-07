module;

export module util:alias;

import std;

namespace hasty {

    export template<typename T>
    using Vec = std::vector<T>;

    export template<typename T>
    using Set = std::set<T>;

    export template<typename K, typename V>
    using UMap = std::unordered_map<K, V>;

    export template<typename T>
    using RefW = std::reference_wrapper<T>;

    export template<typename T>
    using CRefW = std::reference_wrapper<const T>;

    export template<typename T>
    using Opt = std::optional<T>;

    // These are simple aliases to std::make_optional
    export template<typename T>
    constexpr auto make_opt(T&& value) noexcept(std::is_nothrow_constructible_v<std::optional<std::decay_t<T>>, T>) {
        return std::make_optional(std::forward<T>(value));
    }

    // Optional: In-place version
    export template<typename T, typename... Args>
    constexpr auto make_opt(std::in_place_t, Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) {
        return std::make_optional(std::in_place, std::forward<Args>(args)...);
    }

    export using nullopt_t = std::nullopt_t;
    export inline constexpr nullopt_t nullopt = std::nullopt;

    export template<typename T>
    using OptRefW = std::optional<std::reference_wrapper<T>>;

    export template<typename T>
    using OptCRefW = std::optional<std::reference_wrapper<const T>>;

    export template<typename T>
    using SPtr = std::shared_ptr<T>;

    // Simple std::make_unique alias
    export template<typename T, typename... Args>
    constexpr auto make_uptr(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) {
        return std::make_unique<T>(std::forward<Args>(args)...);
    }

    export template<typename T>
    using UPtr = std::unique_ptr<T>;

    // Simple std::make_shared alias
    export template<typename T, typename... Args>
    constexpr auto make_sptr(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }

    export template<typename... Args>
    using Tup = std::tuple<Args...>;

    // Simple std::make_tuple alias
    export template<typename... Ts>
    constexpr auto make_tup(Ts&&... args) noexcept(std::conjunction_v<std::is_nothrow_constructible<std::tuple<std::decay_t<Ts>...>, Ts>...>) {
        return std::make_tuple(std::forward<Ts>(args)...);
    }



}