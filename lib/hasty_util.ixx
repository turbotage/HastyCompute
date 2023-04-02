module;

export module hasty_util;

#ifdef STL_AS_MODULES
import std;
#else
import <memory>;
import <stdexcept>;
import <vector>;
import <string>;
import <optional>;
import <algorithm>;
import <locale>;
#endif

namespace hasty {

    export {
        using i8 = std::int8_t;
        using i16 = std::int16_t;
        using i32 = std::int32_t;
        using i64 = std::int64_t;

        using f32 = float;
        using f64 = double;

        template<typename T>
        using vec = std::vector<T>;

        template<typename T, typename U>
        using vecp = std::vector<std::pair<T, U>>;

        template<typename T>
        using sptr = std::shared_ptr<T>;

        template<typename T>
        using uptr = std::unique_ptr<T>;

    }

    export {

        template<typename T>
        class raw_ptr {
        public:

            raw_ptr() { m_Ptr = nullptr; }
            raw_ptr(T& in) { m_Ptr = &in; }
            raw_ptr(const raw_ptr&) = delete;

            raw_ptr& operator=(const raw_ptr&) = delete;

            raw_ptr(raw_ptr&& other) {
                m_Ptr = other.m_Ptr;
                other.m_Ptr = nullptr;
            }
            void operator=(raw_ptr&& other) {
                m_Ptr = other.m_Ptr;
                other.m_Ptr = nullptr;
            }

            bool is_null() {
                return m_Ptr == nullptr;
            }

            T* get() { return m_Ptr; }

            T* operator->() { return m_Ptr; }

            T* operator->() const { return m_Ptr; }

            T& operator*() { return *m_Ptr; }

        private:
            T* m_Ptr;
        };

        // Used to signal output, functions with these parameters will fill the variable which the
        // reference points to
        template<typename T>
        using out_ref = T&;

        template<typename T>
        using refw = std::reference_wrapper<T>;

        // Used to signal output, functions with these parameters will fill the variable which the
        // reference points to if tc::OptOutRef isn't std::nullopt
        template<typename T>
        using opt_out_ref = std::optional<refw<T>>;

        template<typename T, typename U>
        using opt_out_pair_ref = std::optional<std::pair<refw<T>, refw<U>>>;

        template<typename T>
        using opt_ref = std::optional<refw<T>>;

        template<typename T, typename U>
        using opt_pair_ref = std::optional<std::pair<refw<T>, refw<U>>>;

        template<typename T>
        using opt_u_ptr = std::optional<std::unique_ptr<T>>;

        template<typename T>
        using opt_s_ptr = std::optional<std::shared_ptr<T>>;

        template <typename T>
        struct reversion_wrapper { T& iterable; };

        template <typename T>
        auto begin(reversion_wrapper<T> w) { return std::rbegin(w.iterable); }

        template <typename T>
        auto end(reversion_wrapper<T> w) { return std::rend(w.iterable); }

        template <typename T>
        reversion_wrapper<T> reverse(T&& iterable) { return { iterable }; }

    }

    export class NotImplementedError : public std::runtime_error {
    public:

        NotImplementedError()
            : std::runtime_error("Not Implemented Yet")
        {}

    };


    namespace util {

        export std::size_t replace_all(std::string& inout, std::string_view what, std::string_view with)
        {
            std::size_t count{};
            for (std::string::size_type pos{};
                inout.npos != (pos = inout.find(what.data(), pos, what.length()));
                pos += with.length(), ++count) {
                inout.replace(pos, what.length(), with.data(), with.length());
            }
            return count;
        }

        export std::size_t remove_all(std::string& inout, std::string_view what) {
            return replace_all(inout, what, "");
        }

        export template<typename KeyType, typename HashFunc = std::hash<KeyType>>
            concept Hashable = std::regular_invocable<HashFunc, KeyType>
            && std::convertible_to<std::invoke_result_t<HashFunc, KeyType>, std::size_t>;

        export template <typename KeyType, typename HashFunc = std::hash<KeyType>> requires Hashable<KeyType, HashFunc>
            inline std::size_t hash_combine(const std::size_t& seed, const KeyType& v)
        {
            HashFunc hasher;
            std::size_t ret = seed;
            ret ^= hasher(v) + 0x9e3779b9 + (ret << 6) + (ret >> 2);
            return ret;
        }

        export template <typename KeyType, typename HashFunc = std::hash<KeyType>> requires Hashable<KeyType, HashFunc>
            std::size_t hash_combine(const std::vector<KeyType>& hashes)
        {
            std::size_t ret = hashes.size() > 0 ? hashes.front() : throw std::runtime_error("Can't hash_combine an empty vector");
            for (int i = 1; i < hashes.size(); ++i) {
                ret = hash_combine(ret, hashes[i]);
            }
            return ret;
        }

        export std::string to_lower_case(const std::string& str) {
            std::string ret = str;
            std::transform(ret.begin(), ret.end(), ret.begin(), [](unsigned char c) { return std::tolower(c); });
            return ret;
        }

        export std::string add_whitespace_until(const std::string& str, int until) {
            if (str.size() > until) {
                return std::string(str.begin(), str.begin() + until);
            }

            std::string ret = str;
            ret.reserve(until);
            for (int i = str.size(); i <= until; ++i) {
                ret += ' ';
            }
            return ret;
        }

        export std::string add_after_newline(const std::string& str, const std::string& adder, bool add_start = true)
        {
            std::string ret = str;
            if (add_start) {
                ret.insert(0, adder);
            }
            for (int i = 0; i < ret.size(); ++i) {
                if (ret[i] == '\n') {
                    if (i + 1 > ret.size())
                        return ret;
                    ret.insert(i + 1, adder);
                    i += adder.size() + 2;
                }
            }
            return ret;
        }

        export std::string add_line_numbers(const std::string& str, int max_number_length = 5) {
            int until = max_number_length;
            std::string ret = str;
            ret.insert(0, util::add_whitespace_until(std::to_string(1), until) + "\t|");
            int k = 2;
            for (int i = until; i < ret.size(); ++i) {
                if (ret[i] == '\n') {
                    if (i + 1 > ret.size())
                        return ret;
                    ret.insert(i + 1, util::add_whitespace_until(std::to_string(k), until) + "\t|");
                    i += until + 2;
                    ++k;
                }
            }
            return ret;
        }

        export std::string remove_whitespace(const std::string& str) {
            std::string ret = str;
            ret.erase(std::remove_if(ret.begin(), ret.end(),
                [](char& c) {
                    return std::isspace<char>(c, std::locale::classic());
                }),
                ret.end());
            return ret;
        }

        export template<typename Container> requires std::ranges::range<Container>
            bool container_contains(const Container& c, typename Container::const_reference v)
        {
            return std::find(c.begin(), c.end(), v) != c.end();
        }

        export std::string stupid_compress(std::uint64_t num)
        {
            std::string basec = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            std::string ret;

            auto powlam = [](std::uint64_t base, std::uint32_t exponent) {
                uint64_t retnum = 1;
                for (int i = 0; i < exponent; ++i) {
                    retnum *= base;
                }
                return retnum;
            };

            uint64_t base = std::numeric_limits<uint64_t>::max();
            uint64_t c = (uint64_t)num / base;
            uint64_t rem = num % base;

            for (int i = 10; i >= 0; --i) {
                base = powlam(basec.size(), i);
                c = (uint64_t)num / base;
                rem = num % base;

                if (c > 0)
                    ret += basec[c];
                num = rem;
            }

            return ret;
        }

        export void add_n_str(std::string& str, const std::string& adder, int n)
        {
            for (int i = 0; i < n; ++i) {
                str += adder;
            }
        }

        export std::vector<std::int64_t> broadcast_tensor_shapes(const std::vector<std::int64_t>& shape1, const std::vector<std::int64_t>& shape2)
        {
            if (shape1.size() == 0 || shape2.size() == 0)
                throw std::runtime_error("shapes must have atleast one dimension to be broadcastable");

            auto& small = (shape1.size() > shape2.size()) ? shape2 : shape1;
            auto& big = (shape1.size() > shape2.size()) ? shape1 : shape2;

            std::vector<int64_t> ret(big.size());

            auto retit = ret.rbegin();
            auto smallit = small.rbegin();
            for (auto bigit = big.rbegin(); bigit != big.rend(); ) {
                if (smallit != small.rend()) {
                    if (*smallit == *bigit) {
                        *retit = *bigit;
                    }
                    else if (*smallit > *bigit && *bigit == 1) {
                        *retit = *smallit;
                    }
                    else if (*bigit > *smallit && *smallit == 1) {
                        *retit = *bigit;
                    }
                    else {
                        throw std::runtime_error("shapes where not broadcastable");
                    }
                    ++smallit;
                }
                else {
                    *retit = *bigit;
                }

                ++bigit;
                ++retit;
            }

            return ret;
        }

    }


}