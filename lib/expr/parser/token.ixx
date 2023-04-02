module;

export module token;

#ifdef STL_AS_MODULES
import std.compat;
#else
import <complex>;
import <vector>;
#endif

//import <complex>;
//import <vector>;

import hasty_util;
import defaultexp;

namespace hasty {
	namespace expr {

		export struct FixedTokens {
			enum {
				COMMA_CHAR = (int)',',
				RIGHT_PAREN_CHAR = (int)')',
				LEFT_PAREN_CHAR = (int)'('
			};
		};

		export struct TokenType {
			enum {
				NO_TOKEN_TYPE,
				OPERATOR_TYPE,
				UNARY_OPERATOR_TYPE,
				BINARY_OPERATOR_TYPE,
				FUNCTION_TYPE,
				VARIABLE_TYPE,
				NUMBER_TYPE,
				ZERO_TYPE,
				UNITY_TYPE,
				NEG_UNITY_TYPE,
				NAN_TYPE,
				LEFT_PAREN_TYPE,
				RIGHT_PAREN_TYPE,
				COMMA_TYPE,
			};
		};

		export class Token {
		public:

			virtual std::int32_t get_id() const = 0;

			virtual std::int32_t get_token_type() const = 0;

		};

		export class NoToken : public Token {
		public:
			NoToken() = default;
			NoToken(const NoToken&) = default;

			std::int32_t get_id() const override
			{
				return FixedIDs::NO_TOKEN_ID;
			}


			std::int32_t get_token_type() const override
			{
				return TokenType::NO_TOKEN_TYPE;
			}

		};

		export class LeftParenToken : public Token {
		public:

			LeftParenToken() = default;
			LeftParenToken(const LeftParenToken&) = default;

			std::int32_t get_id() const override
			{
				return FixedIDs::LEFT_PAREN_ID;
			}

			std::int32_t get_token_type() const override
			{
				return TokenType::LEFT_PAREN_TYPE;
			}

		};

		export class RightParenToken : public Token {
		public:

			RightParenToken() = default;
			RightParenToken(const RightParenToken&) = default;

			std::int32_t get_id() const override
			{
				return FixedIDs::RIGHT_PAREN_ID;
			}

			std::int32_t get_token_type() const override
			{
				return TokenType::RIGHT_PAREN_TYPE;
			}

		};

		export class CommaToken : public Token {
		public:

			CommaToken() = default;
			CommaToken(const CommaToken&) = default;

			std::int32_t get_id() const override
			{
				return FixedIDs::COMMA_ID;
			}

			std::int32_t get_token_type() const override
			{
				return TokenType::COMMA_TYPE;
			}

		};

		export class NumberBaseToken : public Token {
		public:

			NumberBaseToken(const NumberBaseToken& other)
				: sizes(other.sizes)
			{}

			NumberBaseToken(const std::vector<std::int64_t>& sizes)
				: sizes(sizes)
			{
			}

			std::vector<std::int64_t> sizes;

		};

		export class NumberToken : public NumberBaseToken {
		public:

			NumberToken()
				: name("DEFAULT_NUMBER"), is_imaginary(false), num(0.0f, 0.0f), NumberBaseToken({ 1 })
			{
			}

			NumberToken(const NumberToken& other)
				: name(other.name), is_imaginary(other.is_imaginary), num(other.num), NumberBaseToken(other.sizes)
			{
			}

			NumberToken(const std::string& realnumberstr, bool is_imaginary)
				: name(realnumberstr), is_imaginary(is_imaginary),
				num(is_imaginary ? std::complex<float>(0.0f, std::atof(name.c_str())) : std::complex<float>(std::atof(name.c_str()), 0.0f)), NumberBaseToken({ 1 })
			{
			}

			NumberToken(const std::string& realnumberstr, bool is_imaginary, const std::vector<int64_t>& sizes)
				: name(realnumberstr), is_imaginary(is_imaginary),
				num(is_imaginary ? std::complex<float>(0.0f, std::atof(name.c_str())) : std::complex<float>(std::atof(name.c_str()), 0.0f)), NumberBaseToken(sizes)
			{
			}

			NumberToken(float number, bool is_imaginary)
				: name(is_imaginary ? std::to_string(number) + "i" : std::to_string(number)), is_imaginary(is_imaginary),
				num(is_imaginary ? std::complex<float>(0.0f, number) : std::complex<float>(number, 0.0f)), NumberBaseToken({ 1 })
			{
			}

			NumberToken(float number, bool is_imaginary, const std::vector<int64_t>& sizes)
				: name(is_imaginary ? std::to_string(number) + "i" : std::to_string(number)), is_imaginary(is_imaginary),
				num(is_imaginary ? std::complex<float>(0.0f, number) : std::complex<float>(number, 0.0f)), NumberBaseToken(sizes)
			{
			}

			NumberToken(std::complex<float> num, bool is_imaginary)
				: name(is_imaginary ? (std::to_string(num.real()) + "+" + std::to_string(num.imag()) + "i") : std::to_string(num.real())), is_imaginary(is_imaginary), num(num), NumberBaseToken({ 1 })
			{
			}

			NumberToken(std::complex<float> num, bool is_imaginary, const std::vector<int64_t>& sizes)
				: name(is_imaginary ? (std::to_string(num.real()) + "+" + std::to_string(num.imag()) + "i") : std::to_string(num.real())), is_imaginary(is_imaginary), num(num), NumberBaseToken(sizes)
			{
			}

			NumberToken(const std::string& numberstr, std::complex<float> num, bool is_imaginary)
				: name(numberstr), num(num), is_imaginary(is_imaginary), NumberBaseToken({ 1 })
			{
			}

			NumberToken(const std::string& numberstr, std::complex<float> num, bool is_imaginary, const std::vector<int64_t>& sizes)
				: name(numberstr), num(num), is_imaginary(is_imaginary), NumberBaseToken(sizes)
			{
			}

			std::string name;
			bool is_imaginary;
			std::complex<float> num;

			std::int32_t get_id() const override
			{
				return FixedIDs::NUMBER_ID;
			}

			std::int32_t get_token_type() const override
			{
				return TokenType::NUMBER_TYPE;
			}

			std::string get_full_name() const
			{
				return name + ((is_imaginary) ? "i" : "");
			}
		};

		export class VariableToken : public Token {
		public:

			VariableToken()
				: name("DEFAULT_VARIABLE")
			{
			}

			VariableToken(const VariableToken& other)
				: Token(other), name(other.name)
			{}

			VariableToken(const std::string& name)
				: name(name)
			{
			}

			friend bool operator==(const VariableToken& lhs, const VariableToken& rhs)
			{
				return lhs.name == rhs.name;
			}

			const std::string name;

			std::int32_t get_id() const override
			{
				return FixedIDs::VARIABLE_ID;
			}

			std::int32_t get_token_type() const override
			{
				return TokenType::VARIABLE_TYPE;
			}

		};

		export std::vector<std::string> strs_from_vartoks(const std::vector<VariableToken>& toks)
		{
			std::vector<std::string> ret;
			ret.reserve(toks.size());
			for (auto& tok : toks) {
				ret.emplace_back(tok.name);
			}
			return ret;
		}

		export class ZeroToken : public NumberBaseToken {
		public:

			ZeroToken()
				: NumberBaseToken({ 1 })
			{
			}

			ZeroToken(const std::vector<int64_t>& sizes)
				: NumberBaseToken(sizes)
			{
			}

			ZeroToken(const ZeroToken&) = default;

			std::int32_t get_id() const override
			{
				return FixedIDs::ZERO_ID;
			}

			std::int32_t get_token_type() const override
			{
				return TokenType::ZERO_TYPE;
			}

		};

		export class UnityToken : public NumberBaseToken {
		public:

			UnityToken()
				: NumberBaseToken({ 1 })
			{
			}


			UnityToken(const std::vector<int64_t>& sizes)
				: NumberBaseToken(sizes)
			{
			}

			UnityToken(const UnityToken&) = default;

			std::int32_t get_id() const override
			{
				return FixedIDs::UNITY_ID;
			}

			std::int32_t get_token_type() const override
			{
				return TokenType::UNITY_TYPE;
			}

		};

		export class NegUnityToken : public NumberBaseToken {
		public:

			NegUnityToken()
				: NumberBaseToken({ 1 })
			{
			}

			NegUnityToken(const std::vector<int64_t>& sizes)
				: NumberBaseToken(sizes)
			{
			}

			NegUnityToken(const NegUnityToken&) = default;

			std::int32_t get_id() const override
			{
				return FixedIDs::NEG_UNITY_ID;
			}

			std::int32_t get_token_type() const override
			{
				return TokenType::NEG_UNITY_TYPE;
			}

		};

		export class NanToken : public NumberBaseToken {
		public:

			NanToken()
				: NumberBaseToken({ 1 })
			{
			}

			NanToken(const std::vector<int64_t>& sizes)
				: NumberBaseToken(sizes)
			{
			}

			NanToken(const NanToken&) = default;

			std::int32_t get_id() const override
			{
				return FixedIDs::NAN_ID;
			}

			std::int32_t get_token_type() const override
			{
				return TokenType::NAN_TYPE;
			}

		};

		export class OperatorToken : public Token {
		public:

			OperatorToken(const OperatorToken& other)
				: Token(other), id(other.id), precedence(other.precedence), is_left_associative(other.is_left_associative)
			{
			}

			OperatorToken(std::int32_t id, std::int32_t precedence, bool is_left_associative)
				: id(id), precedence(precedence), is_left_associative(is_left_associative)
			{
			}

			const std::int32_t id;
			const std::int32_t precedence;
			const bool is_left_associative;

			std::int32_t get_id() const override
			{
				return id;
			}

			std::int32_t get_token_type() const override
			{
				return TokenType::OPERATOR_TYPE;
			}

			virtual std::int32_t get_operator_type() const = 0;

		};

		export class UnaryOperatorToken : public OperatorToken {
		public:

			UnaryOperatorToken(const UnaryOperatorToken& other)
				: OperatorToken(other), allowed_left_tokens(other.allowed_left_tokens)
			{
			}

			UnaryOperatorToken(UnaryOperatorToken&&) = default;

			UnaryOperatorToken(std::int32_t id, std::int32_t precedence, bool is_left_associative,
				const std::vector<std::shared_ptr<expr::Token>>& allowed_left_tokens)
				: OperatorToken(id, precedence, is_left_associative), allowed_left_tokens(allowed_left_tokens)
			{
			}

			const std::vector<std::shared_ptr<expr::Token>> allowed_left_tokens;

			std::int32_t get_operator_type() const override
			{
				return TokenType::UNARY_OPERATOR_TYPE;
			}

		};

		export class BinaryOperatorToken : public OperatorToken {
		public:

			BinaryOperatorToken(const BinaryOperatorToken& other)
				: OperatorToken(other), commutative(other.commutative), anti_commutative(other.anti_commutative), disallowed_left_tokens(other.disallowed_left_tokens)
			{
			}

			BinaryOperatorToken(BinaryOperatorToken&&) = default;

			BinaryOperatorToken(std::int32_t id, std::int32_t precedence, bool is_left_associative,
				bool commutative = false, bool anti_commutative = false)
				: OperatorToken(id, precedence, is_left_associative), commutative(commutative), anti_commutative(anti_commutative)
			{
			}

			BinaryOperatorToken(std::int32_t id, std::int32_t precedence, bool is_left_associative,
				bool commutative, bool anti_commutative,
				const std::vector<std::shared_ptr<expr::Token>>& disallowed_left_tokens)
				: OperatorToken(id, precedence, is_left_associative), commutative(commutative), anti_commutative(anti_commutative), disallowed_left_tokens(disallowed_left_tokens)
			{
			}

			const bool commutative;
			const bool anti_commutative;
			const std::vector<std::shared_ptr<expr::Token>> disallowed_left_tokens;

			std::int32_t get_operator_type() const override
			{
				return TokenType::BINARY_OPERATOR_TYPE;
			}
		};

		export class FunctionToken : public Token {
		public:

			FunctionToken(const FunctionToken& other)
				: Token(other), id(other.id), n_inputs(other.n_inputs), commutative(other.commutative),
				commutative_inputs(other.commutative_inputs), anti_commutative_inputs(other.anti_commutative_inputs)
			{
			}

			FunctionToken(std::int32_t id, std::int32_t n_inputs, bool commutative = false)
				: id(id), n_inputs(n_inputs), commutative(commutative)
			{
			}

			FunctionToken(std::int32_t id, std::int32_t n_inputs, bool commutative,
				const std::vector<std::vector<int>>& commutative_inputs,
				const std::vector<std::pair<int, int>>& anti_commutative_inputs)
				: id(id), n_inputs(n_inputs), commutative(commutative),
				commutative_inputs(commutative_inputs), anti_commutative_inputs(anti_commutative_inputs)
			{
			}

			const std::int32_t id;
			const std::int32_t n_inputs;
			const bool commutative;
			const std::vector<std::vector<int>> commutative_inputs;
			const std::vector<std::pair<int, int>> anti_commutative_inputs;

			std::int32_t get_id() const override
			{
				return id;
			}

			std::int32_t get_token_type() const override
			{
				return TokenType::FUNCTION_TYPE;
			}

		};

	}
}

