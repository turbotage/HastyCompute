module;

export module lexer;

#ifdef STL_AS_MODULES
import std;
#else
import <memory>;
import <stdexcept>;
import <vector>;
import <string>;
import <unordered_map>;
import <regex>;
import <optional>;
#endif

//import <vector>;
//import <unordered_map>;
//import <memory>;
//import <optional>;
//import <stdexcept>;
//import <string>;
//import <regex>;

import hasty_util;
import token;
import defaultexp;

namespace hasty {
	namespace expr {

		export struct LexContext {

			LexContext()
			{
				auto no_token = std::make_shared<NoToken>();
				auto left_paren = std::make_shared<LeftParenToken>();
				auto comma = std::make_shared<CommaToken>();

				// 0
				unary_operators.emplace_back(
					DefaultOperatorIDs::NEG_ID,												// id
					DefaultOperatorPrecedence::NEG_PRECEDENCE,								// precedence
					false,																	// is_left_associative
					std::vector<std::shared_ptr<Token>>{no_token, left_paren, comma});		// allowed_tokens



				// 0
				binary_operators.emplace_back(
					DefaultOperatorIDs::POW_ID,												// id
					DefaultOperatorPrecedence::POW_PRECEDENCE,								// precedence
					false,																	// is_left_associative
					false,																	// commutative
					false,																	// anti_commutative
					std::vector<std::shared_ptr<Token>>{no_token, left_paren, comma});				// disallowed_tokens
				// 1
				binary_operators.emplace_back(
					DefaultOperatorIDs::MUL_ID,												// id
					DefaultOperatorPrecedence::MUL_PRECEDENCE,								// precedence
					true,																	// is_left_associative
					true,																	// commutative
					false,																	// anti_commutative
					std::vector<std::shared_ptr<Token>>{no_token, left_paren, comma});				// disallowed_tokens
				// 2
				binary_operators.emplace_back(
					DefaultOperatorIDs::DIV_ID,												// id
					DefaultOperatorPrecedence::DIV_PRECEDENCE,								// precedence
					true,																	// is_left_associative
					false,																	// commutative
					false,																	// anti_commutative
					std::vector<std::shared_ptr<Token>>{no_token, left_paren, comma});				// disallowed_tokens
				// 3
				binary_operators.emplace_back(
					DefaultOperatorIDs::ADD_ID,												// id
					DefaultOperatorPrecedence::ADD_PRECEDENCE,								// precedence
					true,																	// is_left_associative
					true,																	// commutative
					false,																	// anti_commutative
					std::vector<std::shared_ptr<Token>>{no_token, left_paren, comma});				// disallowed_tokens
				// 4
				binary_operators.emplace_back(
					DefaultOperatorIDs::SUB_ID,												// id
					DefaultOperatorPrecedence::SUB_PRECEDENCE,								// precedence
					true,																	// is_left_associative
					false,																	// commutative
					true,																	// anti_commutative
					std::vector<std::shared_ptr<Token>>{no_token, left_paren, comma});				// disallowed_tokens


				functions.emplace_back(DefaultFunctionIDs::POW_ID, 2);

				functions.emplace_back(DefaultFunctionIDs::ABS_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::SQRT_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::EXP_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::LOG_ID, 1);

				functions.emplace_back(DefaultFunctionIDs::SIN_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::COS_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::TAN_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::ASIN_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::ACOS_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::ATAN_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::SINH_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::COSH_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::TANH_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::ASINH_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::ACOSH_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::ATANH_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::SGN_ID, 1);
				functions.emplace_back(DefaultFunctionIDs::DERIVATIVE_ID, 2);
				functions.emplace_back(DefaultFunctionIDs::SUBS_ID, -1);

				operator_id_name_map = DEFAULT_OPERATOR_MAPS;
				function_id_name_map = DEFAULT_FUNCTION_MAPS;

			}

			LexContext(const LexContext& other)
				: unary_operators(other.unary_operators),
				binary_operators(other.binary_operators),
				functions(other.functions),
				variables(other.variables),
				operator_id_name_map(other.operator_id_name_map),
				function_id_name_map(other.function_id_name_map)
			{
			}

			LexContext& operator=(const LexContext& other)
			{
				{
					std::vector<UnaryOperatorToken> uot(other.unary_operators.begin(), other.unary_operators.end());
					unary_operators.swap(uot);
				}

				{
					std::vector<BinaryOperatorToken> bot(other.binary_operators.begin(), other.binary_operators.end());
					binary_operators.swap(bot);
				}

				{
					std::vector<FunctionToken> fot(other.functions.begin(), other.functions.end());
					functions.swap(fot);
				}

				{
					std::vector<VariableToken> vot(other.variables.begin(), other.variables.end());
					variables.swap(vot);
				}

				operator_id_name_map = other.operator_id_name_map;
				function_id_name_map = other.operator_id_name_map;

				return *this;
			}

			LexContext(LexContext&&) = default;

			// Fixed tokens
			/*
			NoToken				no_token;
			LeftParenToken		left_paren;
			RightParenToken		right_paren;
			CommaToken			comma;
			UnityToken			unity;
			NegUnityToken		neg_unity_token;
			ZeroToken			zero;
			NanToken			nan_token;
			NumberToken			number;
			VariableToken		variable;
			*/


			vec<UnaryOperatorToken>		unary_operators;
			vec<BinaryOperatorToken>	binary_operators;
			vec<FunctionToken>			functions;
			vec<VariableToken>			variables;

			std::unordered_map<int32_t, std::string> operator_id_name_map;
			std::unordered_map<int32_t, std::string> function_id_name_map;

		};

		export LexContext combine_contexts(const LexContext& c1, const LexContext& c2)
		{
			LexContext ret = c1;

			for (auto& var : c2.variables) {
				if (!hasty::util::container_contains(ret.variables, var)) {
					ret.variables.emplace_back(var);
				}
			}

			return ret;
		}

		namespace {
			static std::regex scientific_regex("^([\\d]+(.\\d+)?(?:e-?\\d+)?)?(i?)", std::regex_constants::ECMAScript | std::regex_constants::icase);
		}

		export class Lexer {
		public:

			Lexer(const LexContext& lex_context)
				: m_LexContext(lex_context)
			{
			}

			vec<uptr<Token>> lex(std::string expression) const
			{
				std::vector<uptr<Token>> lexed_tokens;
				lexed_tokens.reserve((expression.length() / 3 + 2));
				lexed_tokens.push_back(std::make_unique<NoToken>());

				std::string_view exprview = expression;
				uptr<Token> rettok;

				int iter = 0;
				while (exprview.length() != 0) {
					std::tie(exprview, rettok) = lex_token(exprview, lexed_tokens);

					if (!rettok)
						throw std::runtime_error("Lexer found no correct token but expression was not empty: " + std::string(exprview));

					lexed_tokens.push_back(std::move(rettok));

					if (exprview.length() == 0)
						break;

					if (iter > expression.size())
						throw std::runtime_error("Lexer ran more iterations than available charecters in expression, something wen't wrong");
				}

				lexed_tokens.erase(lexed_tokens.begin());

				return lexed_tokens;
			}

			const LexContext& context() const {
				return m_LexContext;
			}

		private:

			std::pair<std::string_view, uptr<Token>> lex_token(std::string_view expr,
				const std::vector<uptr<Token>>& lexed_tokens) const
			{
				auto lp_pair = begins_with_left_paren(expr);
				if (lp_pair.second.has_value()) {
					return std::make_pair(lp_pair.first, std::make_unique<LeftParenToken>(std::move(lp_pair.second.value())));
				}

				auto rp_pair = begins_with_right_paren(expr);
				if (rp_pair.second.has_value()) {
					return std::make_pair(rp_pair.first, std::make_unique<RightParenToken>(std::move(rp_pair.second.value())));
				}

				auto comma_pair = begins_with_comma(expr);
				if (comma_pair.second.has_value()) {
					return std::make_pair(comma_pair.first, std::make_unique<CommaToken>(std::move(comma_pair.second.value())));
				}

				auto unop_pair = begins_with_unary_operator(expr, lexed_tokens);
				if (unop_pair.second.has_value()) {
					return std::make_pair(unop_pair.first, std::make_unique<UnaryOperatorToken>(std::move(unop_pair.second.value())));
				}

				auto biop_pair = begins_with_binary_operator(expr, lexed_tokens);
				if (biop_pair.second.has_value()) {
					return std::make_pair(biop_pair.first, std::make_unique<BinaryOperatorToken>(std::move(biop_pair.second.value())));
				}

				auto func_pair = begins_with_function(expr);
				if (func_pair.second.has_value()) {
					return std::make_pair(func_pair.first, std::make_unique<FunctionToken>(std::move(func_pair.second.value())));
				}

				auto var_pair = begins_with_variable(expr);
				if (var_pair.second.has_value()) {
					return std::make_pair(var_pair.first, std::make_unique<VariableToken>(std::move(var_pair.second.value())));
				}

				auto num_pair = begins_with_numberstr(expr);
				if (num_pair.second.has_value()) {
					auto unity = begins_with_unity(num_pair.second.value());
					if (unity.has_value()) {
						return std::make_pair(num_pair.first, std::make_unique<UnityToken>(std::move(unity.value())));
					}

					auto zero = begins_with_zero(num_pair.second.value());
					if (zero.has_value()) {
						return std::make_pair(num_pair.first, std::make_unique<ZeroToken>(std::move(zero.value())));
					}

					return std::make_pair(num_pair.first, std::make_unique<NumberToken>(std::move(num_pair.second.value())));
				}

				return std::make_pair(expr, nullptr);

			}

			std::pair<std::string_view, std::optional<LeftParenToken>> begins_with_left_paren(std::string_view expr) const
			{
				if (expr.at(0) == FixedTokens::LEFT_PAREN_CHAR)
					return std::make_pair(expr.substr(1), LeftParenToken());
				return std::make_pair(expr, std::nullopt);
			}

			std::pair<std::string_view, std::optional<RightParenToken>> begins_with_right_paren(std::string_view expr) const
			{
				if (expr.at(0) == FixedTokens::RIGHT_PAREN_CHAR)
					return std::make_pair(expr.substr(1), RightParenToken());
				return std::make_pair(expr, std::nullopt);
			}

			std::pair<std::string_view, std::optional<CommaToken>> begins_with_comma(std::string_view expr) const
			{
				if (expr.at(0) == FixedTokens::COMMA_CHAR)
					return std::make_pair(expr.substr(1), CommaToken());
				return std::make_pair(expr, std::nullopt);
			}

			std::pair<std::string_view, std::optional<UnaryOperatorToken>> begins_with_unary_operator(std::string_view expr,
				const std::vector<uptr<Token>>& lexed_tokens) const
			{
				for (auto& uop : m_LexContext.unary_operators) {
					std::string opstr = m_LexContext.operator_id_name_map.at(uop.get_id());
					if (expr.rfind(opstr, 0) == 0) {
						int32_t previous_token_id = lexed_tokens.back()->get_id();
						for (auto& allowed_op : uop.allowed_left_tokens) {
							if (allowed_op->get_id() == previous_token_id) {
								return std::make_pair(expr.substr(opstr.length()), uop);
							}
						}
						// The tokens id matched but previous token did not match any left allowed token, might for instance be another unary operator
						return std::make_pair(expr, std::nullopt);
					}
				}
				// No match
				return std::make_pair(expr, std::nullopt);
			}

			std::pair<std::string_view, std::optional<BinaryOperatorToken>> begins_with_binary_operator(std::string_view expr,
				const std::vector<uptr<Token>>& lexed_tokens) const
			{
				for (auto& bop : m_LexContext.binary_operators) {
					std::string opstr = m_LexContext.operator_id_name_map.at(bop.get_id());
					if (expr.rfind(opstr, 0) == 0) {
						int32_t previous_token_id = lexed_tokens.back()->get_id();
						for (auto& disallowed_op : bop.disallowed_left_tokens) {
							if (disallowed_op->get_id() == previous_token_id) {
								throw std::runtime_error("token with token-id: " + std::to_string(previous_token_id) + " is disallowed before binary operator with token-id: " + std::to_string(bop.get_id()));
							}
						}

						return std::make_pair(expr.substr(opstr.length()), bop);
					}
				}
				// No match
				return std::make_pair(expr, std::nullopt);
			}

			std::pair<std::string_view, std::optional<FunctionToken>> begins_with_function(std::string_view expr) const
			{
				for (auto& func : m_LexContext.functions) {
					std::string funcstr = m_LexContext.function_id_name_map.at(func.get_id());
					if (expr.rfind(funcstr, 0) == 0) {
						int name_length = funcstr.length();
						if (expr.at(name_length) != '(')
							throw std::runtime_error("A function must always be followed by a left parenthasis '('");

						// Check if parenthasis are matched
						int32_t parenthasis_diff = 1;
						int32_t number_of_commas = 0;
						for (int i = name_length + 1; parenthasis_diff != 0 && i < expr.length(); ++i) {
							if (expr.at(i) == FixedTokens::LEFT_PAREN_CHAR) {
								parenthasis_diff += 1;
							}
							else if (expr.at(i) == FixedTokens::RIGHT_PAREN_CHAR) {
								parenthasis_diff -= 1;
							}
							else if (expr.at(i) == FixedTokens::COMMA_CHAR) {
								if (parenthasis_diff == 1) {
									number_of_commas += 1;
								}
							}
						}

						if (parenthasis_diff != 0)
							throw std::runtime_error("Parenthasis after function: " + funcstr + ", did not match");

						// This is a variable input function recreate that token with determined number of inputs
						if (func.n_inputs == -1) {
							return std::make_pair(expr.substr(name_length), FunctionToken(
								func.id,
								number_of_commas + 1,
								func.commutative,
								func.commutative_inputs,
								func.anti_commutative_inputs
							));
						}

						if (number_of_commas != (func.n_inputs - 1))
							throw std::runtime_error("Number of commas used in function: " + funcstr + ", was not consistent with expected number of inputs");

						return std::make_pair(expr.substr(name_length), func);
					}
				}
				return std::make_pair(expr, std::nullopt);
			}

			std::pair<std::string_view, std::optional<VariableToken>> begins_with_variable(std::string_view expr) const
			{
				for (auto& var : m_LexContext.variables) {
					if (expr.rfind(var.name, 0) == 0) {
						return std::make_pair(expr.substr(var.name.length()), var);
					}
				}
				return std::make_pair(expr, std::nullopt);
			}

			std::pair<std::string_view, std::optional<NumberToken>> begins_with_numberstr(std::string_view expr) const
			{
				std::cmatch m;
				std::string exprstr(expr);
				if (std::regex_search(exprstr.c_str(), m, scientific_regex, std::regex_constants::match_not_null)) {
					bool is_imaginary = m[3].str().length();
					return std::make_pair(expr.substr(m[0].str().length()), NumberToken(m[1].str(), is_imaginary));
				}
				return std::make_pair(expr, std::nullopt);
			}

			std::optional<ZeroToken> begins_with_zero(const NumberToken& num) const
			{
				if (num.is_imaginary)
					return std::nullopt;

				if (num.num.real() == 0.0f)
					return ZeroToken();

				return std::nullopt;
			}

			std::optional<UnityToken> begins_with_unity(const NumberToken& num) const
			{
				if (num.is_imaginary)
					return std::nullopt;

				if (num.num.real() == 1.0f)
					return UnityToken();

				return std::nullopt;
			}

		private:

			LexContext m_LexContext;

		};

	}
}


