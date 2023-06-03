module;

export module defaultexp;

#ifdef STL_AS_MODULES
import std;
#else
import <unordered_map>;
import <string>;
#endif

namespace hasty {
	namespace expr {

		export struct FixedIDs {
			enum {
				// Fixed Tokens
				NO_TOKEN_ID,
				LEFT_PAREN_ID,
				RIGHT_PAREN_ID,
				COMMA_ID,
				UNITY_ID,
				NEG_UNITY_ID,
				ZERO_ID,
				NAN_ID,
				NUMBER_ID,
				VARIABLE_ID
			};
		};

		export std::unordered_map<std::int32_t, std::string> FIXED_ID_MAPS = {
			// Fixed Tokens
			{FixedIDs::NO_TOKEN_ID, "NO_TOKEN"},
			{FixedIDs::LEFT_PAREN_ID, "("},
			{FixedIDs::RIGHT_PAREN_ID, ")"},
			{FixedIDs::COMMA_ID, ","},
			{FixedIDs::UNITY_ID, "UNITY"},
			{FixedIDs::NEG_UNITY_ID, "NEG_UNITY"},
			{FixedIDs::ZERO_ID, "ZERO"},
			{FixedIDs::NAN_ID, "NAN"},
			{FixedIDs::NUMBER_ID, "NUMBER"},
			{FixedIDs::VARIABLE_ID, "VARIABLE"},
		};

		export struct DefaultOperatorPrecedence {
			enum {
				NEG_PRECEDENCE = 10,
				POW_PRECEDENCE = 10,
				MUL_PRECEDENCE = 5,
				DIV_PRECEDENCE = 5,
				ADD_PRECEDENCE = 3,
				SUB_PRECEDENCE = 3,
			};
		};


		export struct DefaultOperatorIDs {
			enum {
				// Operators
				NEG_ID = FixedIDs::VARIABLE_ID + 1,
				POW_ID,
				MUL_ID,
				DIV_ID,
				ADD_ID,
				SUB_ID,
			};
		};

		export std::unordered_map<int32_t, std::string> DEFAULT_OPERATOR_MAPS = {
			// Operators
			{DefaultOperatorIDs::NEG_ID, "-"},
			{DefaultOperatorIDs::POW_ID, "**"},
			{DefaultOperatorIDs::MUL_ID, "*"},
			{DefaultOperatorIDs::DIV_ID, "/"},
			{DefaultOperatorIDs::ADD_ID, "+"},
			{DefaultOperatorIDs::SUB_ID, "-"},
		};

		export struct DefaultFunctionIDs {
			enum {
				// FunctionTokens
				// 
				// Binary
				POW_ID = DefaultOperatorIDs::SUB_ID + 1,
				MUL_ID,
				DIV_ID,
				ADD_ID,
				SUB_ID,

				// Unary
				NEG_ID,
				ABS_ID,
				SQRT_ID,
				EXP_ID,
				LOG_ID,

				// Trig
				SIN_ID,
				COS_ID,
				TAN_ID,
				ASIN_ID,
				ACOS_ID,
				ATAN_ID,
				SINH_ID,
				COSH_ID,
				TANH_ID,
				ASINH_ID,
				ACOSH_ID,
				ATANH_ID,

				// Special
				SGN_ID,
				DERIVATIVE_ID,
				SUBS_ID,
			};
		};

		export std::unordered_map<int32_t, std::string> DEFAULT_FUNCTION_MAPS = {
			// FunctionTokens
			{DefaultFunctionIDs::POW_ID, "pow"},
			{DefaultFunctionIDs::MUL_ID, "mul"},
			{DefaultFunctionIDs::DIV_ID, "div"},
			{DefaultFunctionIDs::ADD_ID, "add"},
			{DefaultFunctionIDs::SUB_ID, "sub"},

			{DefaultFunctionIDs::NEG_ID, "neg"},
			{DefaultFunctionIDs::ABS_ID, "abs"},
			{DefaultFunctionIDs::SQRT_ID, "sqrt"},
			{DefaultFunctionIDs::EXP_ID, "exp"},
			{DefaultFunctionIDs::LOG_ID, "log"},

			{DefaultFunctionIDs::SIN_ID, "sin"},
			{DefaultFunctionIDs::COS_ID, "cos"},
			{DefaultFunctionIDs::TAN_ID, "tan"},
			{DefaultFunctionIDs::ASIN_ID, "asin"},
			{DefaultFunctionIDs::ACOS_ID, "acos"},
			{DefaultFunctionIDs::ATAN_ID, "atan"},
			{DefaultFunctionIDs::SINH_ID, "sinh"},
			{DefaultFunctionIDs::COSH_ID, "cosh"},
			{DefaultFunctionIDs::TANH_ID, "tanh"},
			{DefaultFunctionIDs::ASINH_ID, "asinh"},
			{DefaultFunctionIDs::ACOSH_ID, "acosh"},
			{DefaultFunctionIDs::ATANH_ID, "atanh"},

			{DefaultFunctionIDs::SGN_ID, "sgn"},
			{DefaultFunctionIDs::DERIVATIVE_ID, "derivative"},
			{DefaultFunctionIDs::SUBS_ID, "subs"}
		};


	}
}

