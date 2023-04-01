module;

export module token_algebra;

import std;

import token;

namespace hasty {
	namespace expr {

		export std::unique_ptr<NumberBaseToken> to_ptr(const ZeroToken& tok);
		export std::unique_ptr<NumberBaseToken> to_ptr(const UnityToken& tok);
		export std::unique_ptr<NumberBaseToken> to_ptr(const NegUnityToken& tok);
		export std::unique_ptr<NumberBaseToken> to_ptr(const NanToken& tok);
		export std::unique_ptr<NumberBaseToken> to_ptr(const NumberToken& tok);

		export NumberToken to_num(const Token& tok);
		export NumberToken to_num(const ZeroToken& tok);
		export NumberToken to_num(const UnityToken& tok);
		export NumberToken to_num(const NegUnityToken& tok);
		export NumberToken to_num(const NanToken& tok);
		export NumberToken to_num(const NumberToken& tok);

		export NumberToken from_number(float number);
		export NumberToken from_number(std::complex<float> number);



		// IMPLEMENTATION

		std::unique_ptr<NumberBaseToken> to_ptr(const ZeroToken& tok)
		{
			return std::make_unique<ZeroToken>(tok);
		}

		std::unique_ptr<NumberBaseToken> to_ptr(const UnityToken& tok)
		{
			return std::make_unique<UnityToken>(tok);
		}

		std::unique_ptr<NumberBaseToken> to_ptr(const NegUnityToken& tok)
		{
			return std::make_unique<NegUnityToken>(tok);
		}

		std::unique_ptr<NumberBaseToken> to_ptr(const NanToken& tok)
		{
			return std::make_unique<NanToken>(tok);
		}

		std::unique_ptr<NumberBaseToken> to_ptr(const NumberToken& tok)
		{
			return std::make_unique<NumberToken>(tok);
		}

		NumberToken to_num(const Token& tok)
		{
			switch (tok.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& ttok = static_cast<const ZeroToken&>(tok);
				return NumberToken(0.0f, false, ttok.sizes);
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& ttok = static_cast<const UnityToken&>(tok);
				return NumberToken(1.0f, false, ttok.sizes);
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& ttok = static_cast<const NegUnityToken&>(tok);
				return NumberToken(-1.0f, false, ttok.sizes);
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& ttok = static_cast<const NanToken&>(tok);
				return NumberToken(std::numeric_limits<float>::quiet_NaN(), false, ttok.sizes);
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& ttok = static_cast<const NumberToken&>(tok);
				return ttok;
			}
			default:
				throw std::runtime_error("Expected Zero, Unity, NegUnity, Nan and Number");
			}
		}

		NumberToken to_num(const ZeroToken& tok)
		{
			return NumberToken(0.0f, false, tok.sizes);
		}

		NumberToken to_num(const UnityToken& tok)
		{
			return NumberToken(1.0f, false, tok.sizes);
		}

		NumberToken to_num(const NegUnityToken& tok)
		{
			return NumberToken(-1.0f, false, tok.sizes);
		}

		NumberToken to_num(const NanToken& tok)
		{
			return NumberToken(std::numeric_limits<float>::quiet_NaN(), false, tok.sizes);
		}

		NumberToken to_num(const NumberToken& tok)
		{
			return tok;
		}

		NumberToken from_number(float number)
		{
			return NumberToken(number, false);
		}

		NumberToken from_number(std::complex<float> number)
		{
			return NumberToken(number, true);
		}

	}
}

