module;

export module neg;

import std;

//import <memory>;
//import <stdexcept>;
//import <complex>;

import token;

namespace hasty {
	namespace expr {

		// Neg

		export std::unique_ptr<NumberBaseToken> operator-(const Token& other);

		export ZeroToken operator-(const ZeroToken& other);
		export UnityToken operator-(const NegUnityToken& other);
		export NegUnityToken operator-(const UnityToken& other);
		export NanToken operator-(const NanToken& other);
		export NumberToken operator-(const NumberToken& other);



		// IMPLEMENTATION

		std::unique_ptr<NumberBaseToken> operator-(const Token& other)
		{
			switch (other.get_token_type()) {
			case TokenType::ZERO_TYPE:
				return std::make_unique<ZeroToken>(-static_cast<const ZeroToken&>(other));
			case TokenType::UNITY_TYPE:
				return std::make_unique<UnityToken>(-static_cast<const NegUnityToken&>(other));
			case TokenType::NEG_UNITY_TYPE:
				return std::make_unique<NegUnityToken>(-static_cast<const UnityToken&>(other));
			case TokenType::NAN_TYPE:
				return std::make_unique<NanToken>(-static_cast<const NanToken&>(other));
			case TokenType::NUMBER_TYPE:
				return std::make_unique<NumberToken>(-static_cast<const NumberToken&>(other));
			default:
				throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
			}
		}

		ZeroToken operator-(const ZeroToken& other)
		{
			return other;
		}

		UnityToken operator-(const NegUnityToken& other)
		{
			return UnityToken(other.sizes);
		}

		NegUnityToken operator-(const UnityToken& other)
		{
			return NegUnityToken(other.sizes);
		}

		NanToken operator-(const NanToken& other)
		{
			return other;
		}

		NumberToken operator-(const NumberToken& other)
		{
			return NumberToken(-other.num, other.is_imaginary, other.sizes);
		}

	}
}
