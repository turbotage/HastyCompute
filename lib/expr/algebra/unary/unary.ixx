module;

export module unary;

import std;

//import <memory>;
//import <stdexcept>;
//import <cmath>;
//import <complex>;

import token;

namespace hasty {
	namespace expr {

		// Sign

		export std::unique_ptr<NumberBaseToken> sgn(const Token& in);
		export ZeroToken sgn(const ZeroToken& in);
		export NegUnityToken sgn(const NegUnityToken& in);
		export UnityToken sgn(const UnityToken& in);
		export NanToken sgn(const NanToken& in);
		export NumberToken sgn(const NumberToken& in);

		// Abs

		export std::unique_ptr<NumberBaseToken> abs(const Token& in);
		export ZeroToken abs(const ZeroToken& in);
		export UnityToken abs(const NegUnityToken& in);
		export UnityToken abs(const UnityToken& in);
		export NanToken abs(const NanToken& in);
		export NumberToken abs(const NumberToken& in);

		// Sqrt

		export std::unique_ptr<NumberBaseToken> sqrt(const Token& in);
		export ZeroToken sqrt(const ZeroToken& in);
		export NumberToken sqrt(const NegUnityToken& in);
		export UnityToken sqrt(const UnityToken& in);
		export NanToken sqrt(const NanToken& in);
		export NumberToken sqrt(const NumberToken& in);

		// Square

		export std::unique_ptr<NumberBaseToken> square(const Token& in);
		export ZeroToken square(const ZeroToken& in);
		export UnityToken square(const NegUnityToken& in);
		export UnityToken square(const UnityToken& in);
		export NanToken square(const NanToken& in);
		export NumberToken square(const NumberToken& in);

		// Exp

		export std::unique_ptr<NumberBaseToken> exp(const Token& in);
		export UnityToken exp(const ZeroToken& in);
		export NumberToken exp(const NegUnityToken& in);
		export NumberToken exp(const UnityToken& in);
		export NanToken exp(const NanToken& in);
		export NumberToken exp(const NumberToken& in);

		// Log

		export std::unique_ptr<NumberBaseToken> log(const Token& in);
		export NanToken log(const ZeroToken& in);
		export NanToken log(const NegUnityToken& in);
		export ZeroToken log(const UnityToken& in);
		export NanToken log(const NanToken& in);
		export NumberToken log(const NumberToken& in);





		// IMPLEMENTATION


		std::unique_ptr<NumberBaseToken> sgn(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<ZeroToken>(sgn(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<UnityToken>(sgn(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NegUnityToken>(sgn(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(sgn(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				if (atok.num.real() == 0.0f && atok.num.imag() == 0.0f) {
					return std::make_unique<ZeroToken>(atok.sizes);
				}
				else if (!atok.is_imaginary) {
					if (atok.num.real() > 0.0f) {
						return std::make_unique<UnityToken>(atok.sizes);
					}
					return std::make_unique<NegUnityToken>(atok.sizes);
				}
				return std::make_unique<NumberToken>(sgn(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		ZeroToken sgn(const ZeroToken& in)
		{
			return in;
		}

		NegUnityToken sgn(const NegUnityToken& in)
		{
			return in;
		}

		UnityToken sgn(const UnityToken& in)
		{
			return in;
		}

		NanToken sgn(const NanToken& in)
		{
			return in;
		}

		NumberToken sgn(const NumberToken& in)
		{
			if (in.num.real() == 0.0f && in.num.imag() == 0.0f) {
				return NumberToken(0.0f, false, in.sizes);
			}

			return NumberToken(in.num / std::abs(in.num), in.is_imaginary, in.sizes);
		}



		// <====================================== ABS ============================================>

		std::unique_ptr<NumberBaseToken> abs(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<ZeroToken>(abs(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<UnityToken>(abs(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<UnityToken>(abs(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(abs(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(abs(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		ZeroToken abs(const ZeroToken& in)
		{
			return in;
		}

		UnityToken abs(const NegUnityToken& in)
		{
			return UnityToken(in.sizes);
		}

		UnityToken abs(const UnityToken& in)
		{
			return in;
		}

		NanToken abs(const NanToken& in)
		{
			return in;
		}

		NumberToken abs(const NumberToken& in)
		{
			return NumberToken(std::abs(in.num), false, in.sizes);
		}


		// <====================================== SQRT ============================================>

		std::unique_ptr<NumberBaseToken> sqrt(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<ZeroToken>(sqrt(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<UnityToken>(sqrt(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NumberToken>(sqrt(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(sqrt(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(sqrt(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		ZeroToken sqrt(const ZeroToken& in)
		{
			return in;
		}

		NumberToken sqrt(const NegUnityToken& in)
		{
			return NumberToken(1.0f, true, in.sizes);
		}

		UnityToken sqrt(const UnityToken& in)
		{
			return in;
		}

		NanToken sqrt(const NanToken& in)
		{
			return in;
		}

		NumberToken sqrt(const NumberToken& in)
		{
			return NumberToken(std::sqrt(in.num), in.is_imaginary, in.sizes);
		}

		// <====================================== SQUARE ============================================>

		std::unique_ptr<NumberBaseToken> square(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<ZeroToken>(square(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<UnityToken>(square(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<UnityToken>(square(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(square(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(square(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		ZeroToken square(const ZeroToken& in)
		{
			return in;
		}

		UnityToken square(const NegUnityToken& in)
		{
			return UnityToken(in.sizes);
		}

		UnityToken square(const UnityToken& in)
		{
			return in;
		}

		NanToken square(const NanToken& in)
		{
			return in;
		}

		NumberToken square(const NumberToken& in)
		{
			return NumberToken(in.num * in.num, in.is_imaginary, in.sizes);
		}

		// <====================================== EXP ============================================>

		std::unique_ptr<NumberBaseToken> exp(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<UnityToken>(exp(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<NumberToken>(exp(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NumberToken>(exp(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(exp(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(exp(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		UnityToken exp(const ZeroToken& in)
		{
			return UnityToken(in.sizes);
		}

		NumberToken exp(const NegUnityToken& in)
		{
			return NumberToken((float)std::exp(-1.0), true, in.sizes);
		}

		NumberToken exp(const UnityToken& in)
		{
			return NumberToken((float)std::exp(1.0), true, in.sizes);
		}

		NanToken exp(const NanToken& in)
		{
			return in;
		}

		NumberToken exp(const NumberToken& in)
		{
			return NumberToken(std::exp(in.num), in.is_imaginary, in.sizes);
		}

		// <====================================== LOG ============================================>

		std::unique_ptr<NumberBaseToken> log(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<NanToken>(log(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<ZeroToken>(log(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NanToken>(log(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(log(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(log(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		NanToken log(const ZeroToken& in)
		{
			return NanToken(in.sizes);
		}

		NanToken log(const NegUnityToken& in)
		{
			return NanToken(in.sizes);
		}

		ZeroToken log(const UnityToken& in)
		{
			return ZeroToken(in.sizes);
		}

		NanToken log(const NanToken& in)
		{
			return in;
		}

		NumberToken log(const NumberToken& in)
		{
			return NumberToken(std::log(in.num), in.is_imaginary, in.sizes);
		}

	}
}

