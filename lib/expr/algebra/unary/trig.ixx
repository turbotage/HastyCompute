module;

export module trig;

import std;

//import <memory>;
//import <stdexcept>;
//import <cmath>;
//import <complex>;

import token;

namespace hasty {
	namespace expr {


		// Sin

		export std::unique_ptr<NumberBaseToken> sin(const Token& in);
		export ZeroToken sin(const ZeroToken& in);
		export NumberToken sin(const NegUnityToken& in);
		export NumberToken sin(const UnityToken& in);
		export NanToken sin(const NanToken& in);
		export NumberToken sin(const NumberToken& in);

		// Cos

		export std::unique_ptr<NumberBaseToken> cos(const Token& in);
		export UnityToken cos(const ZeroToken& in);
		export NumberToken cos(const NegUnityToken& in);
		export NumberToken cos(const UnityToken& in);
		export NanToken cos(const NanToken& in);
		export NumberToken cos(const NumberToken& in);

		// Tan

		export std::unique_ptr<NumberBaseToken> tan(const Token& in);
		export UnityToken tan(const ZeroToken& in);
		export NumberToken tan(const NegUnityToken& in);
		export NumberToken tan(const UnityToken& in);
		export NanToken tan(const NanToken& in);
		export NumberToken tan(const NumberToken& in);

		// Asin

		export std::unique_ptr<NumberBaseToken> asin(const Token& in);
		export UnityToken asin(const ZeroToken& in);
		export NumberToken asin(const NegUnityToken& in);
		export NumberToken asin(const UnityToken& in);
		export NanToken asin(const NanToken& in);
		export NumberToken asin(const NumberToken& in);

		// Acos

		export std::unique_ptr<NumberBaseToken> acos(const Token& in);
		export NumberToken acos(const ZeroToken& in);
		export NumberToken acos(const NegUnityToken& in);
		export ZeroToken acos(const UnityToken& in);
		export NanToken acos(const NanToken& in);
		export NumberToken acos(const NumberToken& in);

		// Atan

		export std::unique_ptr<NumberBaseToken> atan(const Token& in);
		export ZeroToken atan(const ZeroToken& in);
		export NumberToken atan(const NegUnityToken& in);
		export NumberToken atan(const UnityToken& in);
		export NanToken atan(const NanToken& in);
		export NumberToken atan(const NumberToken& in);

		// Sinh

		export std::unique_ptr<NumberBaseToken> sinh(const Token& in);
		export ZeroToken sinh(const ZeroToken& in);
		export NumberToken sinh(const NegUnityToken& in);
		export NumberToken sinh(const UnityToken& in);
		export NanToken sinh(const NanToken& in);
		export NumberToken sinh(const NumberToken& in);

		// Cosh

		export std::unique_ptr<NumberBaseToken> cosh(const Token& in);
		export UnityToken cosh(const ZeroToken& in);
		export NumberToken cosh(const NegUnityToken& in);
		export NumberToken cosh(const UnityToken& in);
		export NanToken cosh(const NanToken& in);
		export NumberToken cosh(const NumberToken& in);

		// Tanh

		export std::unique_ptr<NumberBaseToken> tanh(const Token& in);
		export ZeroToken tanh(const ZeroToken& in);
		export NumberToken tanh(const NegUnityToken& in);
		export NumberToken tanh(const UnityToken& in);
		export NanToken tanh(const NanToken& in);
		export NumberToken tanh(const NumberToken& in);

		// Asinh

		export std::unique_ptr<NumberBaseToken> asinh(const Token& in);
		export ZeroToken asinh(const ZeroToken& in);
		export NumberToken asinh(const NegUnityToken& in);
		export NumberToken asinh(const UnityToken& in);
		export NanToken asinh(const NanToken& in);
		export NumberToken asinh(const NumberToken& in);

		// Acosh

		export std::unique_ptr<NumberBaseToken> acosh(const Token& in);
		export NanToken acosh(const ZeroToken& in);
		export NanToken acosh(const NegUnityToken& in);
		export ZeroToken acosh(const UnityToken& in);
		export NanToken acosh(const NanToken& in);
		export NumberToken acosh(const NumberToken& in);

		// Atanh

		export std::unique_ptr<NumberBaseToken> atanh(const Token& in);
		export ZeroToken atanh(const ZeroToken& in);
		export NanToken atanh(const NegUnityToken& in);
		export NanToken atanh(const UnityToken& in);
		export NanToken atanh(const NanToken& in);
		export NumberToken atanh(const NumberToken& in);



		// IMPLEMENTATION


		std::unique_ptr<NumberBaseToken> sin(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<ZeroToken>(sin(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<NumberToken>(sin(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NumberToken>(sin(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(sin(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(sin(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		ZeroToken sin(const ZeroToken& in)
		{
			return in;
		}

		NumberToken sin(const NegUnityToken& in)
		{
			return NumberToken(std::sin(-1.0), false, in.sizes);
		}

		NumberToken sin(const UnityToken& in)
		{
			return NumberToken(std::sin(1.0), false, in.sizes);
		}

		NanToken sin(const NanToken& in)
		{
			return in;
		}

		NumberToken sin(const NumberToken& in)
		{
			return NumberToken(std::sin(in.num), in.is_imaginary, in.sizes);
		}

		// <====================================== COS ============================================>

		std::unique_ptr<NumberBaseToken> cos(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<UnityToken>(cos(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<NumberToken>(cos(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NumberToken>(cos(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(cos(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(cos(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		UnityToken cos(const ZeroToken& in)
		{
			return UnityToken(in.sizes);
		}

		NumberToken cos(const NegUnityToken& in)
		{
			return NumberToken(std::cos(-1.0), false, in.sizes);
		}

		NumberToken cos(const UnityToken& in)
		{
			return NumberToken(std::cos(1.0), false, in.sizes);
		}

		NanToken cos(const NanToken& in)
		{
			return in;
		}

		NumberToken cos(const NumberToken& in)
		{
			return NumberToken(std::cos(in.num), in.is_imaginary, in.sizes);
		}

		// <====================================== TAN ============================================>

		std::unique_ptr<NumberBaseToken> tan(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<UnityToken>(tan(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<NumberToken>(tan(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NumberToken>(tan(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(tan(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(tan(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		UnityToken tan(const ZeroToken& in)
		{
			return UnityToken(in.sizes);
		}

		NumberToken tan(const NegUnityToken& in)
		{
			return NumberToken(std::tan(-1.0), false, in.sizes);
		}

		NumberToken tan(const UnityToken& in)
		{
			return NumberToken(std::tan(1.0), false, in.sizes);
		}

		NanToken tan(const NanToken& in)
		{
			return in;
		}

		NumberToken tan(const NumberToken& in)
		{
			return NumberToken(std::tan(in.num), in.is_imaginary, in.sizes);
		}

		// <====================================== ASIN ============================================>

		std::unique_ptr<NumberBaseToken> asin(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<UnityToken>(asin(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<NumberToken>(asin(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NumberToken>(asin(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(asin(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(asin(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		UnityToken asin(const ZeroToken& in)
		{
			return UnityToken(in.sizes);
		}

		NumberToken asin(const NegUnityToken& in)
		{
			return NumberToken(std::asin(-1.0), false, in.sizes);
		}

		NumberToken asin(const UnityToken& in)
		{
			return NumberToken(std::asin(-1.0), false, in.sizes);
		}

		NanToken asin(const NanToken& in)
		{
			return in;
		}

		NumberToken asin(const NumberToken& in)
		{
			return NumberToken(std::asin(in.num), in.is_imaginary, in.sizes);
		}

		// <====================================== ACOS ============================================>

		std::unique_ptr<NumberBaseToken> acos(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<NumberToken>(acos(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<ZeroToken>(acos(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NumberToken>(acos(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(acos(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(acos(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		NumberToken acos(const ZeroToken& in)
		{
			return NumberToken(std::acos(0.0), false, in.sizes);
		}

		NumberToken acos(const NegUnityToken& in)
		{
			return NumberToken(std::acos(-1.0), false, in.sizes);
		}

		ZeroToken acos(const UnityToken& in)
		{
			return ZeroToken(in.sizes);
		}

		NanToken acos(const NanToken& in)
		{
			return in;
		}

		NumberToken acos(const NumberToken& in)
		{
			return NumberToken(std::acos(in.num), in.is_imaginary, in.sizes);
		}

		// <====================================== ATAN ============================================>

		std::unique_ptr<NumberBaseToken> atan(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<ZeroToken>(atan(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<NumberToken>(atan(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NumberToken>(atan(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(atan(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(atan(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		ZeroToken atan(const ZeroToken& in)
		{
			return in;
		}

		NumberToken atan(const NegUnityToken& in)
		{
			return NumberToken(std::atan(-1.0), false, in.sizes);
		}

		NumberToken atan(const UnityToken& in)
		{
			return NumberToken(std::atan(1.0), false, in.sizes);
		}

		NanToken atan(const NanToken& in)
		{
			return NanToken();
		}

		NumberToken atan(const NumberToken& in)
		{
			return NumberToken(std::atan(in.num), in.is_imaginary, in.sizes);
		}

		// <====================================== SINH ============================================>

		std::unique_ptr<NumberBaseToken> sinh(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<ZeroToken>(sinh(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<NumberToken>(sinh(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NumberToken>(sinh(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(sinh(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(sinh(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		ZeroToken sinh(const ZeroToken& in)
		{
			return in;
		}

		NumberToken sinh(const NegUnityToken& in)
		{
			return NumberToken(std::sinh(-1.0), false, in.sizes);
		}

		NumberToken sinh(const UnityToken& in)
		{
			return NumberToken(std::sinh(1.0), false, in.sizes);
		}

		NanToken sinh(const NanToken& in)
		{
			return in;
		}

		NumberToken sinh(const NumberToken& in)
		{
			return NumberToken(std::sinh(in.num), in.is_imaginary, in.sizes);
		}

		// <====================================== COSH ============================================>

		std::unique_ptr<NumberBaseToken> cosh(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<UnityToken>(cosh(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<NumberToken>(cosh(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NumberToken>(cosh(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(cosh(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(cosh(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		UnityToken cosh(const ZeroToken& in)
		{
			return UnityToken(in.sizes);
		}

		NumberToken cosh(const NegUnityToken& in)
		{
			return NumberToken(std::cosh(-1.0), false, in.sizes);
		}

		NumberToken cosh(const UnityToken& in)
		{
			return NumberToken(std::cosh(1.0), false, in.sizes);
		}

		NanToken cosh(const NanToken& in)
		{
			return in;
		}

		NumberToken cosh(const NumberToken& in)
		{
			return NumberToken(std::cosh(in.num), in.is_imaginary, in.sizes);
		}

		// <====================================== TANH ============================================>

		std::unique_ptr<NumberBaseToken> tanh(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<ZeroToken>(tanh(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<NumberToken>(tanh(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NumberToken>(tanh(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(tanh(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(tanh(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		ZeroToken tanh(const ZeroToken& in)
		{
			return in;
		}

		NumberToken tanh(const NegUnityToken& in)
		{
			return NumberToken(std::tanh(-1.0), false, in.sizes);
		}

		NumberToken tanh(const UnityToken& in)
		{
			return NumberToken(std::tanh(1.0), false, in.sizes);
		}

		NanToken tanh(const NanToken& in)
		{
			return in;
		}

		NumberToken tanh(const NumberToken& in)
		{
			return NumberToken(std::tanh(in.num), in.is_imaginary, in.sizes);
		}

		// <====================================== ASINH ============================================>

		std::unique_ptr<NumberBaseToken> asinh(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<ZeroToken>(asinh(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<NumberToken>(asinh(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NumberToken>(asinh(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(asinh(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(asinh(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		ZeroToken asinh(const ZeroToken& in)
		{
			return in;
		}

		NumberToken asinh(const NegUnityToken& in)
		{
			return NumberToken(std::asinh(-1.0), false, in.sizes);
		}

		NumberToken asinh(const UnityToken& in)
		{
			return NumberToken(std::asinh(1.0), false, in.sizes);
		}

		NanToken asinh(const NanToken& in)
		{
			return in;
		}

		NumberToken asinh(const NumberToken& in)
		{
			return NumberToken(std::asinh(in.num), in.is_imaginary, in.sizes);
		}

		// <====================================== ACOSH ============================================>

		std::unique_ptr<NumberBaseToken> acosh(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<NanToken>(acosh(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<ZeroToken>(acosh(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NanToken>(acosh(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(acosh(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(acosh(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		NanToken acosh(const ZeroToken& in)
		{
			return NanToken(in.sizes);
		}

		NanToken acosh(const NegUnityToken& in)
		{
			return NanToken(in.sizes);
		}

		ZeroToken acosh(const UnityToken& in)
		{
			return ZeroToken(in.sizes);
		}

		NanToken acosh(const NanToken& in)
		{
			return in;
		}

		NumberToken acosh(const NumberToken& in)
		{
			return NumberToken(std::acosh(in.num), in.is_imaginary, in.sizes);
		}

		// <====================================== ATANH ============================================>

		std::unique_ptr<NumberBaseToken> atanh(const Token& in)
		{
			switch (in.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(in);
				return std::make_unique<ZeroToken>(atanh(atok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(in);
				return std::make_unique<NanToken>(atanh(atok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
				return std::make_unique<NanToken>(atanh(atok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(in);
				return std::make_unique<NanToken>(atanh(atok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(in);
				return std::make_unique<NumberToken>(atanh(atok));
			}
			default:
				throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
			}
		}

		ZeroToken atanh(const ZeroToken& in)
		{
			return in;
		}

		NanToken atanh(const NegUnityToken& in)
		{
			return NanToken(in.sizes);
		}

		NanToken atanh(const UnityToken& in)
		{
			return NanToken(in.sizes);
		}

		NanToken atanh(const NanToken& in)
		{
			return in;
		}

		NumberToken atanh(const NumberToken& in)
		{
			return NumberToken(std::atanh(in.num), in.is_imaginary, in.sizes);
		}


	}
}

