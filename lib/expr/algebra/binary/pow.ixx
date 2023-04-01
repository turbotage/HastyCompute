module;

export module pow;

import std;

//import <memory>;
//import <stdexcept>;
//import <complex>;

import token;
import token_algebra;
import hasty_util;

namespace hasty {
	namespace expr {

		// Pow

		export std::unique_ptr<NumberBaseToken> pow(const Token& a, const Token& b);

		export std::unique_ptr<NumberBaseToken> pow(const ZeroToken& a, const Token& b);
		export std::unique_ptr<NumberBaseToken> pow(const UnityToken& a, const Token& b);
		export std::unique_ptr<NumberBaseToken> pow(const NegUnityToken& a, const Token& b);
		export std::unique_ptr<NumberBaseToken> pow(const NanToken& a, const Token& b);
		export std::unique_ptr<NumberBaseToken> pow(const NumberToken& a, const Token& b);

		export UnityToken pow(const ZeroToken& a, const ZeroToken& b);
		export NanToken pow(const ZeroToken& a, const NegUnityToken& b);
		export ZeroToken pow(const ZeroToken& a, const UnityToken& b);
		export NanToken pow(const ZeroToken& a, const NanToken& b);
		export NumberToken pow(const ZeroToken& a, const NumberToken& b);

		export UnityToken pow(const NegUnityToken& a, const ZeroToken& b);
		export NegUnityToken pow(const NegUnityToken& a, const NegUnityToken& b);
		export NegUnityToken pow(const NegUnityToken& a, const UnityToken& b);
		export NanToken pow(const NegUnityToken& a, const NanToken& b);
		export NumberToken pow(const NegUnityToken& a, const NumberToken& b);

		export UnityToken pow(const UnityToken& a, const ZeroToken& b);
		export UnityToken pow(const UnityToken& a, const NegUnityToken& b);
		export UnityToken pow(const UnityToken& a, const UnityToken& b);
		export NanToken pow(const UnityToken& a, const NanToken& b);
		export NumberToken pow(const UnityToken& a, const NumberToken& b);

		export UnityToken pow(const NanToken& a, const ZeroToken& b);
		export NanToken pow(const NanToken& a, const NegUnityToken& b);
		export NanToken pow(const NanToken& a, const UnityToken& b);
		export NanToken pow(const NanToken& a, const NanToken& b);
		export NanToken pow(const NanToken& a, const NumberToken& b);

		export NumberToken pow(const NumberToken& a, const ZeroToken& b);
		export NumberToken pow(const NumberToken& a, const NegUnityToken& b);
		export NumberToken pow(const NumberToken& a, const UnityToken& b);
		export NanToken pow(const NumberToken& a, const NanToken& b);
		export NumberToken pow(const NumberToken& a, const NumberToken& b);





		// IMPLEMENTATION


		std::unique_ptr<NumberBaseToken> pow(const Token& a, const Token& b)
		{
			switch (a.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(a);
				return pow(atok, b);
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(a);
				return pow(atok, b);
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(a);
				return pow(atok, b);
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(a);
				return pow(atok, b);
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(a);
				return pow(atok, b);
			}
			default:
				throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
			}
		}

		std::unique_ptr<NumberBaseToken> pow(const ZeroToken& a, const Token& b)
		{
			switch (b.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& btok = static_cast<const ZeroToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& btok = static_cast<const UnityToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& btok = static_cast<const NanToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& btok = static_cast<const NumberToken&>(b);
				return to_ptr(pow(a, btok));
			}
			default:
				throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
			}
		}

		std::unique_ptr<NumberBaseToken> pow(const UnityToken& a, const Token& b)
		{
			switch (b.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& btok = static_cast<const ZeroToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& btok = static_cast<const UnityToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& btok = static_cast<const NanToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& btok = static_cast<const NumberToken&>(b);
				return to_ptr(pow(a, btok));
			}
			default:
				throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
			}
		}

		std::unique_ptr<NumberBaseToken> pow(const NegUnityToken& a, const Token& b)
		{
			switch (b.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& btok = static_cast<const ZeroToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& btok = static_cast<const UnityToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& btok = static_cast<const NanToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& btok = static_cast<const NumberToken&>(b);
				return to_ptr(pow(a, btok));
			}
			default:
				throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
			}
		}

		std::unique_ptr<NumberBaseToken> pow(const NanToken& a, const Token& b)
		{
			switch (b.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& btok = static_cast<const ZeroToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& btok = static_cast<const UnityToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& btok = static_cast<const NanToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& btok = static_cast<const NumberToken&>(b);
				return to_ptr(pow(a, btok));
			}
			default:
				throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
			}
		}

		std::unique_ptr<NumberBaseToken> pow(const NumberToken& a, const Token& b)
		{
			switch (b.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& btok = static_cast<const ZeroToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& btok = static_cast<const UnityToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& btok = static_cast<const NanToken&>(b);
				return to_ptr(pow(a, btok));
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& btok = static_cast<const NumberToken&>(b);
				return to_ptr(pow(a, btok));
			}
			default:
				throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
			}
		}


		// 1
		UnityToken pow(const ZeroToken& a, const ZeroToken& b)
		{
			return UnityToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken pow(const ZeroToken& a, const NegUnityToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		ZeroToken pow(const ZeroToken& a, const UnityToken& b)
		{
			return ZeroToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken pow(const ZeroToken& a, const NanToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NumberToken pow(const ZeroToken& a, const NumberToken& b)
		{
			return NumberToken(std::pow(0.0f, b.num), b.is_imaginary, util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		// 2
		UnityToken pow(const NegUnityToken& a, const ZeroToken& b)
		{
			return UnityToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NegUnityToken pow(const NegUnityToken& a, const NegUnityToken& b)
		{
			return NegUnityToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NegUnityToken pow(const NegUnityToken& a, const UnityToken& b)
		{
			return NegUnityToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken pow(const NegUnityToken& a, const NanToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NumberToken pow(const NegUnityToken& a, const NumberToken& b)
		{
			return NumberToken(std::pow(-1.0f, b.num), b.is_imaginary, util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		// 3
		UnityToken pow(const UnityToken& a, const ZeroToken& b)
		{
			return UnityToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		UnityToken pow(const UnityToken& a, const NegUnityToken& b)
		{
			return UnityToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		UnityToken pow(const UnityToken& a, const UnityToken& b)
		{
			return UnityToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken pow(const UnityToken& a, const NanToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NumberToken pow(const UnityToken& a, const NumberToken& b)
		{
			return NumberToken(std::pow(1.0f, b.num), b.is_imaginary, util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		// 4
		UnityToken pow(const NanToken& a, const ZeroToken& b)
		{
			return UnityToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken pow(const NanToken& a, const NegUnityToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken pow(const NanToken& a, const UnityToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken pow(const NanToken& a, const NanToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken pow(const NanToken& a, const NumberToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		// 5
		NumberToken pow(const NumberToken& a, const ZeroToken& b)
		{
			return NumberToken(std::pow(a.num, 0.0f), a.is_imaginary, util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NumberToken pow(const NumberToken& a, const NegUnityToken& b)
		{
			return NumberToken(std::pow(a.num, -1.0f), a.is_imaginary, util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NumberToken pow(const NumberToken& a, const UnityToken& b)
		{
			return NumberToken(std::pow(a.num, 1.0f), a.is_imaginary, util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken pow(const NumberToken& a, const NanToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NumberToken pow(const NumberToken& a, const NumberToken& b)
		{
			return NumberToken(std::pow(a.num, b.num), a.is_imaginary || b.is_imaginary, util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

	}
}