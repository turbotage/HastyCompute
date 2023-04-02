module;

export module div;

#ifdef STL_AS_MODULES
import std;
#else
import <memory>;
import <stdexcept>;
import <complex>;
#endif


import neg;
import token;
import token_algebra;
import hasty_util;

namespace hasty {
	namespace expr {

		export std::unique_ptr<NumberBaseToken> operator/(const Token& a, const Token& b);

		export std::unique_ptr<NumberBaseToken> operator/(const ZeroToken& a, const Token& b);
		export std::unique_ptr<NumberBaseToken> operator/(const UnityToken& a, const Token& b);
		export std::unique_ptr<NumberBaseToken> operator/(const NegUnityToken& a, const Token& b);
		export std::unique_ptr<NumberBaseToken> operator/(const NanToken& a, const Token& b);
		export std::unique_ptr<NumberBaseToken> operator/(const NumberToken& a, const Token& b);

		export NanToken operator/(const ZeroToken& a, const ZeroToken& b);
		export ZeroToken operator/(const ZeroToken& a, const NegUnityToken& b);
		export ZeroToken operator/(const ZeroToken& a, const UnityToken& b);
		export NanToken operator/(const ZeroToken& a, const NanToken& b);
		export NumberToken operator/(const ZeroToken& a, const NumberToken& b);

		export NanToken operator/(const NegUnityToken& a, const ZeroToken& b);
		export UnityToken operator/(const NegUnityToken& a, const NegUnityToken& b);
		export NegUnityToken operator/(const NegUnityToken& a, const UnityToken& b);
		export NanToken operator/(const NegUnityToken& a, const NanToken& b);
		export NumberToken operator/(const NegUnityToken& a, const NumberToken& b);

		export NanToken operator/(const UnityToken& a, const ZeroToken& b);
		export NegUnityToken operator/(const UnityToken& a, const NegUnityToken& b);
		export UnityToken operator/(const UnityToken& a, const UnityToken& b);
		export NanToken operator/(const UnityToken& a, const NanToken& b);
		export NumberToken operator/(const UnityToken& a, const NumberToken& b);

		export NanToken operator/(const NanToken& a, const ZeroToken& b);
		export NanToken operator/(const NanToken& a, const NegUnityToken& b);
		export NanToken operator/(const NanToken& a, const UnityToken& b);
		export NanToken operator/(const NanToken& a, const NanToken& b);
		export NanToken operator/(const NanToken& a, const NumberToken& b);

		export NanToken operator/(const NumberToken& a, const ZeroToken& b);
		export NumberToken operator/(const NumberToken& a, const NegUnityToken& b);
		export NumberToken operator/(const NumberToken& a, const UnityToken& b);
		export NanToken operator/(const NumberToken& a, const NanToken& b);
		export NumberToken operator/(const NumberToken& a, const NumberToken& b);


		// IMPLEMENTATION

		std::unique_ptr<NumberBaseToken> operator/(const Token& a, const Token& b)
		{
			switch (a.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& atok = static_cast<const ZeroToken&>(a);
				return atok / b;
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& atok = static_cast<const UnityToken&>(a);
				return atok / b;
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& atok = static_cast<const NegUnityToken&>(a);
				return atok / b;
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& atok = static_cast<const NanToken&>(a);
				return atok / b;
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& atok = static_cast<const NumberToken&>(a);
				return atok / b;
			}
			default:
				throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
			}
		}

		std::unique_ptr<NumberBaseToken> operator/(const ZeroToken& a, const Token& b)
		{
			switch (b.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& btok = static_cast<const ZeroToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& btok = static_cast<const UnityToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& btok = static_cast<const NanToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& btok = static_cast<const NumberToken&>(b);
				return to_ptr(a / btok);
			}
			default:
				throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
			}
		}

		std::unique_ptr<NumberBaseToken> operator/(const UnityToken& a, const Token& b)
		{
			switch (b.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& btok = static_cast<const ZeroToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& btok = static_cast<const UnityToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& btok = static_cast<const NanToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& btok = static_cast<const NumberToken&>(b);
				return to_ptr(a / btok);
			}
			default:
				throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
			}
		}

		std::unique_ptr<NumberBaseToken> operator/(const NegUnityToken& a, const Token& b)
		{
			switch (b.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& btok = static_cast<const ZeroToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& btok = static_cast<const UnityToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& btok = static_cast<const NanToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& btok = static_cast<const NumberToken&>(b);
				return to_ptr(a / btok);
			}
			default:
				throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
			}
		}

		std::unique_ptr<NumberBaseToken> operator/(const NanToken& a, const Token& b)
		{
			switch (b.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& btok = static_cast<const ZeroToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& btok = static_cast<const UnityToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& btok = static_cast<const NanToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& btok = static_cast<const NumberToken&>(b);
				return to_ptr(a / btok);
			}
			default:
				throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
			}
		}

		std::unique_ptr<NumberBaseToken> operator/(const NumberToken& a, const Token& b)
		{
			switch (b.get_token_type()) {
			case TokenType::ZERO_TYPE:
			{
				const ZeroToken& btok = static_cast<const ZeroToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::UNITY_TYPE:
			{
				const UnityToken& btok = static_cast<const UnityToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NEG_UNITY_TYPE:
			{
				const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NAN_TYPE:
			{
				const NanToken& btok = static_cast<const NanToken&>(b);
				return to_ptr(a / btok);
			}
			case TokenType::NUMBER_TYPE:
			{
				const NumberToken& btok = static_cast<const NumberToken&>(b);
				return to_ptr(a / btok);
			}
			default:
				throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
			}
		}

		// 1
		NanToken operator/(const ZeroToken& a, const ZeroToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		ZeroToken operator/(const ZeroToken& a, const NegUnityToken& b)
		{
			return ZeroToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		ZeroToken operator/(const ZeroToken& a, const UnityToken& b)
		{
			return ZeroToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken operator/(const ZeroToken& a, const NanToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NumberToken operator/(const ZeroToken& a, const NumberToken& b)
		{
			return NumberToken(0.0f / b.num, b.is_imaginary, util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		// 2
		NanToken operator/(const NegUnityToken& a, const ZeroToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		UnityToken operator/(const NegUnityToken& a, const NegUnityToken& b)
		{
			return UnityToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NegUnityToken operator/(const NegUnityToken& a, const UnityToken& b)
		{
			return NegUnityToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken operator/(const NegUnityToken& a, const NanToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NumberToken operator/(const NegUnityToken& a, const NumberToken& b)
		{
			return NumberToken(-1.0f / b.num, b.is_imaginary, util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		// 3
		NanToken operator/(const UnityToken& a, const ZeroToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NegUnityToken operator/(const UnityToken& a, const NegUnityToken& b)
		{
			return NegUnityToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		UnityToken operator/(const UnityToken& a, const UnityToken& b)
		{
			return UnityToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken operator/(const UnityToken& a, const NanToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NumberToken operator/(const UnityToken& a, const NumberToken& b)
		{
			return NumberToken(1.0f / b.num, b.is_imaginary, util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		// 4
		NanToken operator/(const NanToken& a, const ZeroToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken operator/(const NanToken& a, const NegUnityToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken operator/(const NanToken& a, const UnityToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken operator/(const NanToken& a, const NanToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken operator/(const NanToken& a, const NumberToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		// 5
		NanToken operator/(const NumberToken& a, const ZeroToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NumberToken operator/(const NumberToken& a, const NegUnityToken& b)
		{
			return NumberToken(-a.num, a.is_imaginary, util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NumberToken operator/(const NumberToken& a, const UnityToken& b)
		{
			return NumberToken(a.num, a.is_imaginary, util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NanToken operator/(const NumberToken& a, const NanToken& b)
		{
			return NanToken(util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

		NumberToken operator/(const NumberToken& a, const NumberToken& b)
		{
			return NumberToken(a.num / b.num, a.is_imaginary || b.is_imaginary, util::broadcast_tensor_shapes(a.sizes, b.sizes));
		}

	}
}
