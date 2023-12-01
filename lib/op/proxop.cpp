module;

#include <torch/torch.h>

module proxop;

hasty::op::ProximalOperator::ProximalOperator(double base_alpha)
	: _base_alpha(base_alpha)
{

}

double hasty::op::ProximalOperator::get_base_alpha() const
{
	return _base_alpha;
}

void hasty::op::ProximalOperator::set_base_alpha(double base_alpha)
{
	_base_alpha = base_alpha;
}

hasty::op::Vector hasty::op::ProximalOperator::apply(const Vector& input, double alpha) const
{
	double mul = _base_alpha * alpha;
	return _apply(input, mul);
}

void hasty::op::ProximalOperator::apply_inplace(Vector& inout, double alpha) const
{
	if (!has_inplace_apply())
		throw std::runtime_error("This proximal operator does not implement apply_inplace()");
	double mul = _base_alpha * alpha;
	_apply_inplace(inout, mul);
}

bool hasty::op::ProximalOperator::has_inplace_apply() const
{
	return false;
}

hasty::op::Vector hasty::op::ProximalOperator::_apply(const Vector& input, double alpha) const
{
	throw std::runtime_error("_apply() not implemented on base ProximalOperator");
}

void hasty::op::ProximalOperator::_apply_inplace(Vector& input, double alpha) const
{
	throw std::runtime_error("_apply_inplace() not implemented on base ProximalOperator");
}

// ZERO FUNC

hasty::uptr<hasty::op::ZeroFunc> hasty::op::ZeroFunc::Create()
{
	struct creator : public ZeroFunc { creator() : ZeroFunc() {} };
	return std::make_unique<creator>();
}

hasty::op::Vector hasty::op::ZeroFunc::_apply(const Vector& input, double alpha) const
{
	return input;
}

// CONVEX CONJUGATE

hasty::uptr<hasty::op::ConvexConjugate> hasty::op::ConvexConjugate::Create(sptr<ProximalOperator> prox, double base_alpha)
{
	struct creator : public ConvexConjugate { creator(sptr<ProximalOperator> a, double b) : ConvexConjugate(std::move(a), b) {} };
	return std::make_unique<creator>(std::move(prox), base_alpha);
}

hasty::op::ConvexConjugate::ConvexConjugate(sptr<ProximalOperator> prox, double base_alpha)
	: ProximalOperator(base_alpha), _prox(std::move(prox))
{
}

hasty::op::Vector hasty::op::ConvexConjugate::_apply(const Vector& input, double alpha)
{
	return input - alpha * _prox->apply(input / alpha, 1.0 / alpha);
}

// POSTCOMPOSITION

hasty::uptr<hasty::op::Postcomposition> hasty::op::Postcomposition::Create(sptr<ProximalOperator> prox, double base_alpha)
{
	struct creator : public Postcomposition { creator(sptr<ProximalOperator> a, double b) : Postcomposition(std::move(a), b) {} };
	return std::make_unique<creator>(std::move(prox), base_alpha);
}

hasty::op::Postcomposition::Postcomposition(sptr<ProximalOperator> prox, double base_alpha)
	: ProximalOperator(base_alpha), _prox(std::move(prox))
{
}

hasty::op::Vector hasty::op::Postcomposition::_apply(const Vector& input, double alpha)
{
	return apply(input, alpha);
}


// PRECOMPOSITION

hasty::uptr<hasty::op::Precomposition> hasty::op::Precomposition::Create(sptr<ProximalOperator> prox, double a, double b, double base_alpha)
{
	struct creator : public Precomposition { 
		creator(sptr<ProximalOperator> a, double b, double c, double d) 
			: Precomposition(std::move(a), b, c, d) {} 
	};
	return std::make_unique<creator>(std::move(prox), a, b, base_alpha);
}

hasty::op::Precomposition::Precomposition(sptr<ProximalOperator> prox, double a, double b, double base_alpha)
	: ProximalOperator(base_alpha), _prox(std::move(prox)), _a(a), _b(b)
{
}

hasty::op::Vector hasty::op::Precomposition::_apply(const Vector& input, double alpha)
{
	return (1.0 / _a) * (apply(_a*input + _b, _a*_a*alpha) - _b);
}

// UNITARY TRANSFORM

hasty::uptr<hasty::op::UnitaryTransform> hasty::op::UnitaryTransform::Create(sptr<ProximalOperator> prox, sptr<Operator> unitary, sptr<Operator> unitary_adjoint, double base_alpha)
{
	struct creator : public UnitaryTransform {
		creator(sptr<ProximalOperator> a, sptr<Operator> b, sptr<Operator> c, double d)
			: UnitaryTransform(std::move(a), std::move(b), std::move(c), d) {}
	};
	return std::make_unique<creator>(std::move(prox), std::move(unitary), std::move(unitary_adjoint), base_alpha);
}

hasty::op::UnitaryTransform::UnitaryTransform(sptr<ProximalOperator> prox, sptr<Operator> unitary, sptr<Operator> unitary_adjoint, double base_alpha)
	: ProximalOperator(base_alpha), _prox(std::move(prox)), _unitary(std::move(unitary)), _unitary_adjoint(std::move(unitary_adjoint))
{
}

hasty::op::Vector hasty::op::UnitaryTransform::_apply(const Vector& input, double alpha)
{
	return _unitary_adjoint->apply(_prox->apply(_unitary->apply(input), alpha));
}

// AFFINE ADDITION

hasty::uptr<hasty::op::AffineAddition> hasty::op::AffineAddition::Create(sptr<ProximalOperator> prox, const Vector& a, double base_alpha)
{
	struct creator : public AffineAddition {
		creator(sptr<ProximalOperator> a, const Vector& b, double c)
			: AffineAddition(std::move(a), b, c) {}
	};
	return std::make_unique<creator>(std::move(prox), a, base_alpha);
}

hasty::op::AffineAddition::AffineAddition(sptr<ProximalOperator> prox, const Vector& a, double base_alpha)
	: ProximalOperator(base_alpha), _prox(std::move(prox)), _a(a)
{}

hasty::op::Vector hasty::op::AffineAddition::_apply(const Vector& input, double alpha)
{
	return _prox->apply(input - alpha * _a, alpha);
}

 // L2 REG

hasty::uptr<hasty::op::L2Reg> hasty::op::L2Reg::Create(sptr<ProximalOperator> prox, double rho, const opt<Vector>& a, double base_alpha)
{
	struct creator : public L2Reg {
		creator(sptr<ProximalOperator> a, double b, const opt<Vector>& c, double d)
			: L2Reg(std::move(a), b, c, d) {}
	};
	return std::make_unique<creator>(std::move(prox), rho, a, base_alpha);
}

hasty::op::L2Reg::L2Reg(sptr<ProximalOperator> prox, double rho, const opt<Vector>& a, double base_alpha)
	: ProximalOperator(base_alpha), _prox(std::move(prox)), _rho(rho), _a(a)
{
}

hasty::op::Vector hasty::op::L2Reg::_apply(const Vector& input, double alpha)
{
	double mult = alpha * _rho;
	double denom = 1.0 + mult;

	Vector output = input;

	if (_a.has_value())
		output += (*_a) * mult;

	output /= denom;

	if (_prox)
		return _prox->apply(output, alpha / denom);
	return output;
}
