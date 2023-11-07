#include "proxop.hpp"


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

hasty::op::Vector hasty::op::ProximalOperator::operator()(const Vector& input, double alpha) const
{
	return apply(input, alpha);
}


hasty::op::Vector hasty::op::ProximalOperator::_apply(const Vector& input, double alpha) const
{
	throw std::runtime_error("apply() not implemented on base ProximalOperator");
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

void hasty::op::ProximalOperator::_apply_inplace(Vector& input, double alpha) const
{
	throw std::runtime_error("_apply_inplace() not implemented on base ProximalOperator");
}


hasty::op::Vector hasty::op::ZeroFunc::_apply(const Vector& input, double alpha) const
{
	return input;
}


hasty::op::ConvexConjugate::ConvexConjugate(const ProximalOperator& prox, double base_alpha)
	: ProximalOperator(base_alpha), _prox(prox)
{
}

hasty::op::Vector hasty::op::ConvexConjugate::_apply(const Vector& input, double alpha)
{
	return input - alpha * _prox(input / alpha, 1.0 / alpha);
}


hasty::op::Postcomposition::Postcomposition(const ProximalOperator& prox, double base_alpha)
	: ProximalOperator(base_alpha), _prox(prox)
{
}

hasty::op::Vector hasty::op::Postcomposition::_apply(const Vector& input, double alpha)
{
	return apply(input, alpha);
}

hasty::op::Precomposition::Precomposition(const ProximalOperator& prox, double a, double b, double base_alpha)
	: ProximalOperator(base_alpha), _prox(prox), _a(a), _b(b)
{
}

hasty::op::Vector hasty::op::Precomposition::_apply(const Vector& input, double alpha)
{
	return (1.0 / _a) * (apply(_a*input + _b, _a*_a*alpha) - _b);
}

hasty::op::UnitaryTransform::UnitaryTransform(const ProximalOperator& prox, std::shared_ptr<Operator> unitary,
	std::shared_ptr<Operator> unitary_adjoint, double base_alpha)
	: ProximalOperator(base_alpha), _prox(prox), _unitary(std::move(unitary)), _unitary_adjoint(std::move(unitary_adjoint))
{
}

hasty::op::Vector hasty::op::UnitaryTransform::_apply(const Vector& input, double alpha)
{
	return _unitary_adjoint->apply(_prox(_unitary->apply(input), alpha));
}

hasty::op::AffineAddition::AffineAddition(const ProximalOperator& prox, const Vector& a, double base_alpha)
	: ProximalOperator(base_alpha), _prox(prox), _a(a)
{}

hasty::op::Vector hasty::op::AffineAddition::_apply(const Vector& input, double alpha)
{
	return _prox(input - alpha * _a, alpha);
}

hasty::op::L2Reg::L2Reg(const std::optional<ProximalOperator>& prox, double rho, const std::optional<Vector>& a, double base_alpha)
	: ProximalOperator(base_alpha), _prox(prox), _rho(rho), _a(a)
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

	if (_prox.has_value())
		return (*_prox)(output, alpha / denom);
	return output;
}
