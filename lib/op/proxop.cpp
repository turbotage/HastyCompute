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