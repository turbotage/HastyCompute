#include "proxop.hpp"


hasty::op::ProximalOpeartor::ProximalOpeartor(double base_alpha)
	: _base_alpha(base_alpha)
{

}

double hasty::op::ProximalOpeartor::get_base_alpha()
{
	return _base_alpha;
}

void hasty::op::ProximalOpeartor::set_base_alpha(double base_alpha)
{
	_base_alpha = base_alpha;
}

hasty::op::Vector hasty::op::ProximalOpeartor::apply(const Vector& input, double alpha)
{
	double mul = _base_alpha * alpha;
	return _apply(input, mul);
}

hasty::op::Vector hasty::op::ProximalOpeartor::_apply(const Vector& input, double alpha)
{
	throw std::runtime_error("apply() not implemented on base ProximalOperator");
}



hasty::op::Vector hasty::op::ZeroFunc::_apply(const Vector& input, double alpha)
{
	return input;
}

