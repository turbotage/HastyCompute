module;

#include <torch/torch.h>

module op;

namespace {
	std::mutex _oplock;
}


// OPERATOR

hasty::op::Operator::Operator()
	: _should_inplace_apply(false)
{
}

hasty::op::Operator::Operator(bool should_inplace_apply)
	: _should_inplace_apply(should_inplace_apply)
{
}

hasty::op::Vector hasty::op::Operator::operator()(const Vector& in) const
{
	return apply(in);
}

hasty::op::Vector hasty::op::operator*(const Operator& lhs, const Vector& rhs)
{
	return lhs.apply(rhs);
}

void hasty::op::Operator::apply_inplace(Vector& in) const
{
	throw std::runtime_error("apply_inplace() wasn't implemented");
}

bool hasty::op::Operator::has_inplace_apply() const
{
	return false;
}

bool hasty::op::Operator::should_inplace_apply() const
{
	return false;
}
// VECTOR ACCESS

at::Tensor& hasty::op::Operator::access_vectensor(Vector& vec) const
{
	return vec._tensor;
}

const at::Tensor& hasty::op::Operator::access_vectensor(const Vector& vec) const
{
	return vec._tensor;
}

std::vector<hasty::op::Vector>& hasty::op::Operator::access_vecchilds(Vector& vec) const
{
	return vec._children;
}

const std::vector<hasty::op::Vector>& hasty::op::Operator::access_vecchilds(const Vector& vec) const
{
	return vec._children;
}


// ADJOINTABLE OP

hasty::op::AdjointableOp::AdjointableOp()
{

}

hasty::op::AdjointableOp::AdjointableOp(bool should_inplace_apply)
	: Operator(should_inplace_apply)
{
}

// DEFAULT ADJOINTABLE OP

std::unique_ptr<hasty::op::DefaultAdjointableOp> hasty::op::DefaultAdjointableOp::Create(std::shared_ptr<Operator> op, std::shared_ptr<Operator> oph)
{
	struct creator : public DefaultAdjointableOp {
		creator(std::shared_ptr<Operator> a, std::shared_ptr<Operator> b)
			: DefaultAdjointableOp(std::move(a), std::move(b)) {}
	};
	return std::make_unique<creator>(std::move(op), std::move(oph));
}

hasty::op::DefaultAdjointableOp::DefaultAdjointableOp(std::shared_ptr<Operator> op, std::shared_ptr<Operator> oph)
	: _op(std::move(op)), _oph(std::move(oph))
{
}

hasty::op::Vector hasty::op::DefaultAdjointableOp::apply(const Vector& in) const
{
	return _oph->apply(in);
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::op::DefaultAdjointableOp::adjoint() const
{
	return DefaultAdjointableOp::Create(_oph, _op);
}

std::shared_ptr<hasty::op::Operator> hasty::op::DefaultAdjointableOp::to_device(at::Stream stream) const
{
	return DefaultAdjointableOp::Create(std::move(_oph->to_device(stream)), std::move(_op->to_device(stream)));
}


