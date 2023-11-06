#include "opalgebra.hpp"


// ADD OP

hasty::op::AddOp::AddOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{}

hasty::op::Vector hasty::op::AddOp::apply(const Vector& in) const
{
	return _left->apply(in) + _right->apply(in);
}

// ADJOINTABLE ADD OP

hasty::op::AdjointableAddOp::AdjointableAddOp(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{
}

hasty::op::Vector hasty::op::AdjointableAddOp::apply(const Vector& in) const
{
	return _left->apply(in) + _right->apply(in);
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::op::AdjointableAddOp::adjoint() const
{
	return std::make_shared<AdjointableAddOp>(_left->adjoint(), _right->adjoint());
}

// SUB OP

hasty::op::SubOp::SubOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{}

hasty::op::Vector hasty::op::SubOp::apply(const Vector& in) const
{
	return _left->apply(in) - _right->apply(in);
}

// ADJOINTABLE SUB OP

hasty::op::AdjointableSubOp::AdjointableSubOp(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{
}

hasty::op::Vector hasty::op::AdjointableSubOp::apply(const Vector& in) const
{
	return _left->apply(in) - _right->apply(in);
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::op::AdjointableSubOp::adjoint() const
{
	return std::make_shared<AdjointableSubOp>(_left->adjoint(), _left->adjoint());
}

// MUL OP

hasty::op::MulOp::MulOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{}

hasty::op::Vector hasty::op::MulOp::apply(const Vector& in) const
{
	return _left->apply(_right->apply(in));
}

// ADJOINTABLE MUL OP

hasty::op::AdjointableMulOp::AdjointableMulOp(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{
}

hasty::op::Vector hasty::op::AdjointableMulOp::apply(const Vector& in) const
{
	return _left->apply(_right->apply(in));
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::op::AdjointableMulOp::adjoint() const
{
	return std::make_shared<AdjointableMulOp>(_right->adjoint(), _left->adjoint());
}

// SCALE OP

hasty::op::ScaleOp::ScaleOp(const at::Tensor& scalar, std::shared_ptr<Operator> op)
	: _scalar(scalar), _op(std::move(op))
{}

hasty::op::Vector hasty::op::ScaleOp::apply(const Vector& in) const
{
	Vector tmp = _scalar * in;
	if (_op) {
		return _op->apply(tmp);
	}
	return tmp;
}

// ADJOINTABLE MUL OP

hasty::op::AdjointableScaleOp::AdjointableScaleOp(const at::Tensor& scalar, std::shared_ptr<AdjointableOp> op)
	: _scalar(scalar), _op(std::move(op))
{}

hasty::op::Vector hasty::op::AdjointableScaleOp::apply(const Vector& in) const
{
	Vector tmp = _scalar * in;
	if (_op) {
		return _op->apply(tmp);
	}
	return tmp;
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::op::AdjointableScaleOp::adjoint() const
{
	return std::make_shared<AdjointableScaleOp>(_scalar, _op ? _op->adjoint() : nullptr);
}

// STACKED OP

hasty::op::StackedOp::StackedOp(std::vector<std::shared_ptr<Operator>>& ops)
{
	_ops.reserve(ops.size());
	for (auto& op : ops) {
		_ops.push_back(std::move(op));
	}
}

hasty::op::StackedOp::StackedOp(std::vector<std::shared_ptr<Operator>>&& ops)
	: _ops(ops)
{
}


hasty::op::Vector hasty::op::StackedOp::apply(const Vector& in) const
{
	if (!in.has_children()) {
		if (_ops.size() != 1) {
			throw std::runtime_error("StackedOperator stacksize not compatible with number of children in in vector");
		}
		return _ops[0]->apply(in);
	}

	auto& children = access_vecchilds(in);
	if (children.size() != _ops.size()) {
		throw std::runtime_error("StackedOperator stacksize not compatible with number of children in in vector");
	}

	std::vector<op::Vector> newvecs;
	newvecs.reserve(children.size());
	for (int i = 0; i < _ops.size(); ++i) {
		newvecs.push_back(std::move(_ops[i]->apply(children[i])));
	}

	return newvecs;
}

size_t hasty::op::StackedOp::stack_size() const
{
	return _ops.size();
}

hasty::op::Operator& hasty::op::StackedOp::get_slice(size_t idx)
{
	return *_ops[idx];
}

const hasty::op::Operator& hasty::op::StackedOp::get_slice(size_t idx) const
{
	return *_ops[idx];
}

// ADJOINTABLE STACKED OP

hasty::op::AdjointableStackedOp::AdjointableStackedOp(std::vector<std::shared_ptr<AdjointableOp>>& ops)
{
	_ops.reserve(ops.size());
	for (auto& op : ops) {
		_ops.push_back(std::move(op));
	}
}

hasty::op::AdjointableStackedOp::AdjointableStackedOp(std::vector<std::shared_ptr<AdjointableOp>>&& ops)
	: _ops(ops)
{
}

hasty::op::Vector hasty::op::AdjointableStackedOp::apply(const Vector& in) const
{
	if (!in.has_children()) {
		if (_ops.size() != 1) {
			throw std::runtime_error("StackedOperator stacksize not compatible with number of children in in vector");
		}
		return _ops[0]->apply(in);
	}

	auto& children = access_vecchilds(in);
	if (children.size() != _ops.size()) {
		throw std::runtime_error("StackedOperator stacksize not compatible with number of children in in vector");
	}

	std::vector<op::Vector> newvecs;
	newvecs.reserve(children.size());
	for (int i = 0; i < _ops.size(); ++i) {
		newvecs.push_back(std::move(_ops[i]->apply(children[i])));
	}

	return newvecs;
}

size_t hasty::op::AdjointableStackedOp::stack_size() const
{
	return _ops.size();
}

hasty::op::AdjointableOp& hasty::op::AdjointableStackedOp::get_slice(size_t idx)
{
	return *_ops[idx];
}

const hasty::op::AdjointableOp& hasty::op::AdjointableStackedOp::get_slice(size_t idx) const
{
	return *_ops[idx];
}
