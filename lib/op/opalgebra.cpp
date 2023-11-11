module;

#include "../torch_util.hpp"

module opalgebra;

// ADD OP


hasty::op::AddOp::AddOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{}

hasty::op::Vector hasty::op::AddOp::apply(const Vector& in) const
{
	return _left->apply(in) + _right->apply(in);
}

std::shared_ptr<hasty::op::Operator> hasty::op::AddOp::to_device(at::Stream stream) const
{
	return std::make_shared<AddOp>(std::move(_left->to_device(stream)), std::move(_right->to_device(stream)));
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

std::shared_ptr<hasty::op::Operator> hasty::op::AdjointableAddOp::to_device(at::Stream stream) const
{
	return std::make_shared<AdjointableAddOp>(
		std::move(
			std::dynamic_pointer_cast<AdjointableOp>(_left->to_device(stream))), 
		std::move(
			std::dynamic_pointer_cast<AdjointableOp>(_right->to_device(stream))));
}

// SUB OP

hasty::op::SubOp::SubOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{}

hasty::op::Vector hasty::op::SubOp::apply(const Vector& in) const
{
	return _left->apply(in) - _right->apply(in);
}

std::shared_ptr<hasty::op::Operator> hasty::op::SubOp::to_device(at::Stream stream) const
{
	return std::make_shared<SubOp>(std::move(_left->to_device(stream)), std::move(_right->to_device(stream)));
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

std::shared_ptr<hasty::op::Operator> hasty::op::AdjointableSubOp::to_device(at::Stream stream) const
{
	return std::make_shared<AdjointableSubOp>(
		std::move(
			std::dynamic_pointer_cast<AdjointableOp>(_left->to_device(stream))), 
		std::move(
			std::dynamic_pointer_cast<AdjointableOp>(_right->to_device(stream))));
}

// MUL OP

hasty::op::MulOp::MulOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{}

hasty::op::Vector hasty::op::MulOp::apply(const Vector& in) const
{
	return _left->apply(_right->apply(in));
}

std::shared_ptr<hasty::op::Operator> hasty::op::MulOp::to_device(at::Stream stream) const
{
	return std::make_shared<MulOp>(std::move(_left->to_device(stream)), std::move(_right->to_device(stream)));
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

std::shared_ptr<hasty::op::Operator> hasty::op::AdjointableMulOp::to_device(at::Stream stream) const
{
	return std::make_shared<AdjointableMulOp>(
		std::move(
			std::dynamic_pointer_cast<AdjointableOp>(_left->to_device(stream))), 
		std::move(
			std::dynamic_pointer_cast<AdjointableOp>(_right->to_device(stream))));
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

std::shared_ptr<hasty::op::Operator> hasty::op::ScaleOp::to_device(at::Stream stream) const
{
	return std::make_shared<ScaleOp>(_scalar, std::move(_op->to_device(stream)));
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

std::shared_ptr<hasty::op::Operator> hasty::op::AdjointableScaleOp::to_device(at::Stream stream) const
{
	return std::make_shared<AdjointableScaleOp>(_scalar, 
		std::move(std::dynamic_pointer_cast<AdjointableOp>(_op->to_device(stream))));
}

// VERTICALLY STACKED OP

std::unique_ptr<hasty::op::VStackedOp> hasty::op::VStackedOp::Create(const std::vector<std::shared_ptr<Operator>>& ops)
{
	struct creator : public VStackedOp {
		creator(const std::vector<std::shared_ptr<Operator>>& a)
			: VStackedOp(a) {}
	};
	return std::make_unique<creator>(ops);
}

std::unique_ptr<hasty::op::VStackedOp> hasty::op::VStackedOp::Create(std::vector<std::shared_ptr<Operator>>&& ops)
{
	struct creator : public VStackedOp {
		creator(std::vector<std::shared_ptr<Operator>>&& a)
			: VStackedOp(a) {}
	};
	return std::make_unique<creator>(std::move(ops));
}


hasty::op::VStackedOp::VStackedOp(const std::vector<std::shared_ptr<Operator>>& ops)
{
	_ops.reserve(ops.size());
	for (auto& op : ops) {
		_ops.push_back(std::move(op));
	}
}

hasty::op::VStackedOp::VStackedOp(std::vector<std::shared_ptr<Operator>>&& ops)
	: _ops(std::move(ops))
{
}

hasty::op::Vector hasty::op::VStackedOp::apply(const Vector& in) const
{
	std::vector<op::Vector> newvecs;
	newvecs.reserve(_ops.size());

	for (auto& op : _ops) {
		newvecs.push_back(std::move(op->apply(in)));
	}

	return newvecs;
}

size_t hasty::op::VStackedOp::stack_size() const
{
	return _ops.size();
}

hasty::op::Operator& hasty::op::VStackedOp::get_slice(size_t idx)
{
	return *_ops[idx];
}

const hasty::op::Operator& hasty::op::VStackedOp::get_slice(size_t idx) const
{
	return *_ops[idx];
}

std::shared_ptr<hasty::op::Operator> hasty::op::VStackedOp::get_slice_ptr(size_t idx) const
{
	return _ops[idx];
}

const std::vector<std::shared_ptr<hasty::op::Operator>>& hasty::op::VStackedOp::get_stack() const
{
	return _ops;
}

std::shared_ptr<hasty::op::Operator> hasty::op::VStackedOp::to_device(at::Stream stream) const
{
	std::vector<std::shared_ptr<Operator>> ops;
	ops.reserve(_ops.size());
	for (auto& op : _ops) {
		ops.push_back(std::move(op->to_device(stream)));
	}
	return VStackedOp::Create(std::move(ops));
}

// HORIZONTALLY STACKED OP
std::unique_ptr<hasty::op::HStackedOp> hasty::op::HStackedOp::Create(const std::vector<std::shared_ptr<Operator>>& ops)
{
	struct creator : public HStackedOp {
		creator(const std::vector<std::shared_ptr<Operator>>& a)
			: HStackedOp(a) {}
	};
	return std::make_unique<creator>(ops);
}

std::unique_ptr<hasty::op::HStackedOp> hasty::op::HStackedOp::Create(std::vector<std::shared_ptr<Operator>>&& ops)
{
	struct creator : public HStackedOp {
		creator(std::vector<std::shared_ptr<Operator>>&& a)
			: HStackedOp(std::move(a)) {}
	};
	return std::make_unique<creator>(std::move(ops));
}

hasty::op::HStackedOp::HStackedOp(const std::vector<std::shared_ptr<Operator>>& ops)
{
	_ops.reserve(ops.size());
	for (auto& op : ops) {
		_ops.push_back(std::move(op));
	}
}

hasty::op::HStackedOp::HStackedOp(std::vector<std::shared_ptr<Operator>>&& ops)
	: _ops(std::move(ops))
{
}

hasty::op::Vector hasty::op::HStackedOp::apply(const Vector& in) const
{
	if (!in.has_children()) {
		if (_ops.size() != 1) {
			throw std::runtime_error("HStackedOp stacksize not compatible with number of children in in vector");
		}
		return _ops[0]->apply(in);
	}

	auto& children = access_vecchilds(in);
	if (children.size() != _ops.size()) {
		throw std::runtime_error("HStackedOp stacksize not compatible with number of children in in vector");
	}

	Vector out = _ops[0]->apply(children[0]);
	for (int i = 1; i < _ops.size(); ++i) {
		out += _ops[i]->apply(children[i]);
	}
	return out;
}

size_t hasty::op::HStackedOp::stack_size() const
{
	return _ops.size();
}

hasty::op::Operator& hasty::op::HStackedOp::get_slice(size_t idx)
{
	return *_ops[idx];
}

const hasty::op::Operator& hasty::op::HStackedOp::get_slice(size_t idx) const
{
	return *_ops[idx];
}

std::shared_ptr<hasty::op::Operator> hasty::op::HStackedOp::get_slice_ptr(size_t idx) const
{
	return _ops[idx];
}

const std::vector<std::shared_ptr<hasty::op::Operator>>& hasty::op::HStackedOp::get_stack() const
{
	return _ops;
}

std::shared_ptr<hasty::op::Operator> hasty::op::HStackedOp::to_device(at::Stream stream) const
{
	std::vector<std::shared_ptr<Operator>> ops;
	ops.reserve(_ops.size());
	for (auto& op : _ops) {
		ops.push_back(std::move(op->to_device(stream)));
	}
	return HStackedOp::Create(std::move(ops));
}


std::unique_ptr<hasty::op::AdjointableVStackedOp> hasty::op::AdjointableVStackedOp::Create(const std::vector<std::shared_ptr<AdjointableOp>>& ops)
{
	struct creator : public AdjointableVStackedOp {
		creator(const std::vector<std::shared_ptr<AdjointableOp>>& a)
			: AdjointableVStackedOp(a) {}
	};
	return std::make_unique<creator>(ops);
}

std::unique_ptr<hasty::op::AdjointableVStackedOp> hasty::op::AdjointableVStackedOp::Create(std::vector<std::shared_ptr<AdjointableOp>>&& ops)
{
	struct creator : public AdjointableVStackedOp {
		creator(std::vector<std::shared_ptr<AdjointableOp>>&& a)
			: AdjointableVStackedOp(std::move(a)) {}
	};
	return std::make_unique<creator>(std::move(ops));
}

// ADJOINTABLE VSTACKED OP
hasty::op::AdjointableVStackedOp::AdjointableVStackedOp(const std::vector<std::shared_ptr<AdjointableOp>>& ops)
{
	_ops.reserve(ops.size());
	for (auto& op : ops) {
		_ops.emplace_back(op);
	}
}

hasty::op::AdjointableVStackedOp::AdjointableVStackedOp(std::vector<std::shared_ptr<AdjointableOp>>&& ops)
	: _ops(std::move(ops))
{
	
}

hasty::op::Vector hasty::op::AdjointableVStackedOp::apply(const Vector& in) const
{
	std::vector<op::Vector> newvecs;
	newvecs.reserve(_ops.size());

	for (auto& op : _ops) {
		newvecs.push_back(std::move(op->apply(in)));
	}

	return newvecs;
}

size_t hasty::op::AdjointableVStackedOp::stack_size() const
{
	return _ops.size();
}

hasty::op::AdjointableOp& hasty::op::AdjointableVStackedOp::get_slice(size_t idx)
{
	return *_ops[idx];
}

const hasty::op::AdjointableOp& hasty::op::AdjointableVStackedOp::get_slice(size_t idx) const
{
	return *_ops[idx];
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::op::AdjointableVStackedOp::get_slice_ptr(size_t idx) const
{
	return _ops[idx];
}

const std::vector<std::shared_ptr<hasty::op::AdjointableOp>>& hasty::op::AdjointableVStackedOp::get_stack() const
{
	return _ops;
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::op::AdjointableVStackedOp::adjoint() const
{
	std::vector<std::shared_ptr<AdjointableOp>> newops;
	newops.reserve(_ops.size());
	for (auto& op : _ops) {
		newops.push_back(std::move(op->adjoint()));
	}
	return AdjointableHStackedOp::Create(std::move(newops));
}

std::shared_ptr<hasty::op::Operator> hasty::op::AdjointableVStackedOp::to_device(at::Stream stream) const
{
	std::vector<std::shared_ptr<AdjointableOp>> ops;
	ops.reserve(_ops.size());
	for (auto& op : _ops) {
		ops.push_back(std::move(std::dynamic_pointer_cast<AdjointableOp>(op->to_device(stream))));
	}
	return AdjointableVStackedOp::Create(std::move(ops));
}

// ADJOINTABLE HSTACKED OP

std::unique_ptr<hasty::op::AdjointableHStackedOp> hasty::op::AdjointableHStackedOp::Create(const std::vector<std::shared_ptr<AdjointableOp>>& ops)
{
	struct creator : public AdjointableHStackedOp {
		creator(const std::vector<std::shared_ptr<AdjointableOp>>& a)
			: AdjointableHStackedOp(a) {}
	};
	return std::make_unique<creator>(ops);
}

std::unique_ptr<hasty::op::AdjointableHStackedOp> hasty::op::AdjointableHStackedOp::Create(std::vector<std::shared_ptr<AdjointableOp>>&& ops)
{
	struct creator : public AdjointableHStackedOp {
		creator(std::vector<std::shared_ptr<AdjointableOp>>&& a)
			: AdjointableHStackedOp(std::move(a)) {}
	};
	return std::make_unique<creator>(std::move(ops));
}

hasty::op::AdjointableHStackedOp::AdjointableHStackedOp(const std::vector<std::shared_ptr<AdjointableOp>>& ops)
{
	_ops.reserve(ops.size());
	for (auto& op : ops) {
		_ops.emplace_back(op);
	}
}

hasty::op::AdjointableHStackedOp::AdjointableHStackedOp(std::vector<std::shared_ptr<AdjointableOp>>&& ops)
	: _ops(std::move(ops))
{
}

hasty::op::Vector hasty::op::AdjointableHStackedOp::apply(const Vector& in) const
{
	if (!in.has_children()) {
		if (_ops.size() != 1) {
			throw std::runtime_error("HStackedOp stacksize not compatible with number of children in in vector");
		}
		return _ops[0]->apply(in);
	}

	auto& children = access_vecchilds(in);
	if (children.size() != _ops.size()) {
		throw std::runtime_error("HStackedOp stacksize not compatible with number of children in in vector");
	}

	Vector out = _ops[0]->apply(children[0]);
	for (int i = 1; i < _ops.size(); ++i) {
		out += _ops[i]->apply(children[i]);
	}
	return out;
}

size_t hasty::op::AdjointableHStackedOp::stack_size() const
{
	return _ops.size();
}

hasty::op::AdjointableOp& hasty::op::AdjointableHStackedOp::get_slice(size_t idx)
{
	return *_ops[idx];
}

const hasty::op::AdjointableOp& hasty::op::AdjointableHStackedOp::get_slice(size_t idx) const
{
	return *_ops[idx];
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::op::AdjointableHStackedOp::get_slice_ptr(size_t idx) const
{
	return _ops[idx];
}

const std::vector<std::shared_ptr<hasty::op::AdjointableOp>>& hasty::op::AdjointableHStackedOp::get_stack() const
{
	return _ops;
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::op::AdjointableHStackedOp::adjoint() const
{
	std::vector<std::shared_ptr<AdjointableOp>> newops;
	newops.reserve(_ops.size());
	for (auto& op : _ops) {
		newops.push_back(std::move(op->adjoint()));
	}
	return AdjointableVStackedOp::Create(std::move(newops));
}

std::shared_ptr<hasty::op::Operator> hasty::op::AdjointableHStackedOp::to_device(at::Stream stream) const
{
	std::vector<std::shared_ptr<AdjointableOp>> ops;
	ops.reserve(_ops.size());
	for (auto& op : _ops) {
		ops.push_back(std::move(std::dynamic_pointer_cast<AdjointableOp>(op->to_device(stream))));
	}
	return AdjointableHStackedOp::Create(std::move(ops));
}


