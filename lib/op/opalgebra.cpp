module;

#include <torch/torch.h>

module opalgebra;

import op;

// ID OP

std::unique_ptr<hasty::op::IdOp> hasty::op::IdOp::Create()
{
	struct creator : public IdOp { creator() : IdOp() {} };
	return std::make_unique<creator>();
}

hasty::op::IdOp::IdOp() {}

hasty::op::Vector hasty::op::IdOp::apply(const Vector& in) const
{
	return in;
}

hasty::sptr<hasty::op::AdjointableOp> hasty::op::IdOp::adjoint() const
{
	return Create();
}

hasty::sptr<hasty::op::Operator> hasty::op::IdOp::to_device(at::Stream stream) const
{
	return Create();
}

// NEG ID OP
std::unique_ptr<hasty::op::NegIdOp> hasty::op::NegIdOp::Create()
{
	struct creator : public NegIdOp { creator() : NegIdOp() {} };
	return std::make_unique<creator>();
}

hasty::op::NegIdOp::NegIdOp() {}

hasty::op::Vector hasty::op::NegIdOp::apply(const hasty::op::Vector& in) const
{
	return in.neg();
}

hasty::sptr<hasty::op::AdjointableOp> hasty::op::NegIdOp::adjoint() const
{
	return Create();
}

hasty::sptr<hasty::op::Operator> hasty::op::NegIdOp::to_device(at::Stream stream) const
{
	return Create();
}

// ADD OP

std::unique_ptr<hasty::op::AddOp> hasty::op::AddOp::Create(sptr<Operator> lop, sptr<Operator> rop)
{
	struct creator : public AddOp {
		creator(sptr<Operator> a, sptr<Operator> b)
			: AddOp(std::move(a), std::move(b)) {}
	};
	return std::make_unique<creator>(std::move(lop), std::move(rop));
}

hasty::op::AddOp::AddOp(sptr<Operator> lop, sptr<Operator> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{}

hasty::op::Vector hasty::op::AddOp::apply(const Vector& in) const
{
	return _left->apply(in) + _right->apply(in);
}

hasty::sptr<hasty::op::Operator> hasty::op::AddOp::to_device(at::Stream stream) const
{
	return AddOp::Create(std::move(_left->to_device(stream)), std::move(_right->to_device(stream)));
}

// ADJOINTABLE ADD OP

std::unique_ptr<hasty::op::AdjointableAddOp> hasty::op::AdjointableAddOp::Create(sptr<AdjointableOp> lop, sptr<AdjointableOp> rop)
{
	struct creator : public AdjointableAddOp {
		creator(sptr<AdjointableOp> a, sptr<AdjointableOp> b)
			: AdjointableAddOp(std::move(a), std::move(b)) {}
	};
	return std::make_unique<creator>(std::move(lop), std::move(rop));
}

hasty::op::AdjointableAddOp::AdjointableAddOp(sptr<AdjointableOp> lop, sptr<AdjointableOp> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{
}

hasty::op::Vector hasty::op::AdjointableAddOp::apply(const Vector& in) const
{
	return _left->apply(in) + _right->apply(in);
}

hasty::sptr<hasty::op::AdjointableOp> hasty::op::AdjointableAddOp::adjoint() const
{
	return AdjointableAddOp::Create(std::move(_left->adjoint()), std::move(_right->adjoint()));
}

hasty::sptr<hasty::op::Operator> hasty::op::AdjointableAddOp::to_device(at::Stream stream) const
{
	return AdjointableAddOp::Create(
		std::move(
			downcast<AdjointableOp>(std::move(_left->to_device(stream)))),
		std::move(
			downcast<AdjointableOp>(std::move(_right->to_device(stream)))));
}

// SUB OP

std::unique_ptr<hasty::op::SubOp> hasty::op::SubOp::Create(sptr<Operator> lop, sptr<Operator> rop)
{
	struct creator : public SubOp {
		creator(sptr<Operator> a, sptr<Operator> b)
			: SubOp(std::move(a), std::move(b)) {}
	};
	return std::make_unique<creator>(std::move(lop), std::move(rop));
}

hasty::op::SubOp::SubOp(sptr<Operator> lop, sptr<Operator> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{}

hasty::op::Vector hasty::op::SubOp::apply(const Vector& in) const
{
	return _left->apply(in) - _right->apply(in);
}

hasty::sptr<hasty::op::Operator> hasty::op::SubOp::to_device(at::Stream stream) const
{
	return SubOp::Create(std::move(_left->to_device(stream)), std::move(_right->to_device(stream)));
}

// ADJOINTABLE SUB OP

std::unique_ptr<hasty::op::AdjointableSubOp> hasty::op::AdjointableSubOp::Create(sptr<AdjointableOp> lop, sptr<AdjointableOp> rop)
{
	struct creator : public AdjointableSubOp {
		creator(sptr<AdjointableOp> a, sptr<AdjointableOp> b)
			: AdjointableSubOp(std::move(a), std::move(b)) {}
	};
	return std::make_unique<creator>(std::move(lop), std::move(rop));
}

hasty::op::AdjointableSubOp::AdjointableSubOp(sptr<AdjointableOp> lop, sptr<AdjointableOp> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{
}

hasty::op::Vector hasty::op::AdjointableSubOp::apply(const Vector& in) const
{
	return _left->apply(in) - _right->apply(in);
}

hasty::sptr<hasty::op::AdjointableOp> hasty::op::AdjointableSubOp::adjoint() const
{
	return AdjointableSubOp::Create(std::move(_left->adjoint()), std::move(_right->adjoint()));
}

hasty::sptr<hasty::op::Operator> hasty::op::AdjointableSubOp::to_device(at::Stream stream) const
{
	return AdjointableSubOp::Create(
		std::move(
			downcast<AdjointableOp>(std::move(_left->to_device(stream)))),
		std::move(
			downcast<AdjointableOp>(std::move(_right->to_device(stream)))));
}

// MUL OP

std::unique_ptr<hasty::op::MulOp> hasty::op::MulOp::Create(sptr<Operator> lop, sptr<Operator> rop)
{
	struct creator : public MulOp {
		creator(sptr<Operator> a, sptr<Operator> b)
			: MulOp(std::move(a), std::move(b)) {}
	};
	return std::make_unique<creator>(std::move(lop), std::move(rop));
}

hasty::op::MulOp::MulOp(sptr<Operator> lop, sptr<Operator> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{}

hasty::op::Vector hasty::op::MulOp::apply(const Vector& in) const
{
	return _left->apply(_right->apply(in));
}

hasty::sptr<hasty::op::Operator> hasty::op::MulOp::to_device(at::Stream stream) const
{
	return MulOp::Create(std::move(_left->to_device(stream)), std::move(_right->to_device(stream)));
}

// ADJOINTABLE MUL OP

std::unique_ptr<hasty::op::AdjointableMulOp> hasty::op::AdjointableMulOp::Create(sptr<AdjointableOp> lop, sptr<AdjointableOp> rop)
{
	struct creator : public AdjointableMulOp {
		creator(sptr<AdjointableOp> a, sptr<AdjointableOp> b)
			: AdjointableMulOp(std::move(a), std::move(b)) {}
	};
	return std::make_unique<creator>(std::move(lop), std::move(rop));
}

hasty::op::AdjointableMulOp::AdjointableMulOp(sptr<AdjointableOp> lop, sptr<AdjointableOp> rop)
	: _left(std::move(lop)), _right(std::move(rop))
{
}

hasty::op::Vector hasty::op::AdjointableMulOp::apply(const Vector& in) const
{
	return _left->apply(_right->apply(in));
}

hasty::sptr<hasty::op::AdjointableOp> hasty::op::AdjointableMulOp::adjoint() const
{
	return AdjointableMulOp::Create(std::move(_right->adjoint()), std::move(_left->adjoint()));
}

hasty::sptr<hasty::op::Operator> hasty::op::AdjointableMulOp::to_device(at::Stream stream) const
{
	return AdjointableMulOp::Create(
		std::move(
			downcast<AdjointableOp>(std::move(_left->to_device(stream)))),
		std::move(
			downcast<AdjointableOp>(std::move(_right->to_device(stream)))));
}

// SCALE OP

std::unique_ptr<hasty::op::ScaleOp> hasty::op::ScaleOp::Create(const at::Tensor& scalar, sptr<Operator> rop)
{
	struct creator : public ScaleOp {
		creator(const at::Tensor& a, sptr<Operator> b)
			: ScaleOp(a, std::move(b)) {}
	};
	return std::make_unique<creator>(scalar, std::move(rop));
}

hasty::op::ScaleOp::ScaleOp(const at::Tensor& scalar, sptr<Operator> op)
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

hasty::sptr<hasty::op::Operator> hasty::op::ScaleOp::to_device(at::Stream stream) const
{
	return ScaleOp::Create(_scalar, std::move(_op->to_device(stream)));
}

// ADJOINTABLE SCALE OP

std::unique_ptr<hasty::op::AdjointableScaleOp> hasty::op::AdjointableScaleOp::Create(const at::Tensor& scalar, sptr<AdjointableOp> op)
{
	struct creator : public AdjointableScaleOp {
		creator(const at::Tensor& a, sptr<AdjointableOp> b)
			: AdjointableScaleOp(a, std::move(b)) {}
	};
	return std::make_unique<creator>(scalar, std::move(op));
}

hasty::op::AdjointableScaleOp::AdjointableScaleOp(const at::Tensor& scalar, sptr<AdjointableOp> op)
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

hasty::sptr<hasty::op::AdjointableOp> hasty::op::AdjointableScaleOp::adjoint() const
{
	return AdjointableScaleOp::Create(_scalar.conj(), _op ? _op->adjoint() : nullptr);
}

hasty::sptr<hasty::op::Operator> hasty::op::AdjointableScaleOp::to_device(at::Stream stream) const
{
	return AdjointableScaleOp::Create(_scalar, std::move(downcast<AdjointableOp>(std::move(_op->to_device(stream)))));
}

// VERTICALLY STACKED OP

std::unique_ptr<hasty::op::VStackedOp> hasty::op::VStackedOp::Create(const std::vector<sptr<Operator>>& ops)
{
	struct creator : public VStackedOp {
		creator(const std::vector<sptr<Operator>>& a)
			: VStackedOp(a) {}
	};
	return std::make_unique<creator>(ops);
}

std::unique_ptr<hasty::op::VStackedOp> hasty::op::VStackedOp::Create(std::vector<sptr<Operator>>&& ops)
{
	struct creator : public VStackedOp {
		creator(std::vector<sptr<Operator>>&& a)
			: VStackedOp(a) {}
	};
	return std::make_unique<creator>(std::move(ops));
}


hasty::op::VStackedOp::VStackedOp(const std::vector<sptr<Operator>>& ops)
{
	_ops.reserve(ops.size());
	for (auto& op : ops) {
		_ops.push_back(std::move(op));
	}
}

hasty::op::VStackedOp::VStackedOp(std::vector<sptr<Operator>>&& ops)
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

hasty::sptr<hasty::op::Operator> hasty::op::VStackedOp::get_slice_ptr(size_t idx) const
{
	return _ops[idx];
}

const std::vector<hasty::sptr<hasty::op::Operator>>& hasty::op::VStackedOp::get_stack() const
{
	return _ops;
}

hasty::sptr<hasty::op::Operator> hasty::op::VStackedOp::to_device(at::Stream stream) const
{
	std::vector<sptr<Operator>> ops;
	ops.reserve(_ops.size());
	for (auto& op : _ops) {
		ops.push_back(std::move(op->to_device(stream)));
	}
	return VStackedOp::Create(std::move(ops));
}

// HORIZONTALLY STACKED OP
std::unique_ptr<hasty::op::HStackedOp> hasty::op::HStackedOp::Create(const std::vector<sptr<Operator>>& ops)
{
	struct creator : public HStackedOp {
		creator(const std::vector<sptr<Operator>>& a)
			: HStackedOp(a) {}
	};
	return std::make_unique<creator>(ops);
}

std::unique_ptr<hasty::op::HStackedOp> hasty::op::HStackedOp::Create(std::vector<sptr<Operator>>&& ops)
{
	struct creator : public HStackedOp {
		creator(std::vector<sptr<Operator>>&& a)
			: HStackedOp(std::move(a)) {}
	};
	return std::make_unique<creator>(std::move(ops));
}

hasty::op::HStackedOp::HStackedOp(const std::vector<sptr<Operator>>& ops)
{
	_ops.reserve(ops.size());
	for (auto& op : ops) {
		_ops.push_back(std::move(op));
	}
}

hasty::op::HStackedOp::HStackedOp(std::vector<sptr<Operator>>&& ops)
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

hasty::sptr<hasty::op::Operator> hasty::op::HStackedOp::get_slice_ptr(size_t idx) const
{
	return _ops[idx];
}

const std::vector<hasty::sptr<hasty::op::Operator>>& hasty::op::HStackedOp::get_stack() const
{
	return _ops;
}

hasty::sptr<hasty::op::Operator> hasty::op::HStackedOp::to_device(at::Stream stream) const
{
	std::vector<sptr<Operator>> ops;
	ops.reserve(_ops.size());
	for (auto& op : _ops) {
		ops.push_back(std::move(op->to_device(stream)));
	}
	return HStackedOp::Create(std::move(ops));
}


std::unique_ptr<hasty::op::AdjointableVStackedOp> hasty::op::AdjointableVStackedOp::Create(const std::vector<sptr<AdjointableOp>>& ops)
{
	struct creator : public AdjointableVStackedOp {
		creator(const std::vector<sptr<AdjointableOp>>& a)
			: AdjointableVStackedOp(a) {}
	};
	return std::make_unique<creator>(ops);
}

std::unique_ptr<hasty::op::AdjointableVStackedOp> hasty::op::AdjointableVStackedOp::Create(std::vector<sptr<AdjointableOp>>&& ops)
{
	struct creator : public AdjointableVStackedOp {
		creator(std::vector<sptr<AdjointableOp>>&& a)
			: AdjointableVStackedOp(std::move(a)) {}
	};
	return std::make_unique<creator>(std::move(ops));
}

// ADJOINTABLE VSTACKED OP
hasty::op::AdjointableVStackedOp::AdjointableVStackedOp(const std::vector<sptr<AdjointableOp>>& ops)
{
	_ops.reserve(ops.size());
	for (auto& op : ops) {
		_ops.emplace_back(op);
	}
}

hasty::op::AdjointableVStackedOp::AdjointableVStackedOp(std::vector<sptr<AdjointableOp>>&& ops)
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

hasty::sptr<hasty::op::AdjointableOp> hasty::op::AdjointableVStackedOp::get_slice_ptr(size_t idx) const
{
	return _ops[idx];
}

const std::vector<hasty::sptr<hasty::op::AdjointableOp>>& hasty::op::AdjointableVStackedOp::get_stack() const
{
	return _ops;
}

hasty::sptr<hasty::op::AdjointableOp> hasty::op::AdjointableVStackedOp::adjoint() const
{
	std::vector<sptr<AdjointableOp>> newops;
	newops.reserve(_ops.size());
	for (auto& op : _ops) {
		newops.push_back(std::move(op->adjoint()));
	}
	return AdjointableHStackedOp::Create(std::move(newops));
}

hasty::sptr<hasty::op::Operator> hasty::op::AdjointableVStackedOp::to_device(at::Stream stream) const
{
	std::vector<sptr<AdjointableOp>> ops;
	ops.reserve(_ops.size());
	for (auto& op : _ops) {
		ops.push_back(std::move(std::dynamic_pointer_cast<AdjointableOp>(op->to_device(stream))));
	}
	return AdjointableVStackedOp::Create(std::move(ops));
}

// ADJOINTABLE HSTACKED OP

std::unique_ptr<hasty::op::AdjointableHStackedOp> hasty::op::AdjointableHStackedOp::Create(const std::vector<sptr<AdjointableOp>>& ops)
{
	struct creator : public AdjointableHStackedOp {
		creator(const std::vector<sptr<AdjointableOp>>& a)
			: AdjointableHStackedOp(a) {}
	};
	return std::make_unique<creator>(ops);
}

std::unique_ptr<hasty::op::AdjointableHStackedOp> hasty::op::AdjointableHStackedOp::Create(std::vector<sptr<AdjointableOp>>&& ops)
{
	struct creator : public AdjointableHStackedOp {
		creator(std::vector<sptr<AdjointableOp>>&& a)
			: AdjointableHStackedOp(std::move(a)) {}
	};
	return std::make_unique<creator>(std::move(ops));
}

hasty::op::AdjointableHStackedOp::AdjointableHStackedOp(const std::vector<sptr<AdjointableOp>>& ops)
{
	_ops.reserve(ops.size());
	for (auto& op : ops) {
		_ops.emplace_back(op);
	}
}

hasty::op::AdjointableHStackedOp::AdjointableHStackedOp(std::vector<sptr<AdjointableOp>>&& ops)
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

hasty::sptr<hasty::op::AdjointableOp> hasty::op::AdjointableHStackedOp::get_slice_ptr(size_t idx) const
{
	return _ops[idx];
}

const std::vector<hasty::sptr<hasty::op::AdjointableOp>>& hasty::op::AdjointableHStackedOp::get_stack() const
{
	return _ops;
}

hasty::sptr<hasty::op::AdjointableOp> hasty::op::AdjointableHStackedOp::adjoint() const
{
	std::vector<sptr<AdjointableOp>> newops;
	newops.reserve(_ops.size());
	for (auto& op : _ops) {
		newops.push_back(std::move(op->adjoint()));
	}
	return AdjointableVStackedOp::Create(std::move(newops));
}

hasty::sptr<hasty::op::Operator> hasty::op::AdjointableHStackedOp::to_device(at::Stream stream) const
{
	std::vector<sptr<AdjointableOp>> ops;
	ops.reserve(_ops.size());
	for (auto& op : _ops) {
		ops.push_back(std::move(std::dynamic_pointer_cast<AdjointableOp>(op->to_device(stream))));
	}
	return AdjointableHStackedOp::Create(std::move(ops));
}


