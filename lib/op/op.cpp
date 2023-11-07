#include "op.hpp"
#include "opalgebra.hpp"

namespace {
	std::mutex _oplock;
}

hasty::op::VectorShape::VectorShape(const std::initializer_list<int64_t>& dims)
	: _self_shape(dims) {}

hasty::op::VectorShape::VectorShape(const c10::IntArrayRef& dims)
	: _self_shape(std::move(dims.vec())) {}

hasty::op::VectorShape::VectorShape(const std::initializer_list<std::vector<int64_t>>& dims)
{
	_children_shape.reserve(dims.size());
	for (auto& d : dims) {
		_children_shape.emplace_back(d);
	}
}

hasty::op::VectorShape::VectorShape(std::vector<int64_t>&& dims)
	: _self_shape(dims) {}

hasty::op::VectorShape::VectorShape(const std::vector<int64_t>& dims)
	: _self_shape(dims)
{}

hasty::op::VectorShape::VectorShape(const std::vector<int64_t>::iterator& begin, const std::vector<int64_t>::iterator& end)
	: _self_shape(begin, end)
{}

hasty::op::VectorShape::VectorShape(const std::initializer_list<VectorShape>& shapes) 
{
	_children_shape.reserve(shapes.size());
	for (auto& s : shapes) {
		_children_shape.emplace_back(s);
	}
}

hasty::op::VectorShape::VectorShape(const std::vector<VectorShape>& shapes) 
{
	_children_shape.reserve(shapes.size());
	for (auto& s : shapes) {
		_children_shape.emplace_back(s);
	}
}

hasty::op::VectorShape::VectorShape(const std::vector<VectorShape>::iterator& begin, const std::vector<VectorShape>::iterator& end)
	: _children_shape(begin, end)
{}

hasty::op::VectorShape::VectorShape(const std::initializer_list<std::vector<VectorShape>>& shapes)
{
	_children_shape.reserve(shapes.size());
	for (auto& ishapes : shapes) {
		_children_shape.emplace_back(ishapes);
	}
}


hasty::op::Vector::Vector(float scalar)
	: _tensor(at::tensor(scalar)) {}

hasty::op::Vector::Vector(double scalar)
	: _tensor(at::tensor(scalar)) {}

hasty::op::Vector::Vector(std::complex<float> scalar)
	: _tensor(at::tensor(c10::complex<float>(scalar.real(), scalar.imag()))) {}

hasty::op::Vector::Vector(std::complex<double> scalar)
	: _tensor(at::tensor(c10::complex<double>(scalar.real(), scalar.imag()))) {}

hasty::op::Vector::Vector(c10::complex<float> scalar)
	: _tensor(at::tensor(scalar)) {}

hasty::op::Vector::Vector(c10::complex<double> scalar)
	: _tensor(at::tensor(scalar)) {}

hasty::op::Vector::Vector(const at::Tensor& tensor) 
	: _tensor(tensor) {}

hasty::op::Vector::Vector(const std::vector<Vector>& children)
	: _children(children) {}



at::ScalarType hasty::op::Vector::dtype() const
{
	if (_children.empty()) {
		return _tensor.dtype().toScalarType();
	}
	return _children[0].dtype();
}

bool hasty::op::Vector::has_children() const
{
	return !_children.empty();
}

const std::vector<hasty::op::Vector>& hasty::op::Vector::children() const
{
	return _children;
}

const at::Tensor& hasty::op::Vector::tensor() const
{
	return _tensor;
}

hasty::op::Vector::Vector(const Vector& vec)
	: _tensor(vec._tensor), _children(vec._children)
{}

hasty::op::Vector& hasty::op::Vector::operator=(const Vector& vec)
{
	_tensor = vec._tensor;
	_children = vec._children;
	return *this;
}

hasty::op::Vector::Vector(Vector&& vec)
{
	_tensor = std::move(vec._tensor);
	_children = std::move(vec._children);
}

hasty::op::Vector& hasty::op::Vector::operator=(Vector&& vec)
{
	_tensor = std::move(vec._tensor);
	_children = std::move(vec._children);
	return *this;
}

hasty::op::VectorShape hasty::op::Vector::get_shape() const {
	if (_children.empty()) {
		return VectorShape(_tensor.sizes());
	}
	std::vector<VectorShape> shapes;
	shapes.reserve(_children.size());
	for (auto& child : _children) {
		shapes.emplace_back(child.get_shape());
	}
	return shapes;
}

hasty::op::Vector hasty::op::Vector::clone() const
{
	if (_children.empty()) {
		return _tensor;
	}
	std::vector<Vector> newchildren;
	newchildren.reserve(_children.size());
	for (auto& child : _children) {
		newchildren.emplace_back(child.clone());
	}
	return Vector(std::move(newchildren));
}


hasty::op::Vector& hasty::op::Vector::operator+=(const Vector& rhs)
{
#define OPEQ_CHARACTER +=
#define OP_CHARACTER +
	if (_children.empty() && rhs._children.empty()) {
		_tensor OPEQ_CHARACTER rhs._tensor;
		return *this;
	}
	else if (_children.empty()) {
		_children.reserve(rhs._children.size());
		for (auto& rchild : rhs._children) {
			_children.push_back(std::move(*this OP_CHARACTER rchild));
		}
		return *this;
	}
	else if (rhs._children.empty()) {
		for (auto& lchild : _children) {
			lchild OPEQ_CHARACTER rhs;
		}
		return *this;
	}
	else {
		if (_children.size() != rhs._children.size())
			throw std::runtime_error("Uncompatible sizes");
		for (int i = 0; i < _children.size(); ++i) {
			_children[i] OPEQ_CHARACTER rhs._children[i];
		}
		return *this;
	}
#undef OP_CHARACTER
#undef OPEQ_CHARACTER
}

hasty::op::Vector hasty::op::operator+(const Vector& lhs, const Vector& rhs)
{
#define OP_CHARACTER +
	if (lhs._children.empty() && rhs._children.empty()) {
		return lhs._tensor OP_CHARACTER rhs._tensor;
	}
	else if (lhs._children.empty()) {
		std::vector<Vector> new_children;
		new_children.reserve(rhs._children.size());
		for (int i = 0; i < rhs._children.size(); ++i) {
			new_children.emplace_back(lhs._tensor OP_CHARACTER rhs._children[i]);
		}
		return Vector(std::move(new_children));
	}
	else if (rhs._children.empty()) {
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._tensor);
		}
		return Vector(std::move(new_children));
	}
	else {
		if (lhs._children.size() != rhs._children.size())
			throw std::runtime_error("Uncompatible sizes");
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._children[i]);
		}
		return Vector(std::move(new_children));
	}
#undef OP_CHARACTER
}

hasty::op::Vector& hasty::op::Vector::operator-=(const Vector& rhs)
{
#define OPEQ_CHARACTER -=
#define OP_CHARACTER -
	if (_children.empty() && rhs._children.empty()) {
		_tensor OPEQ_CHARACTER rhs._tensor;
		return *this;
	}
	else if (_children.empty()) {
		_children.reserve(rhs._children.size());
		for (auto& rchild : rhs._children) {
			_children.push_back(std::move(*this OP_CHARACTER rchild));
		}
		return *this;
	}
	else if (rhs._children.empty()) {
		for (auto& lchild : _children) {
			lchild OPEQ_CHARACTER rhs;
		}
		return *this;
	}
	else {
		if (_children.size() != rhs._children.size())
			throw std::runtime_error("Uncompatible sizes");
		for (int i = 0; i < _children.size(); ++i) {
			_children[i] OPEQ_CHARACTER rhs._children[i];
		}
		return *this;
	}
#undef OP_CHARACTER
#undef OPEQ_CHARACTER
}

hasty::op::Vector hasty::op::operator-(const Vector& lhs, const Vector& rhs)
{
#define OP_CHARACTER -
	if (lhs._children.empty() && rhs._children.empty()) {
		return lhs._tensor OP_CHARACTER rhs._tensor;
	}
	else if (lhs._children.empty()) {
		std::vector<Vector> new_children;
		new_children.reserve(rhs._children.size());
		for (int i = 0; i < rhs._children.size(); ++i) {
			new_children.emplace_back(lhs._tensor OP_CHARACTER rhs._children[i]);
		}
		return Vector(std::move(new_children));
	}
	else if (rhs._children.empty()) {
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._tensor);
		}
		return Vector(std::move(new_children));
	}
	else {
		if (lhs._children.size() != rhs._children.size())
			throw std::runtime_error("Uncompatible sizes");
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._children[i]);
		}
		return Vector(std::move(new_children));
	}

#undef OP_CHARACTER
}


hasty::op::Vector& hasty::op::Vector::operator*=(const Vector& rhs)
{
#define OPEQ_CHARACTER *=
#define OP_CHARACTER *
	if (_children.empty() && rhs._children.empty()) {
		_tensor OPEQ_CHARACTER rhs._tensor;
		return *this;
	}
	else if (_children.empty()) {
		_children.reserve(rhs._children.size());
		for (auto& rchild : rhs._children) {
			_children.push_back(std::move(*this OP_CHARACTER rchild));
		}
		return *this;
	}
	else if (rhs._children.empty()) {
		for (auto& lchild : _children) {
			lchild OPEQ_CHARACTER rhs;
		}
		return *this;
	}
	else {
		if (_children.size() != rhs._children.size())
			throw std::runtime_error("Uncompatible sizes");
		for (int i = 0; i < _children.size(); ++i) {
			_children[i] OPEQ_CHARACTER rhs._children[i];
		}
		return *this;
	}
#undef OP_CHARACTER
#undef OPEQ_CHARACTER
}

hasty::op::Vector hasty::op::operator*(const Vector& lhs, const Vector& rhs)
{
#define OP_CHARACTER *
	if (lhs._children.empty() && rhs._children.empty()) {
		return lhs._tensor OP_CHARACTER rhs._tensor;
	}
	else if (lhs._children.empty()) {
		std::vector<Vector> new_children;
		new_children.reserve(rhs._children.size());
		for (int i = 0; i < rhs._children.size(); ++i) {
			new_children.emplace_back(lhs._tensor OP_CHARACTER rhs._children[i]);
		}
		return Vector(std::move(new_children));
	}
	else if (rhs._children.empty()) {
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._tensor);
		}
		return Vector(std::move(new_children));
	}
	else {
		if (lhs._children.size() != rhs._children.size())
			throw std::runtime_error("Uncompatible sizes");
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._children[i]);
		}
		return Vector(std::move(new_children));
	}

#undef OP_CHARACTER
}


hasty::op::Vector& hasty::op::Vector::operator/=(const Vector& rhs)
{
#define OPEQ_CHARACTER /=
#define OP_CHARACTER /
	if (_children.empty() && rhs._children.empty()) {
		_tensor OPEQ_CHARACTER rhs._tensor;
		return *this;
	}
	else if (_children.empty()) {
		_children.reserve(rhs._children.size());
		for (auto& rchild : rhs._children) {
			_children.push_back(std::move(*this OP_CHARACTER rchild));
		}
		return *this;
	}
	else if (rhs._children.empty()) {
		for (auto& lchild : _children) {
			lchild OPEQ_CHARACTER rhs;
		}
		return *this;
	}
	else {
		if (_children.size() != rhs._children.size())
			throw std::runtime_error("Uncompatible sizes");
		for (int i = 0; i < _children.size(); ++i) {
			_children[i] OPEQ_CHARACTER rhs._children[i];
		}
		return *this;
	}
#undef OP_CHARACTER
#undef OPEQ_CHARACTER
}

hasty::op::Vector hasty::op::operator/(const Vector& lhs, const Vector& rhs)
{
#define OP_CHARACTER /
	if (lhs._children.empty() && rhs._children.empty()) {
		return lhs._tensor OP_CHARACTER rhs._tensor;
	}
	else if (lhs._children.empty()) {
		std::vector<Vector> new_children;
		new_children.reserve(rhs._children.size());
		for (int i = 0; i < rhs._children.size(); ++i) {
			new_children.emplace_back(lhs._tensor OP_CHARACTER rhs._children[i]);
		}
		return Vector(std::move(new_children));
	}
	else if (rhs._children.empty()) {
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._tensor);
		}
		return Vector(std::move(new_children));
	}
	else {
		if (lhs._children.size() != rhs._children.size())
			throw std::runtime_error("Uncompatible sizes");
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._children[i]);
		}
		return Vector(std::move(new_children));
	}

#undef OP_CHARACTER
}



at::Tensor hasty::op::Vector::norm() const
{
	if (_children.empty()) {
		return at::norm(_tensor);
	}
	else {
		at::Tensor normout = _children[0].norm();
		for (int i = 1; i < _children.size(); ++i) {
			normout += _children[i].norm();
		}
		return normout;
	}
}

hasty::op::Vector hasty::op::Vector::abs() const
{
	if (_children.empty()) {
		return at::abs(_tensor);
	}
	else {
		std::vector<Vector> newchildren;
		newchildren.reserve(_children.size());
		for (auto& child : _children) {
			newchildren.push_back(child.abs());
		}
		return newchildren;
	}
}

hasty::op::Vector hasty::op::Vector::real() const
{
	if (_children.empty()) {
		return at::real(_tensor);
	}
	else {
		std::vector<Vector> newchildren;
		newchildren.reserve(_children.size());
		for (auto& child : _children) {
			newchildren.push_back(child.real());
		}
		return newchildren;
	}
}

hasty::op::Vector hasty::op::Vector::imag() const
{
	if (_children.empty()) {
		return at::imag(_tensor);
	}
	else {
		std::vector<Vector> newchildren;
		newchildren.reserve(_children.size());
		for (auto& child : _children) {
			newchildren.push_back(child.imag());
		}
		return newchildren;
	}
}


hasty::op::Vector hasty::op::Vector::zeros(const VectorShape& shape, const at::TensorOptions& opts)
{
	if (shape._children_shape.empty()) {
		return Vector(at::zeros(at::makeArrayRef(shape._self_shape), opts));
	}
	std::vector<Vector> newchildren;
	for (auto& shape : shape._children_shape) {
		newchildren.emplace_back(Vector::zeros(shape, opts));
	}
	return newchildren;
}

hasty::op::Vector hasty::op::Vector::ones(const VectorShape& shape, const at::TensorOptions& opts)
{
	if (shape._children_shape.empty()) {
		return Vector(at::ones(at::makeArrayRef(shape._self_shape), opts));
	}
	std::vector<Vector> newchildren;
	for (auto& shape : shape._children_shape) {
		newchildren.emplace_back(Vector::ones(shape, opts));
	}
	return newchildren;
}

hasty::op::Vector hasty::op::Vector::rand(const VectorShape& shape, const at::TensorOptions& opts)
{
	if (shape._children_shape.empty()) {
		return Vector(at::rand(at::makeArrayRef(shape._self_shape), opts));
	}
	std::vector<Vector> newchildren;
	for (auto& shape : shape._children_shape) {
		newchildren.emplace_back(Vector::rand(shape, opts));
	}
	return newchildren;
}

hasty::op::Vector hasty::op::Vector::empty(const VectorShape& shape, const at::TensorOptions& opts)
{
	if (shape._children_shape.empty()) {
		return Vector(at::empty(at::makeArrayRef(shape._self_shape), opts));
	}
	std::vector<Vector> newchildren;
	for (auto& shape : shape._children_shape) {
		newchildren.emplace_back(Vector::empty(shape, opts));
	}
	return newchildren;
}


at::Tensor hasty::op::vdot(const Vector& a, const Vector& b)
{
	if (a._children.empty()) {
		return at::vdot(a._tensor.flatten(), b._tensor.flatten());
	}
	if (a._children.size() != b._children.size())
		throw std::runtime_error("Vectors was not the same size in vdot");
	
	at::Tensor sum = vdot(a._children[0], b._children[0]);
	for (int i = 1; i < a._children.size(); ++i) {
		sum += vdot(a._children[i], b._children[i]);
	}
	return sum;
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

hasty::op::Vector hasty::op::Operator::apply(const Vector& in) const
{
	throw std::runtime_error("apply() wasn't implemented");
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
	: Operator()
{
}

hasty::op::AdjointableOp::AdjointableOp(std::shared_ptr<Operator> op, std::shared_ptr<Operator> oph)
	: _op(std::move(op)), _oph(std::move(oph))
{
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::op::AdjointableOp::adjoint() const
{
	return std::make_shared<AdjointableOp>(_oph, _op);
}

std::shared_ptr<hasty::op::Operator> hasty::op::AdjointableOp::to_device(at::Stream stream) const
{
	auto ophcaster = std::dynamic_pointer_cast<AdjointableOp>(_oph->to_device(stream));
	auto opcaster = std::dynamic_pointer_cast<AdjointableOp>(_op->to_device(stream));

	if (!ophcaster)
		throw std::runtime_error("failed dynamic cast _oph in AdjointableOp");
	if (!opcaster)
		throw std::runtime_error("failed dynamic cast _op in AdjointableOp");

	return std::make_shared<AdjointableOp>(std::move(ophcaster), std::move(opcaster));
}

// OPERATOR OPERATORS

// OPERATOR OPERATORS

/*
std::shared_ptr<hasty::op::AdjointableAddOp> hasty::op::add(std::shared_ptr<AdjointableOp> lhs, std::shared_ptr<AdjointableOp> rhs)
{
	return std::make_shared<AdjointableAddOp>(std::move(lhs), std::move(rhs));
}

std::shared_ptr<hasty::op::SubOp> hasty::op::sub(std::shared_ptr<Operator> lhs, std::shared_ptr<Operator> rhs)
{
	return std::make_shared<SubOp>(std::move(lhs), std::move(rhs));
}

std::shared_ptr<hasty::op::AdjointableSubOp> hasty::op::sub(std::shared_ptr<AdjointableOp> lhs, std::shared_ptr<AdjointableOp> rhs)
{
	return std::make_shared<AdjointableSubOp>(std::move(lhs), std::move(rhs));
}

std::shared_ptr<hasty::op::MulOp> hasty::op::mul(std::shared_ptr<Operator> lhs, std::shared_ptr<Operator> rhs)
{
	return std::make_shared<MulOp>(std::move(lhs), std::move(rhs));
}

std::shared_ptr<hasty::op::AdjointableMulOp> hasty::op::mul(std::shared_ptr<AdjointableOp> lhs, std::shared_ptr<AdjointableOp> rhs)
{
	return std::make_shared<AdjointableMulOp>(std::move(lhs), std::move(rhs));
}

std::shared_ptr<hasty::op::ScaleOp> hasty::op::mul(const at::Tensor& lhs, std::shared_ptr<Operator> rhs)
{
	return std::make_shared<ScaleOp>(lhs, std::move(rhs));
}

std::shared_ptr<hasty::op::AdjointableScaleOp> hasty::op::mul(const at::Tensor& lhs, std::shared_ptr<AdjointableOp> rhs)
{
	return std::make_shared<AdjointableScaleOp>(lhs, std::move(rhs));
}

std::shared_ptr<hasty::op::Operator> hasty::op::Operator::to_device(at::Stream stream) const
{
	return nullptr;
}

*/