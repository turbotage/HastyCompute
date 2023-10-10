#include "op.hpp"


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

hasty::op::Vector::Vector(const at::Tensor& tensor) 
	: _tensor(tensor)
{}

hasty::op::Vector::Vector(const std::vector<Vector>& children)
	: _children(children)
{
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


hasty::op::Vector& hasty::op::Vector::operator+=(const Vector& rhs)
{
#define OP_CHARACTER +=
	if (_children.empty()) {
		if (rhs._children.empty()) {
			_tensor OP_CHARACTER rhs._tensor;
			return *this;
		}
		throw std::runtime_error("Uncompatible sizes");
	}
	else {
		if (rhs._children.empty()) {
			for (auto& child : _children) {
				child OP_CHARACTER rhs;
			}
			return *this;
		}
		if (_children.size() != rhs._children.size()) {
			throw std::runtime_error("Uncompatible sizes");
		}
		for (int i = 0; i < _children.size(); ++i) {
			_children[i] OP_CHARACTER rhs._children[i];
		}
		return *this;
	}
#undef OP_CHARACTER
}

hasty::op::Vector hasty::op::operator+(Vector lhs, const Vector& rhs)
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
		return new_children;
	}
	else if (rhs._children.empty()) {
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._tensor);
		}
		return new_children;
	}
	else {
		if (lhs._children.size() != rhs._children.size())
			throw std::runtime_error("Uncompatible sizes");
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._children[i]);
		}
		return new_children;
	}
#undef OP_CHARACTER
}


hasty::op::Vector& hasty::op::Vector::operator-=(const Vector& rhs)
{
#define OP_CHARACTER -=
	if (_children.empty()) {
		if (rhs._children.empty()) {
			_tensor OP_CHARACTER rhs._tensor;
			return *this;
		}
		throw std::runtime_error("Uncompatible sizes");
	}
	else {
		if (rhs._children.empty()) {
			for (auto& child : _children) {
				child OP_CHARACTER rhs;
			}
			return *this;
		}
		if (_children.size() != rhs._children.size()) {
			throw std::runtime_error("Uncompatible sizes");
		}
		for (int i = 0; i < _children.size(); ++i) {
			_children[i] OP_CHARACTER rhs._children[i];
		}
		return *this;
	}
#undef OP_CHARACTER
}

hasty::op::Vector hasty::op::operator-(Vector lhs, const Vector& rhs)
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
		return new_children;
	}
	else if (rhs._children.empty()) {
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._tensor);
		}
		return new_children;
	}
	else {
		if (lhs._children.size() != rhs._children.size())
			throw std::runtime_error("Uncompatible sizes");
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._children[i]);
		}
		return new_children;
	}

#undef OP_CHARACTER
}


hasty::op::Vector& hasty::op::Vector::operator*=(const Vector& rhs)
{
#define OP_CHARACTER *=
	if (_children.empty()) {
		if (rhs._children.empty()) {
			_tensor OP_CHARACTER rhs._tensor;
			return *this;
		}
		throw std::runtime_error("Uncompatible sizes");
	}
	else {
		if (rhs._children.empty()) {
			for (auto& child : _children) {
				child OP_CHARACTER rhs;
			}
			return *this;
		}
		if (_children.size() != rhs._children.size()) {
			throw std::runtime_error("Uncompatible sizes");
		}
		for (int i = 0; i < _children.size(); ++i) {
			_children[i] OP_CHARACTER rhs._children[i];
		}
		return *this;
	}
#undef OP_CHARACTER
}

hasty::op::Vector hasty::op::operator*(Vector lhs, const Vector& rhs)
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
		return new_children;
	}
	else if (rhs._children.empty()) {
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._tensor);
		}
		return new_children;
	}
	else {
		if (lhs._children.size() != rhs._children.size())
			throw std::runtime_error("Uncompatible sizes");
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._children[i]);
		}
		return new_children;
	}

#undef OP_CHARACTER
}


hasty::op::Vector& hasty::op::Vector::operator/=(const Vector& rhs)
{
#define OP_CHARACTER /=
	if (_children.empty()) {
		if (rhs._children.empty()) {
			_tensor OP_CHARACTER rhs._tensor;
			return *this;
		}
		throw std::runtime_error("Uncompatible sizes");
	}
	else {
		if (rhs._children.empty()) {
			for (auto& child : _children) {
				child OP_CHARACTER rhs;
			}
			return *this;
		}
		if (_children.size() != rhs._children.size()) {
			throw std::runtime_error("Uncompatible sizes");
		}
		for (int i = 0; i < _children.size(); ++i) {
			_children[i] OP_CHARACTER rhs._children[i];
		}
		return *this;
	}
#undef OP_CHARACTER
}

hasty::op::Vector hasty::op::operator/(Vector lhs, const Vector& rhs)
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
		return new_children;
	}
	else if (rhs._children.empty()) {
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._tensor);
		}
		return new_children;
	}
	else {
		if (lhs._children.size() != rhs._children.size())
			throw std::runtime_error("Uncompatible sizes");
		std::vector<Vector> new_children;
		new_children.reserve(lhs._children.size());
		for (int i = 0; i < lhs._children.size(); ++i) {
			new_children.emplace_back(lhs._children[i] OP_CHARACTER rhs._children[i]);
		}
		return new_children;
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
		
	}
}





hasty::op::Vector hasty::op::Operator::operator()(const Vector& in) const
{
	return apply(in);
}

hasty::op::Vector hasty::op::Operator::apply(const Vector& in) const
{
	return _apply(in);
}

hasty::op::Operator hasty::op::operator+(Operator lhs, const Operator& rhs)
{
	return AddOp()
}





hasty::op::Vector hasty::op::AddOp::_apply(const Vector& in) const
{
	return left(in) + right(in);
}


