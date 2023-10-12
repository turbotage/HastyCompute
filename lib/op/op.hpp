#pragma once

#include "../torch_util.hpp"

namespace hasty {

	namespace op {

		class VectorShape {
		public:

			VectorShape(const std::initializer_list<int64_t>& dims);
			VectorShape(const c10::IntArrayRef& dims);
			VectorShape(const std::initializer_list<std::vector<int64_t>>& dims);
			VectorShape(std::vector<int64_t>&& dims);
			VectorShape(const std::vector<int64_t>& dims);
			VectorShape(const std::vector<int64_t>::iterator& begin, const std::vector<int64_t>::iterator& end);

			VectorShape(const std::initializer_list<VectorShape>& shapes);
			VectorShape(const std::vector<VectorShape>& shapes);
			VectorShape(const std::vector<VectorShape>::iterator& begin, const std::vector<VectorShape>::iterator& end);
			VectorShape(const std::initializer_list<std::vector<VectorShape>>& shapes);

		private:
			std::vector<int64_t> _self_shape;
			std::vector<VectorShape> _children_shape;

			friend class Vector;
		};

		class Vector {
		public:

			Vector(const at::Tensor& tensor);
			Vector(const std::vector<Vector>& children);

			at::ScalarType dtype() const;

			Vector(const Vector& vec);
			Vector& operator=(const Vector& vec);
			
			Vector(Vector&& vec);
			Vector& operator=(Vector&& vec);

			VectorShape get_shape() const;

			Vector clone() const;

			Vector& operator+=(const Vector& rhs);
			friend Vector operator+(const Vector& lhs, const Vector& rhs);

			Vector& operator-=(const Vector& rhs);

			friend Vector operator-(const Vector& lhs, const Vector& rhs);

			Vector& operator*=(const Vector& rhs);
			friend Vector operator*(const Vector& lhs, const Vector& rhs);

			Vector& operator/=(const Vector& rhs);
			friend Vector operator/(const Vector& lhs, const Vector& rhs);

			at::Tensor norm() const;
			Vector abs() const;
			Vector real() const;
			Vector imag() const;

			static Vector zeros(const VectorShape& shape, const at::TensorOptions& opts);
			static Vector ones(const VectorShape& shape, const at::TensorOptions& opts);
			static Vector rand(const VectorShape& shape, const at::TensorOptions& opts);
			static Vector empty(const VectorShape& shape, const at::TensorOptions& opts);

			friend at::Tensor vdot(const Vector& a, const Vector& b);

		private:
			at::Tensor _tensor;
			std::vector<Vector> _children;
		};

		class Operator {
		public:

			Vector operator()(const Vector& in) const;
			friend Vector operator*(const Operator& lhs, const Vector& rhs);
			
			virtual Vector apply(const Vector& in) const;
			
			virtual void apply_inplace(Vector& in) const;
			virtual bool has_inplace_apply() const;

			friend Operator operator+(const Operator& lhs, const Operator& rhs);
			friend Operator operator-(const Operator& lhs, const Operator& rhs);
			friend Operator operator*(const Operator& lhs, const Operator& rhs);

		private:

		};

		class AddOp : public Operator {
		public:

			AddOp(const Operator& lop, const Operator& rop);

			Vector apply(const Vector& in) const override;

		private:
			Operator _left;
			Operator _right;
		};

		class SubOp : public Operator {
		public:

			SubOp(const Operator& lop, const Operator& rop);

			Vector apply(const Vector& in) const override;

		private:
			Operator _left;
			Operator _right;
		};

		class MulOp : public Operator {
		public:

			MulOp(const Operator& lop, const Operator& rop);

			Vector apply(const Vector& in) const override;

		private:
			Operator _left;
			Operator _right;
		};


	}

}

