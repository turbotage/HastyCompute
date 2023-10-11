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

			Vector(Vector&& vec);

			VectorShape get_shape() const;

			Vector clone() const;

			Vector& operator+=(const Vector& rhs);
			friend Vector operator+(const Vector& lhs, const Vector& rhs);

			Vector& operator-=(const Vector& rhs);

			friend Vector operator-(const Vector& lhs, const Vector& rhs);

			Vector& operator*=(const Vector& rhs);
			friend Vector operator*(const Vector& lhs, const Vector& rhs);
			Vector& operator*=(const at::Tensor& rhs);


			Vector& operator/=(const Vector& rhs);
			friend Vector operator/(const Vector& lhs, const Vector& rhs);

			at::Tensor norm() const;
			Vector abs() const;
			Vector real() const;
			Vector imag() const;

			static Vector zeros(const VectorShape& shape, const at::TensorOptions& opts);
			static Vector ones(const VectorShape& shape, const at::TensorOptions& opts);
			static Vector rand(const VectorShape& shape, const at::TensorOptions& opts);

		private:
			at::Tensor _tensor;
			std::vector<Vector> _children;
		};


		class Operator {
		public:

			Vector operator()(const Vector& in) const;

			Vector apply(const Vector& in) const;

			virtual Vector _apply(const Vector& in) const;

			friend Operator operator+(Operator lhs, const Operator& rhs);
			friend Operator operator-(Operator lhs, const Operator& rhs);
			friend Operator operator*(Operator lhs, const Operator& rhs);


		private:

		};

		class AddOp : public Operator {
		public:

			AddOp(const Operator& lop, const Operator& rop);

			Vector _apply(const Vector& in) const override;

		private:
			Operator _left;
			Operator _right;
		};

		class SubOp : public Operator {
		public:

			SubOp(const Operator& lop, const Operator& rop);

			Vector _apply(const Vector& in) const override;

		private:
			Operator _left;
			Operator _right;
		};

		class MulOp : public Operator {
		public:

			MulOp(const Operator& lop, const Operator& rop);

			Vector _apply(const Vector& in) const override;

		private:
			Operator _left;
			Operator _right;
		};


	}

}

