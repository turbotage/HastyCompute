module;

#include <torch/torch.h>

export module vec;

import torch_util;
import hasty_util;

namespace hasty {
	namespace op {

		export class VectorShape {
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
			friend class Operator;
		};

		class Operator;
		class AdjointableOp;

		export class Vector {
		public:

			Vector(float scalar);
			Vector(double scalar);
			Vector(std::complex<float> scalar);
			Vector(std::complex<double> scalar);
			Vector(c10::complex<float> scalar);
			Vector(c10::complex<double> scalar);

			Vector(const at::Tensor& tensor);
			Vector(const std::vector<Vector>& children);

			at::ScalarType dtype() const;
			at::TensorOptions tensor_opts() const;

			bool has_children() const;

			const std::vector<Vector>& children() const;
			const at::Tensor& tensor() const;

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
			static Vector zeros_like(const Vector& other);

			static Vector ones(const VectorShape& shape, const at::TensorOptions& opts);
			static Vector ones_like(const Vector& other);

			static Vector rand(const VectorShape& shape, const at::TensorOptions& opts);
			static Vector rand_like(const Vector& other);

			static Vector empty(const VectorShape& shape, const at::TensorOptions& opts);
			static Vector empty_like(const Vector& other);

			friend at::Tensor vdot(const Vector& a, const Vector& b);

		private:
			at::Tensor _tensor;
			std::vector<Vector> _children;

			friend class Operator;
			friend class OperatorAlg;
		};

	}
}

