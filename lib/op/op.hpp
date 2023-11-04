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
			friend class Operator;
		};

		class Operator;
		class AdjointableOp;

		class Vector {
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
			static Vector ones(const VectorShape& shape, const at::TensorOptions& opts);
			static Vector rand(const VectorShape& shape, const at::TensorOptions& opts);
			static Vector empty(const VectorShape& shape, const at::TensorOptions& opts);

			friend at::Tensor vdot(const Vector& a, const Vector& b);

		private:
			at::Tensor _tensor;
			std::vector<Vector> _children;

			friend class Operator;
			friend class OperatorAlg;
		};


		class Operator {
		public:

			Operator();
			Operator(bool should_inplace_apply);

			Vector operator()(const Vector& in) const;
			friend Vector operator*(const Operator& lhs, const Vector& rhs);
			
			virtual Vector apply(const Vector& in) const;
			
			virtual void apply_inplace(Vector& in) const;
			virtual bool has_inplace_apply() const;

			virtual bool should_inplace_apply() const;

			friend class AddOp operator+(const Operator& lhs, const Operator& rhs);
			friend class AdjointableAddOp operator+(const AdjointableOp& lhs, const AdjointableOp& rhs);

			friend class SubOp operator-(const Operator& lhs, const Operator& rhs);
			friend class AdjointableSubOp operator-(const AdjointableOp& lhs, const AdjointableOp& rhs);

			friend class MulOp operator*(const Operator& lhs, const Operator& rhs);
			friend class AdjointableMulOp operator*(const AdjointableOp& lhs, const AdjointableOp& rhs);

			friend class ScaleOp operator*(const at::Tensor& lhs, const Operator& rhs);
			friend class AdjointableScaleOp operator*(const at::Tensor& lhs, const AdjointableOp& rhs);

		protected:

			at::Tensor& access_vectensor(Vector& vec) const;
			const at::Tensor& access_vectensor(const Vector& vec) const;

			std::vector<Vector>& access_vecchilds(Vector& vec) const;
			const std::vector<Vector>& access_vecchilds(const Vector& vec) const;

		private:
			bool _should_inplace_apply;

			friend class OperatorAlg;
		};

		class AdjointableOp : public Operator {
		public:

			AdjointableOp();
			AdjointableOp(bool should_inplace_apply);
			AdjointableOp(const at::optional<Operator>& op, const at::optional<Operator>& oph);

			virtual const AdjointableOp& adjoint() const;

		private:
			at::optional<Operator> _op;
			at::optional<Operator> _oph;
		};

		class AddOp : public Operator {
		public:

			AddOp(const Operator& lop, const Operator& rop);

			Vector apply(const Vector& in) const override;

		private:
			Operator _left;
			Operator _right;
		};

		class AdjointableAddOp : public AdjointableOp {
		public:

			AdjointableAddOp(const AdjointableOp& lop, const AdjointableOp& rop);

			Vector apply(const Vector& in) const override;

			const AdjointableAddOp& adjoint() const override;

		private:
			AdjointableOp _left;
			AdjointableOp _right;
		};

		class SubOp : public Operator {
		public:

			SubOp(const Operator& lop, const Operator& rop);

			Vector apply(const Vector& in) const override;

		private:
			Operator _left;
			Operator _right;
		};

		class AdjointableSubOp : public AdjointableOp {
		public:

			AdjointableSubOp(const AdjointableOp& lop, const AdjointableOp& rop);

			Vector apply(const Vector& in) const override;

			const AdjointableSubOp& adjoint() const override;

		private:
			AdjointableOp _left;
			AdjointableOp _right;
		};

		class MulOp : public Operator {
		public:

			MulOp(const Operator& lop, const Operator& rop);

			Vector apply(const Vector& in) const override;

		private:
			Operator _left;
			Operator _right;
		};

		class AdjointableMulOp : public AdjointableOp {
		public:

			AdjointableMulOp(const AdjointableOp& lop, const AdjointableOp& rop);

			Vector apply(const Vector& in) const override;

			const AdjointableMulOp& adjoint() const override;

		private:
			AdjointableOp _left;
			AdjointableOp _right;
		};

		class ScaleOp : public Operator {
		public:

			ScaleOp(const at::Tensor& scalar, const at::optional<Operator>& op);

			Vector apply(const Vector& in) const override;

		private:
			at::Tensor _scalar;
			at::optional<Operator> _op;
		};

		class AdjointableScaleOp : public AdjointableOp {
		public:

			AdjointableScaleOp(const at::Tensor& scalar, const at::optional<AdjointableOp>& op);

			Vector apply(const Vector& in) const override;

			const AdjointableScaleOp& adjoint() const override;

		private:
			at::Tensor _scalar;
			at::optional<AdjointableOp> _op;
		};

		class StackedOp : public Operator {
		public:

			StackedOp(const std::vector<Operator>& ops);

			Vector apply(const Vector& in) const override;

			size_t stack_size() const;

			op::Operator& get_layer(size_t idx);
			const op::Operator& get_layer(size_t idx) const;

		private:
			std::vector<Operator> _ops;
		};

		class AdjointableStackedOp : public AdjointableOp {
		public:

			AdjointableStackedOp(const std::vector<AdjointableOp>& ops);

			Vector apply(const Vector& in) const override;

			size_t stack_size() const;

			op::AdjointableOp& get_layer();

		private:
			std::vector<AdjointableOp> _ops;
		};

	}

}

