#pragma once

#include "op.hpp"

namespace hasty {
	namespace op {

		class AddOp : public Operator {
		public:

			AddOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::shared_ptr<Operator> _left;
			std::shared_ptr<Operator> _right;
		};

		class AdjointableAddOp : public AdjointableOp {
		public:

			AdjointableAddOp(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::shared_ptr<AdjointableOp> _left;
			std::shared_ptr<AdjointableOp> _right;
		};

		class SubOp : public Operator {
		public:

			SubOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::shared_ptr<Operator> _left;
			std::shared_ptr<Operator> _right;
		};

		class AdjointableSubOp : public AdjointableOp {
		public:

			AdjointableSubOp(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::shared_ptr<AdjointableOp> _left;
			std::shared_ptr<AdjointableOp> _right;
		};

		class MulOp : public Operator {
		public:

			MulOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::shared_ptr<Operator> _left;
			std::shared_ptr<Operator> _right;
		};

		class AdjointableMulOp : public AdjointableOp {
		public:

			AdjointableMulOp(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::shared_ptr<AdjointableOp> _left;
			std::shared_ptr<AdjointableOp> _right;
		};

		class ScaleOp : public Operator {
		public:

			ScaleOp(const at::Tensor& scalar, std::shared_ptr<Operator> op);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			at::Tensor _scalar;
			std::shared_ptr<Operator> _op;
		};

		class AdjointableScaleOp : public AdjointableOp {
		public:

			AdjointableScaleOp(const at::Tensor& scalar, std::shared_ptr<AdjointableOp> op);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			at::Tensor _scalar;
			std::shared_ptr<AdjointableOp> _op;
		};

		class StackedOp : public Operator {
		public:

			StackedOp(std::vector<std::shared_ptr<Operator>>& ops);
			StackedOp(std::vector<std::shared_ptr<Operator>>&& ops);

			Vector apply(const Vector& in) const override;

			size_t stack_size() const;

			Operator& get_slice(size_t idx);
			const Operator& get_slice(size_t idx) const;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::vector<std::shared_ptr<Operator>> _ops;
		};

		class AdjointableStackedOp : public AdjointableOp {
		public:

			AdjointableStackedOp(std::vector<std::shared_ptr<AdjointableOp>>& ops);
			AdjointableStackedOp(std::vector<std::shared_ptr<AdjointableOp>>&& ops);

			Vector apply(const Vector& in) const override;

			size_t stack_size() const;

			AdjointableOp& get_slice(size_t idx);
			const AdjointableOp& get_slice(size_t idx) const;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::vector<std::shared_ptr<AdjointableOp>> _ops;
		};

	}
}
