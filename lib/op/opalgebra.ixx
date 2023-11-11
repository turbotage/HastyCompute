module;

#include "../torch_util.hpp"

export module opalgebra;

import op;

namespace hasty {
	namespace op {


		export class AddOp : public Operator {
		public:

			AddOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::shared_ptr<Operator> _left;
			std::shared_ptr<Operator> _right;
		};

		export class AdjointableAddOp : public AdjointableOp {
		public:

			AdjointableAddOp(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::shared_ptr<AdjointableOp> _left;
			std::shared_ptr<AdjointableOp> _right;
		};

		export class SubOp : public Operator {
		public:

			SubOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::shared_ptr<Operator> _left;
			std::shared_ptr<Operator> _right;
		};

		export class AdjointableSubOp : public AdjointableOp {
		public:

			AdjointableSubOp(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::shared_ptr<AdjointableOp> _left;
			std::shared_ptr<AdjointableOp> _right;
		};

		export class MulOp : public Operator {
		public:

			MulOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::shared_ptr<Operator> _left;
			std::shared_ptr<Operator> _right;
		};

		export class AdjointableMulOp : public AdjointableOp {
		public:

			AdjointableMulOp(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			std::shared_ptr<AdjointableOp> _left;
			std::shared_ptr<AdjointableOp> _right;
		};

		export class ScaleOp : public Operator {
		public:

			ScaleOp(const at::Tensor& scalar, std::shared_ptr<Operator> op);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			at::Tensor _scalar;
			std::shared_ptr<Operator> _op;
		};

		export class AdjointableScaleOp : public AdjointableOp {
		public:

			AdjointableScaleOp(const at::Tensor& scalar, std::shared_ptr<AdjointableOp> op);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			at::Tensor _scalar;
			std::shared_ptr<AdjointableOp> _op;
		};

		export class VStackedOp : public Operator {
		public:

			static std::unique_ptr<VStackedOp> Create(const std::vector<std::shared_ptr<Operator>>& ops);

			static std::unique_ptr<VStackedOp> Create(std::vector<std::shared_ptr<Operator>>&& ops);

			Vector apply(const Vector& in) const override;

			size_t stack_size() const;

			Operator& get_slice(size_t idx);
			const Operator& get_slice(size_t idx) const;
			std::shared_ptr<Operator> get_slice_ptr(size_t idx) const;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		protected:
			
			VStackedOp(const std::vector<std::shared_ptr<Operator>>& ops);
			VStackedOp(std::vector<std::shared_ptr<Operator>>&& ops);

		private:
			std::vector<std::shared_ptr<Operator>> _ops;
		};

		export class HStackedOp : public Operator {
		public:

			static std::unique_ptr<HStackedOp> Create(const std::vector<std::shared_ptr<Operator>>& ops);

			static std::unique_ptr<HStackedOp> Create(std::vector<std::shared_ptr<Operator>>&& ops);

			Vector apply(const Vector& in) const override;

			size_t stack_size() const;

			Operator& get_slice(size_t idx);
			const Operator& get_slice(size_t idx) const;
			std::shared_ptr<Operator> get_slice_ptr(size_t idx) const;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		protected:

			HStackedOp(const std::vector<std::shared_ptr<Operator>>& ops);
			HStackedOp(std::vector<std::shared_ptr<Operator>>&& ops);

		private:
			std::vector<std::shared_ptr<Operator>> _ops;
		};

		export class AdjointableVStackedOp : public AdjointableOp {
		public:

			static std::unique_ptr<AdjointableVStackedOp> Create(const std::vector<std::shared_ptr<AdjointableOp>>& ops);

			static std::unique_ptr<AdjointableVStackedOp> Create(std::vector<std::shared_ptr<AdjointableOp>>&& ops);

			Vector apply(const Vector& in) const override;

			size_t stack_size() const;

			AdjointableOp& get_slice(size_t idx);
			const AdjointableOp& get_slice(size_t idx) const;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		protected:

			AdjointableVStackedOp(const std::vector<std::shared_ptr<AdjointableOp>>& ops);
			AdjointableVStackedOp(std::vector<std::shared_ptr<AdjointableOp>>&& ops);

		private:
			std::vector<std::shared_ptr<AdjointableOp>> _ops;
		};

		export class AdjointableHStackedOp : public AdjointableOp {
		public:

			static std::unique_ptr<AdjointableHStackedOp> Create(const std::vector<std::shared_ptr<AdjointableOp>>& ops);

			static std::unique_ptr<AdjointableHStackedOp> Create(std::vector<std::shared_ptr<AdjointableOp>>&& ops);

			Vector apply(const Vector& in) const override;

			size_t stack_size() const;

			AdjointableOp& get_slice(size_t idx);
			const AdjointableOp& get_slice(size_t idx) const;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		protected:

			AdjointableHStackedOp(const std::vector<std::shared_ptr<AdjointableOp>>& ops);
			AdjointableHStackedOp(std::vector<std::shared_ptr<AdjointableOp>>&& ops);

		private:
			std::vector<std::shared_ptr<AdjointableOp>> _ops;
		};


	}
}



