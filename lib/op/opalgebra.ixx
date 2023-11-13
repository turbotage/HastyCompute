module;

#include <torch/torch.h>

export module opalgebra;

import torch_util;
import op;

namespace hasty {
	namespace op {


		export class AddOp : public Operator {
		public:

			static std::unique_ptr<AddOp> Create(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		protected:
			
			AddOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop);

		private:
			std::shared_ptr<Operator> _left;
			std::shared_ptr<Operator> _right;
		};

		export class AdjointableAddOp : public AdjointableOp {
		public:

			static std::unique_ptr<AdjointableAddOp> Create(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		protected:
			
			AdjointableAddOp(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop);

		private:
			std::shared_ptr<AdjointableOp> _left;
			std::shared_ptr<AdjointableOp> _right;
		};

		export class SubOp : public Operator {
		public:

			static std::unique_ptr<SubOp> Create(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		protected:

			SubOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop);

		private:
			std::shared_ptr<Operator> _left;
			std::shared_ptr<Operator> _right;
		};

		export class AdjointableSubOp : public AdjointableOp {
		public:

			static std::unique_ptr<AdjointableSubOp> Create(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;
			
		protected:
			
			AdjointableSubOp(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop);

		private:
			std::shared_ptr<AdjointableOp> _left;
			std::shared_ptr<AdjointableOp> _right;
		};

		export class MulOp : public Operator {
		public:

			static std::unique_ptr<MulOp> Create(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		protected:
			
			MulOp(std::shared_ptr<Operator> lop, std::shared_ptr<Operator> rop);

		private:
			std::shared_ptr<Operator> _left;
			std::shared_ptr<Operator> _right;
		};

		export class AdjointableMulOp : public AdjointableOp {
		public:

			static std::unique_ptr<AdjointableMulOp> Create(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		protected:
			
			AdjointableMulOp(std::shared_ptr<AdjointableOp> lop, std::shared_ptr<AdjointableOp> rop);

		private:
			std::shared_ptr<AdjointableOp> _left;
			std::shared_ptr<AdjointableOp> _right;
		};

		export class ScaleOp : public Operator {
		public:

			static std::unique_ptr<ScaleOp> Create(const at::Tensor& scalar, std::shared_ptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		protected:

			ScaleOp(const at::Tensor& scalar, std::shared_ptr<Operator> op);

		private:
			at::Tensor _scalar;
			std::shared_ptr<Operator> _op;
		};

		export class AdjointableScaleOp : public AdjointableOp {
		public:

			static std::unique_ptr<AdjointableScaleOp> Create(const at::Tensor& scalar, std::shared_ptr<AdjointableOp> op);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;
			
		protected:

			AdjointableScaleOp(const at::Tensor& scalar, std::shared_ptr<AdjointableOp> op);

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

			const std::vector<std::shared_ptr<Operator>>& get_stack() const;

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

			const std::vector<std::shared_ptr<Operator>>& get_stack() const;

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
			std::shared_ptr<AdjointableOp> get_slice_ptr(size_t idx) const;

			const std::vector<std::shared_ptr<AdjointableOp>>& get_stack() const;

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
			std::shared_ptr<AdjointableOp> get_slice_ptr(size_t idx) const;

			const std::vector<std::shared_ptr<AdjointableOp>>& get_stack() const;

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



