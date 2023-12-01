module;

#include <torch/torch.h>

export module opalgebra;

import torch_util;
import op;

namespace hasty {
	namespace op {

		export class IdOp : public AdjointableOp {
		public:

			static std::unique_ptr<IdOp> Create();

			Vector apply(const Vector& in) const override;

			sptr<AdjointableOp> adjoint() const override;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:
			IdOp();
		private:
		};

		export class NegIdOp : public AdjointableOp {
		public:

			static std::unique_ptr<NegIdOp> Create();

			Vector apply(const Vector& in) const override;

			sptr<AdjointableOp> adjoint() const override;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:
			NegIdOp();
		private:
		};

		export class AddOp : public Operator {
		public:

			static std::unique_ptr<AddOp> Create(sptr<Operator> lop, sptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:
			
			AddOp(sptr<Operator> lop, sptr<Operator> rop);

		private:
			sptr<Operator> _left;
			sptr<Operator> _right;
		};

		export class AdjointableAddOp : public AdjointableOp {
		public:

			static std::unique_ptr<AdjointableAddOp> Create(sptr<AdjointableOp> lop, sptr<AdjointableOp> rop);

			Vector apply(const Vector& in) const override;

			sptr<AdjointableOp> adjoint() const override;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:
			
			AdjointableAddOp(sptr<AdjointableOp> lop, sptr<AdjointableOp> rop);

		private:
			sptr<AdjointableOp> _left;
			sptr<AdjointableOp> _right;
		};

		export class SubOp : public Operator {
		public:

			static std::unique_ptr<SubOp> Create(sptr<Operator> lop, sptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:

			SubOp(sptr<Operator> lop, sptr<Operator> rop);

		private:
			sptr<Operator> _left;
			sptr<Operator> _right;
		};

		export class AdjointableSubOp : public AdjointableOp {
		public:

			static std::unique_ptr<AdjointableSubOp> Create(sptr<AdjointableOp> lop, sptr<AdjointableOp> rop);

			Vector apply(const Vector& in) const override;

			sptr<AdjointableOp> adjoint() const override;

			sptr<Operator> to_device(at::Stream stream) const;
			
		protected:
			
			AdjointableSubOp(sptr<AdjointableOp> lop, sptr<AdjointableOp> rop);

		private:
			sptr<AdjointableOp> _left;
			sptr<AdjointableOp> _right;
		};

		export class MulOp : public Operator {
		public:

			static std::unique_ptr<MulOp> Create(sptr<Operator> lop, sptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:
			
			MulOp(sptr<Operator> lop, sptr<Operator> rop);

		private:
			sptr<Operator> _left;
			sptr<Operator> _right;
		};

		export class AdjointableMulOp : public AdjointableOp {
		public:

			static std::unique_ptr<AdjointableMulOp> Create(sptr<AdjointableOp> lop, sptr<AdjointableOp> rop);

			Vector apply(const Vector& in) const override;

			sptr<AdjointableOp> adjoint() const override;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:
			
			AdjointableMulOp(sptr<AdjointableOp> lop, sptr<AdjointableOp> rop);

		private:
			sptr<AdjointableOp> _left;
			sptr<AdjointableOp> _right;
		};

		export class ScaleOp : public Operator {
		public:

			static std::unique_ptr<ScaleOp> Create(const at::Tensor& scalar, sptr<Operator> rop);

			Vector apply(const Vector& in) const override;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:

			ScaleOp(const at::Tensor& scalar, sptr<Operator> op);

		private:
			at::Tensor _scalar;
			sptr<Operator> _op;
		};

		export class AdjointableScaleOp : public AdjointableOp {
		public:

			static std::unique_ptr<AdjointableScaleOp> Create(const at::Tensor& scalar, sptr<AdjointableOp> op);

			Vector apply(const Vector& in) const override;

			sptr<AdjointableOp> adjoint() const override;

			sptr<Operator> to_device(at::Stream stream) const;
			
		protected:

			AdjointableScaleOp(const at::Tensor& scalar, sptr<AdjointableOp> op);

		private:
			at::Tensor _scalar;
			sptr<AdjointableOp> _op;
		};

		export class VStackedOp : public Operator {
		public:

			static std::unique_ptr<VStackedOp> Create(const std::vector<sptr<Operator>>& ops);

			static std::unique_ptr<VStackedOp> Create(std::vector<sptr<Operator>>&& ops);

			Vector apply(const Vector& in) const override;

			size_t stack_size() const;

			Operator& get_slice(size_t idx);
			const Operator& get_slice(size_t idx) const;
			sptr<Operator> get_slice_ptr(size_t idx) const;

			const std::vector<sptr<Operator>>& get_stack() const;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:
			
			VStackedOp(const std::vector<sptr<Operator>>& ops);
			VStackedOp(std::vector<sptr<Operator>>&& ops);

		private:
			std::vector<sptr<Operator>> _ops;
		};

		export class HStackedOp : public Operator {
		public:

			static std::unique_ptr<HStackedOp> Create(const std::vector<sptr<Operator>>& ops);

			static std::unique_ptr<HStackedOp> Create(std::vector<sptr<Operator>>&& ops);

			Vector apply(const Vector& in) const override;

			size_t stack_size() const;

			Operator& get_slice(size_t idx);
			const Operator& get_slice(size_t idx) const;
			sptr<Operator> get_slice_ptr(size_t idx) const;

			const std::vector<sptr<Operator>>& get_stack() const;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:

			HStackedOp(const std::vector<sptr<Operator>>& ops);
			HStackedOp(std::vector<sptr<Operator>>&& ops);

		private:
			std::vector<sptr<Operator>> _ops;
		};

		export class AdjointableVStackedOp : public AdjointableOp {
		public:

			static std::unique_ptr<AdjointableVStackedOp> Create(const std::vector<sptr<AdjointableOp>>& ops);

			static std::unique_ptr<AdjointableVStackedOp> Create(std::vector<sptr<AdjointableOp>>&& ops);

			Vector apply(const Vector& in) const override;

			size_t stack_size() const;

			AdjointableOp& get_slice(size_t idx);
			const AdjointableOp& get_slice(size_t idx) const;
			sptr<AdjointableOp> get_slice_ptr(size_t idx) const;

			const std::vector<sptr<AdjointableOp>>& get_stack() const;

			sptr<AdjointableOp> adjoint() const override;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:

			AdjointableVStackedOp(const std::vector<sptr<AdjointableOp>>& ops);
			AdjointableVStackedOp(std::vector<sptr<AdjointableOp>>&& ops);

		private:
			std::vector<sptr<AdjointableOp>> _ops;
		};

		export class AdjointableHStackedOp : public AdjointableOp {
		public:

			static std::unique_ptr<AdjointableHStackedOp> Create(const std::vector<sptr<AdjointableOp>>& ops);

			static std::unique_ptr<AdjointableHStackedOp> Create(std::vector<sptr<AdjointableOp>>&& ops);

			Vector apply(const Vector& in) const override;

			size_t stack_size() const;

			AdjointableOp& get_slice(size_t idx);
			const AdjointableOp& get_slice(size_t idx) const;
			sptr<AdjointableOp> get_slice_ptr(size_t idx) const;

			const std::vector<sptr<AdjointableOp>>& get_stack() const;

			sptr<AdjointableOp> adjoint() const override;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:

			AdjointableHStackedOp(const std::vector<sptr<AdjointableOp>>& ops);
			AdjointableHStackedOp(std::vector<sptr<AdjointableOp>>&& ops);

		private:
			std::vector<sptr<AdjointableOp>> _ops;
		};


	}
}



