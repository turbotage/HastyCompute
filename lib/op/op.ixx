module;

#include <torch/torch.h>

export module op;

import torch_util;
import hasty_util;
import vec;

namespace hasty {

	namespace op {

		class Operator;
		class AdjointableOp;

		export template<typename T>
		concept OpConcept = std::derived_from<T, Operator>;

		export template<typename T>
		concept NotOpConcept = !std::derived_from<T, Operator>;

		export template<typename T>
		concept AdjointableOpConcept = std::derived_from<T, AdjointableOp>;

		export template<typename T>
		concept NotAdjointableOpConcept = !std::derived_from<T, AdjointableOp>;

		export class Operator : public hasty::inheritable_enable_shared_from_this<Operator> {
		public:

			Vector operator()(const Vector& in) const;
			friend Vector operator*(const Operator& lhs, const Vector& rhs);

			virtual Vector apply(const Vector& in) const = 0;

			virtual void apply_inplace(Vector& in) const;
			virtual bool has_inplace_apply() const;

			virtual bool should_inplace_apply() const;

			/*
			is required to point to an AdjointableOp if this is AdjtointableOp
			*/
			virtual sptr<Operator> to_device(at::Stream stream) const = 0;

			/*
			template<OpConcept Op>
			sptr<Op> downcast() const;
			*/

			template<OpConcept Op1, OpConcept Op2>
			requires NotAdjointableOpConcept<Op1> || NotAdjointableOpConcept<Op2>
			friend sptr<class AddOp> add(sptr<Op1> lhs, sptr<Op2> rhs);

			template<AdjointableOpConcept Op1, AdjointableOpConcept Op2>
			friend sptr<class AdjointableAddOp> add(sptr<Op1> lhs, sptr<Op2> rhs);

			template<OpConcept Op1, OpConcept Op2>
			requires NotAdjointableOpConcept<Op1> || NotAdjointableOpConcept<Op2>
			friend sptr<class SubOp> sub(sptr<Op1> lhs, sptr<Op2> rhs);

			template<AdjointableOpConcept Op1, AdjointableOpConcept Op2>
			friend sptr<class AdjointableSubOp> sub(sptr<Op1> lhs, sptr<Op2> rhs);

			template<OpConcept Op1, OpConcept Op2>
			requires NotAdjointableOpConcept<Op1> || NotAdjointableOpConcept<Op2>
			friend sptr<class MulOp> mul(sptr<Op1> lhs, sptr<Op2> rhs);

			template<AdjointableOpConcept Op1, AdjointableOpConcept Op2>
			friend sptr<class AdjointableMulOp> mul(sptr<Op1> lhs, sptr<Op2> rhs);

			template<OpConcept Op>
			requires NotAdjointableOpConcept<Op>
			friend sptr<class ScaleOp> mul(const at::Tensor& lhs, sptr<Op> rhs);

			template<AdjointableOpConcept Op>
			friend sptr<class AdjointableScaleOp> mul(const at::Tensor& lhs, sptr<Op> rhs);


		protected:

			Operator();
			Operator(bool should_inplace_apply);

			at::Tensor& access_vectensor(Vector& vec) const;
			const at::Tensor& access_vectensor(const Vector& vec) const;

			std::vector<Vector>& access_vecchilds(Vector& vec) const;
			const std::vector<Vector>& access_vecchilds(const Vector& vec) const;

		private:
			bool _should_inplace_apply;

			friend class OperatorAlg;
		};

		export class AdjointableOp : public Operator {
		public:

			virtual sptr<AdjointableOp> adjoint() const = 0;

		protected:

			AdjointableOp();
			AdjointableOp(bool should_inplace_apply);

		};

		export class DefaultAdjointableOp : public AdjointableOp {
		public:

			static uptr<DefaultAdjointableOp> Create(sptr<Operator> op, sptr<Operator> oph);

			Vector apply(const Vector& in) const override;

			sptr<AdjointableOp> adjoint() const override;

			sptr<Operator> to_device(at::Stream stream) const override;

		protected:

			DefaultAdjointableOp(sptr<Operator> op, sptr<Operator> oph);

		private:
			sptr<Operator> _op;
			sptr<Operator> _oph;
		};

		export template<typename RetOp, typename InpOp>
		requires OpConcept<InpOp>&& std::derived_from<RetOp, InpOp>
		sptr<RetOp> downcast(sptr<InpOp> inop)
		{
			return std::dynamic_pointer_cast<RetOp>(std::move(inop));
		}

		export template<typename RetOp, typename InpOp>
		requires std::derived_from<InpOp, RetOp>&& OpConcept<RetOp>
		sptr<RetOp> upcast(sptr<InpOp> inop)
		{
			return std::static_pointer_cast<RetOp>(std::move(inop));
		}

		export template<OpConcept RetOp, OpConcept InpOp>
		sptr<RetOp> staticcast(sptr<InpOp> inop)
		{
			return std::static_pointer_cast<RetOp>(std::move(inop));
		}


		// ADD
		export template<OpConcept Op1, OpConcept Op2>
		requires NotAdjointableOpConcept<Op1> || NotAdjointableOpConcept<Op2>
		sptr<class AddOp> add(sptr<Op1> lhs, sptr<Op2> rhs)
		{
			return AddOp::Create(std::move(lhs), std::move(rhs));
		}

		// ADD ADJ
		export template<AdjointableOpConcept Op1, AdjointableOpConcept Op2>
		sptr<class AdjointableAddOp> add(sptr<Op1> lhs, sptr<Op2> rhs)
		{
			return AdjointableAddOp::Create(std::move(lhs), std::move(rhs));
		}

		// SUB
		export template<OpConcept Op1, OpConcept Op2>
		requires NotAdjointableOpConcept<Op1> || NotAdjointableOpConcept<Op2>
		sptr<class SubOp> sub(sptr<Op1> lhs, sptr<Op2> rhs)
		{
			return SubOp::Create(std::move(lhs), std::move(rhs));
		}

		// SUB ADJ
		export template<AdjointableOpConcept Op1, AdjointableOpConcept Op2>
		sptr<class AdjointableSubOp> sub(sptr<Op1> lhs, sptr<Op2> rhs)
		{
			return AdjointableSubOp::Create(std::move(lhs), std::move(rhs));
		}

		// MUL
		export template<OpConcept Op1, OpConcept Op2>
		requires NotAdjointableOpConcept<Op1> || NotAdjointableOpConcept<Op2>
		sptr<class MulOp> mul(sptr<Op1> lhs, sptr<Op2> rhs)
		{
			return MulOp::Create(std::move(lhs), std::move(rhs));
		}

		// MUL ADJ
		export template<AdjointableOpConcept Op1, AdjointableOpConcept Op2>
		sptr<class AdjointableMulOp> mul(sptr<Op1> lhs, sptr<Op2> rhs)
		{
			return AdjointableMulOp::Create(std::move(lhs), std::move(rhs));
		}

		// SCALE MUL
		export template<OpConcept Op>
		requires NotAdjointableOpConcept<Op>
		sptr<class ScaleOp> mul(const at::Tensor& lhs, sptr<Op> rhs)
		{
			return ScaleOp::Create(lhs, std::move(rhs));
		}

		// SCALE MUL ADJ
		export template<AdjointableOpConcept Op>
		sptr<class AdjointableScaleOp> mul(const at::Tensor& lhs, sptr<Op> rhs)
		{
			return AdjointableScaleOp::Create(lhs, std::move(rhs));
		}

	}

}

