module;

#include "../torch_util.hpp"

export module op;

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

			Operator();
			Operator(bool should_inplace_apply);

			Vector operator()(const Vector& in) const;
			friend Vector operator*(const Operator& lhs, const Vector& rhs);

			virtual Vector apply(const Vector& in) const = 0;

			virtual void apply_inplace(Vector& in) const;
			virtual bool has_inplace_apply() const;

			virtual bool should_inplace_apply() const;

			/*
			is required to point to an AdjointableOp if this is AdjtointableOp
			*/
			virtual std::shared_ptr<Operator> to_device(at::Stream stream) const = 0;

			/*
			template<OpConcept Op>
			std::shared_ptr<Op> downcast() const;
			*/

			template<OpConcept Op1, OpConcept Op2>
				requires NotAdjointableOpConcept<Op1> || NotAdjointableOpConcept<Op2>
			friend std::shared_ptr<class AddOp> add(std::shared_ptr<Op1> lhs, std::shared_ptr<Op2> rhs);

			template<AdjointableOpConcept Op1, AdjointableOpConcept Op2>
			friend std::shared_ptr<class AdjointableAddOp> add(std::shared_ptr<Op1> lhs, std::shared_ptr<Op2> rhs);

			template<OpConcept Op1, OpConcept Op2>
			friend std::shared_ptr<class SubOp> sub(std::shared_ptr<Op1> lhs, std::shared_ptr<Op2> rhs);

			template<AdjointableOpConcept Op1, AdjointableOpConcept Op2>
			friend std::shared_ptr<class AdjointableSubOp> sub(std::shared_ptr<Op1> lhs, std::shared_ptr<Op2> rhs);

			template<OpConcept Op1, OpConcept Op2>
				requires NotAdjointableOpConcept<Op1> || NotAdjointableOpConcept<Op2>
			friend std::shared_ptr<class MulOp> mul(std::shared_ptr<Op1> lhs, std::shared_ptr<Op2> rhs);

			template<AdjointableOpConcept Op1, AdjointableOpConcept Op2>
			friend std::shared_ptr<class AdjointableMulOp> mul(std::shared_ptr<Op1> lhs, std::shared_ptr<Op2> rhs);

			template<OpConcept Op>
			friend std::shared_ptr<class ScaleOp> mul(const at::Tensor& lhs, std::shared_ptr<Op> rhs);

			template<AdjointableOpConcept Op>
			friend std::shared_ptr<class AdjointableScaleOp> mul(const at::Tensor& lhs, std::shared_ptr<Op> rhs);


		protected:

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

			virtual std::shared_ptr<AdjointableOp> adjoint() const = 0;

		};

		export class DefaultAdjointableOp : public AdjointableOp {
		public:

			DefaultAdjointableOp(std::shared_ptr<Operator> op, std::shared_ptr<Operator> oph);

			Vector apply(const Vector& in) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const override;

		private:
			std::shared_ptr<Operator> _op;
			std::shared_ptr<Operator> _oph;
		};

		export template<typename RetOp, typename InpOp>
		requires OpConcept<InpOp>&& std::derived_from<RetOp, InpOp>
		std::shared_ptr<RetOp> downcast(std::shared_ptr<InpOp> inop)
		{
			return std::dynamic_pointer_cast<RetOp>(std::move(inop));
		}

		export template<typename RetOp, typename InpOp>
		requires std::derived_from<InpOp, RetOp>&& OpConcept<RetOp>
		std::shared_ptr<RetOp> upcast(std::shared_ptr<InpOp> inop)
		{
			return std::static_pointer_cast<RetOp>(std::move(inop));
		}

		export template<OpConcept RetOp, OpConcept InpOp>
		std::shared_ptr<RetOp> staticcast(std::shared_ptr<InpOp> inop)
		{
			return std::static_pointer_cast<RetOp>(std::move(inop));
		}


		// ADD
		export template<OpConcept Op1, OpConcept Op2>
		requires NotAdjointableOpConcept<Op1> || NotAdjointableOpConcept<Op2>
		std::shared_ptr<class AddOp> add(std::shared_ptr<Op1> lhs, std::shared_ptr<Op2> rhs)
		{
			return std::make_shared<AddOp>(std::move(lhs), std::move(rhs));
		}

		// ADD ADJ
		export template<AdjointableOpConcept Op1, AdjointableOpConcept Op2>
		std::shared_ptr<class AdjointableAddOp> add(std::shared_ptr<Op1> lhs, std::shared_ptr<Op2> rhs)
		{
			return std::make_shared<AdjointableAddOp>(std::move(lhs), std::move(rhs));
		}

		// SUB
		export template<OpConcept Op1, OpConcept Op2>
		std::shared_ptr<class SubOp> sub(std::shared_ptr<Op1> lhs, std::shared_ptr<Op2> rhs)
		{
			return std::make_shared<SubOp>(std::move(lhs), std::move(rhs));
		}

		// SUB ADJ
		export template<AdjointableOpConcept Op1, AdjointableOpConcept Op2>
		std::shared_ptr<class AdjointableSubOp> sub_adj(std::shared_ptr<Op1> lhs, std::shared_ptr<Op2> rhs)
		{
			return std::make_shared<AdjointableSubOp>(std::move(lhs), std::move(rhs));
		}

		// MUL
		export template<OpConcept Op1, OpConcept Op2>
		requires NotAdjointableOpConcept<Op1> || NotAdjointableOpConcept<Op2>
		std::shared_ptr<class MulOp> mul(std::shared_ptr<Op1> lhs, std::shared_ptr<Op2> rhs)
		{
			return std::make_shared<MulOp>(std::move(lhs), std::move(rhs));
		}

		// MUL ADJ
		export template<AdjointableOpConcept Op1, AdjointableOpConcept Op2>
		std::shared_ptr<class AdjointableMulOp> mul(std::shared_ptr<Op1> lhs, std::shared_ptr<Op2> rhs)
		{
			return std::make_shared<AdjointableMulOp>(std::move(lhs), std::move(rhs));
		}

		// SCALE MUL
		export template<OpConcept Op>
		std::shared_ptr<class ScaleOp> mul(const at::Tensor& lhs, std::shared_ptr<Op> rhs)
		{
			return std::make_shared<ScaleOp>(lhs, std::move(rhs));
		}

		// SCALE MUL ADJ
		export template<AdjointableOpConcept Op>
		std::shared_ptr<class AdjointableScaleOp> mul_adj(const at::Tensor& lhs, std::shared_ptr<Op> rhs)
		{
			return std::make_shared<AdjointableScaleOp>(lhs, std::move(rhs));
		}

	}

}
