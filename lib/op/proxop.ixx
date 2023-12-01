module;

#include <torch/torch.h>

export module proxop;

import <memory>;
import <optional>;

import hasty_util;
import torch_util;
import vec;
import op;

namespace hasty {
	namespace op {

		export class ProximalOperator {
		public:

			double get_base_alpha() const;

			void set_base_alpha(double base_alpha);

			Vector apply(const Vector& input, double alpha = 1.0) const;

			void apply_inplace(Vector& inout, double alpha) const;

			virtual bool has_inplace_apply() const;

		protected:
			
			ProximalOperator(double base_alpha = 1.0);

			virtual Vector _apply(const Vector& input, double alpha) const;

			virtual void _apply_inplace(Vector& input, double alpha) const;

		private:
			double _base_alpha;
		};

		export class ZeroFunc : public ProximalOperator {
		public:

			uptr<ZeroFunc> Create();
			
		protected:

			Vector _apply(const Vector& input, double alpha) const override;

			ZeroFunc() = default;

		private:
		};

		export class ConvexConjugate : public ProximalOperator {
		public:

			uptr<ConvexConjugate> Create(sptr<ProximalOperator> prox, double base_alpha = 1.0);

		protected:

			ConvexConjugate(sptr<ProximalOperator> prox, double base_alpha = 1.0);

			Vector _apply(const Vector& input, double alpha);

		private:
			sptr<ProximalOperator> _prox;
		};

		export class Postcomposition : public ProximalOperator {
		public:

			uptr<Postcomposition> Create(sptr<ProximalOperator> prox, double base_alpha = 1.0);

		protected:

			Postcomposition(sptr<ProximalOperator> prox, double base_alpha = 1.0);

			Vector _apply(const Vector& input, double alpha);

		private:
			sptr<ProximalOperator> _prox;
		};

		export class Precomposition : public ProximalOperator {
		public:

			uptr<Precomposition> Create(sptr<ProximalOperator> prox, double a, double b, double base_alpha = 1.0);

		protected:

			Precomposition(sptr<ProximalOperator> prox, double a, double b, double base_alpha = 1.0);

			Vector _apply(const Vector& input, double alpha);

		private:
			sptr<ProximalOperator> _prox;
			double _a, _b;
		};

		export class UnitaryTransform : public ProximalOperator {
		public:

			uptr<UnitaryTransform> Create(sptr<ProximalOperator> prox, sptr<Operator> unitary,
				sptr<Operator> unitary_adjoint, double base_alpha = 1.0);

		protected:

			UnitaryTransform(sptr<ProximalOperator> prox, sptr<Operator> unitary,
				sptr<Operator> unitary_adjoint, double base_alpha = 1.0);

			Vector _apply(const Vector& input, double alpha);

		private:
			sptr<ProximalOperator>	_prox;
			sptr<Operator>			_unitary;
			sptr<Operator>			_unitary_adjoint;
		};

		export class AffineAddition : public ProximalOperator {
		public:

			uptr<AffineAddition> Create(sptr<ProximalOperator> prox, const Vector& a, double base_alpha = 1.0);

		protected:

			AffineAddition(sptr<ProximalOperator> prox, const Vector& a, double base_alpha = 1.0);

			Vector _apply(const Vector& input, double alpha);

		private:
			sptr<ProximalOperator> _prox;
			Vector _a;
		};

		export class L2Reg : public ProximalOperator {
		public:

			uptr<L2Reg> Create(sptr<ProximalOperator> prox, double rho = 1.0, const opt<Vector>& a = std::nullopt,
				double base_alpha = 1.0);

		protected:

			L2Reg(sptr<ProximalOperator> prox, double rho = 1.0, const opt<Vector>& a = std::nullopt,
				double base_alpha = 1.0);

			Vector _apply(const Vector& input, double alpha);

		private:
			sptr<ProximalOperator> _prox;
			double _rho;
			opt<Vector> _a;
		};

	}
}


