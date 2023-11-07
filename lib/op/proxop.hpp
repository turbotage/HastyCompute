#pragma once

#include "op.hpp"

namespace hasty {
	namespace op {

		class ProximalOperator {
		public:

			ProximalOperator(double base_alpha=1.0);

			double get_base_alpha() const;

			void set_base_alpha(double base_alpha);

			Vector apply(const Vector& input, double alpha=1.0) const;
			Vector operator()(const Vector& input, double alpha = 1.0) const;
			// Must override
			virtual Vector _apply(const Vector& input, double alpha) const;

			void apply_inplace(Vector& inout, double alpha) const;
			
			virtual bool has_inplace_apply() const;
			virtual void _apply_inplace(Vector& input, double alpha) const;

		private:
			double _base_alpha;
		};

		class ZeroFunc : public ProximalOperator {
		public:

			ZeroFunc() = default;

			Vector _apply(const Vector& input, double alpha) const override;

		private:
		};

		class ConvexConjugate : public ProximalOperator {
		public:

			ConvexConjugate(const ProximalOperator& prox, double base_alpha = 1.0);

			Vector _apply(const Vector& input, double alpha);

		private:
			ProximalOperator _prox;
		};

		class Postcomposition : public ProximalOperator {
		public:

			Postcomposition(const ProximalOperator& prox, double base_alpha = 1.0);

			Vector _apply(const Vector& input, double alpha);

		private:
			ProximalOperator _prox;
		};

		class Precomposition : public ProximalOperator {
		public:

			Precomposition(const ProximalOperator& prox, double a, double b, double base_alpha = 1.0);

			Vector _apply(const Vector& input, double alpha);

		private:
			ProximalOperator _prox;
			double _a, _b;
		};

		class UnitaryTransform : public ProximalOperator {
		public:

			UnitaryTransform(const ProximalOperator& prox, std::shared_ptr<Operator> unitary, 
				std::shared_ptr<Operator> unitary_adjoint, double base_alpha = 1.0);

			Vector _apply(const Vector& input, double alpha);

		private:
			ProximalOperator _prox;
			std::shared_ptr<Operator> _unitary;
			std::shared_ptr<Operator> _unitary_adjoint;
		};

		class AffineAddition : public ProximalOperator {
		public:

			AffineAddition(const ProximalOperator& prox, const Vector& a, double base_alpha = 1.0);

			Vector _apply(const Vector& input, double alpha);

		private:
			ProximalOperator _prox;
			Vector _a;
		};

		class L2Reg : public ProximalOperator {
		public:

			L2Reg(const std::optional<ProximalOperator>& prox, double rho = 1.0, const std::optional<Vector>& a = std::nullopt, 
				double base_alpha = 1.0);

			Vector _apply(const Vector& input, double alpha);

		private:
			std::optional<ProximalOperator> _prox;
			double _rho;
			std::optional<Vector> _a;
		};

	}
}
