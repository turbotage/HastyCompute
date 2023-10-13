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

	}
}
