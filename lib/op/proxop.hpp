#pragma once

#include "op.hpp"

namespace hasty {
	namespace op {

		class ProximalOpeartor {
		public:

			ProximalOpeartor(double base_alpha=1.0);

			double get_base_alpha();

			void set_base_alpha(double base_alpha);

			Vector apply(const Vector& input, double alpha=1.0);

			virtual Vector _apply(const Vector& input, double alpha);

		private:
			double _base_alpha;
		};

		class ZeroFunc : public ProximalOpeartor {
		public:

			ZeroFunc() = default;

			Vector _apply(const Vector& input, double alpha);

		private:
		};

		class ConvexConjugate : public ProximalOpeartor {
		public:


		private:


		};

	}
}
