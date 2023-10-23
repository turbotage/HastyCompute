#pragma once

#include "../torch_util.hpp"
#include "../op/op.hpp"

namespace hasty {
	namespace mri {
		
		void stacked_equal_cg(op::Vector& x0, const op::Vector& coord, const op::Vector& kdata,
			const op::Vector& circ_p, const op::Vector& offset);


	}
}

