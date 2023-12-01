module;

#include <torch/torch.h>

export module device;

import torch_util;

namespace hasty {

	export class DeviceContext {
	public:
		virtual const at::Stream& stream() const = 0;
	};

	export template<typename T>
	concept DeviceContextConcept = std::derived_from<T, DeviceContext>;

}



