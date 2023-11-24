#pragma once

#include <torch/torch.h>

#include "skia.hpp"
#include "orthoslicer.hpp"

import thread_pool;

namespace hasty {
namespace viz {

	struct VizAppRenderInfo {
		ThreadPool* tpool;
		uint16_t bufferidx;
	};

	class VizApp {
	public:

		VizApp(SkiaContext& skiactx);
		
		void Render();

	private:
		SkiaContext& _skiactx;
		at::Tensor _tensor;
		VizAppRenderInfo _renderinfo;

		std::unique_ptr<Orthoslicer> _oslicer;

		std::unique_ptr<ThreadPool> _tpool;

		int bufferidx;
	};


}
}
