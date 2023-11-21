#pragma once

#include <torch/torch.h>

#include "skia.hpp"
#include "orthoslicer.hpp"

import thread_pool;

namespace hasty {
namespace viz {

	class VizApp {
	public:

		VizApp(SkiaContext& skiactx);
		
		void Render();

	private:
		SkiaContext& _skiactx;
		at::Tensor _tensor;
		std::unique_ptr<Orthoslicer> _oslicer;

		std::unique_ptr<ThreadPool> _tpool;

		int bufferidx;
	};


}
}
