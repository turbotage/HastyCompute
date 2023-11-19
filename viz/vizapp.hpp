#pragma once

#include <imgui.h>
#include <torch/torch.h>

#include "skia.hpp"
#include "orthoslicer.hpp"

namespace hasty {
namespace viz {

	class VizApp {
	public:

		VizApp(std::shared_ptr<SkiaContext> skiactx);
		
		void Render();

	private:
		std::shared_ptr<SkiaContext> _skiactx;
		at::Tensor _tensor;
		std::unique_ptr<Orthoslicer> _oslicer;
	};


}
}
