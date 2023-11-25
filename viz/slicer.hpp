#pragma once

#include <torch/torch.h>
#include <future>

import thread_pool;

namespace hasty {

	namespace viz {

		struct OrthoslicerRenderInfo;

		enum SliceView : int {
			eAxial,
			eSagital,
			eCoronal
		};

		struct Slicer {

			Slicer(SliceView view, at::Tensor& tensor);

			at::Tensor GetTensorSlice(const std::vector<int64_t>& startslice,
				const std::array<bool, 3>& flip,
				std::array<int64_t, 3> point);

			void HandleCursor(OrthoslicerRenderInfo& renderInfo);

			void SliceUpdate(OrthoslicerRenderInfo& renderInfo);

			void Render(OrthoslicerRenderInfo& renderInfo);

			SliceView view;
			at::Tensor& tensor;

			std::future<at::Tensor> slice0;
			std::future<at::Tensor> slice1;

			std::string slicername;
			std::string plotname;
			std::string heatname;

			std::array<float, 2> scale_minmax_mult;
		};

	}
}
