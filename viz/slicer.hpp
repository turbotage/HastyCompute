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

			at::Tensor GetTensorSlice(const std::vector<int64_t>& startslice, int64_t xpoint, int64_t ypoint, int64_t zpoint);

			void HandleAxialCursor(OrthoslicerRenderInfo& renderInfo);

			void HandleSagitalCursor(OrthoslicerRenderInfo& renderInfo);

			void HandleCoronalCursor(OrthoslicerRenderInfo& renderInfo);

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

			float scale_min_mult = 1.0f;
			float scale_max_mult = 1.0f;
		};

	}
}
