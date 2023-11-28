#include "slicer.hpp"
#include "orthoslicer.hpp"

#include <algorithm>

import thread_pool;

at::Tensor hasty::viz::Slicer::GetTensorSlice(const std::vector<int64_t>& preslice,
	const std::array<bool, 3>& flip,
	std::array<int64_t, 3> point)
{
	int64_t ndim = tensor.ndimension();
	for (int i = 0; i < 3; ++i) {
		if (flip[i]) {
			point[i] = tensor.size(ndim - 3 + i) - point[i];
		}
	}

	using namespace torch::indexing;
	std::vector<TensorIndex> indices;
	indices.reserve(preslice.size() + 3);
	for (auto s : preslice) {
		indices.push_back(s);
	}

	auto slice = Slice();

	switch (view) {
	case eAxial:
		indices.push_back(slice); indices.push_back(slice); indices.push_back(point[2]);
		break;
	case eCoronal:
		indices.push_back(slice); indices.push_back(point[1]); indices.push_back(slice);
		break;
	case eSagital:
		indices.push_back(point[0]); indices.push_back(slice); indices.push_back(slice);
		break;
	default:
		throw std::runtime_error("Only eAxial, eSagital, eCoronal are allowed");
	}

	return tensor.index(at::makeArrayRef(indices));
}

hasty::viz::Slicer::Slicer(SliceView view, at::Tensor& tensor)
	: view(view), tensor(tensor)
{
	switch (view) {
	case SliceView::eAxial:
		slicername = "Axial Slicer";
		plotname = "##Axial Plot";
		heatname = "##Axial Heat";
		break;
	case SliceView::eSagital:
		slicername = "Sagital Slicer";
		plotname = "##Sagital Plot";
		heatname = "##Sagital Heat";
		break;
	case SliceView::eCoronal:
		slicername = "Coronal Slicer";
		plotname = "##Coronal Plot";
		heatname = "##Coronal Heat";
		break;
	default:
		throw std::runtime_error("Only eAxial, eSagital, eCoronal are allowed");
	}

	scale_minmax_mult = { 1.0, 1.0 };

}


void hasty::viz::Slicer::HandleCursor(OrthoslicerRenderInfo& renderInfo)
{
	auto& flip = renderInfo.flip;
	auto& tlen = renderInfo.tensorlen;
	auto& point = renderInfo.point;
	auto& nextpoint = renderInfo.nextpoint;

	if (ImPlot::IsPlotHovered() && ImGui::IsKeyPressed(ImGuiKey_MouseLeft)) {
		ImPlotPoint mousepos = ImPlot::GetPlotMousePos();
		int64_t mousex = static_cast<int64_t>(mousepos.x);
		int64_t mousey = static_cast<int64_t>(mousepos.y);

		switch (view) {
		case eAxial:
		{
			nextpoint[0] = tlen[0] - mousey;
			nextpoint[1] = mousex;
			nextpoint[0] = std::clamp(nextpoint[0], int64_t(0), tlen[0]-1);
			nextpoint[1] = std::clamp(nextpoint[1], int64_t(0), tlen[1]-1);
		}
		break;
		case eCoronal:
		{
			nextpoint[0] = tlen[0] - mousey;
			nextpoint[2] = mousex;
			nextpoint[0] = std::clamp(nextpoint[0], int64_t(0), tlen[0]-1);
			nextpoint[2] = std::clamp(nextpoint[2], int64_t(0), tlen[2]-1);
		}
		break;
		case eSagital:
		{
			nextpoint[1] = tlen[1] - mousey;
			nextpoint[2] = mousex;
			nextpoint[1] = std::clamp(nextpoint[1], int64_t(0), tlen[1]-1);
			nextpoint[2] = std::clamp(nextpoint[2], int64_t(0), tlen[2]-1);
		}
		break;
		default:
			throw std::runtime_error("Unsupported view");
		}

	}

	if (renderInfo.plot_cursor_lines) {
		int64_t xlinepos, ylinepos;
		
		switch (view) {
		case eAxial:
			xlinepos = point[1];
			ylinepos = tlen[0] - point[0];
			break;
		case eCoronal:
			xlinepos = point[2];
			ylinepos = tlen[0] - point[0];
			break;
		case eSagital:
			xlinepos = point[2];
			ylinepos = tlen[1] - point[1];
			break;
		default:
			throw std::runtime_error("Unsupported view");
		}

		ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0, 0.0, 0.0, 0.5));
		ImPlot::PlotInfLines("##Vertical", &xlinepos, 1);
		ImPlot::PlotInfLines("##Horizontal", &ylinepos, 1, ImPlotInfLinesFlags_Horizontal);
		ImPlot::PopStyleColor();
	}
}

void hasty::viz::Slicer::SliceUpdate(OrthoslicerRenderInfo& renderInfo)
{
	auto preslices = renderInfo.preslices;
	auto flip = renderInfo.flip;
	auto point = renderInfo.point;

	if ((renderInfo.bufferidx != 0) && (renderInfo.bufferidx != 1))
	{
		auto slice_lambda = [this, preslices]() mutable
		{
				return GetTensorSlice(preslices, { false, false, false },  {0, 0, 0}).contiguous();
		};

		slice0 = std::async(std::launch::async, slice_lambda);
		slice1 = std::async(std::launch::async, slice_lambda);
		return;
	}

	auto slice_lambda = [this, preslices, flip, point]() mutable
	{
		try {
			return GetTensorSlice(preslices, flip, point).contiguous();
		}
		catch (...) {
			std::cout << "hello";
		}
	};

	if (renderInfo.bufferidx == 0) {
		slice0 = renderInfo.tpool->enqueue(std::move(slice_lambda));
	}
	else {
		slice1 = renderInfo.tpool->enqueue(std::move(slice_lambda));
	}

}

void hasty::viz::Slicer::Render(OrthoslicerRenderInfo& renderInfo)
{
	at::Tensor slice = renderInfo.bufferidx == 0 ? slice1.get() : slice0.get();

	if (ImGui::Begin(slicername.c_str())) {

		ImVec2 windowsize;
		uint32_t tensor_width;
		uint32_t tensor_height;
		float window_width;
		float window_height;

		// Get/Set Window Widths and Heights
		float window_multiplier = 0.95f;
		{
			windowsize = ImGui::GetWindowSize();
			tensor_width = slice.size(0);
			tensor_height = slice.size(1);
			window_width = windowsize.x * window_multiplier;
			window_height = window_width * (tensor_height / tensor_width);
			float width_offset = 10.0f;
			float height_offset = 10.0f;
			if (window_height > windowsize.y * window_multiplier) {
				window_height = windowsize.y * window_multiplier;
				window_width = window_height * (tensor_width / tensor_height);
			}
		}

		ImPlot::PushColormap(renderInfo.map);

		float minscale = renderInfo.current_minmax_scale[0] * scale_minmax_mult[0];
		float maxscale = renderInfo.current_minmax_scale[1] * scale_minmax_mult[1];
		if (minscale > maxscale)
			minscale = maxscale - 0.1f;

		int rows = slice.size(0);
		int cols = slice.size(1);


		static ImPlotFlags plot_flags = ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText | ImPlotFlags_NoMenus;
		if (ImPlot::BeginPlot(plotname.c_str(), ImVec2(window_width, window_height), plot_flags)) {
			static ImPlotAxisFlags axes_flags =
				ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines |
				ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoLabel;
			ImPlot::SetupAxis(ImAxis_X1, nullptr, axes_flags);
			ImPlot::SetupAxis(ImAxis_Y1, nullptr, axes_flags);
			ImPlot::SetupAxisTicks(ImAxis_X1, nullptr, 0, nullptr, false);
			ImPlot::SetupAxisTicks(ImAxis_Y1, nullptr, 0, nullptr, false);
			ImPlot::SetupAxesLimits(0, cols, 0, rows);

			static ImPlotHeatmapFlags hm_flags = 0; //ImPlotHeatmapFlags_ColMajor;
			ImPlot::PlotHeatmap(heatname.c_str(), static_cast<const float*>(slice.const_data_ptr()), rows, cols,
				minscale, maxscale, nullptr, ImPlotPoint(0, 0), ImPlotPoint(cols, rows), hm_flags);

			HandleCursor(renderInfo);
			
			ImPlot::EndPlot();
		}
		ImPlot::PopColormap();

		ImGui::SetNextItemWidth(window_width * window_multiplier);
		ImGui::DragFloatRange2("Multipliers", &scale_minmax_mult[0], &scale_minmax_mult[1], 0.01f, -10.0f, 10.0f);

	}

	ImGui::End();

	SliceUpdate(renderInfo);
}


