#include "orthoslicer.hpp"

import thread_pool;

hasty::viz::Slicer::Slicer(SliceView view, at::Tensor& tensor)
	: view(view), tensor(tensor)
{
}

at::Tensor hasty::viz::Slicer::GetTensorSlice(const std::vector<int64_t>& startslice, int64_t xpoint, int64_t ypoint, int64_t zpoint)
{
	using namespace torch::indexing;
	std::vector<TensorIndex> indices;
	indices.reserve(startslice.size() + 3);
	for (auto s : startslice) {
		indices.push_back(s);
	}

	auto slice = Slice(0, None, 3);

	switch (view) {
	case eAxial:
		indices.push_back(slice); indices.push_back(slice); indices.push_back(zpoint);
		break;
	case eSagital:
		indices.push_back(slice); indices.push_back(ypoint); indices.push_back(slice);
		break;
	case eCoronal:
		indices.push_back(xpoint); indices.push_back(slice); indices.push_back(slice);
		break;
	default:
		throw std::runtime_error("Only eAxial, eSagital, eCoronal are allowed");
	}

	return tensor.index(at::makeArrayRef(indices));
}

hasty::viz::SlicerRenderer::SlicerRenderer(Slicer& slicer)
	: slicer(slicer)
{
	switch (slicer.view) {
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
}


void hasty::viz::SlicerRenderer::HandleAxialCursor(RenderInfo& renderInfo)
{
	if (ImPlot::IsPlotHovered() && ImGui::IsKeyPressed(ImGuiKey_MouseLeft)) {
		ImPlotPoint mousepos = ImPlot::GetPlotMousePos();
		//std::cout << "Hovered  " << "x: " << mousepos.x << " y: " << mousepos.y << std::endl;

		printf("Hovered");

		renderInfo.newxpoint = mousepos.x;
		renderInfo.newypoint = mousepos.y;
	}

	if (renderInfo.plot_cursor_lines) {
		ImPlot::PlotInfLines("##Vertical", &renderInfo.xpoint, 1);
		ImPlot::PlotInfLines("##Horizontal", &renderInfo.ypoint, 1, ImPlotInfLinesFlags_Horizontal);
	}

}

void hasty::viz::SlicerRenderer::HandleSagitalCursor(RenderInfo& renderInfo)
{
	if (ImPlot::IsPlotHovered() && ImGui::IsKeyPressed(ImGuiKey_MouseLeft)) {
		ImPlotPoint mousepos = ImPlot::GetPlotMousePos();
		renderInfo.newxpoint = mousepos.x;
		renderInfo.newzpoint = mousepos.y;
	}
}

void hasty::viz::SlicerRenderer::HandleCoronalCursor(RenderInfo& renderInfo)
{
	if (ImPlot::IsPlotHovered() && ImGui::IsKeyPressed(ImGuiKey_MouseLeft)) {
		ImPlotPoint mousepos = ImPlot::GetPlotMousePos();
		renderInfo.newypoint = mousepos.x;
		renderInfo.newzpoint = mousepos.y;
	}

}

void hasty::viz::SlicerRenderer::HandleCursor(RenderInfo& renderInfo)
{
	switch (slicer.view) {
	case eAxial:
		HandleAxialCursor(renderInfo);
		break;
	case eSagital:
		HandleSagitalCursor(renderInfo);
		break;
	case eCoronal:
		HandleCoronalCursor(renderInfo);
		break;
	default:
		throw std::runtime_error("Only eAxial, eSagital, eCoronal are allowed");
	}
}

void hasty::viz::SlicerRenderer::SliceUpdate(RenderInfo& renderInfo)
{

	if ((renderInfo.bufferidx != 0) && (renderInfo.bufferidx != 1))
	{
		auto slice_lambda = [&slicer_ref = slicer, p = renderInfo.preslices]() mutable
			{
				return slicer_ref.GetTensorSlice(p, 0, 0, 0).contiguous();
			};

		slice0 = std::async(std::launch::async, slice_lambda);
		slice1 = std::async(std::launch::async, slice_lambda);
		return;
	}

	//auto& slicer_ref = slicer;
	auto slice_lambda =	[&slicer_ref = slicer, p = renderInfo.preslices, xpoint = renderInfo.xpoint,
		ypoint = renderInfo.ypoint, zpoint = renderInfo.zpoint]() mutable
		{
			try {
				return slicer_ref.GetTensorSlice(p, xpoint, ypoint, zpoint).contiguous();
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

void hasty::viz::SlicerRenderer::Render(RenderInfo& renderInfo)
{
	at::Tensor slice = renderInfo.bufferidx == 0 ? slice1.get() : slice0.get();

	if (ImGui::Begin(slicername.c_str())) {

		auto windowsize = ImGui::GetWindowSize();
		uint32_t tensor_width = slice.size(0); 
		uint32_t tensor_height = slice.size(1);

		float window_width = windowsize.x * 0.95f;
		float window_height = window_width * (tensor_height / tensor_width);
		if (window_height > windowsize.y * 0.95f) {
			window_height = windowsize.y * 0.95f;
			window_width = window_height * (tensor_width / tensor_height);
		}


		ImGui::SetNextItemWidth(window_width);
		ImGui::DragFloatRange2("Min Multiplier / Max Multiplier", &scale_min_mult, &scale_max_mult, 0.01f, -10.0f, 10.0f);

		ImPlot::PushColormap(renderInfo.map);

		float minscale = renderInfo.min_scale * scale_min_mult;
		float maxscale = renderInfo.max_scale * scale_max_mult;
		if (minscale > maxscale)
			minscale = maxscale - 0.1f;
		
		int rows = slice.size(0);
		int cols = slice.size(1);

		static ImPlotAxisFlags axes_flags = 
			ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines | 
			ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoLabel;

		static ImPlotFlags plot_flags = ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText;
		if (ImPlot::BeginPlot(plotname.c_str(), ImVec2(window_width, window_height), plot_flags)) {
			ImPlot::SetupAxes(nullptr, nullptr, axes_flags, axes_flags);
			ImPlot::SetupAxesLimits(0, cols, 0, rows);
			static ImPlotHeatmapFlags hm_flags = 0;
			ImPlot::PlotHeatmap(heatname.c_str(), static_cast<const float*>(slice.const_data_ptr()), rows, cols,
				minscale, maxscale, nullptr, ImPlotPoint(0, 0), ImPlotPoint(cols, rows), hm_flags);
			HandleCursor(renderInfo);

			ImPlot::EndPlot();
		}
		ImGui::SameLine();
		ImPlot::ColormapScale("##HeatScale", minscale, maxscale, ImVec2(50, window_height));

		ImGui::SameLine();

		ImPlot::PopColormap();

	}

	ImGui::End();

	SliceUpdate(renderInfo);
}

hasty::viz::Orthoslicer::Orthoslicer(at::Tensor tensor)
	: _tensor(tensor),
	_axialSlicer(SliceView::eAxial, _tensor),
	_sagitalSlicer(SliceView::eSagital, _tensor),
	_coronalSlicer(SliceView::eCoronal, _tensor),
	_axialSlicerRenderer(_axialSlicer),
	_sagitalSlicerRenderer(_sagitalSlicer),
	_coronalSlicerRenderer(_coronalSlicer)
{

	auto ndim = tensor.ndimension();
	std::vector<int64_t> preslices(ndim - 3, 0);
	_renderInfo.preslices = std::move(preslices);

	_renderInfo.xpoint = 0;
	_renderInfo.ypoint = 0;
	_renderInfo.zpoint = 0;

	_renderInfo.newxpoint = 0;
	_renderInfo.newypoint = 0;
	_renderInfo.newzpoint = 0;

	_renderInfo.min_scale = -1.0f;
	_renderInfo.max_scale = 1.0f;

	_renderInfo.map = ImPlotColormap_Viridis;

	_renderInfo.plot_cursor_lines = true;

	_renderInfo.bufferidx = -1;
	_renderInfo.tpool = nullptr;

	_axialSlicerRenderer.SliceUpdate(_renderInfo);
	_sagitalSlicerRenderer.SliceUpdate(_renderInfo);
	_coronalSlicerRenderer.SliceUpdate(_renderInfo);
}

void hasty::viz::Orthoslicer::Render(const RenderInfo& rinfo)
{
	_renderInfo.bufferidx = rinfo.bufferidx;
	_renderInfo.tpool = rinfo.tpool;

	ImGui::ShowMetricsWindow();

	if (ImGui::Begin("Orthoslicer")) {

		ImGuiWindowClass slicer_class;
		slicer_class.ClassId = ImGui::GetID("SlicerWindowClass");
		slicer_class.DockingAllowUnclassed = false;

		static ImGuiID dockid = ImGui::GetID("Orthoslicer Dockspace");
		ImGui::DockSpace(dockid, ImVec2(0.0f, 0.0f), 0, &slicer_class);

		ImGui::SetNextWindowClass(&slicer_class);
		_axialSlicerRenderer.Render(_renderInfo);
		ImGui::SetNextWindowClass(&slicer_class);
		_sagitalSlicerRenderer.Render(_renderInfo);
		ImGui::SetNextWindowClass(&slicer_class);
		_coronalSlicerRenderer.Render(_renderInfo);

		//ImGui::EndChild();
	}

	ImGui::End();

	_renderInfo.xpoint = _renderInfo.newxpoint;
	_renderInfo.ypoint = _renderInfo.newypoint;
	_renderInfo.zpoint = _renderInfo.newzpoint;
}

