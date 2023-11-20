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

	switch (view) {
	case eAxial:
		indices.push_back(Slice()); indices.push_back(Slice()); indices.push_back(zpoint);
		break;
	case eSagital:
		indices.push_back(Slice()); indices.push_back(ypoint); indices.push_back(Slice());
		break;
	case eCoronal:
		indices.push_back(xpoint); indices.push_back(Slice()); indices.push_back(Slice());
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

void hasty::viz::SlicerRenderer::Render(RenderInfo& renderInfo)
{
	auto beg = std::chrono::high_resolution_clock::now();
	if (renderInfo.doubleBuffer) {
		slice0 = slicer.GetTensorSlice({ 0, 0 }, 40, 40, 40).contiguous();
	}
	else {
		slice1 = slicer.GetTensorSlice({ 0, 0 }, 40, 40, 40).contiguous();
	}
	auto end1 = std::chrono::high_resolution_clock::now();
	auto duration = duration_cast<std::chrono::milliseconds>(end1 - beg);
	printf("Time: %d\n", duration.count());

	at::Tensor& slice = renderInfo.doubleBuffer ? slice0 : slice1;

	if (ImGui::Begin(slicername.c_str())) {

		auto windowsize = ImGui::GetWindowSize();
		uint32_t tensor_width = slice.size(0); uint32_t tensor_height = slice.size(1);
		float hw_tensor_ratio = tensor_height / tensor_width;

		float window_width = windowsize.x * 0.95f;
		float window_height = window_width * hw_tensor_ratio;

		static ImPlotHeatmapFlags hm_flags = 0;

		ImGui::SetNextItemWidth(window_width);
		ImGui::DragFloatRange2("Min Multiplier / Max Multiplier", &scale_min_mult, &scale_max_mult, 0.01f, -10.0f, 10.0f);

		ImPlot::PushColormap(renderInfo.map);

		float minscale = renderInfo.min_scale * scale_min_mult;
		float maxscale = renderInfo.max_scale * scale_max_mult;
		if (minscale > maxscale)
			minscale = maxscale - 0.1f;
		

		static ImPlotAxisFlags axes_flags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;
		if (ImPlot::BeginPlot(plotname.c_str(), ImVec2(window_width, window_height), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
			ImPlot::SetupAxes(nullptr, nullptr, axes_flags, axes_flags);
			ImPlot::PlotHeatmap(heatname.c_str(), (const float*)slice.const_data_ptr(), slice.size(0), slice.size(1),
				minscale, maxscale, nullptr, ImPlotPoint(0, 0), ImPlotPoint(slice.size(0), slice.size(1)), hm_flags);

			HandleCursor(renderInfo);

			ImPlot::EndPlot();
		}
		ImGui::SameLine();
		ImPlot::ColormapScale("##HeatScale", minscale, maxscale, ImVec2(0.05 * windowsize.x, window_height));

		ImGui::SameLine();

		ImPlot::PopColormap();
	}


	ImGui::End();
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

	_renderInfo.min_scale = -1.0f;
	_renderInfo.max_scale = 1.0f;

}

void hasty::viz::Orthoslicer::Render(const RenderInfo& rinfo)
{
	_renderInfo.doubleBuffer = rinfo.doubleBuffer;
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

