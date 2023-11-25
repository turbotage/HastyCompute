#include "orthoslicer.hpp"
#include "vizapp.hpp"

#include "implot_internal.h"

import thread_pool;


hasty::viz::Orthoslicer::Orthoslicer(at::Tensor tensor)
	: _tensor(tensor),
	_axialSlicer(std::make_unique<Slicer>(SliceView::eAxial, _tensor)),
	_sagitalSlicer(std::make_unique<Slicer>(SliceView::eSagital, _tensor)),
	_coronalSlicer(std::make_unique<Slicer>(SliceView::eCoronal, _tensor))
{
	int64_t ndim = tensor.ndimension();
	std::vector<int64_t> preslices(ndim - 3, 0);
	_renderInfo.preslices = std::move(preslices);
	_renderInfo.tensorlen = { _tensor.size(ndim - 3), _tensor.size(ndim - 2), _tensor.size(ndim - 1) };
	_renderInfo.point = { 0, 0, 0 };
	_renderInfo.nextpoint = { 0, 0, 0 };
	_renderInfo.flip = { false, false, false };

	_renderInfo.minmax_scale = { _tensor.min().item<float>(), _tensor.max().item<float>() };
	_renderInfo.current_minmax_scale = _renderInfo.minmax_scale;

	_renderInfo.map = ImPlotColormap_Viridis;
	_renderInfo.plot_cursor_lines = true;

	_renderInfo.bufferidx = -1;
	_renderInfo.tpool = nullptr;

	_axialSlicer->SliceUpdate(_renderInfo);
	_sagitalSlicer->SliceUpdate(_renderInfo);
	_coronalSlicer->SliceUpdate(_renderInfo);
}

void hasty::viz::Orthoslicer::RenderSlicerViewport()
{
	if (ImGui::Begin("Slicers")) {

		auto windowsize = ImGui::GetWindowSize();

		ImPlot::PushColormap(_renderInfo.map);
		ImPlot::ColormapScale("##HeatScale", _renderInfo.current_minmax_scale[0], _renderInfo.current_minmax_scale[1], ImVec2(80, windowsize.y * 0.95f));
		ImPlot::PopColormap();

		ImGui::SameLine();

		ImGuiWindowClass slicer_class;
		slicer_class.ClassId = ImGui::GetID("SlicerWindowClass");
		slicer_class.DockingAllowUnclassed = false;

		static ImGuiID slicer_dockid = ImGui::GetID("Slicers Dockspace");
		ImGui::DockSpace(slicer_dockid, ImVec2(0.0f, 0.0f), 0, &slicer_class);

		ImGui::SetNextWindowClass(&slicer_class);
		_axialSlicer->Render(_renderInfo);
		ImGui::SetNextWindowClass(&slicer_class);
		_sagitalSlicer->Render(_renderInfo);
		ImGui::SetNextWindowClass(&slicer_class);
		_coronalSlicer->Render(_renderInfo);

	}
	ImGui::End();
}

void hasty::viz::Orthoslicer::RenderColormapSelector() {
	auto& map = _renderInfo.map;
	auto windowsize = ImGui::GetWindowSize();
	if (ImGui::BeginCombo("Colormap Selector", ImPlot::GetColormapName(map))) {
		for (int i = 0; i < ImPlot::GetColormapCount(); ++i) {

			if (ImPlot::ColormapButton(ImPlot::GetColormapName(i), ImVec2(windowsize.x*0.75, 0), i)) {
				map = i;
				ImPlot::BustColorCache(_axialSlicer->heatname.c_str());
				ImPlot::BustColorCache(_sagitalSlicer->heatname.c_str());
				ImPlot::BustColorCache(_coronalSlicer->heatname.c_str());
			}
		}
		ImGui::EndCombo();
	}
}

void hasty::viz::Orthoslicer::RenderGlobalOptions()
{
	if (ImGui::Begin("Global Slicer Options")) {

		auto windowsize = ImGui::GetWindowSize();

		RenderColormapSelector();

		ImGui::SetNextItemWidth(0.75 * windowsize.x);
		ImGui::DragFloatRange2("Global Min / Max",
			&_renderInfo.current_minmax_scale[0], &_renderInfo.current_minmax_scale[1],
			(_renderInfo.minmax_scale[1] - _renderInfo.minmax_scale[0]) / 1000.0f,
			_renderInfo.minmax_scale[0], _renderInfo.minmax_scale[1]);

		auto position_text = "x: " + std::to_string(_renderInfo.point[0]) + " y: " + std::to_string(_renderInfo.point[1])
			+ " z: " + std::to_string(_renderInfo.point[2]);

		ImGui::Text(position_text.c_str());

		if (ImGui::RadioButton("Flip X", _renderInfo.flip[0])) {
			_renderInfo.flip[0] = !_renderInfo.flip[0];
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("Flip Y", _renderInfo.flip[1])) {
			_renderInfo.flip[1] = !_renderInfo.flip[1];
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("Flip Z", _renderInfo.flip[2])) {
			_renderInfo.flip[2] = !_renderInfo.flip[2];
		}

	}
	ImGui::End();
}

void hasty::viz::Orthoslicer::Render(const VizAppRenderInfo& rinfo)
{
	_renderInfo.bufferidx = rinfo.bufferidx;
	_renderInfo.tpool = rinfo.tpool;

	// Actual Orthoslicer Window
	if (ImGui::Begin("Orthoslicer")) {

		ImGuiWindowClass ortho_class;
		ortho_class.ClassId = ImGui::GetID("OrthoWindowClass");
		ortho_class.DockingAllowUnclassed = false;

		static ImGuiID ortho_dockid = ImGui::GetID("Orthoslicer Dockspace");
		ImGui::DockSpace(ortho_dockid, ImVec2(0, 0), 0, &ortho_class);

		// Slicers Viewport
		ImGui::SetNextWindowClass(&ortho_class);
		RenderSlicerViewport();

		// Global Settings Viewport
		ImGui::SetNextWindowClass(&ortho_class);
		RenderGlobalOptions();
		
	}

	ImGui::End();

	_renderInfo.point = _renderInfo.nextpoint;
}

