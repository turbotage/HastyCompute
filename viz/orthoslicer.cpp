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

		/*
		auto position_text = "x: " + std::to_string(_renderInfo.point[0]) + " y: " + std::to_string(_renderInfo.point[1])
			+ " z: " + std::to_string(_renderInfo.point[2]);
		ImGui::Text(position_text.c_str());
		*/

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

		static int pos[3] = { 0, 0, 0 };
		pos[0] = (int)_renderInfo.point[0];
		pos[1] = (int)_renderInfo.point[1];
		pos[2] = (int)_renderInfo.point[2];
		if (ImGui::InputInt3("Position", pos)) {

			pos[0] = std::clamp(pos[0], 0, (int)_renderInfo.tensorlen[0]-1);
			pos[1] = std::clamp(pos[1], 0, (int)_renderInfo.tensorlen[1]-1);
			pos[2] = std::clamp(pos[2], 0, (int)_renderInfo.tensorlen[2]-1);

			_renderInfo.nextpoint[0] = pos[0];
			_renderInfo.nextpoint[1] = pos[1];
			_renderInfo.nextpoint[2] = pos[2];
		}


	}
	ImGui::End();
}

void hasty::viz::Orthoslicer::RenderPreslicer()
{
	using namespace at::indexing;
	std::vector<TensorIndex> indices;
	indices.reserve(4);
	indices.push_back("...");
	indices.push_back(_renderInfo.point[0]);
	indices.push_back(_renderInfo.point[1]);
	indices.push_back(_renderInfo.point[2]);

	at::Tensor slice = _tensor.index(at::makeArrayRef(indices)).contiguous();

	int64_t rows = slice.size(0);
	int64_t cols = slice.size(1);

	if (ImGui::Begin("Preslicer")) {

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

		ImPlot::PushColormap(ImPlotColormap_Greys);

		static ImPlotFlags plot_flags = ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText | ImPlotFlags_NoMenus;
		if (ImPlot::BeginPlot("##Preslice", ImVec2(window_width, window_height), plot_flags)) {
			static ImPlotAxisFlags axes_flags =
				ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines |
				ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoLabel;
			ImPlot::SetupAxis(ImAxis_Y1, nullptr, axes_flags);
			ImPlot::SetupAxis(ImAxis_X1, nullptr, axes_flags);
			ImPlot::SetupAxisTicks(ImAxis_X1, nullptr, 0, nullptr, false);
			ImPlot::SetupAxisTicks(ImAxis_Y1, nullptr, 0, nullptr, false);
			ImPlot::SetupAxesLimits(0, cols, 0, rows);

			static ImPlotHeatmapFlags hm_flags = 0; //ImPlotHeatmapFlags_ColMajor;
			ImPlot::PlotHeatmap("##PreslicerHeatmap", static_cast<const float*>(slice.const_data_ptr()), rows, cols,
				0, 0, nullptr, ImPlotPoint(0, 0), ImPlotPoint(cols, rows), hm_flags);

			if (ImPlot::IsPlotHovered() && ImGui::IsKeyPressed(ImGuiKey_MouseLeft)) {
				ImPlotPoint mousepos = ImPlot::GetPlotMousePos();
				int64_t mousex = static_cast<int64_t>(mousepos.x);
				int64_t mousey = rows - static_cast<int64_t>(mousepos.y);

				_renderInfo.preslices = { std::clamp(mousey, int64_t(0), rows - 1), std::clamp(mousex, int64_t(0), cols-1) };
			
				/*
				double rect[4];
				rect[0] = _renderInfo.preslices[0];
				rect[1] = _renderInfo.preslices[1];
				rect[2] = windowsize.x / cols;
				rect[3] = windowsize.y / rows;
				ImPlot::DragRect(ImGui::GetID("##PreslicerRect"), &rect[0], &rect[1], &rect[2], &rect[3],
					ImVec4(0.0, 1.0, 0.0, 0.5), ImPlotDragToolFlags_NoInputs);

				*/
				ImDrawList* draw_list = ImGui::GetForegroundDrawList();

			}

			ImPlot::EndPlot();
		}

		ImPlot::PopColormap();

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
		
		ImGui::SetNextWindowClass(&ortho_class);
		RenderPreslicer();

	}

	ImGui::End();

	_renderInfo.point = _renderInfo.nextpoint;
}

