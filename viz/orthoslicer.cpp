#include "orthoslicer.hpp"
#include "vizapp.hpp"

import thread_pool;


hasty::viz::Orthoslicer::Orthoslicer(at::Tensor tensor)
	: _tensor(tensor),
	_axialSlicer(std::make_unique<Slicer>(SliceView::eAxial, _tensor)),
	_sagitalSlicer(std::make_unique<Slicer>(SliceView::eSagital, _tensor)),
	_coronalSlicer(std::make_unique<Slicer>(SliceView::eCoronal, _tensor))
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

	_renderInfo.min_scale = _tensor.min().item<float>();
	_renderInfo.max_scale = _tensor.max().item<float>();
	_renderInfo.current_min_scale = _renderInfo.min_scale;
	_renderInfo.current_max_scale = _renderInfo.max_scale;

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

void hasty::viz::Orthoslicer::RenderGlobalOptions()
{
	if (ImGui::Begin("Global Slicer Options")) {
		auto windowsize = ImGui::GetWindowSize();

		if (ImPlot::ColormapButton(ImPlot::GetColormapName(_renderInfo.map), ImVec2(windowsize.x, 0), _renderInfo.map))
		{
			_renderInfo.map = (_renderInfo.map + 1) % ImPlot::GetColormapCount();
			ImPlot::BustColorCache(_axialSlicer->heatname.c_str());
			ImPlot::BustColorCache(_sagitalSlicer->heatname.c_str());
			ImPlot::BustColorCache(_coronalSlicer->heatname.c_str());
		}

		ImGui::SetNextItemWidth(windowsize.x);
		ImGui::DragFloatRange2("GLobal Min / Max",
			&_renderInfo.current_min_scale, &_renderInfo.current_max_scale,
			(_renderInfo.max_scale - _renderInfo.min_scale) / 1000.0f,
			_renderInfo.current_min_scale, _renderInfo.current_max_scale);

	}
	ImGui::End();
}

void hasty::viz::Orthoslicer::Render(const VizAppRenderInfo& rinfo)
{
	_renderInfo.bufferidx = rinfo.bufferidx;
	_renderInfo.tpool = rinfo.tpool;

	ImGui::ShowMetricsWindow();

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

	_renderInfo.xpoint = _renderInfo.newxpoint;
	_renderInfo.ypoint = _renderInfo.newypoint;
	_renderInfo.zpoint = _renderInfo.newzpoint;
}

