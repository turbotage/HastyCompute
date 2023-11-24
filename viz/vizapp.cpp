#include "vizapp.hpp"
#include "orthoslicer.hpp"
#include "slicer.hpp"

import thread_pool;
import hasty_util;
import hdf5;


hasty::viz::VizApp::VizApp(SkiaContext& skiactx)
	: _skiactx(skiactx)
{
	_tensor = at::abs(hasty::import_tensor(
		"D:\\4DRecon\\dat\\dat2\\my_full_real_45.h5",
		"image"
	)).contiguous();

	_tensor /= _tensor.max();

	_oslicer = std::make_unique<Orthoslicer>(_tensor);
	_tpool = std::make_unique<ThreadPool>(2);

	_renderinfo.tpool = _tpool.get();
	_renderinfo.bufferidx = 0;
}

void hasty::viz::VizApp::Render()
{
	ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
	ImGuiViewport* viewport = ImGui::GetMainViewport();

	_oslicer->Render(_renderinfo);

	_renderinfo.bufferidx = _renderinfo.bufferidx == 1 ? 0 : 1;
}












/*
void hasty::viz::Application::Render(ImGuiIO& io)
{
	ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
	
	// 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
	bool show_demo_window = true;
	bool show_another_window = false;
	// Our state
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	ImGui::ShowDemoWindow(&show_demo_window);

	{

	}

	// 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
	{
		static float f = 0.0f;
		static int counter = 0;

		ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

		ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
		ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
		ImGui::Checkbox("Another Window", &show_another_window);

		ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
		ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

		if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
			counter++;
		ImGui::SameLine();
		ImGui::Text("counter = %d", counter);

		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
		ImGui::End();
	}

	// 3. Show another simple window.
	if (show_another_window)
	{
		ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
		ImGui::Text("Hello from another window!");
		if (ImGui::Button("Close Me"))
			show_another_window = false;
		ImGui::End();
	}
}


*/