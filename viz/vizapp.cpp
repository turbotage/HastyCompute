#include "vizapp.hpp"

import thread_pool;
import hasty_util;

#include <highfive/H5Easy.hpp>

namespace {
	at::Tensor import_tensor() {
		at::InferenceMode imode;
		
		H5Easy::File file("D:\\4DRecon\\dat\\dat2\\images_encs_20f_cropped_interpolated.h5", H5Easy::File::ReadOnly);

		HighFive::DataSet dataset = file.getDataSet("images");

		HighFive::DataType dtype = dataset.getDataType();
		std::string dtype_str = dtype.string();
		size_t dtype_size = dtype.getSize();
		std::vector<int64_t> dims = hasty::util::vector_cast<int64_t>(dataset.getDimensions());
		size_t nelem = dataset.getElementCount();

		std::vector<std::byte> tensorbytes(dataset.getStorageSize());

		at::ScalarType scalar_type;
		if (dtype_str == "Float32") {
			scalar_type = at::ScalarType::Float;
		}
		else if (dtype_str == "Float64") {
			scalar_type = at::ScalarType::Double;
		}
		else if (dtype_str == "Compound64") {
			HighFive::CompoundType ctype(std::move(dtype));
			auto members = ctype.getMembers();
			if (members.size() != 2)
				throw std::runtime_error("HighFive reported an Compound64 type");
			scalar_type = at::ScalarType::ComplexFloat;
		}
		else if (dtype_str == "Compound128") {
			HighFive::CompoundType ctype(std::move(dtype));
			auto members = ctype.getMembers();
			if (members.size() != 2)
				throw std::runtime_error("HighFive reported an Compound64 type");
			scalar_type = at::ScalarType::ComplexDouble;
		}
		else {
			throw std::runtime_error("disallowed dtype");
		}
			
		at::Tensor blobtensor = at::from_blob(tensorbytes.data(), at::makeArrayRef(dims), scalar_type);

		return blobtensor.detach().clone();

	}
}


hasty::viz::VizApp::VizApp(SkiaContext& skiactx)
	: _skiactx(skiactx)
{
	_tensor = at::abs(import_tensor()).contiguous();

	std::cout << "Tensor max: " << _tensor.max() << std::endl;

	_tensor /= _tensor.max();

	_oslicer = std::make_unique<Orthoslicer>(_tensor);
	_tpool = std::make_unique<ThreadPool>(2);
}

void hasty::viz::VizApp::Render()
{
	ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
	ImGuiViewport* viewport = ImGui::GetMainViewport();
	
	Orthoslicer::RenderInfo renderInfo;
	renderInfo.tpool = _tpool.get();
	renderInfo.bufferidx = 0;

	_oslicer->Render(renderInfo);

	renderInfo.bufferidx = renderInfo.bufferidx == 1 ? 0 : 1;
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