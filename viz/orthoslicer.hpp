#pragma once

#include "base/VulkanBase.hpp"

#include <torch/torch.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <implot.h>

#include "skia.hpp"

namespace hasty {
	namespace viz {

		enum SliceView : int {
			eAxial,
			eSagital,
			eCoronal
		};

		struct Slicer {

			Slicer(SliceView view, at::Tensor& tensor)
				: view(view), tensor(tensor)
			{

			}

			at::Tensor GetTensorSlice(const std::vector<int64_t>& startslice, int64_t xpoint, int64_t ypoint, int64_t zpoint)
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

			SliceView view;

			at::Tensor& tensor;

		};

		struct SliceRenderInfo {

			std::vector<int64_t> preslices;

			float xpoint;
			float ypoint;
			float zpoint;

			float newxpoint;
			float newypoint;
			float newzpoint;

			float min_scale = 0.0;
			float max_scale = 1.0;
			ImPlotColormap map = ImPlotColormap_Viridis;

			bool plot_cursor_lines = true;

			bool doubleBuffer;
		};

		struct SlicerRenderer {

			SlicerRenderer(Slicer& slicer)
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

			void HandleAxialCursor(SliceRenderInfo& renderInfo)
			{
				if (ImPlot::IsPlotHovered() && ImGui::IsKeyPressed(ImGuiKey_MouseLeft)) {
					ImPlotPoint mousepos = ImPlot::GetPlotMousePos();
					//std::cout << "Hovered  " << "x: " << mousepos.x << " y: " << mousepos.y << std::endl;

					printf("Hovered");

					renderInfo.newxpoint = mousepos.x;
					renderInfo.newypoint = mousepos.y;
				}

			}

			void HandleSagitalCursor(SliceRenderInfo& renderInfo)
			{
				if (ImPlot::IsPlotHovered() && ImGui::IsKeyPressed(ImGuiKey_MouseLeft)) {
					ImPlotPoint mousepos = ImPlot::GetPlotMousePos();
					renderInfo.newxpoint = mousepos.x;
					renderInfo.newzpoint = mousepos.y;
				}
			}

			void HandleCoronalCursor(SliceRenderInfo& renderInfo)
			{
				if (ImPlot::IsPlotHovered() && ImGui::IsKeyPressed(ImGuiKey_MouseLeft)) {
					ImPlotPoint mousepos = ImPlot::GetPlotMousePos();
					renderInfo.newypoint = mousepos.x;
					renderInfo.newzpoint = mousepos.y;
				}

			}

			void HandleCursor(SliceRenderInfo& renderInfo)
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

			void Render(SliceRenderInfo& renderInfo) {

				if (renderInfo.doubleBuffer) {
					slice0 = slicer.GetTensorSlice({ 0 }, 40, 40, 40).contiguous();;
				}
				else {
					slice1 = slicer.GetTensorSlice({ 0 }, 40, 40, 40).contiguous();;
				}

				at::Tensor& slice = renderInfo.doubleBuffer ? slice0 : slice1;
				
				if (ImGui::Begin(slicername.c_str())) {

					auto windowsize = ImGui::GetWindowSize();
					uint32_t tensor_width = slice.size(0); uint32_t tensor_height = slice.size(1);
					float hw_tensor_ratio = tensor_height / tensor_width;

					float window_width = windowsize.x * 0.9f;
					float window_height = window_width * hw_tensor_ratio;
					
					static ImPlotHeatmapFlags hm_flags = 0;

					ImGui::SetNextItemWidth(window_width);
					ImGui::DragFloatRange2("Min Multiplier / Max Multiplier", &scale_min_mult, &scale_max_mult, 0.01f, -40, 40);

					ImPlot::PushColormap(renderInfo.map);

					float minscale = renderInfo.min_scale * scale_min_mult;
					float maxscale = renderInfo.max_scale * scale_max_mult;

					static ImPlotAxisFlags axes_flags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;
					if (ImPlot::BeginPlot(plotname.c_str(), ImVec2(window_width, window_height), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
						ImPlot::SetupAxes(nullptr, nullptr, axes_flags, axes_flags);
						ImPlot::PlotHeatmap(heatname.c_str(), (const float*)slice.const_data_ptr(), slice.size(0), slice.size(1),
							minscale, maxscale, nullptr, ImPlotPoint(0, 0), ImPlotPoint(slice.size(0), slice.size(1)), hm_flags);
					
						HandleCursor(renderInfo);

						ImPlot::EndPlot();
					}
					ImGui::SameLine();
					ImPlot::ColormapScale("##HeatScale", minscale, maxscale, ImVec2(0.08 * windowsize.x, window_height));

					ImGui::SameLine();

					ImPlot::PopColormap();
				}


				ImGui::End();
			}

			Slicer& slicer;
			at::Tensor slice0;
			at::Tensor slice1;

			std::string slicername;
			std::string plotname;
			std::string heatname;

			float scale_min_mult = 1.0f;
			float scale_max_mult = 1.0f;
		};

		class Orthoslicer {
		public:

			struct RenderInfo {
				bool doubleBuffer;
			};

			Orthoslicer(at::Tensor tensor)
				: _tensor(tensor),
				_axialSlicer(SliceView::eAxial, _tensor),
				_sagitalSlicer(SliceView::eSagital, _tensor),
				_coronalSlicer(SliceView::eCoronal, _tensor),
				_axialSlicerRenderer(_axialSlicer),
				_sagitalSlicerRenderer(_sagitalSlicer),
				_coronalSlicerRenderer(_coronalSlicer)
			{

			}

			void Render(const RenderInfo& rinfo) {
				
				_renderInfo.doubleBuffer = rinfo.doubleBuffer;

				ImGuiWindowClass slicer_class;
				slicer_class.ClassId = ImGui::GetID("SlicerWindowClass");
				slicer_class.DockingAllowUnclassed = false;


				static ImGuiID dockid = ImGui::GetID("Orthoslicer Dockspace");

				if (ImGui::Begin("Orthoslicer")) {

					//ImGui::BeginChild("SlicerAnchor");

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

		private:
			at::Tensor _slice;

			at::Tensor _tensor;
			Slicer _axialSlicer;
			Slicer _sagitalSlicer;
			Slicer _coronalSlicer;

			SlicerRenderer _axialSlicerRenderer;
			SlicerRenderer _sagitalSlicerRenderer;
			SlicerRenderer _coronalSlicerRenderer;

			SliceRenderInfo _renderInfo;
		};



		/*
		struct SkiaSlicer : public Slicer {

			SkiaSlicer(SkiaContext& skiaCtx, SliceView view, at::Tensor& tensor)
				: skiaCtx(skiaCtx), Slicer(view, tensor)
			{
				imageInfo = SkImageInfo::Make(width, height, SkColorType::kRGBA_8888_SkColorType, SkAlphaType::kUnpremul_SkAlphaType);
				rgbaBuffer.resize(width * height * 4);
			}

			sk_sp<SkImage> GetSlice(const std::vector<int64_t>& startslice,
				int64_t xpoint, int64_t ypoint, int64_t zpoint,
				std::function<void(const at::Tensor&, std::vector<char>&)> colormap)
			{
				auto slice = GetTensorSlice(startslice, xpoint, ypoint, zpoint);
				colormap(slice, rgbaBuffer);
				auto skia_tensordata = SkData::MakeWithoutCopy(rgbaBuffer.data(), rgbaBuffer.size());
				return SkImages::RasterFromData(imageInfo, skia_tensordata, width);
			}

			SkiaContext& skiaCtx;
			SkImageInfo imageInfo;
			sk_sp<SkSurface> slicerSurface;
			std::vector<char> rgbaBuffer;
		};
		*/

		/*
		class SkiaOrthoslicer {
		public:

			SkiaOrthoslicer(std::shared_ptr<SkiaContext> pSkiaCtx, at::Tensor tensor)
				: _tensor(tensor), _pSkiaCtx(pSkiaCtx),
				_axialSlicer(*pSkiaCtx, SliceView::eAxial, _tensor),
				_sagitalSlicer(*pSkiaCtx, SliceView::eSagital, _tensor),
				_coronalSlicer(*pSkiaCtx, SliceView::eCoronal, _tensor)
			{

			}

		private:
			at::Tensor _tensor;
			std::shared_ptr<SkiaContext> _pSkiaCtx;
			SkiaSlicer _axialSlicer;
			SkiaSlicer _sagitalSlicer;
			SkiaSlicer _coronalSlicer;
		};
		*/
















		/*

		struct SlicerTexture {
			uint32_t width, height;
			
			VkBuffer stagingBuffer;
			VkDeviceMemory stagingMemory;
			
			VkSampler srcSampler;
			VkImage srcImage;
			VkImageLayout srcImageLayout;
			VkDeviceMemory srcImageMemory;
			VkImageView srcView;

			VkSampler nearestSampler;
			VkSampler linearSampler;
			VkImage dstImage;
			VkImageLayout dstImageLayout;
			VkDeviceMemory dstImageMemory;
			VkImageView dstView;

		};

		class Orthoslicer {
		public:

			Orthoslicer(uint32_t xextent, uint32_t yextent, uint32_t zextent, vks::VulkanDevice& device);

		private:

			void CreateStagingBuffer(uint32_t width, uint32_t height, VkBuffer& buffer, VkDeviceMemory& mem)
			{
				// STAGING BUFFER
				VkBufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo();
				bufferCreateInfo.size = width * height * sizeof(float);
				bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
				bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
				VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer));

				VkMemoryRequirements memReqs;
				vkGetBufferMemoryRequirements(device, buffer, &memReqs);

				VkMemoryAllocateInfo allocInfo = vks::initializers::memoryAllocateInfo();
				allocInfo.allocationSize = memReqs.size;
				VkBool32 success;
				allocInfo.memoryTypeIndex = device.getMemoryType(memReqs.memoryTypeBits,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					&success);
				if (!success)
					throw std::runtime_error("Found no suitable memory type");

				VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &mem));
				
				VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, mem, 0));
			}

			void CreateImage(uint32_t width, uint32_t height, VkFormat format, VkImage image, VkDeviceMemory mem)
			{
				VkImageCreateInfo imageInfo = vks::initializers::imageCreateInfo();
				imageInfo.imageType = VK_IMAGE_TYPE_2D;
				imageInfo.extent.width = width;
				imageInfo.extent.height = height;
				imageInfo.extent.depth = 1;
				imageInfo.mipLevels = 1;
				imageInfo.arrayLayers = 1;
				imageInfo.format = format;
				imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
				imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
				imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
				imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
				imageInfo.flags = 0;
				VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, image));

				VkMemoryRequirements memReqs = {};
				vkGetImageMemoryRequirements(device, image, &memReqs);

				VkMemoryAllocateInfo allocInfo = vks::initializers::memoryAllocateInfo();
				allocInfo.allocationSize = memReqs.size;
				VkBool32 success;
				allocInfo.memoryTypeIndex = device.getMemoryType(memReqs.memoryTypeBits,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &success);
				if (!success)
					throw std::runtime_error("Failed to create slicer texture");

				if (vkAllocateMemory(device, &allocInfo, nullptr, &mem) != VK_SUCCESS) {
					throw std::runtime_error("failed to allocate image memory!");
				}
				vkBindImageMemory(device, image, mem, 0);
			}

			void CreateImageView(VkImage image, VkFormat format, VkImageView& imageView)
			{
				VkImageViewCreateInfo viewInfo = vks::initializers::imageViewCreateInfo();
				viewInfo.image = image;
				viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
				viewInfo.format = format;
				viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				viewInfo.subresourceRange.baseMipLevel = 0;
				viewInfo.subresourceRange.levelCount = 1;
				viewInfo.subresourceRange.baseArrayLayer = 0;
				viewInfo.subresourceRange.layerCount = 1;

				if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
					throw std::runtime_error("failed to create texture image view!");
				}
			}

			void CreateSampler(VkSampler& sampler, VkFilter magFilter, VkFilter minFilter)
			{
				VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();
				samplerInfo.magFilter = magFilter;
				samplerInfo.minFilter = minFilter;
				samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
				samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
				samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
				samplerInfo.anisotropyEnable = VK_FALSE;
				samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
				samplerInfo.unnormalizedCoordinates = VK_FALSE;
				samplerInfo.compareEnable = VK_FALSE;
				samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
				samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
				samplerInfo.mipLodBias = 0.0f;
				samplerInfo.minLod = 0.0;
				samplerInfo.maxLod = 0.0f;

				if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
					throw std::runtime_error("failed to create texture sampler!");
				}
			}

			void CreateSlicerTexture(SlicerTexture& texture) {

				CreateStagingBuffer(texture.width, texture.height, texture.stagingBuffer, texture.stagingMemory);

				// SRC IMAGE
				CreateImage(texture.width, texture.height, VK_FORMAT_R32_SFLOAT, texture.srcImage, texture.srcImageMemory);
				CreateImageView(texture.srcImage, VK_FORMAT_R32_SFLOAT, texture.srcView);
				CreateSampler(texture.srcSampler, VK_FILTER_NEAREST, VK_FILTER_NEAREST);

				// DST IMAGE
				CreateImage(texture.width, texture.height, VK_FORMAT_R8G8B8A8_SRGB, texture.dstImage, texture.dstImageMemory);
				CreateImageView(texture.dstImage, VK_FORMAT_R8G8B8A8_SRGB, texture.dstView);
				CreateSampler(texture.nearestSampler, VK_FILTER_NEAREST, VK_FILTER_NEAREST);
				CreateSampler(texture.linearSampler, VK_FILTER_NEAREST, VK_FILTER_NEAREST);

			}

			vks::VulkanDevice& device;

			SlicerTexture axial_texture;
			SlicerTexture coronal_texture;
			SlicerTexture sagital_texture;

		};
		*/

	}
}