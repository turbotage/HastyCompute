#pragma once

#include "base/VulkanBase.hpp"

#include <torch/torch.h>

#include <imgui.h>
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
				uint64_t ndim = tensor.ndimension();
				uint64_t xlen = tensor.size(ndim - 3);
				uint64_t ylen = tensor.size(ndim - 2);
				uint64_t zlen = tensor.size(ndim - 1);

				switch (view) {
				case eAxial:
					width = xlen;
					height = ylen;
					break;
				case eSagital:
					width = xlen;
					height = zlen;
					break;
				case eCoronal:
					width = ylen;
					height = zlen;
					break;
				default:
					throw std::runtime_error("View can only be eAxial, eSagital or eCoronal");
				}

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
				{
					indices.push_back(Slice()); indices.push_back(Slice()); indices.push_back(zpoint);
				}
				break;
				case eSagital:
				{
					indices.push_back(Slice()); indices.push_back(ypoint); indices.push_back(Slice());
				}
				break;
				case eCoronal:
				{
					indices.push_back(xpoint); indices.push_back(Slice()); indices.push_back(Slice());
				}
				break;
				default:
					throw std::runtime_error("Only eAxial, eSagital, eCoronal are allowed");
				}

				return tensor.index(at::makeArrayRef(indices)).contiguous();
			}

			SliceView view;

			uint32_t width;
			uint32_t height;

			at::Tensor& tensor;

		};

		struct SlicerRenderer {

			SlicerRenderer(Slicer& slicer)
				: slicer(slicer)
			{

			}

			void RenderSlice() {
				static ImPlotHeatmapFlags hm_flags = 0;
				static float scale_min = 0.0f;
				static float scale_max = 1.0f;

				ImGui::SetNextItemWidth(225);
				ImGui::DragFloatRange2("Min / Max", &scale_min, &scale_max, 0.01f, -20, 20);

				static ImPlotColormap map = ImPlotColormap_Viridis;

				ImPlot::PushColormap(map);

				static ImPlotAxisFlags axes_flags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;
				if (ImPlot::BeginPlot("##Heatmap1", ImVec2(500, 500), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
					ImPlot::SetupAxes(nullptr, nullptr, axes_flags, axes_flags);
					ImPlot::PlotHeatmap("heat", (const float*)_slice.const_data_ptr(), _slice.size(0), _slice.size(1), scale_min, scale_max, nullptr, ImPlotPoint(0, 0), ImPlotPoint(1, 1), hm_flags);
					ImPlot::EndPlot();
				}
				ImGui::SameLine();
				ImPlot::ColormapScale("##HeatScale", scale_min, scale_max, ImVec2(60, 225));

				ImGui::SameLine();

				ImPlot::PopColormap();
			}

			Slicer& slicer;

		};

		class Orthoslicer {
		public:

			Orthoslicer(at::Tensor tensor)
				: _tensor(tensor),
				_axialSlicer(SliceView::eAxial, _tensor),
				_sagitalSlicer(SliceView::eSagital, _tensor),
				_coronalSlicer(SliceView::eCoronal, _tensor)
			{

				
			}

			void RenderSlice() {
				static ImPlotHeatmapFlags hm_flags = 0;
				static float scale_min = 0.0f;
				static float scale_max = 1.0f;

				ImGui::SetNextItemWidth(225);
				ImGui::DragFloatRange2("Min / Max", &scale_min, &scale_max, 0.01f, -20, 20);

				static ImPlotColormap map = ImPlotColormap_Viridis;

				ImPlot::PushColormap(map);

				static ImPlotAxisFlags axes_flags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;
				if (ImPlot::BeginPlot("##Heatmap1", ImVec2(500, 500), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
					ImPlot::SetupAxes(nullptr, nullptr, axes_flags, axes_flags);
					//ImPlot::SetupAxisTicks(ImAxis_X1, 0 + 1.0 / 14.0, 1 - 1.0 / 14.0, 7, nullptr);
					//ImPlot::SetupAxisTicks(ImAxis_Y1, 1 - 1.0 / 14.0, 0 + 1.0 / 14.0, 7, nullptr);
					ImPlot::PlotHeatmap("heat", (const float*)_slice.const_data_ptr(), _slice.size(0), _slice.size(1), scale_min, scale_max, nullptr, ImPlotPoint(0, 0), ImPlotPoint(1, 1), hm_flags);
					ImPlot::EndPlot();
				}
				ImGui::SameLine();
				ImPlot::ColormapScale("##HeatScale", scale_min, scale_max, ImVec2(60, 225));

				ImGui::SameLine();

				ImPlot::PopColormap();
			}

			void Render() {
				
				_slice = _axialSlicer.GetTensorSlice({ 0 }, 40, 40, 40);

				
			}

		private:
			at::Tensor _slice;

			at::Tensor _tensor;
			Slicer _axialSlicer;
			Slicer _sagitalSlicer;
			Slicer _coronalSlicer;
		};




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