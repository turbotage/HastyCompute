#pragma once

#include "base/VulkanBase.hpp"

#include <skia/include/gpu/vk/GrVkBackendContext.h>
#include <skia/include/gpu/vk/GrVkVulkan.h>
#include <skia/include/gpu/vk/GrVkExtensions.h>

#include <skia/include/gpu/vk/>

namespace hasty {
	namespace viz {

		class SkiaWrapper {
		public:

			SkiaWrapper(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
				VkQueue graphicsQueue, uint32_t graphicsQueueIndex, VkPhysicalDeviceFeatures features) {
				_backendContext.fInstance = instance;
				_backendContext.fPhysicalDevice = physicalDevice;
				_backendContext.fDevice = device;
				_backendContext.fQueue = graphicsQueue;
				_backendContext.fGraphicsQueueIndex = graphicsQueueIndex;
				
				
				
			
			}

		private:

			GrVkBackendContext _backendContext;
		};

		class SkiaOrthoslicer {

			SkiaOrthoslicer() {
				GrVkBackendContext backendContext;


			}

		};
























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

	}
}