#pragma once

#include <skia/gpu/vk/VulkanBackendContext.h>
#include <skia/gpu/vk/VulkanExtensions.h>
#include <skia/gpu/vk/GrVkBackendContext.h>

#include <skia/gpu/GrDirectContext.h>


#include <skia/core/SkImage.h>
#include <skia/core/SkSurface.h>
#include <skia/core/SkCanvas.h>
#include <skia/core/SkBitmap.h>
#include <skia/core/SkPaint.h>
#include <skia/core/SkColor.h>
#include <skia/core/SkShader.h>

#include <skia/gpu/ganesh/SkSurfaceGanesh.h>
#include <skia/gpu/ganesh/SkImageGanesh.h>

namespace hasty {
	namespace viz {

		class SkiaContext {
		public:

			SkiaContext(
				VkInstance instance,
				VkPhysicalDevice physicalDevice,
				VkDevice device,
				VkQueue graphicsQueue,
				uint32_t graphicsQueueIndex,
				VkPhysicalDeviceFeatures* pFeatures,
				VkPhysicalDeviceFeatures2* pFeatures2,
				uint32_t apiVersion);

			operator GrDirectContext* () const
			{
				return _directContext.get();
			};

		private:
			GrVkBackendContext _grVkBackendContext;
			skgpu::VulkanBackendContext _vulkanBackendContext;
			sk_sp<GrDirectContext> _directContext;
		};

	}
}


