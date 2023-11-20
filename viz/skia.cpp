#include "skia.hpp"

hasty::viz::SkiaContext::SkiaContext(
	VkInstance instance,
	VkPhysicalDevice physicalDevice,
	VkDevice device,
	VkQueue graphicsQueue,
	uint32_t graphicsQueueIndex,
	VkPhysicalDeviceFeatures* pFeatures,
	VkPhysicalDeviceFeatures2* pFeatures2,
	uint32_t apiVersion)
{
	_vulkanBackendContext.fInstance = instance;
	_vulkanBackendContext.fPhysicalDevice = physicalDevice;
	_vulkanBackendContext.fDevice = device;
	_vulkanBackendContext.fQueue = graphicsQueue;
	_vulkanBackendContext.fGraphicsQueueIndex = graphicsQueueIndex;
	_vulkanBackendContext.fMaxAPIVersion = apiVersion;

	_vulkanBackendContext.fDeviceFeatures = pFeatures;
	_vulkanBackendContext.fDeviceFeatures2 = pFeatures2;

	_vulkanBackendContext.fGetProc = [](const char* proc_name, VkInstance instance, VkDevice device) {
		if (device != VK_NULL_HANDLE) {
			return vkGetDeviceProcAddr(device, proc_name);
		}
		return vkGetInstanceProcAddr(instance, proc_name);
		};

	_vulkanBackendContext.fProtectedContext = skgpu::Protected::kNo;

	_grVkBackendContext.fInstance = _vulkanBackendContext.fInstance;
	_grVkBackendContext.fPhysicalDevice = _vulkanBackendContext.fPhysicalDevice;
	_grVkBackendContext.fDevice = _vulkanBackendContext.fDevice;
	_grVkBackendContext.fQueue = _vulkanBackendContext.fQueue;
	_grVkBackendContext.fGraphicsQueueIndex = _vulkanBackendContext.fGraphicsQueueIndex;
	_grVkBackendContext.fMaxAPIVersion = _vulkanBackendContext.fMaxAPIVersion;
	_grVkBackendContext.fVkExtensions = _vulkanBackendContext.fVkExtensions;
	_grVkBackendContext.fDeviceFeatures = _vulkanBackendContext.fDeviceFeatures;
	_grVkBackendContext.fDeviceFeatures2 = _vulkanBackendContext.fDeviceFeatures2;
	_grVkBackendContext.fMemoryAllocator = _vulkanBackendContext.fMemoryAllocator;
	_grVkBackendContext.fGetProc = _vulkanBackendContext.fGetProc;
	_grVkBackendContext.fProtectedContext = _vulkanBackendContext.fProtectedContext;
	_grVkBackendContext.fOwnsInstanceAndDevice = false;

	_directContext = GrDirectContext::MakeVulkan(_grVkBackendContext);
	if (!_directContext) {
		throw std::runtime_error("Failed to create direct context");
	}
}


