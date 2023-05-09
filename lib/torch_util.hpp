#pragma once

#include <torch/torch.h>

namespace hasty {

	struct DeviceOptions {
		DeviceOptions(const c10::Device& device, const c10::Stream& stream)
		{
			devices.emplace_back(device, std::vector{ stream });
		}

		void push_back_device(const c10::Device& device, const c10::Stream& stream)
		{
			for (auto& dev : devices) {
				if (dev.first == device) {
					dev.second.push_back(stream);
				}
			}
		}

		std::vector<std::pair<c10::Device, std::vector<c10::Stream>>> devices;
	};

	namespace torch_util {
		std::stringstream print_4d_xyz(const at::Tensor& toprint);

		std::vector<int64_t> nmodes_from_tensor(const at::Tensor& tensor);

	}

}

