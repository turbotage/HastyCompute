#pragma once

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

namespace hasty {

	namespace torch_util {

		std::vector<c10::Stream> get_streams(const at::optional<std::vector<c10::Stream>>& streams);

		std::stringstream print_4d_xyz(const at::Tensor& toprint);

		std::vector<int64_t> nmodes_from_tensor(const at::Tensor& tensor);

        template<typename T>
        std::vector<int64_t> argsort(const std::vector<T>& array) {
            std::vector<int64_t> indices(array.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                [&array](int left, int right) -> bool {
                    // sort indices according to corresponding array element
                    return array[left] < array[right];
                });

            return indices;
        }

        template<typename T>
        std::vector<T> apply_permutation(const std::vector<T>& v, const std::vector<int64_t>& indices)
        {
            std::vector<T> v2(v.size());
            for (size_t i = 0; i < v.size(); i++) {
                v2[i] = v[indices[i]];
            }
            return v2;
        }

	}

}

