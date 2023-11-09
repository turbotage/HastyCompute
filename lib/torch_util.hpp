#pragma once

#include <torch/torch.h>

#include <future>

#include "export.hpp"

namespace hasty {

	namespace torch_util {

		std::vector<at::Stream> LIB_EXPORT get_streams(const at::optional<std::vector<at::Stream>>& streams);

        std::vector<at::Stream> LIB_EXPORT get_streams(const at::optional<at::ArrayRef<at::Stream>>& streams);

        std::stringstream print_4d_xyz(const at::Tensor& toprint);

		std::vector<int64_t> nmodes_from_tensor(const at::Tensor& tensor);

        template<typename T>
        std::vector<int64_t> argsort(const std::vector<T>& array) {
            std::vector<int64_t> indices(array.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                [&array](int64_t left, int64_t right) -> bool {
                    // sort indices according to corresponding array element
                    return array[left] < array[right];
                });

            return indices;
        }

        at::ScalarType complex_type(at::ScalarType real_type, std::initializer_list<at::ScalarType> allowed_types);

        at::ScalarType real_type(at::ScalarType complex_type, std::initializer_list<at::ScalarType> allowed_types);

        template<typename T>
        std::vector<T> apply_permutation(const std::vector<T>& v, const std::vector<int64_t>& indices)
        {
            std::vector<T> v2(v.size());
            for (size_t i = 0; i < v.size(); i++) {
                v2[i] = v[indices[i]];
            }
            return v2;
        }

        void LIB_EXPORT future_catcher(std::future<void>& fut);

        void LIB_EXPORT future_catcher(const std::function<void()>& func);

        at::Tensor upscale_with_zeropad(const at::Tensor& input, const std::vector<int64_t>& newsize);

        at::Tensor upscale_with_zeropad(const at::Tensor& input, const at::ArrayRef<int64_t>& newsize);

        at::Tensor resize(const at::Tensor& input, const std::vector<int64_t>& newsize);

        at::Tensor resize(const at::Tensor& input, const at::ArrayRef<int64_t>& newsize);
	}

}

