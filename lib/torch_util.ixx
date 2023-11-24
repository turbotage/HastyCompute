module;

#include <torch/torch.h>

export module torch_util;

import <future>;

namespace hasty {

	namespace torch_util {

		export std::vector<at::Stream> get_streams(const at::optional<std::vector<at::Stream>>& streams);

		export std::vector<at::Stream> get_streams(const at::optional<at::ArrayRef<at::Stream>>& streams);

		export std::stringstream print_4d_xyz(const at::Tensor& toprint);

		export std::vector<int64_t> nmodes_from_tensor(const at::Tensor& tensor);

		export template<typename T>
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

		export at::ScalarType complex_type(at::ScalarType real_type, std::initializer_list<at::ScalarType> allowed_types);

		export at::ScalarType real_type(at::ScalarType complex_type, std::initializer_list<at::ScalarType> allowed_types);

		export template<typename T>
		std::vector<T> apply_permutation(const std::vector<T>& v, const std::vector<int64_t>& indices)
		{
			std::vector<T> v2(v.size());
			for (size_t i = 0; i < v.size(); i++) {
				v2[i] = v[indices[i]];
			}
			return v2;
		}

		export template<typename T>
		T future_catcher(std::future<T>& fut)
		{
			try {
				return fut.get();
			}
			catch (c10::Error& e) {
				std::string err = e.what();
				std::cerr << err << std::endl;
				throw std::runtime_error(err);
			}
			catch (std::exception& e) {
				std::string err = e.what();
				std::cerr << err << std::endl;
				throw std::runtime_error(err);
			}
			catch (...) {
				std::cerr << "caught something strange: " << std::endl;
				throw std::runtime_error("caught something strange: ");
			}
		}

		export template<typename T>
		T future_catcher(const std::function<T()>& func)
		{
			try {
				return func();
			}
			catch (c10::Error& e) {
				std::string err = e.what();
				std::cerr << err << std::endl;
				throw std::runtime_error(err);
			}
			catch (std::exception& e) {
				std::string err = e.what();
				std::cerr << err << std::endl;
				throw std::runtime_error(err);
			}
			catch (...) {
				std::cerr << "caught something strange: " << std::endl;
				throw std::runtime_error("caught something strange: ");
			}
		}

		export void future_catcher(std::future<void>& fut);

		export void future_catcher(const std::function<void()>& func);

		export at::Tensor upscale_with_zeropad(const at::Tensor& input, const std::vector<int64_t>& newsize);

		export at::Tensor upscale_with_zeropad(const at::Tensor& input, const at::ArrayRef<int64_t>& newsize);

		export at::Tensor resize(const at::Tensor& input, const std::vector<int64_t>& newsize);

		export at::Tensor resize(const at::Tensor& input, const at::ArrayRef<int64_t>& newsize);
	}

}

