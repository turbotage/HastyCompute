#pragma once

#include <ATen/ATen.h>
#include "block.hpp"

namespace hasty {
	namespace cuda {

		using CTensorVec = std::vector<std::reference_wrapper<const at::Tensor>>;
		using TensorVec = std::vector<std::reference_wrapper<at::Tensor>>;

		using CTensorVecVec = std::vector<CTensorRefVec>;
		using TensorVecVec = std::vector<TensorRefVec>;


		// tensors outer vector are frames inner are encodes
		at::Tensor extract_block(const at::Tensor& tensor, const Block<4>& block);

		at::Tensor svt(const at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor,at::Tensor,at::Tensor>> storage);

		void svt_inplace(at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

		void insert_block(const TensorRefVecVec& tensors, 
			const std::pair<std::vector<int64_t>, std::vector<int64_t>>& block, const at::Tensor& block_tensor);
		
		// Runs 
		class LLRecon_4D_Encodes {
		public:

			LLRecon_4D_Encodes(at::Tensor& image,
					const CTensorVecVec& coords,
					const at::Tensor& smaps,
					const CTensorVecVec& kdata);

			void iter();

		private:
			at::Tensor& _image;
			const at::Tensor& _coords;
			const at::Tensor& _smaps;
			const at::Tensor& _kdata;
		};

	}
}