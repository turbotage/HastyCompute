#pragma once

#include "ffi_defines.hpp"
#include "../torch_util.hpp"

namespace hasty {

	namespace ffi {

		// A^HA
		LIB_EXPORT
		void batched_sense(
			at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps,
			const std::vector<at::Tensor>& coords);

		// A^HWAx
		LIB_EXPORT
		void batched_sense_weighted(at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps,
			const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights);

		// A^H(Ax-b)
		LIB_EXPORT
		void batched_sense_kdata(at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps,
			const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas);

		// A^HW(Ax-b)
		LIB_EXPORT
		void batched_sense_weighted_kdata(at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps,
			const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, const std::vector<at::Tensor>& kdatas);



		// A^HA = F^HDF D calculated from coords
		LIB_EXPORT
		void batched_sense_toeplitz(
			at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps, 
			const std::vector<at::Tensor>& coords);

		// A^HA = F^HDF with D given
		LIB_EXPORT
		void batched_sense_toeplitz_diagonals(at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps,
			const at::Tensor& diagonals);






		LIB_EXPORT
		void random_blocks_svt(at::Tensor& input, int32_t nblocks, int32_t block_size, int32_t rank);

	}
}
