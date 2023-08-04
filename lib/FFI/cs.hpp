#pragma once

#include "ffi_defines.hpp"
#include "../torch_util.hpp"

namespace hasty {

	namespace ffi {


		// FORWARD

		LIB_EXPORT
		at::Tensor batched_sense_forward(const at::Tensor& input, at::TensorList output, const std::optional<std::vector<std::vector<int64_t>>>& coils,
			const at::Tensor& smaps, const std::vector<at::Tensor>& coords, 
			bool sum, bool sumnorm, const std::vector<c10::Stream>& streams);




		LIB_EXPORT
		at::Tensor batched_sense_forward_weighted(const at::Tensor& input, std::vector<at::Tensor>& output, const std::optional<std::vector<std::vector<int64_t>>>& coils,
			const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, 
			bool sum, bool sumnorm, const std::vector<c10::Stream>& streams);

		LIB_EXPORT
		at::Tensor batched_sense_forward_kdata(const at::Tensor& input, std::vector<at::Tensor>& output, const std::optional<std::vector<std::vector<int64_t>>>& coils,
			const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas,
			bool sum, bool sumnorm, const std::vector<c10::Stream>& streams);

		LIB_EXPORT
		at::Tensor batched_sense_forward_weighted_kdata(const at::Tensor& input, std::vector<at::Tensor>& output, const std::optional<std::vector<std::vector<int64_t>>>& coils,
			const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, const std::vector<at::Tensor>& kdatas,
			bool sum, bool sumnorm, const std::vector<c10::Stream>& streams);

		// ADJOINT

		LIB_EXPORT
		at::Tensor batched_sense_adjoint(const std::vector<at::Tensor>& input, at::Tensor& output, const std::optional<std::vector<std::vector<int64_t>>>& coils,
			const at::Tensor& smaps, const std::vector<at::Tensor>& coords,
			bool sum, bool sumnorm, const std::vector<c10::Stream>& streams);

		LIB_EXPORT
		at::Tensor batched_sense_adjoint_weighted(const std::vector<at::Tensor>& input, at::Tensor& output, const std::optional<std::vector<std::vector<int64_t>>>& coils,
			const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights,
			bool sum, bool sumnorm, const std::vector<c10::Stream>& streams);

		LIB_EXPORT
		at::Tensor batched_sense_adjoint_kdata(const std::vector<at::Tensor>& input, at::Tensor& output, const std::optional<std::vector<std::vector<int64_t>>>& coils,
			const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas,
			bool sum, bool sumnorm, const std::vector<c10::Stream>& streams);

		LIB_EXPORT
		at::Tensor batched_sense_adjoint_weighted_kdata(const std::vector<at::Tensor>& input, at::Tensor& output, const std::optional<std::vector<std::vector<int64_t>>>& coils,
			const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, const std::vector<at::Tensor>& kdatas,
			bool sum, bool sumnorm, const std::vector<c10::Stream>& streams);


		// NORMAL

		// A^HA
		LIB_EXPORT
		void batched_sense_normal(
			at::Tensor& input, const std::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
			const std::vector<at::Tensor>& coords, const std::vector<c10::Stream>& streams);

		// A^HWAx
		LIB_EXPORT
		void batched_sense_normal_weighted(at::Tensor& input, const std::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
			const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, const std::vector<c10::Stream>& streams);

		// A^H(Ax-b)
		LIB_EXPORT
		void batched_sense_normal_kdata(at::Tensor& input, const std::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
			const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas, const std::vector<c10::Stream>& streams);

		// A^HW(Ax-b)
		LIB_EXPORT
		void batched_sense_normal_weighted_kdata(at::Tensor& input, const std::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
			const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, 
			const std::vector<at::Tensor>& kdatas, const std::vector<c10::Stream>& streams);



		// A^HA = F^HDF D calculated from coords
		LIB_EXPORT
		void batched_sense_toeplitz(
			at::Tensor& input, const std::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
			const std::vector<at::Tensor>& coords, const std::vector<c10::Stream>& streams);

		// A^HA = F^HDF with D given
		LIB_EXPORT
		void batched_sense_toeplitz_diagonals(at::Tensor& input, const std::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
			const at::Tensor& diagonals, const std::vector<c10::Stream>& streams);


		// LLR

		LIB_EXPORT
		void random_blocks_svt(at::Tensor& input, int64_t nblocks, int64_t block_size,
			double thresh, bool soft, const std::vector<c10::Stream>& streams);

		LIB_EXPORT
		void normal_blocks_svt(at::Tensor& input, std::vector<int64_t> block_strides, std::vector<int64_t> block_shapes,
			int64_t block_iter, double thresh, bool soft, const std::vector<c10::Stream>& streams);

	}
}
