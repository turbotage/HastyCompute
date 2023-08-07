#pragma once

#include "py_util.hpp"


/*
namespace bs { 

	// FORWARD 

	void batched_sense_forward(const at::Tensor& input, at::TensorList output, const at::optional<std::vector<std::vector<int64_t>>>& coils,
		const at::Tensor& smaps, const std::vector<at::Tensor>& coords,
		bool sum, bool sumnorm, const at::optional<std::vector<c10::Stream>>& streams)
	{
		std::optional<std::vector<std::vector<int64_t>>> coilss = coils.has_value() ? std::make_optional(*coils) : std::nullopt;
		hasty::ffi::batched_sense_forward(input, output, coilss, smaps, coords, sum, sumnorm, hasty::ffi::get_streams(streams));
	}
	
	void batched_sense_forward_weighted(const at::Tensor& input, std::vector<at::Tensor>& output, const at::optional<std::vector<std::vector<int64_t>>>& coils,
		const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights,
		bool sum, bool sumnorm, const at::optional<std::vector<c10::Stream>>& streams)
	{
		std::optional<std::vector<std::vector<int64_t>>> coilss = coils.has_value() ? std::make_optional(*coils) : std::nullopt;
		hasty::ffi::batched_sense_forward_weighted(input, output, coilss, smaps, coords, weights, sum, sumnorm, hasty::ffi::get_streams(streams));
	}

	void batched_sense_forward_kdata(const at::Tensor& input, std::vector<at::Tensor>& output, const at::optional<std::vector<std::vector<int64_t>>>& coils,
		const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas,
		bool sum, bool sumnorm, const at::optional<std::vector<c10::Stream>>& streams)
	{
		std::optional<std::vector<std::vector<int64_t>>> coilss = coils.has_value() ? std::make_optional(*coils) : std::nullopt;
		hasty::ffi::batched_sense_forward_kdata(input, output, coilss, smaps, coords, kdatas, sum, sumnorm, hasty::ffi::get_streams(streams));
	}

	void batched_sense_forward_weighted_kdata(const at::Tensor& input, std::vector<at::Tensor>& output, const at::optional<std::vector<std::vector<int64_t>>>& coils,
		const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, const std::vector<at::Tensor>& kdatas,
		bool sum, bool sumnorm, const at::optional<std::vector<c10::Stream>>& streams)
	{
		std::optional<std::vector<std::vector<int64_t>>> coilss = coils.has_value() ? std::make_optional(*coils) : std::nullopt;
		hasty::ffi::batched_sense_forward_weighted_kdata(input, output, coilss, smaps, coords, weights, kdatas, sum, sumnorm, hasty::ffi::get_streams(streams));
	}
	
	// ADJOINT

	at::Tensor batched_sense_adjoint(const std::vector<at::Tensor>& input, at::Tensor& output, const at::optional<std::vector<std::vector<int64_t>>>& coils,
		const at::Tensor& smaps, const std::vector<at::Tensor>& coords,
		bool sum, bool sumnorm, const at::optional<std::vector<c10::Stream>>& streams)
	{
		std::optional<std::vector<std::vector<int64_t>>> coilss = coils.has_value() ? std::make_optional(*coils) : std::nullopt;
		return hasty::ffi::batched_sense_adjoint(input, output, coilss, smaps, coords, sum, sumnorm, hasty::ffi::get_streams(streams));
	}

	at::Tensor batched_sense_adjoint_weighted(const std::vector<at::Tensor>& input, at::Tensor& output, const at::optional<std::vector<std::vector<int64_t>>>& coils,
		const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights,
		bool sum, bool sumnorm, const at::optional<std::vector<c10::Stream>>& streams)
	{
		std::optional<std::vector<std::vector<int64_t>>> coilss = coils.has_value() ? std::make_optional(*coils) : std::nullopt;
		return hasty::ffi::batched_sense_adjoint_weighted(input, output, coilss, smaps, coords, weights, sum, sumnorm, hasty::ffi::get_streams(streams));
	}

	at::Tensor batched_sense_adjoint_kdata(const std::vector<at::Tensor>& input, at::Tensor& output, const at::optional<std::vector<std::vector<int64_t>>>& coils,
		const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas,
		bool sum, bool sumnorm, const at::optional<std::vector<c10::Stream>>& streams)
	{
		std::optional<std::vector<std::vector<int64_t>>> coilss = coils.has_value() ? std::make_optional(*coils) : std::nullopt;
		return hasty::ffi::batched_sense_adjoint_kdata(input, output, coilss, smaps, coords, kdatas, sum, sumnorm, hasty::ffi::get_streams(streams));
	}

	at::Tensor batched_sense_adjoint_weighted_kdata(const std::vector<at::Tensor>& input, at::Tensor& output, const at::optional<std::vector<std::vector<int64_t>>>& coils,
		const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, const std::vector<at::Tensor>& kdatas,
		bool sum, bool sumnorm, const at::optional<std::vector<c10::Stream>>& streams)
	{ 
		std::optional<std::vector<std::vector<int64_t>>> coilss = coils.has_value() ? std::make_optional(*coils) : std::nullopt;
		return hasty::ffi::batched_sense_adjoint_weighted_kdata(input, output, coilss, smaps, coords, weights, kdatas, sum, sumnorm, hasty::ffi::get_streams(streams));
	}

	// NORMAL

	void batched_sense_normal(at::Tensor& input, const at::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
		const std::vector<at::Tensor>& coords, const at::optional<std::vector<c10::Stream>>& streams)
	{
		std::optional<std::vector<std::vector<int64_t>>> coilss = coils.has_value() ? std::make_optional(*coils) : std::nullopt;
		hasty::ffi::batched_sense_normal(input, coilss, smaps, coords, hasty::ffi::get_streams(streams));
	}

	void batched_sense_normal_weighted(at::Tensor& input, const at::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
		const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, const at::optional<std::vector<c10::Stream>>& streams)
	{
		std::optional<std::vector<std::vector<int64_t>>> coilss = coils.has_value() ? std::make_optional(*coils) : std::nullopt;
		hasty::ffi::batched_sense_normal_weighted(input, coilss, smaps, coords, weights, hasty::ffi::get_streams(streams));
	}

	void batched_sense_normal_kdata(at::Tensor& input, const at::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
		const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas, const at::optional<std::vector<c10::Stream>>& streams)
	{
		std::optional<std::vector<std::vector<int64_t>>> coilss = coils.has_value() ? std::make_optional(*coils) : std::nullopt;
		hasty::ffi::batched_sense_normal_kdata(input, coilss, smaps, coords, kdatas, hasty::ffi::get_streams(streams));
	}

	void batched_sense_normal_weighted_kdata(at::Tensor& input, const at::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
		const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, 
		const std::vector<at::Tensor>& kdatas, const at::optional<std::vector<c10::Stream>>& streams)
	{
		std::optional<std::vector<std::vector<int64_t>>> coilss = coils.has_value() ? std::make_optional(*coils) : std::nullopt;
		hasty::ffi::batched_sense_normal_weighted_kdata(input, coilss, smaps, coords, weights, kdatas, hasty::ffi::get_streams(streams));
	}

	// TOEPLITZ

	void batched_sense_toeplitz(at::Tensor& input, const std::vector<std::vector<int64_t>>& coils, const at::Tensor& smaps,
		const std::vector<at::Tensor>& coords, const at::optional<std::vector<c10::Stream>>& streams)
	{
		hasty::ffi::batched_sense_toeplitz(input, coils, smaps, coords, hasty::ffi::get_streams(streams));
	}

	void batched_sense_toeplitz_diagonals(at::Tensor& input, const std::vector<std::vector<int64_t>>& coils, const at::Tensor& smaps,
		const at::Tensor& diagonals, const at::optional<std::vector<c10::Stream>>& streams)
	{
		hasty::ffi::batched_sense_toeplitz_diagonals(input, coils, smaps, diagonals, hasty::ffi::get_streams(streams));
	}
}
*/

namespace hasty {
	namespace ffi {

		class LIB_EXPORT BatchedSense : public torch::CustomClassHolder {
		public:

			BatchedSense(const at::TensorList& coords, const at::Tensor& smaps,
				const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights,
				const at::optional<at::ArrayRef<at::Stream>>& streams);

			void apply(const at::Tensor& in, at::TensorList out,
				const at::optional<std::vector<std::vector<int64_t>>>& coils);

		private:
			std::unique_ptr<hasty::BatchedSense> _bs;
		};

		class LIB_EXPORT BatchedSenseAdjoint : public torch::CustomClassHolder {
		public:

			BatchedSenseAdjoint(const at::TensorList& coords, const at::Tensor& smaps,
				const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights,
				const at::optional<at::ArrayRef<at::Stream>>& streams);

			void apply(const at::TensorList& in, at::Tensor out,
				const at::optional<std::vector<std::vector<int64_t>>>& coils);

		private:
			std::unique_ptr<hasty::BatchedSenseAdjoint> _bs;
		};

		class LIB_EXPORT BatchedSenseNormal : public torch::CustomClassHolder {
		public:

			BatchedSenseNormal(const at::TensorList& coords, const at::Tensor& smaps,
				const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights,
				const at::optional<at::ArrayRef<at::Stream>>& streams);

			void apply(const at::Tensor& in, at::Tensor out,
				const at::optional<std::vector<std::vector<int64_t>>>& coils);

		private:
			std::unique_ptr<hasty::BatchedSenseNormal> _bs;
		};

		class LIB_EXPORT BatchedSenseNormalAdjoint : public torch::CustomClassHolder {
		public:

			BatchedSenseNormalAdjoint(const at::TensorList& coords, const at::Tensor& smaps,
				const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights,
				const at::optional<at::ArrayRef<at::Stream>>& streams);

			void apply(const at::TensorList& in, at::TensorList out,
				const at::optional<std::vector<std::vector<int64_t>>>& coils);

		private:
			std::unique_ptr<hasty::BatchedSenseNormalAdjoint> _bs;
		};

	}
}

namespace dummy {

	void LIB_EXPORT dummy(at::TensorList tensorlist);

	void LIB_EXPORT stream_dummy(const at::optional<at::ArrayRef<at::Stream>>& streams, const torch::Tensor& in);
}

/*
	bs.class_<hasty::ffi::BatchedSense>("BatchedSense")
		.def(torch::init<const at::TensorList&, const at::Tensor&,
			const at::optional<at::TensorList>&, const at::optional<at::TensorList>&,
			const at::optional<at::ArrayRef<at::Stream>>&>())
		.def("apply", &hasty::ffi::BatchedSense::apply);
	*/

/*
.def(py::init<const at::TensorList&, const at::Tensor&,
			const at::optional<at::TensorList>&, const at::optional<at::TensorList>&,
			const at::optional<std::vector<at::Stream>>&>())
*/


/*
using namespace at;

TORCH_LIBRARY(HastySense, m) {
	
	m.def("dummy", dummy::dummy);

	m.def("nufft1", nufft::nufft1);
	m.def("nufft2", nufft::nufft2);
	m.def("nufft21", nufft::nufft21);
	m.def("nufft12", nufft::nufft12);
	
	m.def("batched_sense_forward", bs::batched_sense_forward);

	m.def("batched_sense_forward_weighted", bs::batched_sense_forward_weighted);
	m.def("batched_sense_forward_kdata", bs::batched_sense_forward_kdata);
	m.def("batched_sense_forward_weighted_kdata", bs::batched_sense_forward_weighted_kdata);

	
	m.def("batched_sense_adjoint", bs::batched_sense_adjoint);
	m.def("batched_sense_adjoint_weighted", bs::batched_sense_adjoint_weighted);
	m.def("batched_sense_adjoint_kdata", bs::batched_sense_adjoint_kdata);
	m.def("batched_sense_adjoint_weighted_kdata", bs::batched_sense_adjoint_weighted_kdata);

	m.def("batched_sense_normal", bs::batched_sense_normal);
	m.def("batched_sense_normal_weighted", bs::batched_sense_normal_weighted);
	m.def("batched_sense_normal_kdata", bs::batched_sense_normal_kdata);
	m.def("batched_sense_normal_weighted_kdata", bs::batched_sense_normal_weighted_kdata);
	
	m.def("batched_sense_toeplitz", bs::batched_sense_toeplitz);
	m.def("batched_sense_toeplitz_diagonals", bs::batched_sense_toeplitz_diagonals);
	
}
*/







