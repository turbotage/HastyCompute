#pragma once

#include "../torch_util.hpp"
#include <cufinufft_opts.h>


struct cufinufft_plan_s;
typedef cufinufft_plan_s* cufinufft_plan;

struct cufinufftf_plan_s;
typedef cufinufftf_plan_s* cufinufftf_plan;


namespace hasty {

	enum NufftType {
		eType1 = 1,
		eType2 = 2,
		eType3 = 3
	};

	class NufftOptions {
	public:

		inline static NufftOptions type1(double tol = 1e-6) { return { NufftType::eType1, true, tol }; }

		inline static NufftOptions type2(double tol = 1e-6) { return { NufftType::eType2, false, tol }; }

	public:
		NufftType type = NufftType::eType1;
		bool positive = true;
		double tol = 1e-6;

		const NufftType& get_type() const { return type; }

		int get_positive() const { return positive ? 1 : -1; }

		double get_tol() const { return tol; }
	};

	class Nufft {
	public:

		Nufft(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const NufftOptions& opts = NufftOptions{});

		void close();
		~Nufft();

		void apply(const at::Tensor& in, at::Tensor& out) const;

		c10::ScalarType coord_type();

		c10::ScalarType complex_type();

		int32_t nfreq();

		int32_t ndim();

	protected:

		void make_plan_set_pts();

		void apply_type1(const at::Tensor& in, at::Tensor& out) const;

		void apply_type2(const at::Tensor& in, at::Tensor& out) const;

		c10::ScalarType			_type;
		int32_t					_ndim;
		int32_t					_ntransf;
		int32_t					_nfreq;
		const at::Tensor		_coords;
		std::vector<int64_t>	_nmodes;
		std::array<int32_t, 3>	_nmodes_flipped;
		NufftOptions			_opts;

		bool _closed = false;
		cufinufft_plan			_plan;
		cufinufftf_plan			_planf;

		cufinufft_opts _finufft_opts;

	};

	class NufftNormal {
	public:

		NufftNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const NufftOptions& forward_ops, const NufftOptions& backward_ops);

		void apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage, std::optional<std::function<void(at::Tensor&)>> func_between) const;

		void apply_inplace(at::Tensor& in, at::Tensor& storage, std::optional<std::function<void(at::Tensor&)>> func_between) const;

		c10::ScalarType coord_type();

		c10::ScalarType complex_type();

		int32_t nfreq();

		int32_t ndim();

		const Nufft& get_forward();

		const Nufft& get_backward();

	protected:
		Nufft				_forward;
		Nufft				_backward;
	};

	class NormalNufftToeplitz {
	public:

		static void build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, double tol, at::Tensor& diagonal);

		static void build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, double tol, at::Tensor& diagonal,
			at::Tensor& storage, bool storage_is_frequency);

		static void build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, double tol, at::Tensor& diagonal,
			at::Tensor& frequency_storage, at::Tensor& input_storage);

	public:

		NormalNufftToeplitz(const at::Tensor& coords, const std::vector<int64_t>& nmodes, std::optional<double> tol,
			std::optional<std::reference_wrapper<at::Tensor>> diagonal, 
			std::optional<std::reference_wrapper<at::Tensor>> frequency_storage,
			std::optional<std::reference_wrapper<at::Tensor>> input_storage);

		NormalNufftToeplitz(at::Tensor&& diagonal, const std::vector<int64_t>& nmodes);

		at::Tensor apply(const at::Tensor& in, at::Tensor& storage1, at::Tensor& storage2) const;

		void apply_add(const at::Tensor& in, at::Tensor& add_to, at::Tensor& storage1, at::Tensor& storage2) const;

		void apply_addcmul(const at::Tensor& in, at::Tensor& add_to, const at::Tensor& mul, at::Tensor& storage1, at::Tensor& storage2) const;

		void apply_inplace(at::Tensor& in, at::Tensor& storage1, at::Tensor& storage2) const;

		at::Tensor get_diagonal();

	private:
			
		bool _created_from_diagonal;

		c10::ScalarType								_type;
		int32_t										_ntransf;
		int32_t										_ndim;
		int32_t										_nfreq;

		std::vector<int64_t>						_nmodes;
		std::vector<int64_t>						_transdims;
		std::vector<int64_t>						_nmodes_ns;

		c10::IntArrayRef							_transform_dims;
		c10::IntArrayRef							_expanded_dims;
		std::vector<at::indexing::TensorIndex>		_index_vector;
		c10::ArrayRef<at::indexing::TensorIndex>	_indices;

		at::Tensor									_diagonal;
	};

}



