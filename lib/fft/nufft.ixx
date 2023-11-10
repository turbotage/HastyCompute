module;

#include "../torch_util.hpp"
#include <cufinufft_opts.h>
#include <finufft_opts.h>

export module nufft;

typedef struct finufft_plan_s* finufft_plan;
typedef struct finufftf_plan_s* finufftf_plan;

typedef struct cufinufft_plan_s* cufinufft_plan;
typedef struct cufinufft_fplan_s* cufinufftf_plan;

namespace hasty {
	namespace nufft {

		export enum NufftType {
			eType1 = 1, // ADJOINT
			eType2 = 2, // FORWARD
			eType3 = 3
		};

		export class NufftOptions {
		public:

			NufftOptions() = default;

			NufftOptions(NufftType intype, const at::optional<bool>& inpositive, const at::optional<double>& intol)
				: type(intype)
			{
				if (inpositive.has_value())
					positive = *inpositive;
				if (intol.has_value())
					tol = *intol;
			}

			inline static NufftOptions type1() { return { NufftType::eType1, true, 1e-5 }; }

			inline static NufftOptions type2() { return { NufftType::eType2, false, 1e-5 }; }

			inline static NufftOptions type1(double tol) { return { NufftType::eType1, true, tol }; }

			inline static NufftOptions type2(double tol) { return { NufftType::eType2, false, tol }; }

			inline static NufftOptions type1(bool positive) { return { NufftType::eType1, positive, 1e-5 }; }

			inline static NufftOptions type2(bool positive) { return { NufftType::eType2, positive, 1e-5 }; }

			inline static NufftOptions type1(bool positive, double tol) { return { NufftType::eType1, positive, tol }; }

			inline static NufftOptions type2(bool positive, double tol) { return { NufftType::eType2, positive, tol }; }

		public:
			NufftType type = NufftType::eType1;
			bool positive = true;
			double tol = 1e-5;

			const NufftType& get_type() const { return type; }

			int get_positive() const { return positive ? 1 : -1; }

			double get_tol() const { return tol; }
		};

		export at::Tensor allocate_out(const at::Tensor& coords, int ntransf);

		export at::Tensor allocate_adjoint_out(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

		export class Nufft {
		public:

			Nufft(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const NufftOptions& opts = NufftOptions{});

			void close();
			~Nufft();

			void apply(const at::Tensor& in, at::Tensor& out) const;

			at::ScalarType coord_type();

			at::ScalarType complex_type();

			int32_t nfreq();

			int32_t ndim();

		protected:

			void make_plan_set_pts();

			void apply_type1(const at::Tensor& in, at::Tensor& out) const;

			void apply_type2(const at::Tensor& in, at::Tensor& out) const;

			at::ScalarType			_type;
			int32_t					_ndim;
			int32_t					_ntransf;
			int64_t					_nfreq;
			const at::Tensor		_coords;
			std::vector<int64_t>	_nmodes;
			std::array<int64_t, 3>	_nmodes_flipped;
			NufftOptions			_opts;

			bool _closed = false;
			finufft_plan			_plan;
			finufftf_plan			_planf;

			finufft_opts _finufft_opts;

			int32_t _nvoxels;
		};

		export class CUDANufft {
		public:

			CUDANufft(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const NufftOptions& opts = NufftOptions{});

			void close();
			~CUDANufft();

			void apply(const at::Tensor& in, at::Tensor& out) const;

			at::ScalarType coord_type();

			at::ScalarType complex_type();

			int32_t nfreq();

			int32_t ndim();

		protected:

			void make_plan_set_pts();

			void apply_type1(const at::Tensor& in, at::Tensor& out) const;

			void apply_type2(const at::Tensor& in, at::Tensor& out) const;

			at::ScalarType			_type;
			int32_t					_ndim;
			int32_t					_ntransf;
			int32_t					_nfreq;
			const at::Tensor		_coords;
			std::vector<int64_t>	_nmodes;
			std::array<int64_t, 3>	_nmodes_flipped;
			NufftOptions			_opts;

			bool _closed = false;
			cufinufft_plan			_plan;
			cufinufftf_plan			_planf;

			cufinufft_opts _finufft_opts;

			int32_t _nvoxels;
		};

		export at::Tensor allocate_normal_out(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

		export at::Tensor allocate_normal_storage(const at::Tensor& coords, int ntransf);

		export at::Tensor allocate_normal_adjoint_out(const at::Tensor& coords, int ntransf);

		export at::Tensor allocate_normal_adjoint_storage(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

		export class NufftNormal {
		public:

			NufftNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const NufftOptions& forward_ops, const NufftOptions& backward_ops);

			void apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage, at::optional<std::function<void(at::Tensor&)>> func_between) const;

			void apply_inplace(at::Tensor& in, at::Tensor& storage, at::optional<std::function<void(at::Tensor&)>> func_between) const;

			void apply_forward(const at::Tensor& in, at::Tensor& out);

			void apply_backward(const at::Tensor& in, at::Tensor& out);

			at::ScalarType coord_type();

			at::ScalarType complex_type();

			int32_t nfreq();

			int32_t ndim();

			const Nufft& get_forward();

			const Nufft& get_backward();

		protected:
			Nufft				_forward;
			Nufft				_backward;
		};

		export class CUDANufftNormal {
		public:

			CUDANufftNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const NufftOptions& forward_ops, const NufftOptions& backward_ops);

			void apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage, at::optional<std::function<void(at::Tensor&)>> func_between) const;

			void apply_inplace(at::Tensor& in, at::Tensor& storage, at::optional<std::function<void(at::Tensor&)>> func_between) const;

			void apply_forward(const at::Tensor& in, at::Tensor& out);

			void apply_backward(const at::Tensor& in, at::Tensor& out);

			at::ScalarType coord_type();

			at::ScalarType complex_type();

			int32_t nfreq();

			int32_t ndim();

			const CUDANufft& get_forward();

			const CUDANufft& get_backward();

		protected:
			CUDANufft				_forward;
			CUDANufft				_backward;
		};


		class CUDANormalNufftToeplitz {
		public:

			static void build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, double tol, at::Tensor& diagonal);

			static void build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, double tol, at::Tensor& diagonal,
				at::Tensor& storage, bool storage_is_frequency);

			static void build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, double tol, at::Tensor& diagonal,
				at::Tensor& frequency_storage, at::Tensor& input_storage);

		public:

			CUDANormalNufftToeplitz(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const at::optional<double>& tol,
				const at::optional<std::reference_wrapper<at::Tensor>>& diagonal,
				const at::optional<std::reference_wrapper<at::Tensor>>& frequency_storage,
				const at::optional<std::reference_wrapper<at::Tensor>>& input_storage);

			CUDANormalNufftToeplitz(at::Tensor&& diagonal, const std::vector<int64_t>& nmodes);

			at::Tensor apply(const at::Tensor& in, at::Tensor& storage1, at::Tensor& storage2) const;

			void apply_add(const at::Tensor& in, at::Tensor& add_to, at::Tensor& storage1, at::Tensor& storage2) const;

			void apply_addcmul(const at::Tensor& in, at::Tensor& add_to, const at::Tensor& mul, at::Tensor& storage1, at::Tensor& storage2) const;

			void apply_inplace(at::Tensor& in, at::Tensor& storage1, at::Tensor& storage2) const;

			at::Tensor get_diagonal();

		private:

			bool _created_from_diagonal;

			at::ScalarType								_type;
			int32_t										_ntransf;
			int32_t										_ndim;
			int32_t										_nfreq;

			std::vector<int64_t>						_nmodes;
			std::vector<int64_t>						_transdims;
			std::vector<int64_t>						_nmodes_ns;

			at::IntArrayRef								_transform_dims;
			at::IntArrayRef								_expanded_dims;
			std::vector<at::indexing::TensorIndex>		_index_vector;
			at::ArrayRef<at::indexing::TensorIndex>		_indices;

			at::Tensor									_diagonal;
		};

	}

	namespace grid {

		class Gridding {
		public:

			Gridding(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

			void close();

			void apply(const at::Tensor& in, at::Tensor& out) const;

			at::ScalarType coord_type();

			at::ScalarType complex_type();

			int32_t nfreq();

			int32_t ndim();

		private:
			at::ScalarType			_type;
			int32_t					_ndim;
			int32_t					_ntransf;
			int64_t					_nfreq;
			const at::Tensor		_coords;
			std::vector<int64_t>	_nmodes;
			std::array<int64_t, 3>	_nmodes_flipped;

			bool _closed = false;
			finufft_plan			_plan;
			finufftf_plan			_planf;

			finufft_opts _finufft_opts;

			int32_t _nvoxels;
		};

	}

}


