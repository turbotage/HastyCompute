//module;
//#include <torch/torch.h>;
#include <torch/torch.h>

//export module nufft_cu;
#ifdef STL_AS_MODULES
import std.compat;
#else
import <memory>;
import <vector>;
import <string>;
import <stdexcept>;
import <array>;
import <functional>;
import <optional>;
#endif


struct cufinufft_plan_s;
typedef cufinufft_plan_s* cufinufft_plan;

struct cufinufftf_plan_s;
typedef cufinufftf_plan_s* cufinufftf_plan;


namespace hasty {

	namespace cuda {

		enum NufftType {
			eType1 = 1,
			eType2 = 2,
			eType3 = 3
		};

		class NufftOptions {
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

			~Nufft();

			void apply(const at::Tensor& in, at::Tensor& out) const;

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

			cufinufft_plan			_plan;
			cufinufftf_plan			_planf;

		};

		class NufftNormal {
		public:

			NufftNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const NufftOptions& forward_ops, const NufftOptions& backward_ops);

			void apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage, std::optional<std::function<void(at::Tensor&)>> func_between) const;

			const Nufft& get_forward();

			const Nufft& get_backward();

		protected:
			const at::Tensor	_coords;
			Nufft				_forward;
			Nufft				_backward;
		};

		class ToeplitzNormalNufft {
		public:

			static void build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, at::Tensor& diagonal);

			static void build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, at::Tensor& diagonal,
				at::Tensor& storage, bool storage_is_frequency);

			static void build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, at::Tensor& diagonal,
				at::Tensor& frequency_storage, at::Tensor& input_storage);

		public:

			ToeplitzNormalNufft(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				std::optional<std::reference_wrapper<at::Tensor>> diagonal, 
				std::optional<std::reference_wrapper<at::Tensor>> frequency_storage,
				std::optional<std::reference_wrapper<at::Tensor>> input_storage);

			void apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage1, at::Tensor& storage2) const;
			

		private:
			
			c10::ScalarType			_type;
			int32_t					_ntransf;
			int32_t					_ndim;
			int32_t					_nfreq;

			std::vector<int64_t>	_nmodes;
			std::vector<int64_t>	_transdims;
			std::vector<int64_t>	_nmodes_ns;

			at::Tensor				_diagonal;
		};

	}
}



