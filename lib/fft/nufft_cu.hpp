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
#endif


struct cufinufft_plan_s;
typedef cufinufft_plan_s* cufinufft_plan;

struct cufinufftf_plan_s;
typedef cufinufftf_plan_s* cufinufftf_plan;


namespace hasty {

	namespace cuda {

		class NufftOptions {
		public:
			bool positive = false;
			double tol = 1e-6;

			int get_positive() const { return positive ? 1 : -1; }

			double get_tol() const { return tol; }
		};

		enum NufftType {
			eType1 = 1,
			eType2 = 2,
			eType3 = 3
		};

		class Nufft {
		public:

			Nufft(const at::Tensor& coords, const std::vector<int32_t>& nmodes, const NufftType& type, const NufftOptions& opts = NufftOptions{});

			~Nufft();

			void apply(const at::Tensor& in, at::Tensor& out);

		protected:

			void make_plan_set_pts();

			void apply_type1(const at::Tensor& in, at::Tensor& out);

			void apply_type2(const at::Tensor& in, at::Tensor& out);

			c10::ScalarType			_type;
			NufftType				_nufftType;
			int32_t					_ndim;
			int32_t					_ntransf;
			int32_t					_nfreq;
			at::Tensor				_coords;
			std::vector<int32_t>	_nmodes;
			std::array<int32_t, 3>	_nmodes_flipped;
			NufftOptions			_opts;

			cufinufft_plan			_plan;
			cufinufftf_plan			_planf;

		};

		class ToeplitzNormalNufft : protected Nufft {
		public:

			ToeplitzNormalNufft(const at::Tensor& coords, const std::vector<int32_t>& nmodes);

			void apply(const at::Tensor& in, at::Tensor& out);

		private:
			at::Tensor _diagonal;
		};

	}
}



