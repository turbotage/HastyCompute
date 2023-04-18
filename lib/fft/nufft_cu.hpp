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
			bool positive = false;
			double tol = 1e-6;

			const NufftType& get_type() const { return type; }

			int get_positive() const { return positive ? 1 : -1; }

			double get_tol() const { return tol; }
		};

		class Nufft {
		public:

			Nufft(const at::Tensor& coords, const std::vector<int32_t>& nmodes, const NufftOptions& opts = NufftOptions{});

			~Nufft();

			void apply(const at::Tensor& in, at::Tensor& out);

		protected:

			void make_plan_set_pts();

			void apply_type1(const at::Tensor& in, at::Tensor& out);

			void apply_type2(const at::Tensor& in, at::Tensor& out);

			c10::ScalarType			_type;
			int32_t					_ndim;
			int32_t					_ntransf;
			int32_t					_nfreq;
			const at::Tensor		_coords;
			std::vector<int32_t>	_nmodes;
			std::array<int32_t, 3>	_nmodes_flipped;
			NufftOptions			_opts;

			cufinufft_plan			_plan;
			cufinufftf_plan			_planf;

		};

		class NufftNormal {
		public:

			NufftNormal(const at::Tensor& coords, const std::vector<int32_t>& nmodes, const NufftOptions& forward_ops, const NufftOptions& backward_ops);

			void apply_1to2(const at::Tensor& in, at::Tensor& out, at::Tensor& storage, std::optional<std::function<void(at::Tensor&)>> func_between);

			void apply_2to1(const at::Tensor& in, at::Tensor& out, at::Tensor& storage, std::optional<std::function<void(at::Tensor&)>> func_between);

		protected:


			const at::Tensor _coords;
			Nufft _forward;
			Nufft _backward;
			
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



