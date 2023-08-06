#pragma once

#include "../fft/nufft.hpp"
#include "../threading/thread_pool.hpp"


namespace hasty {

	using TensorVec = std::vector<at::Tensor>;

	using CoilApplier = std::function<void(at::Tensor&, int32_t)>;
	struct CoilManipulator {
		CoilManipulator() = default;

		CoilManipulator& setPreApply(const CoilApplier& apply) {
			if (preapplier.has_value())
				throw std::runtime_error("Tried to set non nullopt applier");
			preapplier = std::make_optional(apply);
			return *this;
		}

		CoilManipulator& setMidApply(const CoilApplier& apply) {
			if (midapplier.has_value())
				throw std::runtime_error("Tried to set non nullopt applier");
			midapplier = std::make_optional(apply);
			return *this;
		}

		CoilManipulator& setPostApply(const CoilApplier& apply) {
			if (postapplier.has_value())
				throw std::runtime_error("Tried to set non nullopt applier");
			postapplier = std::make_optional(apply);
			return *this;
		}

		std::optional<CoilApplier> preapplier;
		std::optional<CoilApplier> midapplier;
		std::optional<CoilApplier> postapplier;
	};

	class Sense {
	public:

		Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

		void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
			const std::optional<at::Tensor>& imspace_storage, const std::optional<at::Tensor>& kspace_storage,
			const std::optional<CoilApplier>& premanip,
			const std::optional<CoilApplier>& postmanip);

	private:
		Nufft _nufft;
		std::vector<int64_t> _nmodes;
	};

	class SenseAdjoint {
	public:

		SenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

		void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
			const std::optional<at::Tensor>& imspace_storage, const std::optional<at::Tensor>& kspace_storage,
			const std::optional<CoilApplier>& premanip,
			const std::optional<CoilApplier>& postmanip);

	private:
		Nufft _nufft;
		std::vector<int64_t> _nmodes;
	};

	class SenseNormal {
	public:

		SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

		void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
			const std::optional<at::Tensor>& imspace_storage, const std::optional<at::Tensor>& kspace_storage,
			const std::optional<CoilApplier>& premanip,
			const std::optional<CoilApplier>& midmanip,
			const std::optional<CoilApplier>& postmanip);

	private:

		NufftNormal _normal_nufft;
		std::vector<int64_t> _nmodes;

	};

	class SenseNormalAdjoint {
	public:

		SenseNormalAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

		void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
			const std::optional<at::Tensor>& imspace_storage,
			const std::optional<CoilApplier>& premanip,
			const std::optional<CoilApplier>& midmanip,
			const std::optional<CoilApplier>& postmanip);

	private:

		NufftNormal _normal_nufft;
		std::vector<int64_t> _nmodes;
	};

	class SenseNormalToeplitz {
	public:

		SenseNormalToeplitz(const at::Tensor& coords, const std::vector<int64_t>& nmodes, double tol);

		SenseNormalToeplitz(at::Tensor&& diagonal, const std::vector<int64_t>& nmodes);

		void apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage1, at::Tensor& storage2,
			const at::Tensor& smaps, const std::vector<int64_t>& coils) const;

	private:

		NormalNufftToeplitz _normal_nufft;

	};

}


