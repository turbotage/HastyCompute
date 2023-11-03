#include "batch_cg.hpp"

#include "../op/mriop.hpp"

#include <c10/cuda/CUDAGuard.h>

hasty::mri::SenseAdmmLoader::SenseAdmmLoader(
	const std::vector<at::Tensor>& coords, const std::vector<int64_t>& nmodes,
	const std::vector<at::Tensor>& kdata, const at::Tensor& smaps, double rho, 
	const at::optional<std::vector<at::Tensor>>& preconds)
	: _coords(coords), _nmodes(nmodes), _kdata(kdata), _smaps(smaps), _rho(rho), _preconds(preconds)
{

}

hasty::op::ConjugateGradientLoadResult hasty::mri::SenseAdmmLoader::load(SenseDeviceContext& dctx, size_t idx)
{
	c10::InferenceMode im_guard;
	c10::cuda::CUDAStreamGuard guard(dctx.stream);
	
	auto device = dctx.stream.device();

	auto coords = _coords[idx].to(device, true);
	auto kdata = _kdata[idx].to(device, true);

	std::vector<int64_t> coils(dctx.smaps.size(0));
	std::generate(coils.begin(), coils.end(), [n = 0]() mutable { return n++; });

	op::SenseN senseop(coords, _nmodes, dctx.smaps, coils);
	op::Operator A = senseop + op::ScaleOp(at::tensor(_rho));

	op::Vector b = senseop.apply_backward(op::Vector(kdata));
}

