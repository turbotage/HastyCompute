#include "batch_cg.hpp"

#include "../op/mriop.hpp"

#include <c10/cuda/CUDAGuard.h>

hasty::mri::SenseAdmmLoader::SenseAdmmLoader(
	const std::vector<at::Tensor>& coords, const std::vector<int64_t>& nmodes,
	const std::vector<at::Tensor>& kdata, const at::Tensor& smaps,
	const std::shared_ptr<op::Admm::Context>& ctx,
	const at::optional<std::vector<at::Tensor>>& preconds)
	: _coords(coords), _nmodes(nmodes), _kdata(kdata), _smaps(smaps), _ctx(ctx), _preconds(preconds)
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

	std::shared_ptr<op::Operator> CGOp;

	std::shared_ptr<op::AdjointableOp> SHS = std::make_shared<op::SenseNOp>(coords, _nmodes, dctx.smaps, coils);
	std::shared_ptr<op::AdjointableOp> AHA;

	{
		std::unique_lock<std::mutex> lock(_ctx->ctxmut);

		auto AH = std::dynamic_pointer_cast<op::AdjointableOp>(_ctx->A->adjoint()->to_device(dctx.stream));

		if (_ctx->AHA != nullptr) {
			AHA = std::dynamic_pointer_cast<op::AdjointableOp>(_ctx->AHA->to_device(dctx.stream));
		}
		else {
			
			auto A = std::dynamic_pointer_cast<op::AdjointableOp>(_ctx->A->to_device(dctx.stream));
			AHA = std::static_pointer_cast<op::AdjointableOp>(op::mul_adj(std::move(AH), std::move(A)));
		}

		if (!AHA)
			throw std::runtime_error("AHA could not be cast to an AdjointableOp on this device");

		AHA = std::static_pointer_cast<op::AdjointableOp>(
			std::make_shared<op::AdjointableScaleOp>(at::tensor(_ctx->rho / 2.0), AHA)));

		CGOp = std::static_pointer_cast<op::AdjointableOp>(op::add(SHS, AHA));

		

	}



	//std::shared_ptr<op::
	//op::Operator A = SHS + op::ScaleOp(at::tensor(_ctx->rho), );
	//op::Vector b = SHS.apply_backward(op::Vector(kdata));


}


