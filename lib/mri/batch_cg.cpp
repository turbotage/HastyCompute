module;

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

module batch_cg;


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
	std::generate(coils.begin(), coils.end(), [n = int64_t(0)]() mutable { return n++; });

	std::shared_ptr<op::Operator> CGOp;

	std::shared_ptr<op::AdjointableOp> SHS = op::SenseNOp::Create(coords, _nmodes, dctx.smaps, coils);
	std::shared_ptr<op::AdjointableOp> AHA;
	op::Vector CGvec;

	{
		std::unique_lock<std::mutex> lock(_ctx->ctxmut);

		auto AH = op::staticcast<op::AdjointableOp>(std::move(_ctx->A->adjoint()->to_device(dctx.stream)));
		auto A = op::staticcast<op::AdjointableOp>(std::move(_ctx->A->to_device(dctx.stream)));
		auto B = op::staticcast<op::AdjointableOp>(std::move(_ctx->B->to_device(dctx.stream)));

		if (_ctx->AHA != nullptr) {
			AHA = op::staticcast<op::AdjointableOp>(std::move(_ctx->AHA->to_device(dctx.stream)));
		}
		else {
			AHA = op::upcast<op::AdjointableOp>(std::move(op::mul(AH, A)));
		}

		if (!AHA)
			throw std::runtime_error("AHA could not be cast to an AdjointableOp on this device");

		AHA = op::mul(at::tensor(_ctx->rho), AHA);

		CGOp = op::staticcast<op::AdjointableOp>(op::add(std::move(SHS), std::move(AHA)));

		CGvec = _ctx->rho * AH->apply(*_ctx->c - A->apply(*_ctx->x) - B->apply(*_ctx->z) - *_ctx->u);
		
		CGvec += SHS->apply_backward(op::Vector(kdata));

	}
	

	//std::shared_ptr<op::
	//op::Operator A = SHS + op::ScaleOp(at::tensor(_ctx->rho), );
	//op::Vector b = SHS.apply_backward(op::Vector(kdata));


}



