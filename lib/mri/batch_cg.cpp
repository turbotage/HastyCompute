module;

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

module batch_cg;

import precond;

hasty::mri::SenseAdmmLoader::SenseAdmmLoader(
	const std::vector<at::Tensor>& coords, const std::vector<int64_t>& nmodes,
	const std::vector<at::Tensor>& kdata, const at::Tensor& smaps,
	std::shared_ptr<op::Admm::Context> ctx,
	const at::optional<std::vector<at::Tensor>>& preconds)
	: _coords(coords), _nmodes(nmodes), _kdata(kdata), _smaps(smaps), _ctx(std::move(ctx)), _preconds(preconds)
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
	
	std::shared_ptr<op::AdjointableOp> AH;
	std::shared_ptr<op::AdjointableOp> A;
	std::shared_ptr<op::AdjointableOp> B;
	
	op::Vector z;
	op::Vector u;
	op::Vector c;

	op::Vector CGvec;

	{
		std::unique_lock<std::mutex> lock(_ctx->ctxmut);

		auto localA = op::downcast<op::AdjointableVStackedOp>(_ctx->A);
		if (!localA)
			throw std::runtime_error("SenseAdmmLoader requires A to be AdjointableVStackedOp");
		A = op::downcast<op::AdjointableOp>(localA->get_slice_ptr(idx)->to_device(dctx.stream));
		AH = A->adjoint();

		auto localB = op::downcast<op::AdjointableVStackedOp>(_ctx->B);
		if (!localB)
			throw std::runtime_error("SenseAdmmLoader requires B to be AdjointableVStackedOp");
		B = op::downcast<op::AdjointableOp>(localB->get_slice_ptr(idx)->to_device(dctx.stream));

		if (_ctx->AHA != nullptr) {
			auto localAHA = op::downcast<op::AdjointableVStackedOp>(_ctx->AHA);
			if (!localAHA)
				throw std::runtime_error("SenseAdmmLoader requires AHA to be AdjointableVStackedOp");

			AHA = op::downcast<op::AdjointableOp>(std::move(localAHA->get_slice_ptr(idx)->to_device(dctx.stream)));
		}
		else {
			AHA = op::upcast<op::AdjointableOp>(std::move(op::mul(AH, A)));
		}

		if (!AHA)
			throw std::runtime_error("AHA could not be cast to an AdjointableOp on this device");

		z = _ctx->z[idx].copy().to_device(dctx.stream);
		u = _ctx->u[idx].copy().to_device(dctx.stream);
		c = _ctx->c[idx].copy().to_device(dctx.stream);
	}
	
	AHA = op::mul(at::tensor(0.5 * _ctx->rho), AHA);

	CGOp = op::staticcast<op::AdjointableOp>(op::add(std::move(SHS), std::move(AHA)));

	if (c)
		CGvec = _ctx->rho * AH->apply(c - B->apply(z) - u);
	else
		CGvec = (0.5*_ctx->rho) * AH->apply(B->apply(z) - u);
		
	CGvec += SHS->apply_backward(op::Vector(kdata));

	return ConjugateGradientLoadResult{ std::move(CGop), std::move(CGvec), _preconds.has_value() ?
		CirculantPreconditionerOp::Create((*_preconds)[idx], true, at::nullopt, false) : nullptr
	};
}



