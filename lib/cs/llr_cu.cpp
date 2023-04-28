#include "llr_cu.hpp"

#include <random>

hasty::cuda::LLR_4DEncodes::LLR_4DEncodes(
	const LLR_4DEncodesOptions& options,
	at::Tensor& image,
	const CTensorVecVec& coords,
	const at::Tensor& smaps,
	const CTensorVecVec& kdata)
	:
	_options(options),
	_image(image),
	_coords(coords),
	_smaps(smaps),
	_kdata(kdata)
{
	
}
