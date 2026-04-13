module;

export module mri_mod:forward;

import tensor_mod;
import hasty_util_mod;

namespace hasty {
namespace mri {

Tensor forward(
    const Tensor& magnetization, 
    const Tensor& sensitivity_maps, 
    const Tensor& rate_map,
    const Tensor& timestamps,
    const Tensor& kspace_trajectory,
    const Tensor& nonlin_gradient_waveforms,
    const Tensor& nonlin_gradient_basis
) {
    
}

}
}