module;

export module mri_mod:forward;

import hasty_util_mod;
import hasty_tensor_mod;

namespace hasty {
namespace mri {

export Tensor forward(
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