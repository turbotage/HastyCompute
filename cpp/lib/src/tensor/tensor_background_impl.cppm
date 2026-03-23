module;

export module hasty_tensor_mod:background_impl;

import :background;
import :tensor;

hasty::TensorIndex::TensorIndex(const hasty::Tensor& tensor)
    : m_torch_index(tensor.to_torch())
{}

hasty::Tensor hasty::TensorIndex::tensor() const {
    return hasty::Tensor(m_torch_index.tensor());
}

