module;

export module hasty_tensor_mod:masking;

import std;
import hasty_util_mod;
import :tensor;
import :external;

namespace hasty {

namespace {

void sphere_offsets_rec(
    std::vector<std::vector<i64>>& result,
    std::vector<i64>& cur,
    i64 radius2, i64 ndim, i64 depth, i64 dist2_so_far)
{
    if (depth == ndim) {
        result.push_back(cur);
        return;
    }
    i64 r = static_cast<i64>(std::sqrt(static_cast<double>(radius2 - dist2_so_far)));
    for (i64 d = -r; d <= r; ++d) {
        i64 new_dist2 = dist2_so_far + d * d;
        if (new_dist2 <= radius2) {
            cur.push_back(d);
            sphere_offsets_rec(result, cur, radius2, ndim, depth + 1, new_dist2);
            cur.pop_back();
        }
    }
}

std::vector<std::vector<i64>> sphere_offsets(i64 radius, i64 ndim)
{
    std::vector<std::vector<i64>> result;
    std::vector<i64> cur;
    sphere_offsets_rec(result, cur, radius * radius, ndim, 0, 0LL);
    return result;
}

struct MorphPrep {
    Tensor           flat;
    std::vector<i64> pre_flat;
    std::vector<i64> perm;
    i64              n_batch;
};

MorphPrep prep_morph(Tensor t, const std::vector<i64>& batch_dims)
{
    i64 ndim = t.ndimension();

    std::vector<i64> bdims;
    bdims.reserve(batch_dims.size());
    for (auto d : batch_dims)
        bdims.push_back(d < 0 ? d + ndim : d);

    std::set<i64> bset(bdims.begin(), bdims.end());

    std::vector<i64> perm;
    perm.reserve(ndim);
    for (auto d : bdims) perm.push_back(d);
    for (i64 d = 0; d < ndim; ++d)
        if (!bset.count(d)) perm.push_back(d);

    bool is_id = true;
    for (i64 i = 0; i < ndim; ++i)
        if (perm[i] != i) { is_id = false; break; }

    Tensor pt = is_id ? std::move(t) : t.permute(ArrayRef<i64>(perm));
    std::vector<i64> pre_flat(pt.sizes().begin(), pt.sizes().end());

    i64 n_batch = (i64)bdims.size();
    i64 batch_size = 1;
    for (i64 i = 0; i < n_batch; ++i) batch_size *= pre_flat[i];

    std::vector<i64> flat_shape = {batch_size};
    for (i64 i = n_batch; i < ndim; ++i) flat_shape.push_back(pre_flat[i]);

    return { pt.reshape(ArrayRef<i64>(flat_shape)), pre_flat, perm, n_batch };
}

Tensor unprep_morph(Tensor t, const MorphPrep& prep)
{
    i64 ndim = (i64)prep.perm.size();
    t = t.reshape(ArrayRef<i64>(prep.pre_flat));

    bool is_id = true;
    for (i64 i = 0; i < ndim; ++i)
        if (prep.perm[i] != i) { is_id = false; break; }
    if (is_id) return t;

    std::vector<i64> inv(ndim);
    for (i64 i = 0; i < ndim; ++i) inv[prep.perm[i]] = i;
    return t.permute(ArrayRef<i64>(inv));
}

// Build spherical convolution kernel [1,1,sz,...,sz] with 1.0 inside L2 ball.
// Built on CPU then moved to target device.
Tensor make_sphere_kernel(i64 radius, i64 spatial_ndim, Device dev)
{
    i64 sz = 2 * radius + 1;
    std::vector<i64> shape(spatial_ndim, sz);
    auto kernel = zeros(ArrayRef<i64>(shape), Opt<Device>{}, Opt<eScalarType>(eScalarType::Float));
    auto one    = ones( ArrayRef<i64>({1}),  Opt<Device>{}, Opt<eScalarType>(eScalarType::Float));

    auto offsets = sphere_offsets(radius, spatial_ndim);
    for (const auto& off : offsets) {
        std::vector<TensorIndex> idx;
        idx.reserve(spatial_ndim);
        for (i64 d = 0; d < spatial_ndim; ++d)
            idx.emplace_back(TensorIndex(off[d] + radius));
        kernel.index_put_(ArrayRef<TensorIndex>(idx), one);
    }

    std::vector<i64> kern_shape = {1, 1};
    for (i64 d = 0; d < spatial_ndim; ++d) kern_shape.push_back(sz);
    return kernel.reshape(ArrayRef<i64>(kern_shape)).to(dev);
}

// GPU-native dilation/erosion via single convolution with spherical kernel.
// Dispatches to conv1d/2d/3d for spatial_ndim ≤ 3 via wrapped convolution().
Tensor morph_conv(const Tensor& mask, i64 radius, i64 spatial_ndim, bool dilate)
{
    auto kernel     = make_sphere_kernel(radius, spatial_ndim, mask.device());
    i64  kernel_sum = static_cast<i64>(kernel.sum().item<float>() + 0.5f);

    // [B, s0,...,sN-1] → [B, 1, s0,...,sN-1]
    Tensor m = mask.to(eScalarType::Float).unsqueeze(1);

    std::vector<i64> stride(spatial_ndim, 1);
    std::vector<i64> padding(spatial_ndim, radius);
    std::vector<i64> dil(spatial_ndim, 1);

    Tensor out = convolution(
        m, kernel,
        ArrayRef<i64>(stride), ArrayRef<i64>(padding), ArrayRef<i64>(dil)
    ).squeeze(1);

    if (dilate)
        return out > Scalar(0.5f);
    else
        return out >= Scalar(static_cast<float>(kernel_sum) - 0.5f);
}

// Roll-based fallback for spatial_ndim > 3 (convolution doesn't support >3D).
// Runs on the tensor's current device — no CPU transfer.
Tensor morph_roll(const Tensor& mask, i64 radius, i64 spatial_ndim, bool dilate)
{
    Tensor m = dilate ? mask.to(eScalarType::Float)
                      : mask.logical_not().to(eScalarType::Float);
    auto result  = zeros_like(m);
    auto offsets = sphere_offsets(radius, spatial_ndim);

    for (const auto& off : offsets) {
        std::vector<i64> shifts, dims;
        for (i64 d = 0; d < spatial_ndim; ++d) {
            if (off[d] != 0) {
                shifts.push_back(off[d]);
                dims.push_back(d + 1);
            }
        }

        Tensor shifted = shifts.empty()
            ? m
            : m.roll(ArrayRef<i64>(shifts), ArrayRef<i64>(dims));

        for (i64 d = 0; d < spatial_ndim; ++d) {
            if (off[d] == 0) continue;
            i64 dim = d + 1;
            i64 sz = shifted.size(dim);
            if (off[d] > 0)
                shifted.narrow(dim, 0, off[d]).fill_(Scalar(0.0f));
            else
                shifted.narrow(dim, sz + off[d], -off[d]).fill_(Scalar(0.0f));
        }

        result = result + shifted;
    }

    Tensor out = result > Scalar(0.5f);
    return dilate ? out : out.logical_not();
}

Tensor dilate_impl(const Tensor& mask, i64 radius, i64 spatial_ndim)
{
    return spatial_ndim <= 3
        ? morph_conv(mask, radius, spatial_ndim, true)
        : morph_roll(mask, radius, spatial_ndim, true);
}

Tensor erode_impl(const Tensor& mask, i64 radius, i64 spatial_ndim)
{
    return spatial_ndim <= 3
        ? morph_conv(mask, radius, spatial_ndim, false)
        : morph_roll(mask, radius, spatial_ndim, false);
}

} // namespace

// Morphological dilation with spherical (L2) structuring element.
// Uses GPU convolution for spatial_ndim ≤ 3; roll-based for higher dims.
export Tensor mask_dilate(Tensor mask, i64 radius = 1, std::vector<i64> batch_dims = {})
{
    if (mask.scalar_type() != eScalarType::Bool)
        throw std::runtime_error("[mask_dilate] mask must be bool");

    i64 ndim = mask.ndimension();
    i64 spatial_ndim = ndim - (i64)batch_dims.size();
    if (spatial_ndim < 1)
        throw std::runtime_error("[mask_dilate] no spatial dims remaining after batch dims");

    auto prep = prep_morph(std::move(mask), batch_dims);
    return unprep_morph(dilate_impl(prep.flat, radius, spatial_ndim), prep);
}

// Morphological erosion with spherical (L2) structuring element.
export Tensor mask_erode(Tensor mask, i64 radius = 1, std::vector<i64> batch_dims = {})
{
    if (mask.scalar_type() != eScalarType::Bool)
        throw std::runtime_error("[mask_erode] mask must be bool");

    i64 ndim = mask.ndimension();
    i64 spatial_ndim = ndim - (i64)batch_dims.size();
    if (spatial_ndim < 1)
        throw std::runtime_error("[mask_erode] no spatial dims remaining after batch dims");

    auto prep = prep_morph(std::move(mask), batch_dims);
    return unprep_morph(erode_impl(prep.flat, radius, spatial_ndim), prep);
}

}
