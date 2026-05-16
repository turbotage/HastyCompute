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

// offsets passed in — caller already has them, avoids recomputing.
// kernel_sum = offsets.size() on CPU, no .item() GPU sync needed.
Tensor make_sphere_kernel(
    i64 radius, i64 spatial_ndim, Device dev,
    const std::vector<std::vector<i64>>& offsets)
{
    i64 sz = 2 * radius + 1;
    std::vector<i64> shape(spatial_ndim, sz);
    auto kernel = zeros(ArrayRef<i64>(shape), Opt<Device>{}, Opt<eScalarType>(eScalarType::Float));
    auto one    = ones( ArrayRef<i64>({1}),   Opt<Device>{}, Opt<eScalarType>(eScalarType::Float));

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

// Conv-based dilation/erosion for spatial_ndim <= 3.
// Kernel and offsets built once, reused across n_iters — no redundant H2D or .item() syncs.
Tensor morph_conv(const Tensor& mask, i64 radius, i64 spatial_ndim, bool dilate, i64 n_iters)
{
    auto offsets    = sphere_offsets(radius, spatial_ndim);
    auto kernel     = make_sphere_kernel(radius, spatial_ndim, mask.device(), offsets);
    i64  kernel_sum = (i64)offsets.size(); // offsets.size() == number of 1s in kernel, no GPU sync

    std::vector<i64> stride(spatial_ndim, 1);
    std::vector<i64> padding(spatial_ndim, radius);
    std::vector<i64> dil(spatial_ndim, 1);
    Scalar lo_thresh(0.5f);
    Scalar hi_thresh(static_cast<float>(kernel_sum) - 0.5f);

    // [B, s...] → [B, 1, s...] for convolution; kept in this shape between iterations
    Tensor cur = mask.to(eScalarType::Float).unsqueeze(1);
    for (i64 i = 0; i < n_iters; ++i) {
        Tensor conv_out = convolution(
            cur, kernel,
            ArrayRef<i64>(stride), ArrayRef<i64>(padding), ArrayRef<i64>(dil));

        Tensor bool_out = dilate ? (conv_out > lo_thresh) : (conv_out >= hi_thresh);

        if (i < n_iters - 1)
            cur = bool_out.to(eScalarType::Float); // keep [B, 1, s...] for next conv
        else
            cur = bool_out.squeeze(1);             // [B, s...] bool on final iter
    }
    return cur;
}

// Roll-based dilation/erosion for spatial_ndim > 3.
// offsets computed once and reused across n_iters.
// In-place result.add_() avoids per-offset tensor allocation.
Tensor morph_roll(const Tensor& mask, i64 radius, i64 spatial_ndim, bool dilate, i64 n_iters)
{
    auto offsets = sphere_offsets(radius, spatial_ndim);
    Tensor cur = mask;

    for (i64 iter = 0; iter < n_iters; ++iter) {
        Tensor m = dilate ? cur.to(eScalarType::Float)
                          : cur.logical_not().to(eScalarType::Float);

        auto result = zeros_like(m);
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
                i64 sz  = shifted.size(dim);
                if (off[d] > 0)
                    shifted.narrow(dim, 0, off[d]).fill_(Scalar(0.0f));
                else
                    shifted.narrow(dim, sz + off[d], -off[d]).fill_(Scalar(0.0f));
            }

            result.add_(shifted);
        }

        Tensor out = result > Scalar(0.5f);
        cur = dilate ? out : out.logical_not();
    }
    return cur;
}

Tensor dilate_impl(const Tensor& mask, i64 radius, i64 spatial_ndim, i64 n_iters)
{
    return spatial_ndim <= 3
        ? morph_conv(mask, radius, spatial_ndim, true,  n_iters)
        : morph_roll(mask, radius, spatial_ndim, true,  n_iters);
}

Tensor erode_impl(const Tensor& mask, i64 radius, i64 spatial_ndim, i64 n_iters)
{
    return spatial_ndim <= 3
        ? morph_conv(mask, radius, spatial_ndim, false, n_iters)
        : morph_roll(mask, radius, spatial_ndim, false, n_iters);
}

} // namespace

export Tensor ellipsoid_mask(
    std::vector<i64>    shape,
    std::vector<double> semiaxes,
    std::vector<double> offset  = {},
    TensorOptions       opts    = TensorOptions())
{
    i64 ndim = (i64)shape.size();
    if ((i64)semiaxes.size() != ndim)
        throw std::runtime_error("[ellipsoid_mask] shape and semiaxes must have same length");
    if (!offset.empty() && (i64)offset.size() != ndim)
        throw std::runtime_error("[ellipsoid_mask] offset must be empty or match ndim");

    TensorOptions float_opts = opts.dtype(eScalarType::Float);

    // Accumulate sum((coord_d / semiaxis_d)^2) via broadcasting — no full meshgrid
    Tensor dist2 = zeros(ArrayRef<i64>({1}), float_opts);
    for (i64 d = 0; d < ndim; ++d) {
        double center = (shape[d] - 1) * 0.5 + (offset.empty() ? 0.0 : offset[d]);
        Tensor coord = arange(shape[d], float_opts);
        coord = (coord - Scalar(center)) / Scalar(semiaxes[d]);
        coord = coord * coord;

        std::vector<i64> view_shape(ndim, 1LL);
        view_shape[d] = shape[d];
        dist2 = dist2 + coord.view(ArrayRef<i64>(view_shape));
    }

    return dist2 <= Scalar(1.0f);
}

export Tensor mask_dilate(Tensor mask, i64 radius = 1, i64 n_iters = 1, std::vector<i64> batch_dims = {})
{
    if (mask.scalar_type() != eScalarType::Bool)
        throw std::runtime_error("[mask_dilate] mask must be bool");
    if (radius < 1)
        throw std::runtime_error("[mask_dilate] radius must be >= 1");
    if (n_iters < 1)
        throw std::runtime_error("[mask_dilate] n_iters must be >= 1");

    i64 ndim         = mask.ndimension();
    i64 spatial_ndim = ndim - (i64)batch_dims.size();
    if (spatial_ndim < 1)
        throw std::runtime_error("[mask_dilate] no spatial dims remaining after batch dims");

    auto prep = prep_morph(std::move(mask), batch_dims);
    return unprep_morph(dilate_impl(prep.flat, radius, spatial_ndim, n_iters), prep);
}

export Tensor mask_erode(Tensor mask, i64 radius = 1, i64 n_iters = 1, std::vector<i64> batch_dims = {})
{
    if (mask.scalar_type() != eScalarType::Bool)
        throw std::runtime_error("[mask_erode] mask must be bool");
    if (radius < 1)
        throw std::runtime_error("[mask_erode] radius must be >= 1");
    if (n_iters < 1)
        throw std::runtime_error("[mask_erode] n_iters must be >= 1");

    i64 ndim         = mask.ndimension();
    i64 spatial_ndim = ndim - (i64)batch_dims.size();
    if (spatial_ndim < 1)
        throw std::runtime_error("[mask_erode] no spatial dims remaining after batch dims");

    auto prep = prep_morph(std::move(mask), batch_dims);
    return unprep_morph(erode_impl(prep.flat, radius, spatial_ndim, n_iters), prep);
}

}
