module;

export module hasty_mri_mod:coordinate_warp;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_fft_mod;

namespace hasty {
namespace mri {

// ── CoordinateWarp ────────────────────────────────────────────────────────────
//
// Builds the q-space density that makes the MRI forward model's continuous
// signal integral exact under the substitution q = u(r):
//
//   S(k) = ∫ ρ(r)·φ_res(r)·exp(-i·k·u(r)) dr
//        = ∫ m_q(q)·exp(-i·k·q) dq,   m_q(q) = ρ(u⁻¹(q))·φ_res(u⁻¹(q)) / |det J_u(u⁻¹(q))|
//
// This is a genuine change of integration variable, so |det J_u| DOES belong
// here — unlike a discrete phase relabeling (Σ_n ρ[n]·exp(-ik·u(r_n)), which
// needs no Jacobian since it never changes the quadrature nodes), this
// approximates the CONTINUOUS integral on a q-grid whose own resolution can
// be refined independently of the original r-grid's voxel size. That's the
// whole point: oversampling q-space should shrink discretization error
// relative to the true continuous signal, not just relative to a
// fixed-resolution discrete reference.
//
// u(r) generally mixes all axes (e.g. spherical-harmonic GNL terms), so u⁻¹
// is NOT separable per axis — it's found per q-grid point via batched
// Newton's method (every iteration is one tensor op across all points at
// once; each point's 3×3 linear solve is closed-form/Cramer's rule, no
// per-point loop). The inversion is computed ONCE per CoordinateWarp
// instance (cached at construction) since it depends only on the warp
// itself, not on which image is being transformed — warp_field_r_to_q calls
// after the first just interpolate, using the cached result.
//
// Each channel contributes an axis-aligned displacement
//   Δq_axis = (c_phys / dx_axis) · field(r)
// where field MAY depend on all of r (not just r_axis) — e.g. xz, x²-y², or
// any spherical-harmonic-like term. Channel::field_fn is therefore a
// CALLABLE (value + full gradient at arbitrary, possibly off-grid, batched
// query positions) rather than a fixed per-r-voxel array — Newton's method
// needs to evaluate it at iterate positions that generally aren't on the
// r-grid at all, and a discrete array would force an extra interpolation
// (with its own error) on every Newton step.
//
// The full (generally non-diagonal) Jacobian matrix
//   J_u[axis, e] = δ_{axis,e} + Σ_{channels on axis} c_pix · ∂field/∂r_e
// is needed both for det(J_u) (the density correction above, and the
// diffeomorphism diagnostic) and for each Newton step's linear solve.
//
// Dimension-agnostic up to cuFINUFFT's own limit: r_shape/q_shape may be 1D,
// 2D, or 3D (warp_field_r_to_q/warp_field_q_to_r throw beyond that — the
// theory generalizes, the underlying NUFFT library doesn't).
//
// Image-domain interpolation (both directions) is FFT+full-type-2-NUFFT
// bandlimited reconstruction, accuracy independently configurable per call
// via three knobs (see warp_field_r_to_q):
//   tol           — NUFFT kernel tolerance (the dominant cost driver: kernel
//                   width, hence interp cost, scales with -log(tol)).
//   n_subsamples  — sub-voxel supersampling within each q-voxel, averaged
//                   down to q_shape's OWN resolution afterward. Lets you
//                   capture more of the warp's spectral spread (anti-
//                   aliasing) WITHOUT inflating q_shape itself (which would
//                   make every downstream gradient step/Toeplitz kernel
//                   bigger too). n_subsamples=1 (default): no supersampling,
//                   evaluate once at the voxel center.
//   refine_iters  — for n_subsamples>1, each sub-voxel offset's inverse warp
//                   is found by a cheap LOCAL correction (one linear solve
//                   using the Jacobian already cached at the voxel center)
//                   followed by `refine_iters` Newton steps warm-started
//                   from that guess — stays correct even for severe GNL
//                   (where a linear-only correction would break down)
//                   without paying for a from-scratch Newton solve per
//                   sub-offset.
// Defaults (tol=1e-6, n_subsamples=1, refine_iters=2) are accurate but not
// supersampled. For repeated/approximate calls (e.g. inside a PnP/ISTA
// proximal step) loosen tol (e.g. 1e-2 — the denoiser's own error already
// exceeds that). For one-time high-accuracy transforms (e.g. coil
// sensitivity maps) raise n_subsamples (e.g. 8) for anti-aliasing.
//
// q_shape may exceed r_shape in TWO independent, explicit ways, controlled
// separately so neither silently mixes into the other:
//   margin       — pure padding: extra array bounds at the SAME voxel size
//                  (dx_Q == dx_R), giving a displaced voxel room to land
//                  without being clamped. Just pick q_shape bigger than
//                  oversample_factor·r_shape; the excess is margin.
//   oversample_factor — a genuine resolution increase (per axis): q-grid
//                  voxels are dx_R/oversample_factor, finer than r-grid
//                  voxels, over the same (or padded) FOV. Refining this is
//                  what should shrink the q-space integral's discretization
//                  error toward the true continuous signal.
// oversample_factor defaults to 1 (no resolution change, pure margin) if not
// given. dx_axis in Channel is always the R-GRID voxel size — CoordinateWarp
// applies oversample_factor internally, callers never convert it themselves.
//
// Pixel-space convention throughout: integer-centered indices n_d = i_d − N_d/2,
// shared by both grids (same center/isocenter) — so an UNDISPLACED r-voxel's
// coordinate is identical whether expressed in r-grid or q-grid terms. (If
// isocenter isn't at the array center, that's a property of how field_fn is
// written, not of this convention — field_fn receives positions in this same
// centered convention and is free to subtract its own isocenter offset.)
//
// The gradient-step part of solving for the image can live entirely in
// q-space (Toeplitz normal operator on whatever grid Upsilon/mask_idx/
// voxel_to_bin live on, no special-casing) — but proximal/denoising steps
// (e.g. PnP/ISTA) that need r-space content run warp_field_q_to_r/
// warp_field_r_to_q every iteration; that's an inherent cost (see `fast`),
// not something this class can hide.
//
// Density correction (density_correction_q/density_correction_r) is
// DELIBERATELY separate from the warp calls — apply it exactly once,
// exactly where you're assembling the actual integral. ρ(r)·φ(r) is a
// single integrand evaluated at one r-position; the substitution applies
// once to that whole product, not once per field warped.

// ── Timing diagnostic ────────────────────────────────────────────────────────
//
// Cumulative wall-clock breakdown across all CoordinateWarp activity in this
// process, to answer "is the Newton solve or the NUFFT the actual bottleneck"
// without guessing. Three buckets, matching the three genuinely separable
// costs in get_or_build_profile/warp_field_r_to_q:
//   newton_s      — invert_warp/invert_warp_from (both ctor cold-start and
//                   get_or_build_profile's refine_iters warm-start).
//   nufft_build_s — make_bandlimited_executor's plan construction + setpts
//                   (paid once per (tol,n_subsamples,refine_iters) cache miss).
//   nufft_apply_s — apply_bandlimited_executor's actual plan->execute call
//                   only (paid on EVERY warp_field_r_to_q/q_to_r call).
//   fft_s         — apply_bandlimited_executor's fftn/fftshift forward-FFT
//                   step immediately before plan->execute — kept separate
//                   since it's a different op entirely (whole-grid FFT, not
//                   nonuniform interpolation), also paid every call.
// All CUDA-synchronized before/after each timed region — otherwise these
// would just measure host-side kernel-launch dispatch time, not actual
// device work.
export struct WarpTimingStats {
    double newton_s = 0.0;
    double nufft_build_s = 0.0;
    double nufft_apply_s = 0.0;
    double fft_s = 0.0;
    i64 newton_calls = 0;
    i64 nufft_build_calls = 0;
    i64 nufft_apply_calls = 0;
    i64 fft_calls = 0;
};

namespace {
WarpTimingStats g_warp_timing;
}

export WarpTimingStats get_warp_timing_stats() { return g_warp_timing; }
export void reset_warp_timing_stats() { g_warp_timing = WarpTimingStats{}; }

export class CoordinateWarp {
public:

    // Evaluates one GNL field and its full gradient at a batch of query
    // positions — arbitrary, possibly off the r-grid (Newton iterates land
    // off-grid in general).
    //   pos  : [N, D] float, r-grid INTEGER-CENTERED PIXEL positions (same
    //          convention as the r-grid itself; may be fractional/off-grid).
    //   returns {value [N], grad [N, D]} — value in physical units (e.g. m²
    //          for an xz-type term), grad[n,e] = d(value)/d(pos[n,e]) i.e.
    //          the gradient w.r.t. PIXEL position (chain rule through the
    //          caller's own physical-to-pixel scaling already applied).
    using FieldFn = std::function<std::pair<Tensor, Tensor>(const Tensor&)>;

    // Specification for one separable-by-axis (but not necessarily
    // separable-by-coordinate-dependence) GNL channel.
    //   field_fn : value + full gradient at arbitrary positions — see FieldFn.
    //   axis     : which q-axis this channel's displacement adds to, in [0,D-1].
    //              field_fn may depend on ALL axes; only the displacement's
    //              OWN axis is fixed by this field.
    //   c_phys   : physical coupling (e.g. [m⁻¹]): Δr_axis_m = c_phys · field
    //   dx_axis  : R-GRID voxel size [m] along axis (oversample_factor, if
    //              any, is applied internally — never folded in by the caller).
    struct Channel {
        FieldFn field_fn;
        i64     axis;
        float   c_phys;
        float   dx_axis;
    };

    CoordinateWarp() = default;

    // Build the warp from one or more GNL channels.
    // r_shape/q_shape: arbitrary-length (D = r_shape.size() == q_shape.size()),
    // C order (dim 0 varies slowest). q_shape must be >= ceil(oversample_factor·r_shape)
    // per axis; any excess is margin/padding. oversample_factor: per-axis
    // resolution multiplier, empty (default) = all 1 (pure margin, no
    // resolution change) — see class doc.
    //
    // Precomputes (and caches) the per-q-grid-point inverse warp r=u⁻¹(q) via
    // batched Newton's method — this is the expensive one-time cost that
    // every warp_field_r_to_q call afterward reuses.
    //
    // dev: device to build all internal tensors on. Can't be inferred from
    // the channels — field_fn is a callable, not a stored tensor, and may
    // only capture plain scalars (no device-resident data to query).
    explicit CoordinateWarp(std::vector<Channel> channels,
                             ArrayRef<i64> r_shape,
                             ArrayRef<i64> q_shape,
                             Device dev,
                             ArrayRef<float> oversample_factor = {},
                             i64 newton_iters = 6);

    // ── Coordinate queries ────────────────────────────────────────────────────

    // Precomputed q = u(r) for ALL N_r r-voxels. Shape [N_r, D], q-grid integer-centered pixels.
    const Tensor& all_q_pix() const { return q_pix_all_; }

    // Warped coordinates for a MASKED subset [N_mask] of r-voxels. Returns [N_mask, D].
    Tensor warp_voxels(const Tensor& mask_idx) const {
        return q_pix_all_.index_select(0, mask_idx).contiguous();
    }

    // det(J_u) for a MASKED subset of r-voxels. Returns [N_mask].
    Tensor masked_jacobian(const Tensor& mask_idx) const {
        return jacobian_.index_select(0, mask_idx).contiguous();
    }

    // det(J_u) for ALL r-voxels [N_r]. The FULL (generally non-diagonal)
    // Jacobian's determinant — must stay > 0 for u to remain a diffeomorphism.
    const Tensor& all_jacobian() const { return jacobian_; }

    // ── Image-space r ↔ q transfers ──────────────────────────────────────────
    //
    // Pure relabeling ONLY — field_q(q) = field_r(u⁻¹(q)), field_r(r) =
    // field_q(u(r)). NO Jacobian, NO oversample factor, for EITHER direction,
    // for ANY field (including ρ). Whether a density correction is needed at
    // all depends entirely on what you're about to DO with the result —
    // ρ(r)·φ(r) is a single integrand evaluated at the SAME r-position, and
    // the substitution applies ONCE to that whole product, not once per
    // field separately. So: warp every field you need with these, then
    // multiply by density_correction_q()/density_correction_r() yourself,
    // exactly once, at whichever point you're actually assembling/using the
    // integral — not buried inside the warp call.
    //
    // The expensive part (inverting u at every q-grid point) is cached at
    // construction — these calls are just an interpolation.
    //
    // tol: NUFFT kernel tolerance — the dominant cost lever (kernel width,
    //   hence interp cost over every query point, scales with -log(tol)).
    // n_subsamples: sub-voxel supersampling per q-voxel, averaged back down
    //   to q_shape's resolution — anti-aliasing without inflating q_shape.
    //   Must be a perfect D-th power (e.g. 8=2^3, 27=3^3 in 3D); 1 (default)
    //   = no supersampling, single evaluation at the voxel center.
    // refine_iters: for n_subsamples>1, Newton iterations refining each
    //   sub-offset's inverse warp, warm-started from a cheap local-Jacobian
    //   correction — keeps this correct even for severe GNL (where the
    //   linear correction alone would break down) without a from-scratch
    //   Newton solve per sub-offset. Unused when n_subsamples=1.
    //
    // Executors for a given (tol, n_subsamples, refine_iters) combination
    // are built once and cached (lazily, on first use) — repeated calls with
    // the SAME settings reuse them; different settings for different fields
    // (e.g. tight for coil maps, loose for a PnP proximal step) each get
    // their own cached profile.
    Tensor warp_field_r_to_q(const Tensor& field_r, double tol = 1e-6,
                             i64 n_subsamples = 1, i64 refine_iters = 2) const;
    Tensor warp_field_q_to_r(const Tensor& field_q, double tol = 1e-6,
                             i64 n_subsamples = 1, i64 refine_iters = 2) const;

    // Multiplicative density-correction factors — apply exactly once,
    // exactly where you're assembling the actual integral (e.g.
    // rho_q = warp_field_r_to_q(rho_r) * density_correction_q()).
    //
    //   density_correction_q(): 1 / (|det J_u(u⁻¹(q))| · oversample^D)
    //     — what a density ρ(r)dr picks up under q=u(r): see class doc.
    //   density_correction_r(): |det J_u(r)| · oversample^D
    //     — the inverse direction.
    // [q_shape]/[r_shape] real tensors respectively (cached, no recompute).
    Tensor density_correction_q() const;
    Tensor density_correction_r() const;

    // ── Accessors ─────────────────────────────────────────────────────────────
    ArrayRef<i64>   r_shape() const { return r_shape_; }
    ArrayRef<i64>   q_shape() const { return q_shape_; }
    ArrayRef<float> oversample_factor() const { return oversample_factor_; }
    i64 ndim()       const { return (i64)r_shape_.size(); }
    i64 n_r_voxels() const { return prod(r_shape_); }
    i64 n_q_voxels() const { return prod(q_shape_); }

    // Max |displacement| in Q-GRID PIXEL units (i.e. already scaled by
    // oversample_factor), per axis, over all r-voxels. Use this to size
    // q_shape with enough margin that warped voxels stay well inside the
    // q-grid boundary — FINUFFT's kernel has finite support, and points too
    // close to (or past) the edge lose accuracy, a real source of error if
    // q_shape has no margin beyond oversample_factor·r_shape.
    const std::vector<float>& max_abs_displacement_pix() const { return max_abs_disp_; }

private:
    static i64 prod(const std::vector<i64>& shape) {
        i64 p = 1;
        for (i64 s : shape) p *= s;
        return p;
    }

    std::vector<Channel> channels_;
    std::vector<i64>     r_shape_;
    std::vector<i64>     q_shape_;
    std::vector<float>   oversample_factor_;  // [D] — per-axis resolution multiplier
    Tensor               os_t_;          // [D] — oversample_factor_ as a device tensor (cached, built once)
    Tensor               r_pix_all_;     // [N_r, D] — r-grid's own centered pixel positions
    Tensor               q_pix_all_;     // [N_r, D] — forward map u(r), at every r-voxel
    Tensor               jacobian_;      // [N_r]    — TRUE det(J_u) at every r-voxel
    std::vector<float>   max_abs_disp_;  // [D] — max |displacement| in q-pixel units, per axis

    std::vector<i64> padded_r_shape_;  // r_shape + zero-pad margin (see pad_centered doc)
    Tensor r_found_q_;       // [N_q, D] — cached u⁻¹(q) for every q-grid point (voxel-center, n_subsamples=1 case)
    Tensor abs_det_at_found_; // [N_q]    — cached |det J_u(r_found_q_)|

    // Lazily-built, cached per (tol, n_subsamples, refine_iters) — see
    // get_or_build_profile doc. Each profile holds one executor per
    // sub-voxel offset (n_subsamples of them; just 1 for the default,
    // voxel-center-only case) for EACH direction.
    struct AccuracyProfile {
        std::vector<std::function<Tensor(const Tensor&)>> r_execs;  // r_to_q direction
        std::vector<std::function<Tensor(const Tensor&)>> q_execs;  // q_to_r direction
    };
    mutable std::map<std::tuple<double, i64, i64>, AccuracyProfile> profile_cache_;
    const AccuracyProfile& get_or_build_profile(double tol, i64 n_subsamples, i64 refine_iters) const;
};

// ── Implementation ────────────────────────────────────────────────────────────

namespace {

i64 prodvec(const std::vector<i64>& shape)
{
    i64 p = 1;
    for (i64 s : shape) p *= s;
    return p;
}

// Build integer-centered pixel positions for a C-order grid of the given
// shape: coord[n, d] = i_d − N_d/2. n indexes the grid in C order (dim 0
// varies slowest). Used for both the r-grid and the q-grid — purely a
// labeling convention, doesn't know which grid it's building for.
Tensor make_pix_centered(ArrayRef<i64> shape, Device dev)
{
    const i64 D = (i64)shape.size();
    const TensorOptions opts{dev, eScalarType::Float};

    std::vector<i64> full_shape(shape.begin(), shape.end());

    std::vector<Tensor> coords;
    coords.reserve(D);
    for (i64 d = 0; d < D; ++d) {
        const i64 n_d = shape[d];

        std::vector<i64> bshape(D, 1);
        bshape[d] = n_d;

        auto c = arange(n_d, opts).sub(Scalar((f32)(n_d / 2)))
                     .reshape(ArrayRef<i64>(bshape))
                     .expand(ArrayRef<i64>(full_shape))
                     .reshape({(i64)(prodvec(full_shape))});
        coords.push_back(std::move(c));
    }

    return stack(coords, 1).contiguous();  // [N, D]
}

// ── Batched evaluation of u(r) = r + Σ c_pix·field(r) and its full Jacobian ──

struct WarpEval {
    Tensor disp;                  // [N, D] — Σ_channels c_pix·field, per axis
    std::vector<Tensor> jac_rows; // D tensors, each [N, D] — row `axis` of J_u per point
};

WarpEval evaluate_warp(const std::vector<CoordinateWarp::Channel>& channels,
                       const Tensor& pos, i64 D, Device dev)
{
    const i64 N = pos.size(0);
    const TensorOptions opts{dev, eScalarType::Float};

    auto disp = zeros({N, D}, opts);
    std::vector<Tensor> jac_rows(D);
    for (i64 d = 0; d < D; ++d) {
        auto row = zeros({N, D}, opts);
        row.select(1, d).add_(Scalar(1.0f));  // identity row d
        jac_rows[d] = row;
    }

    for (auto& ch : channels) {
        auto [val, grad] = ch.field_fn(pos);   // val:[N], grad:[N,D]
        const float c_pix = ch.c_phys / ch.dx_axis;
        disp.select(1, ch.axis).add_(val.mul(Scalar(c_pix)));
        jac_rows[ch.axis].add_(grad.mul(Scalar(c_pix)));
    }

    return WarpEval{std::move(disp), std::move(jac_rows)};
}

Tensor jac_entry(const std::vector<Tensor>& rows, i64 i, i64 j) { return rows[i].select(1, j); }

// det(J), J given as D row-tensors (Cramer's rule building block). D in {1,2,3}.
Tensor det_jac(const std::vector<Tensor>& rows, i64 D)
{
    if (D == 1) {
        return jac_entry(rows, 0, 0);
    } else if (D == 2) {
        auto a = jac_entry(rows,0,0), b = jac_entry(rows,0,1);
        auto c = jac_entry(rows,1,0), d = jac_entry(rows,1,1);
        return a.mul(d).sub(b.mul(c));
    } else if (D == 3) {
        auto a = jac_entry(rows,0,0), b = jac_entry(rows,0,1), c = jac_entry(rows,0,2);
        auto d = jac_entry(rows,1,0), e = jac_entry(rows,1,1), f = jac_entry(rows,1,2);
        auto g = jac_entry(rows,2,0), h = jac_entry(rows,2,1), i_ = jac_entry(rows,2,2);
        return a.mul(e.mul(i_).sub(f.mul(h)))
                .sub(b.mul(d.mul(i_).sub(f.mul(g))))
                .add(c.mul(d.mul(h).sub(e.mul(g))));
    }
    throw std::invalid_argument("CoordinateWarp: only 1D/2D/3D supported");
}

// Solve J·x = rhs for x via Cramer's rule (closed-form, no per-point loop —
// every operation below is a single elementwise tensor op over all N points
// at once). rhs: [N,D]. Returns [N,D]. D in {1,2,3}.
Tensor solve_jac(const std::vector<Tensor>& rows, const Tensor& rhs, i64 D)
{
    auto det = det_jac(rows, D);
    auto det_safe = det.add(Scalar(1e-20f));  // warp kept near-diffeomorphic (det>0) by design

    if (D == 1) {
        return rhs.select(1,0).div(det_safe).unsqueeze(1);
    } else if (D == 2) {
        auto a = jac_entry(rows,0,0), b = jac_entry(rows,0,1);
        auto c = jac_entry(rows,1,0), d = jac_entry(rows,1,1);
        auto p = rhs.select(1,0), q = rhs.select(1,1);
        auto x0 = p.mul(d).sub(b.mul(q)).div(det_safe);
        auto x1 = a.mul(q).sub(p.mul(c)).div(det_safe);
        return stack({x0, x1}, 1);
    } else if (D == 3) {
        auto a = jac_entry(rows,0,0), b = jac_entry(rows,0,1), c = jac_entry(rows,0,2);
        auto d = jac_entry(rows,1,0), e = jac_entry(rows,1,1), f = jac_entry(rows,1,2);
        auto g = jac_entry(rows,2,0), h = jac_entry(rows,2,1), i_ = jac_entry(rows,2,2);
        auto p = rhs.select(1,0), q = rhs.select(1,1), r = rhs.select(1,2);

        auto det0 = p.mul(e.mul(i_).sub(f.mul(h)))
                     .sub(b.mul(q.mul(i_).sub(f.mul(r))))
                     .add(c.mul(q.mul(h).sub(e.mul(r))));
        auto det1 = a.mul(q.mul(i_).sub(f.mul(r)))
                     .sub(p.mul(d.mul(i_).sub(f.mul(g))))
                     .add(c.mul(d.mul(r).sub(q.mul(g))));
        auto det2 = a.mul(e.mul(r).sub(q.mul(h)))
                     .sub(b.mul(d.mul(r).sub(q.mul(g))))
                     .add(p.mul(d.mul(h).sub(e.mul(g))));

        return stack({det0.div(det_safe), det1.div(det_safe), det2.div(det_safe)}, 1);
    }
    throw std::invalid_argument("CoordinateWarp: only 1D/2D/3D supported");
}

// Batched Newton solve of u(r) = q_target for r, warm-started from r0.
// q_target/r0: [N,D], r-grid-pixel units. Every iteration is O(1) tensor ops
// across all N points simultaneously — no per-point loop anywhere.
Tensor invert_warp_from(const std::vector<CoordinateWarp::Channel>& channels,
                        const Tensor& r0, const Tensor& q_target, i64 D, Device dev, i64 n_iters)
{
    if (dev.type == eDeviceType::CUDA) cuda::synchronize(dev);
    auto t0 = std::chrono::steady_clock::now();

    auto r = r0.clone();
    for (i64 it = 0; it < n_iters; ++it) {
        auto ev = evaluate_warp(channels, r, D, dev);
        auto residual = r.add(ev.disp).sub(q_target);     // F(r) = u(r) - q_target
        auto step = solve_jac(ev.jac_rows, residual, D);
        r = r.sub(step);
    }

    if (dev.type == eDeviceType::CUDA) cuda::synchronize(dev);
    g_warp_timing.newton_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++g_warp_timing.newton_calls;
    return r;
}

// Cold-start version: r0 = q_target (good guess, warp is mild, J_u ≈ I).
Tensor invert_warp(const std::vector<CoordinateWarp::Channel>& channels,
                   const Tensor& q_target, i64 D, Device dev, i64 n_iters)
{
    return invert_warp_from(channels, q_target, q_target, D, dev, n_iters);
}

// Cartesian D-dimensional sub-grid of offsets within one voxel, in
// [-0.5,0.5) units of one voxel — n_subsamples MUST be a perfect D-th power
// (e.g. 8=2^3, 27=3^3 in 3D). n_subsamples=1 gives a single offset of 0
// (voxel center, no supersampling) with no special-casing needed: the
// formula (j+0.5)/1 - 0.5 = 0 for the only j=0.
std::vector<std::vector<float>> subgrid_offsets(i64 n_subsamples, i64 D)
{
    i64 n_per_axis = (i64)std::llround(std::pow((double)n_subsamples, 1.0 / (double)D));
    i64 check = 1;
    for (i64 d = 0; d < D; ++d) check *= n_per_axis;
    if (check != n_subsamples)
        throw std::invalid_argument("CoordinateWarp: n_subsamples must be a perfect D-th power (e.g. 1,8,27 in 3D)");

    std::vector<float> axis_offsets(n_per_axis);
    for (i64 j = 0; j < n_per_axis; ++j)
        axis_offsets[j] = (float)(j + 0.5) / (float)n_per_axis - 0.5f;

    std::vector<std::vector<float>> offsets;
    offsets.reserve(n_subsamples);
    std::vector<i64> idx(D, 0);
    for (i64 s = 0; s < n_subsamples; ++s) {
        std::vector<float> off(D);
        for (i64 d = 0; d < D; ++d) off[d] = axis_offsets[idx[d]];
        offsets.push_back(std::move(off));
        for (i64 d = D - 1; d >= 0; --d) {
            if (++idx[d] < n_per_axis) break;
            idx[d] = 0;
        }
    }
    return offsets;
}

// [-π,π] NUFFT coords from integer-centered pixel coords, axis-reversed to
// match cufinufft's mode ordering (coords row 0 pairs with the LAST/fastest
// C-order axis — same reversed-nmodes convention used everywhere else in
// this codebase's NUFFT calls).
template<i64 D>
Tensor make_nufft_coords(const Tensor& pix, const std::array<i64, D>& shape)
{
    using namespace std::numbers;
    const Device dev = pix.device();
    auto coords = zeros({D, pix.size(0)}, TensorOptions(dev, eScalarType::Float));
    for (i64 d = 0; d < D; ++d) {
        const float scale = 2.0f * (float)pi_v<f64> / (float)shape[D - 1 - d];
        coords.select(0, d).copy_(pix.select(1, D - 1 - d).mul(Scalar(scale)));
    }
    return coords.contiguous();
}

// Bandlimited reconstruction of a regularly-sampled field at arbitrary
// (off-grid) positions — the mathematically correct replacement for raw
// spread/interp-only kernel convolution.
//
// nufft_interp_from_grid (SPREAD_ONLY/interp-only) computes a DIRECT
// kernel-weighted blend of grid_flat with NO deconvolution and NO internal
// transform — its accuracy is bounded by the kernel's own finite width, NOT
// by `tol` (tol describes the FULL pipeline's accuracy, deconvolve+FFT+
// interp together; skipping deconvolve+FFT throws that guarantee away).
// This is why a 1e-6 tol gave a 3.5e-3 floor: tol was never actually being
// honored.
//
// Correct approach: type-2 NUFFT computes f(x_j) = Sum_k c_k*exp(sign*i*k*x_j)
// for known Fourier coefficients c_k on a uniform grid, to tol accuracy
// (deconvolve+ifft+kernel-interp together, by design). So: forward-FFT
// grid_flat ourselves to get c_k, feed THAT to a full (non-spread-only)
// type-2 plan with sign=POS (matches the inverse-transform convention,
// since we're reconstructing FROM Fourier coefficients) and
// mode_order=FFT (matches torch fftn's natural DC-first ordering, so c_k
// needs no reshuffling) — this IS exactly the bandlimited reconstruction of
// grid_flat at x_j, accurate to `tol`.
//
// Plan/setpts cost is non-trivial for the full (non-spread-only) pipeline —
// it needs an internal upsampled cuFFT plan, unlike spread-only which skips
// that entirely. Query positions (r_found_q_/q_pix_all_) and grid shape are
// FIXED per CoordinateWarp instance and reused across many scatter/gather
// calls (once per warped quantity: rho, z_eff, each conc channel, ...) — so
// build the plan + setpts ONCE and cache the resulting executor, rather than
// rebuilding it on every call.
template<i64 D>
std::function<Tensor(const Tensor&)> make_bandlimited_executor(
    const Tensor& pix, const std::array<i64, D>& grid_shape, Device dev, double tol)
{
    using namespace fft;

    // pix is in the physical CENTERED convention (i - shape/2, same as
    // make_pix_centered) — but fftn/fftshift's CMCL indexing pairs array
    // index j with k_eff=j-shape/2 relative to the FFT's own LITERAL
    // (0-based) array-index origin, not the physical center. Shift back to
    // literal coords before building NUFFT coordinates, or every frequency
    // picks up a constant per-axis phase error (alternating (-1)^k_eff,
    // since the shift is exactly shape/2) — this was the source of the
    // near-zero-correlation garbled reconstruction.
    auto pix_literal = pix.clone();
    for (i64 d = 0; d < D; ++d)
        pix_literal.select(1, d).add_(Scalar((float)(grid_shape[d] / 2)));

    auto coords = make_nufft_coords<D>(pix_literal, grid_shape);

    NufftOptions<cuda_t, f32, UTN> opts;
    opts.ntransf = 1;
    opts.tol = tol;   // the actual kernel-width/cost lever — see class doc
    // mode_order left at DEFAULT (= CMCL/centered, modeord=0) — the
    // convention already proven correct elsewhere in this codebase (outer
    // forward-model NUFFT calls). c_k is fftshift'd by the caller to match
    // (DC-first -> DC-centered) rather than relying on the untested FFT/
    // modeord=1 branch, which combined with sign=POS had never been
    // exercised anywhere in this codebase before and is the likely source
    // of the garbled (near-zero-correlation) result.
    opts.sign = decltype(opts)::eNufftSign::POS;   // inverse-transform convention: exp(+i*k*x)
    // upsampling_factor left at DEFAULT (2.0). The actual cost driver here is
    // kernel width during spread/interp over millions of query points, NOT
    // FFT size (FFT is fast even at hundreds of millions of elements) — a
    // narrower upsampfac needs a WIDER kernel for the same tol, which makes
    // the real bottleneck worse while shrinking something (the internal FFT
    // grid) that was never slow. 2.0 gives the narrowest kernel for a given
    // tol, i.e. the cheapest interp step.
    // spread_interp_method left at DEFAULT: full deconvolve+ifft+interp pipeline, tol-accurate.

    if (dev.type == eDeviceType::CUDA) cuda::synchronize(dev);
    auto t0 = std::chrono::steady_clock::now();

    auto plan = std::make_shared<NufftPlan<cuda_t, f32, D, UTN>>(grid_shape, opts);
    plan->setpts(coords);

    if (dev.type == eDeviceType::CUDA) cuda::synchronize(dev);
    g_warp_timing.nufft_build_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++g_warp_timing.nufft_build_calls;

    const i64 Nq = pix.size(0);
    std::vector<i64> in_shape{1};
    for (i64 d = 0; d < D; ++d) in_shape.push_back(grid_shape[d]);

    return [plan, in_shape, Nq, dev](const Tensor& c_k_flat) -> Tensor {
        if (dev.type == eDeviceType::CUDA) cuda::synchronize(dev);
        auto t0 = std::chrono::steady_clock::now();

        auto input  = c_k_flat.reshape(ArrayRef<i64>(in_shape)).contiguous();
        auto output = zeros({(i64)1, Nq}, TensorOptions(dev, eScalarType::ComplexFloat)).contiguous();
        plan->execute(input, output);

        if (dev.type == eDeviceType::CUDA) cuda::synchronize(dev);
        g_warp_timing.nufft_apply_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++g_warp_timing.nufft_apply_calls;
        return output.reshape({Nq});
    };
}

// Forward-FFT grid_flat, run it through a cached bandlimited executor (see
// make_bandlimited_executor), undo the FFT's unnormalized scaling.
Tensor apply_bandlimited_executor(const std::function<Tensor(const Tensor&)>& exec,
                                  const Tensor& grid_flat, ArrayRef<i64> grid_shape)
{
    const i64 N = grid_flat.size(0);
    const Device dev = grid_flat.device();

    if (dev.type == eDeviceType::CUDA) cuda::synchronize(dev);
    auto t0 = std::chrono::steady_clock::now();

    // fftn gives DC-first order; fftshift to DC-centered to match the
    // plan's CMCL mode_order (see make_bandlimited_executor doc).
    auto c_k = fftshift(fftn(grid_flat.reshape(grid_shape))).reshape({N}).contiguous();

    if (dev.type == eDeviceType::CUDA) cuda::synchronize(dev);
    g_warp_timing.fft_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++g_warp_timing.fft_calls;
    // Forward FFT was unnormalized (norm="backward" default); the inverse
    // relation rho(x) = (1/N)*Sum_k c_k*exp(+i*k*x) needs the 1/N here.
    return exec(c_k).div(Scalar((float)N));
}

// Embed `src` (shape src_shape) into a zero-padded buffer of shape
// padded_shape, CENTERED — both shapes share the same integer-centered-pixel
// convention (coord = i - N/2), so centering the smaller array inside the
// larger one means a query position expressed in src_shape's own convention
// is ALSO directly valid against the padded buffer, no coordinate shift
// needed (the same physical point — e.g. isocenter — maps to the same
// centered coordinate in both).
//
// This is the principled fix for FINUFFT's periodicity: rather than
// clamping/zeroing query points after the fact (which can't fix points that
// are nominally in-bounds but close enough to the edge for the KERNEL's own
// support to wrap), pad the SOURCE with real zeros, generously beyond
// whatever margin the kernel's support needs, so wraparound only ever reads
// zeros — never real content from the opposite edge. Standard way to get
// non-periodic (Dirichlet) boundary behavior out of an inherently periodic
// FFT/NUFFT-based method.
Tensor pad_centered(const Tensor& src_flat, ArrayRef<i64> src_shape,
                    const std::vector<i64>& padded_shape, i64 D, Device dev)
{
    auto padded_flat = zeros({prodvec(padded_shape)}, TensorOptions(dev, eScalarType::ComplexFloat));
    auto padded_view = padded_flat.reshape(ArrayRef<i64>(padded_shape));
    auto src_view    = src_flat.reshape(src_shape);

    Tensor dst = padded_view;
    for (i64 d = 0; d < D; ++d) {
        const i64 start = padded_shape[d] / 2 - src_shape[d] / 2;
        dst = dst.narrow(d, start, src_shape[d]);
    }
    dst.copy_(src_view);
    return padded_flat;
}

} // anonymous namespace

CoordinateWarp::CoordinateWarp(std::vector<Channel> channels,
                                ArrayRef<i64> r_shape,
                                ArrayRef<i64> q_shape,
                                Device dev,
                                ArrayRef<float> oversample_factor,
                                i64 newton_iters)
    : channels_(std::move(channels)),
      r_shape_(r_shape.begin(), r_shape.end()),
      q_shape_(q_shape.begin(), q_shape.end())
{
    if (channels_.empty())
        throw std::invalid_argument("CoordinateWarp: at least one channel required");
    if (r_shape.size() != q_shape.size())
        throw std::invalid_argument("CoordinateWarp: r_shape and q_shape must have the same dimensionality");

    const i64 D = (i64)r_shape.size();
    if (D < 1 || D > 3)
        throw std::invalid_argument("CoordinateWarp: only 1D/2D/3D supported (cuFINUFFT limit)");

    for (auto& ch : channels_) {
        if (ch.axis < 0 || ch.axis >= D)
            throw std::invalid_argument("CoordinateWarp: axis must be in [0, ndim-1]");
    }

    if (oversample_factor.empty())
        oversample_factor_.assign(D, 1.0f);
    else if ((i64)oversample_factor.size() == D)
        oversample_factor_.assign(oversample_factor.begin(), oversample_factor.end());
    else
        throw std::invalid_argument("CoordinateWarp: oversample_factor must have one entry per axis (or be empty)");

    // ── Forward map + TRUE Jacobian at every r-voxel (diagnostics + gather) ──
    r_pix_all_ = make_pix_centered(r_shape_, dev);   // [N_r, D]
    auto ev_r = evaluate_warp(channels_, r_pix_all_, D, dev);

    max_abs_disp_.resize(D);
    for (i64 d = 0; d < D; ++d)
        max_abs_disp_[d] = ev_r.disp.select(1, d).abs().max().item<float>() * oversample_factor_[d];

    jacobian_ = det_jac(ev_r.jac_rows, D).contiguous();

    os_t_ = Tensor::from_blob(oversample_factor_.data(), {D}, eScalarType::Float, Device{eDeviceType::CPU})
                .clone().to(dev);
    q_pix_all_ = r_pix_all_.add(ev_r.disp).mul(os_t_.unsqueeze(0)).contiguous();

    // ── Inverse map at every q-grid point, cached (the expensive one-time
    //    cost every warp_field_r_to_q call afterward reuses) ───────────────
    auto q_array  = make_pix_centered(q_shape_, dev);              // [N_q, D]
    auto q_target = q_array.div(os_t_.unsqueeze(0));               // r-pixel units
    r_found_q_ = invert_warp(channels_, q_target, D, dev, newton_iters).contiguous();

    auto ev_found = evaluate_warp(channels_, r_found_q_, D, dev);
    abs_det_at_found_ = det_jac(ev_found.jac_rows, D).abs().contiguous();

    // Padded r-grid for warp_field_r_to_q's gather step (see pad_centered doc):
    // big enough to contain every reachable r_found query (bounded by
    // q_shape/oversample_factor) plus a generous kernel-half-width safety
    // buffer, so FINUFFT's periodic wraparound only ever reads zeros.
    constexpr i64 kKernelBuffer = 32;  // generous; actual FINUFFT kernel half-width is smaller
    padded_r_shape_.resize(D);
    for (i64 d = 0; d < D; ++d) {
        const i64 reach = (i64)std::ceil((float)q_shape_[d] / oversample_factor_[d]) + 2 * kKernelBuffer;
        padded_r_shape_[d] = std::max(reach, r_shape_[d] + 2 * kKernelBuffer);
    }

    // Note: no executors built here anymore — they're lazily built per
    // (tol, n_subsamples, refine_iters) combination on first use, see
    // get_or_build_profile.
}

// Lazily builds and caches one AccuracyProfile (a set of executors, one per
// sub-voxel offset) for a given (tol, n_subsamples, refine_iters)
// combination — see class doc for what each knob does.
//
// For the n_subsamples=1 (no supersampling) case, the single offset is
// exactly 0 (see subgrid_offsets doc), so this reuses the already-cached
// r_found_q_/q_pix_all_ directly rather than redundantly re-deriving them.
//
// For n_subsamples>1: r_to_q direction needs a NEW inverse warp per
// sub-offset — found via a cheap local-Jacobian linear correction (one
// linear solve, using the Jacobian already evaluated at r_found_q_) as a
// warm start, refined by `refine_iters` Newton steps (stays correct for
// severe GNL, where the linear correction alone would break down). q_to_r
// direction needs no inversion at all — it's the forward map, evaluated
// directly at each offset r-position.
const CoordinateWarp::AccuracyProfile& CoordinateWarp::get_or_build_profile(
    double tol, i64 n_subsamples, i64 refine_iters) const
{
    auto key = std::make_tuple(tol, n_subsamples, refine_iters);
    auto found = profile_cache_.find(key);
    if (found != profile_cache_.end())
        return found->second;

    const i64 D = ndim();
    const Device dev = r_found_q_.device();
    auto offsets = subgrid_offsets(n_subsamples, D);

    auto ev_found = evaluate_warp(channels_, r_found_q_, D, dev);  // jac_rows at r_found_q_, for the linear correction
    auto q_array  = make_pix_centered(q_shape_, dev);              // [N_q, D]

    AccuracyProfile prof;
    prof.r_execs.reserve(offsets.size());
    prof.q_execs.reserve(offsets.size());

    for (auto& off : offsets) {
        std::vector<float> off_buf(off.begin(), off.end());
        auto off_t = Tensor::from_blob(off_buf.data(), {D}, eScalarType::Float, Device{eDeviceType::CPU})
                         .clone().to(dev);

        // ── r_to_q direction: needs the inverse warp at q_array + off ──
        Tensor r_found_k;
        if (n_subsamples == 1) {
            r_found_k = r_found_q_;
        } else {
            auto delta_q = off_t.unsqueeze(0).expand({n_q_voxels(), D}).contiguous();
            auto q_target_k    = q_array.add(delta_q).div(os_t_.unsqueeze(0));
            auto delta_r_units = delta_q.div(os_t_.unsqueeze(0));   // same shift, r-pixel-equivalent units
            auto r_guess = r_found_q_.add(solve_jac(ev_found.jac_rows, delta_r_units, D));
            r_found_k = invert_warp_from(channels_, r_guess, q_target_k, D, dev, refine_iters);
        }
        switch (D) {
            case 1: prof.r_execs.push_back(make_bandlimited_executor<1>(r_found_k, std::array<i64,1>{padded_r_shape_[0]}, dev, tol)); break;
            case 2: prof.r_execs.push_back(make_bandlimited_executor<2>(r_found_k, std::array<i64,2>{padded_r_shape_[0], padded_r_shape_[1]}, dev, tol)); break;
            case 3: prof.r_execs.push_back(make_bandlimited_executor<3>(r_found_k, std::array<i64,3>{padded_r_shape_[0], padded_r_shape_[1], padded_r_shape_[2]}, dev, tol)); break;
            default: throw std::invalid_argument("CoordinateWarp: only supports 1D/2D/3D (cuFINUFFT limit)");
        }

        // ── q_to_r direction: forward map, exact, no inversion needed ──
        Tensor q_pix_k;
        if (n_subsamples == 1) {
            q_pix_k = q_pix_all_;
        } else {
            auto delta_r = off_t.unsqueeze(0).expand({n_r_voxels(), D}).contiguous();
            auto r_guess_pix = r_pix_all_.add(delta_r);
            auto ev_off = evaluate_warp(channels_, r_guess_pix, D, dev);
            q_pix_k = r_guess_pix.add(ev_off.disp).mul(os_t_.unsqueeze(0)).contiguous();
        }
        switch (D) {
            case 1: prof.q_execs.push_back(make_bandlimited_executor<1>(q_pix_k, std::array<i64,1>{q_shape_[0]}, dev, tol)); break;
            case 2: prof.q_execs.push_back(make_bandlimited_executor<2>(q_pix_k, std::array<i64,2>{q_shape_[0], q_shape_[1]}, dev, tol)); break;
            case 3: prof.q_execs.push_back(make_bandlimited_executor<3>(q_pix_k, std::array<i64,3>{q_shape_[0], q_shape_[1], q_shape_[2]}, dev, tol)); break;
            default: throw std::invalid_argument("CoordinateWarp: only supports 1D/2D/3D (cuFINUFFT limit)");
        }
    }

    auto inserted = profile_cache_.emplace(key, std::move(prof));
    return inserted.first->second;
}

// Pure relabeling: field_q(q) = field_r(u⁻¹(q)). NO density correction —
// see class doc: that's the caller's job, applied exactly once, exactly
// where the actual integral is assembled (density_correction_q()).
Tensor CoordinateWarp::warp_field_r_to_q(const Tensor& field_r, double tol,
                                         i64 n_subsamples, i64 refine_iters) const
{
    const i64 D = ndim();
    const Device dev = field_r.device();
    auto field_flat = field_r.reshape({n_r_voxels()}).to(eScalarType::ComplexFloat);

    // Zero-pad before interpolating — see pad_centered doc. r_found_q_ is
    // already expressed in r_shape_'s own centered convention, which is
    // directly valid against the padded buffer too (same isocenter, just a
    // wider legitimate range) — no coordinate shift needed.
    auto field_padded = pad_centered(field_flat, r_shape_, padded_r_shape_, D, dev);

    const auto& prof = get_or_build_profile(tol, n_subsamples, refine_iters);
    Tensor acc;
    for (auto& exec : prof.r_execs) {
        auto v = apply_bandlimited_executor(exec, field_padded, padded_r_shape_);
        acc = acc.defined() ? acc.add(v) : v;
    }
    auto interp_flat = acc.div(Scalar((float)prof.r_execs.size()));

    auto result = interp_flat.reshape(q_shape_);
    if (!field_r.is_complex())
        return result.real().contiguous();
    return result.contiguous();
}

// Pure relabeling: field_r(r) = field_q(u(r)). NO density correction — see
// warp_field_r_to_q.
Tensor CoordinateWarp::warp_field_q_to_r(const Tensor& field_q, double tol,
                                         i64 n_subsamples, i64 refine_iters) const
{
    auto field_flat = field_q.reshape({n_q_voxels()}).to(eScalarType::ComplexFloat);

    const auto& prof = get_or_build_profile(tol, n_subsamples, refine_iters);
    Tensor acc;
    for (auto& exec : prof.q_execs) {
        auto v = apply_bandlimited_executor(exec, field_flat, q_shape_);
        acc = acc.defined() ? acc.add(v) : v;
    }
    auto interp_flat = acc.div(Scalar((float)prof.q_execs.size()));

    auto result = interp_flat.reshape(r_shape_);
    if (!field_q.is_complex())
        return result.real().contiguous();
    return result.contiguous();
}

// 1 / (|det J_u(u⁻¹(q))| · oversample^D) — see class doc for the
// derivation (the continuous-integral density correction under q=u(r)) and
// the warp_field_r_to_q doc for why this is a SEPARATE step from relabeling.
// Cached inputs (abs_det_at_found_) — this is just a multiply, cheap.
Tensor CoordinateWarp::density_correction_q() const
{
    float volume_ratio = 1.0f;
    for (float os : oversample_factor_) volume_ratio *= os;
    return (Scalar(1.0f) / abs_det_at_found_.mul(Scalar(volume_ratio)).add(Scalar(1e-20f)))
               .reshape(q_shape_);
}

// |det J_u(r)| · oversample^D — inverse-direction density correction.
Tensor CoordinateWarp::density_correction_r() const
{
    float volume_ratio = 1.0f;
    for (float os : oversample_factor_) volume_ratio *= os;
    return jacobian_.abs().mul(Scalar(volume_ratio)).reshape(r_shape_);
}

} // namespace mri
} // namespace hasty
