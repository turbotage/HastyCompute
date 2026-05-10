module;

// PETSC_SKIP_REAL___FLOAT128 is injected via CMake INTERFACE_COMPILE_DEFINITIONS
// on PETSc::petsc — no need to redefine it here.
#include <petscsys.h>
#include <petscvec.h>
#include <petscmat.h>
#include <slepcsvd.h>
#if defined(PETSC_HAVE_CUDA)
#  include <cuda_runtime.h>
#endif

// Module implementation unit — no `export` keyword.
module hasty_linalg_mod;

import std;
import hasty_util_mod;
import hasty_tensor_mod;

// PETSc's Vec/Mat/SVD live in global namespace but 'Vec' clashes with
// hasty::Vec (alias template for std::vector).  Hoist aliases before
// entering the hasty namespace so we can spell them unambiguously.
using PetscVec = ::Vec;
using PetscMat = ::Mat;
using PetscSVD = ::SVD;

namespace hasty {
namespace linalg {

// ─── SLEPc lifetime ──────────────────────────────────────────────────────────

namespace detail {

struct SlepcLifetime {
    SlepcLifetime() {
        int argc = 0; char** argv = nullptr;
        SlepcInitialize(&argc, &argv, nullptr, nullptr);
    }
    ~SlepcLifetime() { SlepcFinalize(); }
};

inline void ensure_init() {
    static SlepcLifetime g;
}

// ─── Shell-matrix context ─────────────────────────────────────────────────────

struct ShellCtx {
    const LinearOperator* op;
    bool cuda;
    i32  cuda_dev;
};

// Wrap a PetscScalar* (complex-double) as a Tensor of op dtype.
static Tensor petsc_to_op(const PetscScalar* data, PetscInt n,
                           eScalarType op_dtype, Device dev)
{
    auto cd = Tensor::from_blob(
        const_cast<void*>(reinterpret_cast<const void*>(data)),
        {(i64)n}, eScalarType::ComplexDouble, dev);
    return (op_dtype == eScalarType::ComplexDouble) ? cd : cd.to(op_dtype);
}

// Write op-dtype result into a PETSc CPU buffer.
static void op_to_petsc_cpu(const Tensor& result, PetscScalar* dst, PetscInt n)
{
    Tensor cd = (result.dtype() == eScalarType::ComplexDouble)
        ? result.contiguous()
        : result.to(eScalarType::ComplexDouble).contiguous();
    std::memcpy(dst, cd.const_data_ptr(),
                static_cast<std::size_t>(n) * sizeof(PetscScalar));
}

// ─── CPU shell operations ─────────────────────────────────────────────────────

static PetscErrorCode cpu_mult(PetscMat A, PetscVec x, PetscVec y)
{
    ShellCtx* ctx; PetscCall(MatShellGetContext(A, &ctx));
    PetscInt nx; PetscCall(VecGetLocalSize(x, &nx));
    PetscInt ny; PetscCall(VecGetLocalSize(y, &ny));

    const PetscScalar* xp; PetscCall(VecGetArrayRead(x, &xp));
    auto xt = petsc_to_op(xp, nx, ctx->op->dtype(), ctx->op->device());
    PetscCall(VecRestoreArrayRead(x, &xp));

    auto res = ctx->op->matvec(xt);

    PetscScalar* yp; PetscCall(VecGetArray(y, &yp));
    op_to_petsc_cpu(res, yp, ny);
    PetscCall(VecRestoreArray(y, &yp));
    return PETSC_SUCCESS;
}

static PetscErrorCode cpu_mult_hermitian(PetscMat A, PetscVec x, PetscVec y)
{
    ShellCtx* ctx; PetscCall(MatShellGetContext(A, &ctx));
    PetscInt nx; PetscCall(VecGetLocalSize(x, &nx));
    PetscInt ny; PetscCall(VecGetLocalSize(y, &ny));

    const PetscScalar* xp; PetscCall(VecGetArrayRead(x, &xp));
    auto xt = petsc_to_op(xp, nx, ctx->op->dtype(), ctx->op->device());
    PetscCall(VecRestoreArrayRead(x, &xp));

    auto res = ctx->op->rmatvec(xt);

    PetscScalar* yp; PetscCall(VecGetArray(y, &yp));
    op_to_petsc_cpu(res, yp, ny);
    PetscCall(VecRestoreArray(y, &yp));
    return PETSC_SUCCESS;
}

// ─── CUDA shell operations ────────────────────────────────────────────────────

#if defined(PETSC_HAVE_CUDA)
static PetscErrorCode cuda_mult(PetscMat A, PetscVec x, PetscVec y)
{
    ShellCtx* ctx; PetscCall(MatShellGetContext(A, &ctx));
    PetscInt nx; PetscCall(VecGetLocalSize(x, &nx));
    PetscInt ny; PetscCall(VecGetLocalSize(y, &ny));

    Device dev{eDeviceType::CUDA, static_cast<DeviceIndex>(ctx->cuda_dev)};
    const PetscScalar* xd; PetscCall(VecCUDAGetArrayRead(x, &xd));
    auto xt = petsc_to_op(xd, nx, ctx->op->dtype(), dev);
    PetscCall(VecCUDARestoreArrayRead(x, &xd));

    auto res = ctx->op->matvec(xt);
    auto cd  = (res.dtype() == eScalarType::ComplexDouble)
        ? res.contiguous()
        : res.to(eScalarType::ComplexDouble).contiguous();

    PetscScalar* yd; PetscCall(VecCUDAGetArrayWrite(y, &yd));
    cudaMemcpy(yd, cd.const_data_ptr(),
               static_cast<std::size_t>(ny) * sizeof(PetscScalar),
               cudaMemcpyDeviceToDevice);
    PetscCall(VecCUDARestoreArrayWrite(y, &yd));
    return PETSC_SUCCESS;
}

static PetscErrorCode cuda_mult_hermitian(PetscMat A, PetscVec x, PetscVec y)
{
    ShellCtx* ctx; PetscCall(MatShellGetContext(A, &ctx));
    PetscInt nx; PetscCall(VecGetLocalSize(x, &nx));
    PetscInt ny; PetscCall(VecGetLocalSize(y, &ny));

    Device dev{eDeviceType::CUDA, static_cast<DeviceIndex>(ctx->cuda_dev)};
    const PetscScalar* xd; PetscCall(VecCUDAGetArrayRead(x, &xd));
    auto xt = petsc_to_op(xd, nx, ctx->op->dtype(), dev);
    PetscCall(VecCUDARestoreArrayRead(x, &xd));

    auto res = ctx->op->rmatvec(xt);
    auto cd  = (res.dtype() == eScalarType::ComplexDouble)
        ? res.contiguous()
        : res.to(eScalarType::ComplexDouble).contiguous();

    PetscScalar* yd; PetscCall(VecCUDAGetArrayWrite(y, &yd));
    cudaMemcpy(yd, cd.const_data_ptr(),
               static_cast<std::size_t>(ny) * sizeof(PetscScalar),
               cudaMemcpyDeviceToDevice);
    PetscCall(VecCUDARestoreArrayWrite(y, &yd));
    return PETSC_SUCCESS;
}
#endif // PETSC_HAVE_CUDA

// ─── Extract one converged triplet into U[:,i] and Vh[i,:] ───────────────────

static void extract_triplet(PetscVec u_vec, PetscVec v_vec, i64 i, i64 m, i64 n,
                             Tensor& U, Tensor& Vh, bool cuda, i32 cuda_dev)
{
    if (!cuda) {
        const PetscScalar* up; VecGetArrayRead(u_vec, &up);
        auto u_src = Tensor::from_blob(
            const_cast<void*>(reinterpret_cast<const void*>(up)),
            {m}, eScalarType::ComplexDouble, Device{eDeviceType::CPU});
        U.select(1, i).copy_(u_src);
        VecRestoreArrayRead(u_vec, &up);

        const PetscScalar* vp; VecGetArrayRead(v_vec, &vp);
        auto v_src = Tensor::from_blob(
            const_cast<void*>(reinterpret_cast<const void*>(vp)),
            {n}, eScalarType::ComplexDouble, Device{eDeviceType::CPU});
        Vh.select(0, i).copy_(v_src.conj());   // Vh[i,:] = conj(v)
        VecRestoreArrayRead(v_vec, &vp);
    }
#if defined(PETSC_HAVE_CUDA)
    else {
        Device dev{eDeviceType::CUDA, static_cast<DeviceIndex>(cuda_dev)};

        const PetscScalar* ud; VecCUDAGetArrayRead(u_vec, &ud);
        auto u_src = Tensor::from_blob(
            const_cast<void*>(reinterpret_cast<const void*>(ud)),
            {m}, eScalarType::ComplexDouble, dev);
        U.select(1, i).copy_(u_src);
        VecCUDARestoreArrayRead(u_vec, &ud);

        const PetscScalar* vd; VecCUDAGetArrayRead(v_vec, &vd);
        auto v_src = Tensor::from_blob(
            const_cast<void*>(reinterpret_cast<const void*>(vd)),
            {n}, eScalarType::ComplexDouble, dev);
        Vh.select(0, i).copy_(v_src.conj());
        VecCUDARestoreArrayRead(v_vec, &vd);
    }
#endif
}

} // namespace detail


// ─── operator_svd ────────────────────────────────────────────────────────────

SVDResult operator_svd(const LinearOperator& op, i64 k, i64 ncv, i64 mpd)
{
    detail::ensure_init();

    bool on_gpu = (op.device().type == eDeviceType::CUDA);
    i64  m = op.m(), n = op.n();

    detail::ShellCtx ctx{&op, on_gpu,
        on_gpu ? static_cast<i32>(op.device().index) : i32{-1}};

    // ── Shell matrix ──────────────────────────────────────────────────────────
    PetscMat A;
    PetscCallAbort(PETSC_COMM_SELF,
        MatCreateShell(PETSC_COMM_SELF,
                       static_cast<PetscInt>(m), static_cast<PetscInt>(n),
                       static_cast<PetscInt>(m), static_cast<PetscInt>(n),
                       &ctx, &A));

    if (on_gpu) {
#if defined(PETSC_HAVE_CUDA)
        PetscCallAbort(PETSC_COMM_SELF,
            MatShellSetOperation(A, MATOP_MULT,
                reinterpret_cast<void(*)(void)>(detail::cuda_mult)));
        PetscCallAbort(PETSC_COMM_SELF,
            MatShellSetOperation(A, MATOP_MULT_HERMITIAN_TRANSPOSE,
                reinterpret_cast<void(*)(void)>(detail::cuda_mult_hermitian)));
        PetscCallAbort(PETSC_COMM_SELF, MatSetVecType(A, VECCUDA));
#else
        MatDestroy(&A);
        throw std::runtime_error(
            "operator_svd: op is on CUDA but PETSc was built without CUDA");
#endif
    } else {
        PetscCallAbort(PETSC_COMM_SELF,
            MatShellSetOperation(A, MATOP_MULT,
                reinterpret_cast<void(*)(void)>(detail::cpu_mult)));
        PetscCallAbort(PETSC_COMM_SELF,
            MatShellSetOperation(A, MATOP_MULT_HERMITIAN_TRANSPOSE,
                reinterpret_cast<void(*)(void)>(detail::cpu_mult_hermitian)));
    }

    // ── SVD solver ────────────────────────────────────────────────────────────
    PetscInt p_ncv = (ncv < 0) ? PETSC_DETERMINE : static_cast<PetscInt>(ncv);
    PetscInt p_mpd = (mpd < 0) ? PETSC_DETERMINE : static_cast<PetscInt>(mpd);

    PetscSVD svd;
    PetscCallAbort(PETSC_COMM_SELF, SVDCreate(PETSC_COMM_SELF, &svd));
    PetscCallAbort(PETSC_COMM_SELF, SVDSetOperators(svd, A, nullptr));
    PetscCallAbort(PETSC_COMM_SELF, SVDSetType(svd, SVDTRLANCZOS));
    PetscCallAbort(PETSC_COMM_SELF,
        SVDSetDimensions(svd, static_cast<PetscInt>(k), p_ncv, p_mpd));
    PetscCallAbort(PETSC_COMM_SELF, SVDSetFromOptions(svd));
    PetscCallAbort(PETSC_COMM_SELF, SVDSolve(svd));

    PetscInt nconv;
    PetscCallAbort(PETSC_COMM_SELF, SVDGetConverged(svd, &nconv));
    i64 k_got = std::min(static_cast<i64>(nconv), k);

    // ── Allocate result buffers ───────────────────────────────────────────────
    Device res_dev = on_gpu
        ? Device{eDeviceType::CUDA, op.device().index}
        : Device{eDeviceType::CPU};

    auto U_cd  = empty({m,     k_got}, TensorOptions{res_dev, eScalarType::ComplexDouble});
    auto Vh_cd = empty({k_got, n    }, TensorOptions{res_dev, eScalarType::ComplexDouble});
    // S_d gathered on CPU first: PETSc sigma values are host scalars and
    // mutable_data_ptr on a CUDA tensor would return a device pointer.
    auto S_cpu = empty({k_got}, TensorOptions{Device{eDeviceType::CPU}, eScalarType::Double});

    // ── Extract converged triplets ────────────────────────────────────────────
    PetscVec u_vec, v_vec;
    PetscCallAbort(PETSC_COMM_SELF, MatCreateVecs(A, &v_vec, &u_vec));

    auto* s_ptr = S_cpu.mutable_data_ptr<double>();
    for (i64 i = 0; i < k_got; ++i) {
        PetscReal sigma;
        PetscCallAbort(PETSC_COMM_SELF,
            SVDGetSingularTriplet(svd, static_cast<PetscInt>(i),
                                  &sigma, u_vec, v_vec));
        s_ptr[i] = static_cast<double>(sigma);
        detail::extract_triplet(u_vec, v_vec, i, m, n, U_cd, Vh_cd, on_gpu,
                                on_gpu ? static_cast<i32>(op.device().index) : i32{-1});
    }
    auto S_d = on_gpu ? S_cpu.to(res_dev) : S_cpu;

    VecDestroy(&u_vec);
    VecDestroy(&v_vec);
    SVDDestroy(&svd);
    MatDestroy(&A);

    // ── Cast to op dtype ──────────────────────────────────────────────────────
    bool want_float = (op.dtype() == eScalarType::ComplexFloat);
    return {
        want_float ? U_cd.to(eScalarType::ComplexFloat)  : std::move(U_cd),
        want_float ? S_d.to(eScalarType::Float)          : std::move(S_d),
        want_float ? Vh_cd.to(eScalarType::ComplexFloat) : std::move(Vh_cd),
    };
}

} // namespace linalg
} // namespace hasty
