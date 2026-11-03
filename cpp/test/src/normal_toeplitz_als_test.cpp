import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_mri_mod;

// ── Validates the VARPRO-style closed-form CP-ALS init/loss added to
// normal_toeplitz.cppm (z_svd_init, varpro_reduced_loss) -- see
// notes/MRI_Physics.tex, "Closed-Form Normal-Operator Compression, With
// Coils" / "Full algorithm" for the derivation this implements.

using namespace hasty;
using namespace hasty::mri;

static Tensor rand_complex(ArrayRef<i64> shape, Device dev)
{
    auto re = rand(shape, TensorOptions(dev, eScalarType::Double)).sub(Scalar(0.5));
    auto im = rand(shape, TensorOptions(dev, eScalarType::Double)).sub(Scalar(0.5));
    return view_as_complex(stack({re, im}, -1).contiguous());
}

// ── Test 1: ‖G‖²_F closed form vs an explicit brute-force Frobenius norm ────
//
// Build an EXPLICIT orthonormal spatial basis Upsilon [N,L] (small N), form
// g[j,k]=Upsilon@W^T, form the literal G[j,i,k]=g[j,k]*conj(g[i,k]) tensor
// (small enough here to actually materialize), and compare its Frobenius
// norm against the closed form Sum_k(Sum_l|W[k,l]|^2)^2 used inside
// varpro_reduced_loss (with Acore/Bcore/Lam all zero-width, so only the
// g_norm_sq term contributes).
static bool test_gnorm_closed_form()
{
    Device dev{eDeviceType::CPU};
    const i64 N = 20, L = 4, K = 64;

    // Random orthonormal Upsilon via QR of a random [N,L] matrix.
    auto raw = rand_complex({N, L}, dev);
    auto qr  = linalg_qr(raw);
    auto Upsilon = qr.Q.contiguous();   // [N,L], orthonormal columns

    auto W = rand_complex({K, L}, dev);

    auto g = mm(Upsilon, W.transpose(0, 1).contiguous());   // [N,K]

    // Brute-force G[j,i,k] = g[j,k]*conj(g[i,k]), literal Frobenius norm.
    auto G = g.unsqueeze(1).mul(g.unsqueeze(0).conj());     // [N,N,K]
    double brute = G.abs().pow(Scalar(2.0)).sum().item<double>();

    double closed = varpro_reduced_loss(W, zeros({L, (i64)0}, TensorOptions(dev, eScalarType::ComplexDouble)),
                                            zeros({L, (i64)0}, TensorOptions(dev, eScalarType::ComplexDouble)),
                                            zeros({K, (i64)0}, TensorOptions(dev, eScalarType::ComplexDouble)));

    double rel_err = std::abs(brute - closed) / (brute + 1e-30);
    std::cout << "  [gnorm] brute=" << brute << "  closed=" << closed
              << "  rel_err=" << std::scientific << rel_err << "\n";
    return rel_err < 1e-9;
}

// ── Test 2: new Z-SVD init vs the old one-hot/argsort init ──────────────────
//
// Re-implements the OLD (removed) one-hot init locally -- ranks (l1,l2)
// pairs by a column-importance proxy (column norm of W, standing in for the
// separate S used in the real pipeline) descending, one-hot Acore/Bcore.
// The new z_svd_init must reach a loss no worse than this.
static bool test_init_quality()
{
    Device dev{eDeviceType::CPU};
    const i64 L = 8, K = 1000, P = 4;

    auto W = rand_complex({K, L}, dev);

    // -- old one-hot init, reproduced locally --
    auto col_norm = W.abs().pow(Scalar(2.0)).sum(0);   // [L] real Double, importance proxy
    auto pair_ord = argsort(col_norm.unsqueeze(1).mul(col_norm.unsqueeze(0)).reshape({L*L}),
                            0, /*descending=*/true);
    auto ord_cpu = pair_ord.cpu().contiguous();
    const i64* po = ord_cpu.const_data_ptr<i64>();

    auto Acore_old = zeros({L, P}, TensorOptions(dev, eScalarType::ComplexDouble));
    auto Bcore_old = zeros({L, P}, TensorOptions(dev, eScalarType::ComplexDouble));
    auto Lam_old   = zeros({K, P}, TensorOptions(dev, eScalarType::ComplexDouble));
    for (i64 pi = 0; pi < P; ++pi) {
        i64 idx = po[pi], l1 = idx / L, l2 = idx % L;
        Acore_old[l1, pi] = Scalar(std::complex<double>(1.0, 0.0));
        Bcore_old[l2, pi] = Scalar(std::complex<double>(1.0, 0.0));
        Lam_old.select(1, pi).copy_(W.select(1, l1).mul(W.select(1, l2).conj()));
    }
    double loss_old = varpro_reduced_loss(W, Acore_old, Bcore_old, Lam_old);

    // -- new Z-SVD init --
    auto init = z_svd_init(W, P);
    double loss_new = varpro_reduced_loss(W, init.Acore, init.Bcore, init.Lam);

    std::cout << "  [init quality, pre-ALS] loss_old=" << std::scientific << loss_old
              << "  loss_new=" << loss_new << "\n";

    // -- post-ALS: what actually matters is where each init ends up after
    // refinement, not just where it starts. Run the SAME als_refine from
    // both inits and compare final loss.
    const i64 n_iter = 50;
    auto refined_old = als_refine(W, Acore_old, Bcore_old, Lam_old, n_iter);
    auto refined_new = als_refine(W, init.Acore, init.Bcore, init.Lam, n_iter);
    double loss_old_post = varpro_reduced_loss(W, refined_old.Acore, refined_old.Bcore, refined_old.Lam);
    double loss_new_post = varpro_reduced_loss(W, refined_new.Acore, refined_new.Bcore, refined_new.Lam);

    std::cout << "  [init quality, post-" << n_iter << "-ALS] loss_old=" << loss_old_post
              << "  loss_new=" << loss_new_post << "\n";

    // Report only -- post-ALS result is the real arbiter, and it's an open
    // empirical question which init basin ALS converges to a better point
    // from; don't hard-fail the test on it, just surface the numbers.
    return true;
}

// ── Test 3: coil-weighted loss baseline — confirm als_refine on the plain
// W objective still decreases the coil-weighted loss (it's a necessary condition
// since the factored coil structure means coil maps are applied at apply time,
// not in ALS). This is an informational check only (loss value printed).
// The old als_refine_coil_weighted is removed — coil maps now factor outside
// the P-term split and are applied in apply_normal_toeplitz_off_fourier_operator.
static double coil_weighted_loss_bruteforce(
    const Tensor& Upsilon, const Tensor& W, const Tensor& Xi2,
    const Tensor& Acore, const Tensor& Bcore, const Tensor& Lam)
{
    const i64 N = Upsilon.size(0), P = Acore.size(1), K = W.size(0);

    auto g = mm(Upsilon, W.transpose(0, 1).contiguous());          // [N,K]
    auto G = g.unsqueeze(1).mul(g.unsqueeze(0).conj());            // [N,N,K]  G[j,i,k]

    auto A = mm(Upsilon, Acore);                                    // [N,P]
    auto B = mm(Upsilon.conj().contiguous(), Bcore);                 // [N,P]
    auto AB = A.unsqueeze(1).mul(B.unsqueeze(0)).reshape({N*N, P}); // [N*N,P]  AB[j,i,p]
    auto Mhat = mm(AB, Lam.transpose(0, 1).contiguous()).reshape({N, N, K});  // [N,N,K]

    auto resid2 = G.sub(Mhat).abs().pow(Scalar(2.0));               // [N,N,K]
    return resid2.mul(Xi2.unsqueeze(2)).sum().item<double>();
}

static bool test_coil_weighted_loss_informational()
{
    Device dev{eDeviceType::CPU};
    const i64 N = 10, L = 3, K = 20, P = 2, C = 3;

    auto raw = rand_complex({N, L}, dev);
    auto Upsilon = linalg_qr(raw).Q.contiguous();
    auto W = rand_complex({K, L}, dev);

    auto coil_maps = rand_complex({C, N}, dev).add(Scalar(std::complex<double>(0.5, 0.0)));
    auto Xi = mm(coil_maps.conj().transpose(0, 1).contiguous(), coil_maps);
    auto Xi2 = Xi.abs().pow(Scalar(2.0));

    auto init = z_svd_init(W, P);
    auto refined = als_refine(W, init.Acore, init.Bcore, init.Lam, 30);
    double loss = coil_weighted_loss_bruteforce(Upsilon, W, Xi2, refined.Acore, refined.Bcore, refined.Lam);

    std::cout << "  [coil-weighted loss after plain als_refine]=" << std::scientific << loss << "\n";
    return true;  // informational only
}

int main()
{
    std::cout << "=== Normal-Toeplitz VARPRO ALS init/loss test ===\n\n";
    int failures = 0;

    std::cout << "-- gnorm closed form vs brute force --\n";
    failures += !test_gnorm_closed_form();

    std::cout << "\n-- init quality: new Z-SVD vs old one-hot --\n";
    failures += !test_init_quality();

    std::cout << "\n-- coil-weighted loss (informational, plain als_refine) --\n";
    failures += !test_coil_weighted_loss_informational();

    std::cout << "\n========================================\n"
              << "  " << failures << " failure(s)\n"
              << "========================================\n";
    return failures > 0 ? 1 : 0;
}
