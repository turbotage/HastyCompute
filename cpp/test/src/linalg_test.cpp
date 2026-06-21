import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_linalg_mod;

// ─── Reference SVD data ───────────────────────────────────────────────────────
// Builds A = U diag(S) Vh with known singular values on CPU.
// U (m,rank), S (rank,) real, Vh (rank,n) — all complex double.

struct RefSVD { hasty::Tensor A, U, S, Vh; };

static RefSVD make_ref_svd(hasty::i64 m, hasty::i64 n, hasty::i64 rank)
{
    std::mt19937_64 rng{1234};
    std::normal_distribution<double> nd;

    // Build orthonormal U columns and V columns via Gram-Schmidt.
    using cvec = std::vector<std::complex<double>>;
    std::vector<cvec> ucols(rank), vcols(rank);

    auto ortho_norm = [](cvec& v,
                         const std::vector<cvec>& basis, hasty::i64 cnt) {
        for (hasty::i64 j = 0; j < cnt; ++j) {
            std::complex<double> dot{};
            for (std::size_t k = 0; k < v.size(); ++k)
                dot += std::conj(basis[j][k]) * v[k];
            for (std::size_t k = 0; k < v.size(); ++k)
                v[k] -= dot * basis[j][k];
        }
        double nrm = 0;
        for (auto& x : v) nrm += std::norm(x);
        nrm = std::sqrt(nrm);
        if (nrm > 1e-14) for (auto& x : v) x /= nrm;
    };

    for (hasty::i64 r = 0; r < rank; ++r) {
        ucols[r].resize(m); for (auto& x : ucols[r]) x = {nd(rng), nd(rng)};
        ortho_norm(ucols[r], ucols, r);
        vcols[r].resize(n); for (auto& x : vcols[r]) x = {nd(rng), nd(rng)};
        ortho_norm(vcols[r], vcols, r);
    }

    // Singular values: 10, 9, …, 10 - (rank-1).
    hasty::TensorOptions cd{hasty::Device{hasty::eDeviceType::CPU}, hasty::eScalarType::ComplexDouble};
    hasty::TensorOptions rd{hasty::Device{hasty::eDeviceType::CPU}, hasty::eScalarType::Double};

    auto U_t  = hasty::zeros({m, rank    }, cd);
    auto Vh_t = hasty::zeros({rank, n    }, cd);
    auto S_t  = hasty::zeros({rank       }, rd);

    auto* Up  = U_t.mutable_data_ptr<std::complex<double>>();
    auto* Vhp = Vh_t.mutable_data_ptr<std::complex<double>>();
    auto* Sp  = S_t.mutable_data_ptr<double>();

    for (hasty::i64 r = 0; r < rank; ++r) {
        Sp[r] = 10.0 - static_cast<double>(r);   // 10, 9, 8, …
        for (hasty::i64 i = 0; i < m; ++i) Up [i * rank + r] = ucols[r][i];
        for (hasty::i64 j = 0; j < n; ++j) Vhp[r * n    + j] = std::conj(vcols[r][j]);
    }

    // A = U * S_c * Vh  via:  A[i,j] = sum_r U[i,r] * S[r] * Vh[r,j]
    auto A_t = hasty::zeros({m, n}, cd);
    auto* Ap = A_t.mutable_data_ptr<std::complex<double>>();
    for (hasty::i64 i = 0; i < m; ++i)
        for (hasty::i64 j = 0; j < n; ++j) {
            std::complex<double> acc{};
            for (hasty::i64 r = 0; r < rank; ++r)
                acc += Up[i * rank + r] * Sp[r] * Vhp[r * n + j];
            Ap[i * n + j] = acc;
        }

    return {std::move(A_t), std::move(U_t), std::move(S_t), std::move(Vh_t)};
}

// ─── Matvec helpers (always operate on CPU data, handle device transfer) ──────

// Builds a LinearOperator wrapping the (m,n) CPU matrix A, placed on `dev`.
// The shell callbacks receive tensors on `dev`; we move to CPU, multiply, move back.
static hasty::linalg::LinearOperator make_op(const hasty::Tensor& A_cpu, hasty::Device dev)
{
    hasty::i64 m = A_cpu.size(0), n = A_cpu.size(1);

    // A_cpu is captured by value (shared-ptr reference counting keeps it alive).
    auto mv_fn = [A_cpu, m, n, dev](const hasty::Tensor& v) -> hasty::Tensor {
        auto vc = v.cpu().contiguous();
        auto result = hasty::zeros({m}, hasty::TensorOptions{hasty::Device{hasty::eDeviceType::CPU},
                                               hasty::eScalarType::ComplexDouble});
        const auto* Ap = A_cpu.const_data_ptr<std::complex<double>>();
        const auto* vp = vc.const_data_ptr<std::complex<double>>();
        auto*       rp = result.mutable_data_ptr<std::complex<double>>();
        for (hasty::i64 i = 0; i < m; ++i) {
            std::complex<double> acc{};
            for (hasty::i64 j = 0; j < n; ++j) acc += Ap[i * n + j] * vp[j];
            rp[i] = acc;
        }
        return result.to(hasty::TensorOptions{dev, hasty::eScalarType::ComplexDouble});
    };

    auto rmv_fn = [A_cpu, m, n, dev](const hasty::Tensor& u) -> hasty::Tensor {
        auto uc = u.cpu().contiguous();
        auto result = hasty::zeros({n}, hasty::TensorOptions{hasty::Device{hasty::eDeviceType::CPU},
                                               hasty::eScalarType::ComplexDouble});
        const auto* Ap = A_cpu.const_data_ptr<std::complex<double>>();
        const auto* up = uc.const_data_ptr<std::complex<double>>();
        auto*       rp = result.mutable_data_ptr<std::complex<double>>();
        for (hasty::i64 j = 0; j < n; ++j) {
            std::complex<double> acc{};
            for (hasty::i64 i = 0; i < m; ++i)
                acc += std::conj(Ap[i * n + j]) * up[i];
            rp[j] = acc;
        }
        return result.to(hasty::TensorOptions{dev, hasty::eScalarType::ComplexDouble});
    };

    return hasty::linalg::LinearOperator(m, n, std::move(mv_fn), std::move(rmv_fn),
                          hasty::eScalarType::ComplexDouble, dev);
}

// ─── Check CUDA availability without torch headers ────────────────────────────

static bool cuda_available()
{
    try {
        auto t = hasty::empty({1}, hasty::TensorOptions{hasty::Device{hasty::eDeviceType::CUDA, 0},
                                           hasty::eScalarType::Float});
        return true;
    } catch (...) { return false; }
}

// ─── Accuracy test ────────────────────────────────────────────────────────────

static bool test_svd_accuracy(hasty::i64 m, hasty::i64 n, hasty::i64 rank, hasty::Device dev,
                               double sv_rtol   = 1e-4,
                               double recon_tol = 2e-3)
{
    std::string label = (dev.type == hasty::eDeviceType::CUDA) ? "CUDA" : "CPU";
    std::cout << "\n[accuracy] " << m << "x" << n << " rank=" << rank
              << "  k=" << rank << "  dev=" << label << "\n";

    auto ref = make_ref_svd(m, n, rank);
    auto op  = make_op(ref.A, dev);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto [U, S, Vh] = hasty::linalg::operator_svd(op, rank);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Singular value relative error (sorted descending in both).
    auto S_got = S.cpu();
    auto S_ref = ref.S.cpu();
    double sv_err = 0;
    const auto* gp = S_got.const_data_ptr<double>();
    const auto* rp = S_ref.const_data_ptr<double>();
    for (hasty::i64 i = 0; i < rank; ++i)
        sv_err = std::max(sv_err, std::abs(gp[i] - rp[i]) / (rp[i] + 1e-30));

    // Frobenius reconstruction error  ||U S Vh - A||_F / ||A||_F
    auto S_c    = S.to(hasty::eScalarType::ComplexDouble).cpu();
    auto U_cpu  = U.cpu();
    auto Vh_cpu = Vh.cpu();
    auto US = U_cpu.mul(S_c.unsqueeze(0));         // (m, rank)
    auto A_approx = hasty::mm(US, Vh_cpu);                // (m, n)

    auto diff_norm = A_approx.sub(ref.A).norm().item<double>();
    auto A_norm    = ref.A.norm().item<double>();
    double recon_err = diff_norm / A_norm;

    bool sv_ok    = sv_err   < sv_rtol;
    bool recon_ok = recon_err < recon_tol;

    std::cout << "  sv_rel_err = " << sv_err
              << (sv_ok    ? "  OK" : "  FAIL") << "\n"
              << "  recon_err  = " << recon_err
              << (recon_ok ? "  OK" : "  FAIL") << "\n"
              << "  time       = " << ms << " ms\n"
              << "  STATUS: " << ((sv_ok && recon_ok) ? "PASSED" : "FAILED") << "\n";

    return sv_ok && recon_ok;
}

// ─── Performance test ─────────────────────────────────────────────────────────

static void test_svd_perf(hasty::i64 m, hasty::i64 n, hasty::i64 k, hasty::Device dev, int reps = 3)
{
    std::string label = (dev.type == hasty::eDeviceType::CUDA) ? "CUDA" : "CPU";
    std::cout << "\n[perf]     " << m << "x" << n << " k=" << k
              << "  dev=" << label << "\n";

    auto ref = make_ref_svd(m, n, k);
    auto op  = make_op(ref.A, dev);

    double total = 0;
    for (int rep = 0; rep < reps; ++rep) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto [U, S, Vh] = hasty::linalg::operator_svd(op, k);
        (void)U; (void)S; (void)Vh;
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  rep " << rep << ": " << ms << " ms\n";
        total += ms;
    }
    std::cout << "  avg: " << total / reps << " ms\n";
}

// ─── Batched linalg_svd check (A0 for butterfly factorization) ────────────────
//
// hasty::linalg_svd (tensor_external_linalg.cppm) wraps hat::linalg_svd, which
// is libtorch's own at::linalg_svd — that batches over leading dims natively.
// This codebase has never called it batched before; confirm a [batch,m,n]
// call matches a loop of single-matrix calls exactly (same underlying op,
// just exercised in the shape this codebase hasn't used yet).

static bool test_batched_svd(hasty::i64 batch, hasty::i64 m, hasty::i64 n)
{
    std::cout << "\n[batched-svd] batch=" << batch << " " << m << "x" << n << "\n";

    hasty::Device cpu{hasty::eDeviceType::CPU};
    hasty::TensorOptions cd{cpu, hasty::eScalarType::ComplexDouble};

    std::mt19937_64 rng{99};
    std::normal_distribution<double> nd;

    auto A = hasty::zeros({batch, m, n}, cd);
    auto* Ap = A.mutable_data_ptr<std::complex<double>>();
    for (hasty::i64 i = 0; i < batch * m * n; ++i) Ap[i] = {nd(rng), nd(rng)};

    // Batched call.
    auto [Ub, Sb, Vhb] = hasty::linalg_svd(A, false);

    // Loop of single-matrix calls.
    double max_sv_err = 0.0;
    for (hasty::i64 b = 0; b < batch; ++b) {
        auto Ai = A.select(0, b).contiguous();
        auto [Ui, Si, Vhi] = hasty::linalg_svd(Ai, false);
        auto Sb_i = Sb.select(0, b).cpu();
        auto Si_cpu = Si.cpu();
        const auto* gp = Sb_i.const_data_ptr<double>();
        const auto* rp = Si_cpu.const_data_ptr<double>();
        hasty::i64 rank = std::min(m, n);
        for (hasty::i64 r = 0; r < rank; ++r)
            max_sv_err = std::max(max_sv_err, std::abs(gp[r] - rp[r]) / (rp[r] + 1e-30));
    }

    bool ok = max_sv_err < 1e-8;
    std::cout << "  max_sv_rel_err (batched vs looped) = " << max_sv_err
              << (ok ? "  OK" : "  FAIL") << "\n";
    return ok;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main()
{
    std::cout << "====================================================\n"
              << "  SLEPc LinearOperator SVD — accuracy + performance\n"
              << "====================================================\n";

    hasty::Device cpu{hasty::eDeviceType::CPU};
    int failures = 0;

    // ── Accuracy: square, tall, wide ────────────────────────────────────────
    failures += !test_svd_accuracy( 64,  48,  8, cpu);
    failures += !test_svd_accuracy( 48,  64,  8, cpu);         // wide (n > m)
    failures += !test_svd_accuracy(128,  96, 12, cpu);
    failures += !test_svd_accuracy( 96, 128, 12, cpu);
    failures += !test_svd_accuracy(256, 256, 16, cpu, 1e-3, 5e-3);

    // ── Batched linalg_svd (A0 for butterfly factorization) ─────────────────
    failures += !test_batched_svd(8, 16, 16);
    failures += !test_batched_svd(16, 12, 20);   // wide (n > m)
    failures += !test_batched_svd(4, 32, 32);

    // ── Performance: matvec is O(mn) loops so use moderate sizes ────────────
    test_svd_perf(256, 192, 16, cpu, 3);
    test_svd_perf(192, 256, 16, cpu, 3);

    // ── CUDA ─────────────────────────────────────────────────────────────────
    if (cuda_available()) {
        hasty::Device cuda0{hasty::eDeviceType::CUDA, 0};
        failures += !test_svd_accuracy( 64,  48,  8, cuda0);
        failures += !test_svd_accuracy( 48,  64,  8, cuda0);
        failures += !test_svd_accuracy(128,  96, 12, cuda0);
        test_svd_perf(256, 192, 16, cuda0, 3);
    } else {
        std::cout << "\n[CUDA] not available, skipping GPU tests.\n";
    }

    std::cout << "\n====================================================\n"
              << "  " << failures << " failure(s)\n"
              << "====================================================\n";
    return failures > 0 ? 1 : 0;
}
