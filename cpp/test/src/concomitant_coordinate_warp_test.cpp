import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_linalg_mod;

// ── How much of the concomitant-field phase can be absorbed into a cheap,
// EVALUATABLE coordinate-warp-style correction, before it ever reaches
// phi_lowrank's SVD?
//
// Real (Bernstein/King) concomitant field along z for a general 3-axis
// gradient G(t)=(Gx,Gy,Gz):
//   B_conc(r,t) = (1/2B0)[Gz(t)^2 (x^2+y^2)/4 + Gx(t)^2 z^2 + Gy(t)^2 z^2
//                         - Gx(t)Gz(t) xz - Gy(t)Gz(t) yz]
// Accumulated phase: Phi_conc(r,t) = m(r)^T c(t), m=[x^2+y^2,z^2,z^2,xz,yz]
// (Q=5 EXACT spatial monomials), c(t) the time-integrated gradient-product
// coefficients -- matches ConcomitantBasis's convention exactly.
//
// u(r) must be EVALUATABLE at arbitrary r (CoordinateWarp evaluates spatial
// maps on the full image grid, not just sampled points) -- so we do NOT fit
// u as a free per-voxel value. Since Phi_conc(r,t) is EXACTLY confined to
// span{m_q(r)}, the Eckart-Young-optimal rank-1 u(r) is PROVABLY a linear
// combination u(r)=Sum_q alpha_q*m_q(r) (low-rank approximants of a matrix
// always live in its own row/column space) -- so solve for (w(t), alpha)
// directly in the tiny [K,5] coefficient space instead of the huge [K,N]
// phase matrix. The relevant norm there is G-weighted (G[q,q']=mean_r
// m_q*m_q', the spatial Gram of the monomials over the FOV ball), not plain
// Euclidean -- whiten by G's Cholesky factor, do an ordinary SVD, unwhiten.
// Exact, not approximate -- gives the SAME answer as fitting in voxel space
// would, far cheaper, and produces a genuinely evaluatable closed-form u(r).
//
// FOV: 22cm diameter -> R_fov=11cm radius ball is the FULL FOV here (not a
// restricted central ROI -- this is the ball over which we minimize, voxels
// fill all of it, no excluded edge region in this test).

using namespace hasty;

int main()
{
    Device dev{eDeviceType::CPU};

    const i64 N = 20000;    // voxels, uniform in the FOV ball (for estimating G)
    const i64 K = 2000;     // readout samples, one representative spoke
    const double R_fov   = 0.11;     // 11cm radius = 22cm diameter FOV
    const double G_amp   = 0.04;     // T/m, representative gradient amplitude
    const double B0      = 1.5;      // T -- 1.5T is the WORSE case (term scales 1/B0)
    const double T_read   = 0.015;   // 15ms readout
    const double gamma_p = 2.675e8;  // rad/s/T, proton gyromagnetic ratio

    std::cout << "=== Concomitant-field coordinate-warp absorption test ===\n";
    std::cout << "  N=" << N << " (Gram estimate)  K=" << K << "  R_fov=" << R_fov
              << "m (FULL FOV)  G=" << G_amp << "T/m  B0=" << B0 << "T  T_read=" << T_read << "s\n\n";

    auto opts_d = TensorOptions(dev, eScalarType::Double);

    // Real 3D positions, uniform in the FOV ball (rejection sampling) --
    // only used to estimate G, the spatial Gram of the 5 monomials over the
    // actual FOV geometry. Never materialize the full [K,N] phase matrix.
    Tensor pos;
    {
        std::vector<double> buf;
        buf.reserve((std::size_t)N * 3);
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        i64 got = 0;
        while (got < N) {
            double x = u(rng), y = u(rng), z = u(rng);
            if (x*x + y*y + z*z <= 1.0) {
                buf.push_back(x * R_fov);
                buf.push_back(y * R_fov);
                buf.push_back(z * R_fov);
                ++got;
            }
        }
        pos = Tensor::from_blob(buf.data(), {N, 3}, eScalarType::Double, Device{eDeviceType::CPU})
                  .clone().to(dev);
    }
    auto x = pos.select(1, 0), y = pos.select(1, 1), z = pos.select(1, 2);

    // M: [N,5] monomials at real FOV positions.
    auto M = stack({
        x.mul(x).add(y.mul(y)),   // x^2+y^2
        z.mul(z),                  // z^2 (Gx^2 term)
        z.mul(z),                  // z^2 (Gy^2 term)
        x.mul(z),                  // xz
        y.mul(z)                   // yz
    }, 1);   // [N,5]

    // G = mean_r(m_q*m_q'), the spatial Gram, [5,5].
    auto G = mm(M.transpose(0, 1).contiguous(), M).div(Scalar((double)N));

    // Representative 3-axis gradient waveform: oscillating with different
    // phases per axis (mimics a rotating/oblique readout direction, as in
    // spiral/radial trajectories where all 3 axes are simultaneously active).
    auto t = arange(K, opts_d).mul(Scalar(T_read / (double)(K - 1)));   // [K] seconds
    using namespace std::numbers;
    const double f = 3.0 / T_read;
    auto Gx = t.mul(Scalar(2.0 * pi_v<f64> * f)).cos().mul(Scalar(G_amp));
    auto Gy = t.mul(Scalar(2.0 * pi_v<f64> * f)).add(Scalar(2.0 * pi_v<f64> / 3.0)).cos().mul(Scalar(G_amp));
    auto Gz = t.mul(Scalar(2.0 * pi_v<f64> * f)).add(Scalar(4.0 * pi_v<f64> / 3.0)).cos().mul(Scalar(G_amp));

    auto dt = Scalar(T_read / (double)(K - 1));
    auto cumtrapz = [&](const Tensor& f_t) -> Tensor { return f_t.mul(dt).cumsum(0); };

    // C: [K,5] time-integrated gradient-product coefficients c_q(t).
    auto C = stack({
        cumtrapz(Gz.mul(Gz)).mul(Scalar(gamma_p / (8.0 * B0))),
        cumtrapz(Gx.mul(Gx)).mul(Scalar(gamma_p / (2.0 * B0))),
        cumtrapz(Gy.mul(Gy)).mul(Scalar(gamma_p / (2.0 * B0))),
        cumtrapz(Gx.mul(Gz)).mul(Scalar(-gamma_p / (2.0 * B0))),
        cumtrapz(Gy.mul(Gz)).mul(Scalar(-gamma_p / (2.0 * B0)))
    }, 1);   // [K,5]

    // Var(Phi) = sum_{q,q'} G[q,q'] * mean_k(c_q(k)c_q'(k))  -- never touches [K,N].
    auto var_of = [&](const Tensor& Ck) -> double {
        auto Ctmp = mm(Ck.transpose(0, 1).contiguous(), Ck).div(Scalar((double)K));   // [5,5]
        return G.mul(Ctmp).sum().item<double>();
    };

    double var_total = var_of(C);
    std::cout << "  (1) total:                    var=" << std::scientific << var_total
              << "  rms=" << std::fixed << std::sqrt(var_total) << " rad\n";

    // -- Step 2: remove time-average (DC) per coefficient -- exact, closed-form --
    auto c_bar = C.mean(0);              // [5]
    auto dC    = C.sub(c_bar.unsqueeze(0));  // [K,5], zero-mean
    double var_ac = var_of(dC);
    std::cout << "  (2) after DC removal:         var=" << std::scientific << var_ac
              << "  rms=" << std::fixed << std::sqrt(var_ac) << " rad"
              << "   (" << std::setprecision(1) << (100.0 * (1.0 - var_ac / var_total)) << "% of variance removed)\n";

    // -- Step 3: best rank-R fit w(t)*(M*alpha) to Phi_AC, solved in the tiny
    // G-whitened [K,5] coefficient space (exact, not approximate). G is a
    // Gram matrix (symmetric PSD by construction) -- Cholesky G=L L^T gives
    // the whitening factor directly.
    auto L = linalg_cholesky(G);                    // lower-triangular, [5,5], G = L L^T

    auto I5 = zeros({25}, opts_d);
    auto diag_idx = arange(5, TensorOptions(dev, eScalarType::Long)).mul(Scalar((i64)6));
    I5.scatter_(0, diag_idx, ones({5}, opts_d));
    I5 = I5.reshape({5, 5});

    auto L_inv = linalg_solve(L, I5);               // explicit inverse -- 5x5, cheap, fine here
    auto L_T_inv = L_inv.transpose(0, 1).contiguous();

    auto dC_white = mm(dC, L);                    // [K,5], Euclidean-equivalent
    auto svd = linalg_svd(dC_white, false);        // U:[K,5] S:[5] Vh:[5,5]

    std::cout << "\n  Singular values (G-whitened coefficient space): ";
    auto S_cpu = svd.S.cpu().contiguous();
    const double* sp = S_cpu.const_data_ptr<double>();
    for (i64 i = 0; i < 5; ++i) std::cout << std::scientific << std::setprecision(3) << sp[i] << " ";
    std::cout << "\n\n";

    for (i64 rank : {1, 2, 3, 5}) {
        auto Ur = svd.U.narrow(1, 0, rank);                 // [K,rank]
        auto Sr = svd.S.narrow(0, 0, rank);                 // [rank]
        auto Vhr = svd.Vh.narrow(0, 0, rank);                // [rank,5]
        auto beta = Sr.unsqueeze(1).mul(Vhr);                // [rank,5], in whitened coords
        auto alpha = mm(beta, L_T_inv.transpose(0,1).contiguous());  // unwhiten -> [rank,5], u(r)=M*alpha^T per channel
        auto dC_approx = mm(Ur, alpha);                       // [K,5], reconstruction of dC
        auto resid = dC.sub(dC_approx);
        double var_r = var_of(resid);
        std::cout << "  rank=" << rank << "  resid var=" << std::scientific << var_r
                  << "  (" << std::fixed << std::setprecision(1) << (100.0 * (1.0 - var_r / var_total))
                  << "% of TOTAL variance removed, DC+rank" << rank << " combined)\n";
        if (rank == 1) {
            std::cout << "    u(r) coefficients [x^2+y^2, z^2(Gx), z^2(Gy), xz, yz] alpha = [";
            auto a_cpu = alpha.select(0,0).cpu().contiguous();
            const double* ap = a_cpu.const_data_ptr<double>();
            for (i64 q = 0; q < 5; ++q) std::cout << std::scientific << std::setprecision(3) << ap[q] << (q<4?", ":"");
            std::cout << "]\n";
        }
    }

    std::cout << "\n========================================\n";
    return 0;
}
