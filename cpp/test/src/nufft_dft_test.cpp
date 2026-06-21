import std;
impory hasty_util_mod;
import hasty_tensor_mod;
import hasty_fft_mod;


void nufft_dft_consistency_test()
{
    using namespace hasty;
    using namespace hasty::fft;
    using namespace std::numbers;

    const i64 nx = 224, ny = 320, nz = 280;
    const i64 M  = 200;

    auto cuda_dev = Device{eDeviceType::CUDA, 0};
    auto cpu_dev  = Device{eDeviceType::CPU};

    const TensorOptions opts_f{cuda_dev, eScalarType::Float};
    const TensorOptions opts_c{cuda_dev, eScalarType::ComplexFloat};

    std::cout << "NUFFT/DFT consistency test\n"
              << "  image: " << nx << "x" << ny << "x" << nz
              << "  M=" << M << " random frequencies\n\n";

    auto img_r = rand({nx, ny, nz}, opts_f);
    auto img_i = rand({nx, ny, nz}, opts_f);
    auto img   = view_as_complex(stack({img_r, img_i}, -1).contiguous());

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> udist(
        -(float)pi_v<double> * 0.99f, (float)pi_v<double> * 0.99f);
    std::vector<float> xi_buf(M * 3);
    for (auto& v : xi_buf) v = udist(rng);
    auto xi = Tensor::from_blob(xi_buf.data(), {M, 3}, eScalarType::Float, cpu_dev)
                  .clone().to(cuda_dev);

    auto dft_cfg = make_dft_config({nx, ny, nz}, /*batch_size=*/1, cuda_dev);
    auto t0_dft  = std::chrono::steady_clock::now();
    auto F_dft   = dft(dft_cfg, img, xi);
    double dft_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0_dft).count();

    auto coords = zeros({3, M}, opts_f);
    coords.select(0, 0).copy_(xi.select(1, 2));
    coords.select(0, 1).copy_(xi.select(1, 1));
    coords.select(0, 2).copy_(xi.select(1, 0));
    coords = coords.contiguous();

    NufftOptions<cuda_t, f32, UTN> nufft_opts;
    nufft_opts.ntransf = 1;
    NufftPlan<cuda_t, f32, 3, UTN> plan({nx, ny, nz}, nufft_opts);
    plan.setpts(coords);

    auto img_batch   = img.unsqueeze(0).contiguous();
    auto F_nufft_out = zeros({1, M}, opts_c).contiguous();
    auto t0_nufft    = std::chrono::steady_clock::now();
    plan.execute(img_batch, F_nufft_out);
    double nufft_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0_nufft).count();
    auto F_nufft = F_nufft_out.select(0, 0);

    {
        auto dc_coords = zeros({3, 1}, opts_f);
        NufftPlan<cuda_t, f32, 3, UTN> dc_plan({nx, ny, nz}, nufft_opts);
        dc_plan.setpts(dc_coords);
        auto dc_out = zeros({1, 1}, opts_c).contiguous();
        dc_plan.execute(img_batch, dc_out);
        float dc_nufft = dc_out.real().item<float>();

        auto xi_dc  = zeros({1, 3}, opts_f);
        auto F_dc   = dft(dft_cfg, img, xi_dc);
        float dc_dft  = F_dc.real().item<float>();
        float dc_true = img.real().sum().item<float>();

        std::cout << "  DC sanity:\n"
                  << "    sum(img)     = " << std::fixed << std::setprecision(2) << dc_true  << "\n"
                  << "    NUFFT(0,0,0) = " << dc_nufft << "  |err|=" << std::abs(dc_nufft - dc_true) << "\n"
                  << "    DFT(0,0,0)   = " << dc_dft   << "  |err|=" << std::abs(dc_dft   - dc_true) << "\n\n";
    }

    auto abs_err       = F_dft.sub(F_nufft).abs();
    float mean_abs_err = abs_err.mean().item<float>();
    float max_abs_err  = abs_err.max().item<float>();
    float ref_norm     = F_dft.abs().mean().item<float>();
    float rel_err      = mean_abs_err / (ref_norm + 1e-30f);

    auto phase_diff = F_dft.angle().sub(F_nufft.angle());
    float phase_circ_r = phase_diff.cos().mean().item<float>();

    std::cout << "  NUFFT vs DFT (" << M << " random freqs):\n"
              << "    mean |err|   = " << std::scientific << std::setprecision(3) << mean_abs_err << "\n"
              << "    max  |err|   = " << max_abs_err << "\n"
              << "    rel err      = " << rel_err << "\n"
              << "    phase circ-r = " << std::fixed << std::setprecision(6) << phase_circ_r << "\n\n"
              << "  timing: DFT=" << std::setprecision(2) << dft_s
              << "s  NUFFT=" << nufft_s << "s\n\n";

    bool pass = (phase_circ_r > 0.9999f) && (rel_err < 1e-3f);
    std::cout << (pass ? "PASS" : "FAIL") << "\n";
}


int main(){

    nufft_dft_consistency_test();

}