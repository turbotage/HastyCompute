import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_fft_mod;
import hasty_generic_value_mod;
import hasty_server_mod;

import hasty_viz_mod;

void server_test() {
    std::cout << "Starting server..." << std::endl;

    GenericValueBank bank;
    CommandRegistry registry;

    // function_id 0: element-wise tensor add
    registry.register_command(0, "add",
        [](const std::string&, std::vector<hasty::GenericValue> inputs)
            -> std::vector<hasty::GenericValue>
        {
            if (inputs.size() != 2 || !inputs[0].is_tensor() || !inputs[1].is_tensor())
                throw std::runtime_error("add: requires 2 tensor inputs");
            return { hasty::GenericValue(inputs[0].as_tensor().add(inputs[1].as_tensor())) };
        });

    auto handle = start_server(bank, registry, "0.0.0.0:50051");
    //auto handle = start_server(bank, registry, "unix:///tmp/hasty.sock");

    // Signal readiness to any waiting test runner
    std::cout << "READY" << std::endl;

    handle.wait();
}

void viz_test()
{
    std::cout << "Generating example plot..." << std::endl;
    auto fig = hasty::viz::example_plot();
    fig.show();
}

// Helper: L2 norm of a complex flat tensor via ATen
static double l2_norm(const hasty::Tensor& t)
{
    return t.to_torch().norm().item<double>();
}

// Test that toeplitz_multiplication(x) ≈ A^H W A x for a random 2-D problem.
//
// A   = UTN (uniform → non-uniform, type-2 NUFFT)
// A^H = NTU (non-uniform → uniform, type-1 NUFFT)
// W   = diagonal density weights (real, all-ones here)
//
// im_size = {NY, NX};  nmodes for cufinufft = {NX, NY}  (x fastest)
void test_toeplitz_multiplication()
{
    using namespace hasty;
    using namespace hasty::fft;

    std::cout << "test_toeplitz_multiplication: running 2-D test...\n";

    constexpr i64 NY   = 32;
    constexpr i64 NX   = 32;
    constexpr i64 npts = 1000;

    Device cuda0(eDeviceType::CUDA, 0);

    // ── k-space trajectory: [2, npts] float, uniform in [-pi, pi] ────────────
    // rand gives [0,1); scale to [-pi, pi)
    constexpr float pi = 3.14159265358979f;
    Tensor coords = rand({2, npts}, TensorOptions(cuda0, eScalarType::Float));
    coords.mul_(Scalar{2.0f * pi});
    coords.add_(Scalar{-pi});

    // ── Density weights: all ones (real) ─────────────────────────────────────
    Tensor weights = ones({npts}, TensorOptions(cuda0, eScalarType::Float));

    // ── Random complex image, flat [NY*NX] ───────────────────────────────────
    Tensor x_flat = view_as_complex(
        stack({rand({NY * NX}, TensorOptions(cuda0, eScalarType::Float)),
               rand({NY * NX}, TensorOptions(cuda0, eScalarType::Float))}, -1).contiguous());

    // ── Direct A^H W A x ─────────────────────────────────────────────────────

    // Forward NUFFT: x [NY*NX] → kspace [npts]
    // nmodes = {NX, NY} so cufinufft outputs x-fastest, matching flat x_flat layout
    NufftPlan<cuda_t, f32, 2, UTN> plan_utn({NX, NY});
    plan_utn.setpts(coords);
    Tensor kspace = zeros({npts}, TensorOptions(cuda0, eScalarType::ComplexFloat));
    plan_utn.execute(x_flat, kspace);

    // Apply real weights W (promote to complex)
    Tensor w_cplx = view_as_complex(
        stack({weights, zeros_like(weights)}, -1).contiguous());
    kspace.mul_(w_cplx);

    // Adjoint NUFFT: kspace [npts] → y_direct [NY*NX]
    NufftPlan<cuda_t, f32, 2, NTU> plan_ntu({NX, NY});
    plan_ntu.setpts(coords);
    Tensor y_direct = zeros({NY * NX}, TensorOptions(cuda0, eScalarType::ComplexFloat));
    plan_ntu.execute(kspace, y_direct);

    // ── Toeplitz approach ─────────────────────────────────────────────────────

    // Build and transform kernel (shape {2*NY, 2*NX} → VkFFT convolution format)
    Tensor kernel = create_toeplitz_kernel(coords, weights, {NY, NX});
    transform_toeplitz_kernel(kernel);

    // Input needs a batch dimension: [1, NY, NX]
    Tensor x_img      = x_flat.view({1, NY, NX});
    Tensor y_toeplitz = zeros({1, NY, NX}, TensorOptions(cuda0, eScalarType::ComplexFloat));

    toeplitz_multiplication(
        x_img, y_toeplitz, kernel,
        std::nullopt, std::nullopt, std::nullopt,
        ToeplitzMultType::NONE,
        ToeplitzMultType::MULT,      // input_mult1_type  (unused, mult1=null)
        ToeplitzMultType::MULT_CONJ, // output_mult1_type (unused)
        ToeplitzMultType::MULT,      // input_mult2_type  (unused, mult2=null)
        ToeplitzMultType::MULT_CONJ, // output_mult2_type (unused)
        ToeplitzAccumulateType::NONE
    );

    // ── Compare ───────────────────────────────────────────────────────────────
    Tensor y_toeplitz_flat = y_toeplitz.view({NY * NX});
    Tensor diff            = y_direct.add(y_toeplitz_flat, Scalar{-1});

    double norm_ref  = l2_norm(y_direct);
    double norm_diff = l2_norm(diff);
    double rel_err   = (norm_ref > 0.0) ? norm_diff / norm_ref : norm_diff;

    std::cout << "  ||y_direct||  = " << norm_ref  << "\n";
    std::cout << "  ||y_toeplitz|| = " << l2_norm(y_toeplitz_flat) << "\n";
    std::cout << "  rel_err       = " << rel_err << "\n";

    if (rel_err < 1e-2)
        std::cout << "  PASS\n";
    else
        std::cout << "  FAIL (rel_err too large)\n";
}

int main() {
    //server_test();
    //viz_test();
    test_toeplitz_multiplication();
    return 0;
}
