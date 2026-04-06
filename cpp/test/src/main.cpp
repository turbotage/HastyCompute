#include <configure_file_settings.hpp>

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_fft_mod;
import hasty_generic_value_mod;
import hasty_server_mod;
import hasty_io_mod;

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

// Helper: L2 norm of a complex flat tensor via ATen
static double l2_norm(const hasty::Tensor& t)
{
    return t.to_torch().norm().item<double>();
}

std::pair<bool, std::string> test_nufft_normal_identity()
{
    using namespace hasty;
    using namespace hasty::fft;

    std::cout << "test_nufft_normal_identity: running 2-D test...\n";

    constexpr i64 NY   = 16;
    constexpr i64 NX   = 16;
    constexpr i64 npts = 100;

    Device cuda0(eDeviceType::CUDA, 0);

    // ── k-space trajectory: [2, npts] float, uniform in [-pi, pi] ────────────
    // rand gives [0,1); scale to [-pi, pi)
    constexpr float pi = 3.14159265358979f;
    Tensor coords = rand({2, npts}, TensorOptions(cuda0, eScalarType::Float));
    coords.mul_(Scalar{2.0f * pi});
    coords.add_(Scalar{-pi});

    // ── Density weights: all ones (real) ─────────────────────────────────────
    Tensor weights = ones({npts}, TensorOptions(cuda0, eScalarType::Float));

    return {false, ""};
}

// Test that toeplitz_multiplication(x) ≈ A^H W A x for a random 2-D problem.
//
// A   = UTN (uniform → non-uniform, type-2 NUFFT)
// A^H = NTU (non-uniform → uniform, type-1 NUFFT)
// W   = diagonal density weights (real, all-ones here)
//
// im_size = {NY, NX};  nmodes for cufinufft = {NX, NY}  (x fastest)
// Returns a [N, npts] float tensor of Cartesian k-space coordinates in [-pi, pi).
// nmodes[0] is the fastest-varying dimension (x), matching the cufinufft/NUFFT convention.
// npts = prod(nmodes).  Pair with create_toeplitz_kernel(..., im_size={nmodes[N-1],...,nmodes[0]}).
template<std::size_t N>
hasty::Tensor create_cartesian_coords(const std::array<hasty::i64, N>& nmodes, hasty::Device device)
{
    using namespace hasty;
    constexpr float pi = 3.14159265358979f;

    i64 npts = 1;
    for (auto n : nmodes) npts *= n;

    std::vector<Tensor> coord_rows;
    coord_rows.reserve(N);

    for (std::size_t d = 0; d < N; ++d) {
        i64 n = nmodes[d];

        // inner = nmodes[0] * ... * nmodes[d-1]  → makes dim 0 cycle fastest
        i64 inner = 1;
        for (std::size_t k = 0; k < d; ++k) inner *= nmodes[k];

        // outer = nmodes[d+1] * ... * nmodes[N-1]
        i64 outer = 1;
        for (std::size_t k = d + 1; k < N; ++k) outer *= nmodes[k];

        // 1D grid for this dimension: 2*pi*k/n - pi, k = 0 .. n-1
        Tensor c1d = arange(n, TensorOptions(device, eScalarType::Float));
        c1d = c1d.mul(Scalar{2.0f * pi / static_cast<float>(n)}).add(Scalar{-pi});

        // Indices for one tile: each of the n values repeated `inner` times
        // [0,0,...(inner), 1,1,...(inner), ..., n-1,...(inner)]
        // Use float arange + divide + truncate-to-long (safe for non-negative values)
        Tensor fidx = arange(n * inner, TensorOptions(device, eScalarType::Float));
        Tensor idx  = fidx.div(Scalar{static_cast<float>(inner)}).to(eScalarType::Long);
        Tensor tile = c1d.index_select(0, idx);  // size = n * inner

        // Repeat the tile `outer` times along dim 0
        std::vector<Tensor> tiles;
        tiles.reserve(outer);
        for (i64 o = 0; o < outer; ++o) tiles.push_back(tile);

        coord_rows.push_back(cat(tiles, 0));  // size = npts
    }

    return hasty::stack(coord_rows, 0);  // [N, npts]
}



void test_toeplitz_multiplication()
{
    using namespace hasty;
    using namespace hasty::fft;

    std::cout << "test_toeplitz_multiplication: running 2-D test...\n";

    Device cuda0(eDeviceType::CUDA, 0);

    Tensor input;
    {
        auto img_path = std::string(HASTY_DATA_DIR) + "/imgs/images_1.h5";
        std::cout << "Loading image from " << img_path << std::endl;
    
        //hasty::GenericValue gv = hasty::io::hdf5::read_generic_value_entry(img_path, "coins_303x384", false);
        hasty::GenericValue gv = hasty::io::hdf5::read_generic_value_entry(img_path, "astronaut_luma_512x512", false);
    
        input = gv.as_tensor();
        //input = input.transpose(0, 1).contiguous();

        //input = rand({64, 128}, TensorOptions(eScalarType::Float));

        hasty::viz::default_heatmap(hasty::viz::DefaultHeatmapOptions<1,1>{
            .z = {{input.spanning_view()}},
            .titles = {{"Input"}}
        }).show();

        input = input.to(TensorOptions(cuda0, eScalarType::ComplexFloat)).contiguous().unsqueeze(0);
    }

    i64 NY   = input.size(1);
    i64 NX   = input.size(2);
    
    bool cartesion_coords = false;
    Tensor coords;
    if (cartesion_coords) {
        coords = create_cartesian_coords<2>({NX, NY}, cuda0);
    } else {
        coords = rand({2, 10000}, TensorOptions(cuda0, eScalarType::Float));
        coords.mul_(Scalar{2*3.141592f});
        coords.add_(Scalar{-3.141592f});
    }
    
    
    i64 npts = coords.size(1);
    
    Tensor weights = ones({npts}, TensorOptions(cuda0, eScalarType::ComplexFloat));


    //Tensor kernel = create_toeplitz_kernel(coords, weights, {NY, NX}, true);
    Tensor kernel = create_toeplitz_kernel_standard(coords, weights, {NY, NX});

    {
        auto kernel_real_cpu = kernel.real().cpu().contiguous();
        auto kernel_imag_cpu = kernel.imag().cpu().contiguous();
        
        std::cout << "Kernel real max value: " << kernel_real_cpu.max().item<float>() << std::endl;
        std::cout << "Kernel real min value: " << kernel_real_cpu.min().item<float>() << std::endl;
        std::cout << "Kernel imag max value: " << kernel_imag_cpu.max().item<float>() << std::endl;
        std::cout << "Kernel imag min value: " << kernel_imag_cpu.min().item<float>() << std::endl;
        std::cout << "Kernel real mean value: " << kernel_real_cpu.mean().item<float>() << std::endl;
        std::cout << "Kernel imag mean value: " << kernel_imag_cpu.mean().item<float>() << std::endl;
    }

    {

        auto kernel_real_cpu = kernel.real().cpu().contiguous();
        auto kernel_imag_cpu = kernel.imag().cpu().contiguous();
        viz::default_heatmap(viz::DefaultHeatmapOptions<1, 2>{
            .z = {{kernel_real_cpu.spanning_view(), kernel_imag_cpu.spanning_view()}},
            .titles = {{"Kernel Real Part", "Kernel Imaginary Part"}}
        }).show();
    }

    Tensor output_toep = zeros({1, NY, NX}, TensorOptions(cuda0, eScalarType::ComplexFloat));
    Tensor output_nufft = zeros_like(output_toep);

    {
        Tensor intermediate_output = zeros({1, npts}, TensorOptions(cuda0, eScalarType::ComplexFloat));
        {
            NufftOptions<cuda_t, f32, UTN> opts;
            NufftPlan<cuda_t, f32, 2, UTN> plan({NX, NY}, opts);
            plan.setpts(coords);
            plan.execute(input, intermediate_output);

        }
        intermediate_output.mul_(Scalar{1.0f / static_cast<float>(NY * NX)});  // scale for unnormalized FFT
        {
            NufftOptions<cuda_t, f32, NTU> opts;
            NufftPlan<cuda_t, f32, 2, NTU> plan({NX, NY}, opts);
            plan.setpts(coords);
            plan.execute(intermediate_output, output_nufft);
        }
    }

    toeplitz_multiplication(
        input, output_toep, kernel,
        std::nullopt, std::nullopt, std::nullopt,
        ToeplitzMultType::NONE,
        ToeplitzMultType::NONE,      // input_mult1_type  (unused, mult1=null)
        ToeplitzMultType::NONE, // output_mult1_type (unused)
        ToeplitzMultType::NONE,      // input_mult2_type  (unused, mult2=null)
        ToeplitzMultType::NONE, // output_mult2_type (unused)
        ToeplitzAccumulateType::NONE
    );

    auto ratio = output_toep.real().mean().item<double>() / output_nufft.real().mean().item<double>();
    std::cout << "Mean ratio (toep/NUFFT): " << ratio << std::endl;

    output_toep = output_toep.view({NY, NX}).cpu();
    output_nufft = output_nufft.view({NY, NX}).cpu();

    auto output_toep_real = output_toep.real().contiguous();
    auto output_toep_imag = output_toep.imag().contiguous();

    auto output_nufft_real = output_nufft.real().contiguous();
    auto output_nufft_imag = output_nufft.imag().contiguous();


    {
        auto straight_output = zeros({2*NY, 2*NX}, TensorOptions(cuda0, eScalarType::ComplexFloat));
        straight_output[Slice(0, NY), Slice(0, NX)] = input.squeeze(0);
        straight_output = fftn(straight_output);
        straight_output.mul_(Scalar{1.0f / static_cast<float>(4 * NY * NX)});  // scale for unnormalized FFT
        straight_output.mul_(kernel);
        straight_output = ifftn(straight_output);
        straight_output = straight_output[Slice(0, NY), Slice(0, NX)];

        auto straight_output_real = straight_output.real().cpu().contiguous();
        auto straight_output_imag = straight_output.imag().cpu().contiguous();

        viz::default_heatmap(viz::DefaultHeatmapOptions<2, 2>{
            .z = Arr{
                    Arr{straight_output_real.spanning_view(), output_nufft_real.spanning_view()},
                    Arr{straight_output_imag.spanning_view(), output_nufft_imag.spanning_view()}
                },
            .titles = Arr{
                Arr<std::string,2>{"straight_output_real", "output_nufft_real"}, 
                Arr<std::string,2>{"straight_output_imag", "output_nufft_imag"}
            }
        }).show();
    }

    viz::default_heatmap(viz::DefaultHeatmapOptions<2, 2>{
        .z = Arr{
            Arr{output_toep_real.spanning_view(), output_nufft_real.spanning_view()},
            Arr{output_toep_imag.spanning_view(), output_nufft_imag.spanning_view()}
        },
        .titles = Arr{
            Arr<std::string,2>{"output_toep_real", "output_nufft_real"}, 
            Arr<std::string,2>{"output_toep_imag", "output_nufft_imag"}
        }
    }).show();

    // ── Compare ───────────────────────────────────────────────────────────────
    if (false) {
        output_toep = output_toep.contiguous().view({NY,NX}).cpu();
    
        auto output_real = output_toep.real().contiguous();
        auto output_imag = output_toep.imag().contiguous();
    
        auto input_real = input.view({NY, NX}).cpu().real().contiguous();
        auto input_imag = input.view({NY, NX}).cpu().imag().contiguous();
    
        viz::default_heatmap(viz::DefaultHeatmapOptions<1, 2>{
            .z = {{input_real.spanning_view(), output_real.spanning_view()}},
            .titles = {{"Input Real Part", "Toeplitz Output Real Part"}}
        }).show();
    
        viz::default_heatmap(viz::DefaultHeatmapOptions<1, 2>{
            .z = {{input_imag.spanning_view(), output_imag.spanning_view()}},
            .titles = {{"Input Imag Part", "Toeplitz Output Imaginary Part"}}
        }).show();
    }





}

void test_slider_heatmap() {
    using namespace hasty;
    using namespace hasty::viz;

    // Sanity-check: synthetic volume where slice k is filled with value k.
    // If the slider works, the heatmap brightness should change as you drag.
    {
        constexpr i64 NZ = 16, NY = 64, NX = 64;
        Tensor synth = zeros({NZ, NY, NX}, TensorOptions(eScalarType::Float));
        for (i64 k = 0; k < NZ; ++k)
            synth.select(0, k).fill_(Scalar{static_cast<float>(k)});

        default_heatmap_slider(DefaultHeatmapSliderOptions<1, 1>{
            .z = {{synth.spanning_view()}},
            .titles = {{"Slider sanity-check (brightness = slice index)"}},
            .slider_prefix = "z = "
        }).show();
    }

    auto img_path = std::string(HASTY_DATA_DIR) + "/imgs/images_1.h5";
    std::cout << "Loading shepp_logan_3d_64x400x400 from " << img_path << "\n";

    GenericValue gv = hasty::io::hdf5::read_generic_value_entry(img_path, "shepp_logan_3d_64x400x400", false);
    Tensor vol = gv.as_tensor().contiguous();  // [64, 400, 400] on CPU

    std::cout << "Loaded shape: [" << vol.size(0) << ", " << vol.size(1) << ", " << vol.size(2) << "]\n";

    default_heatmap_slider(DefaultHeatmapSliderOptions<1, 1>{
        .z = {{vol.spanning_view()}},
        .titles = {{"Shepp-Logan 3D"}},
        .slider_prefix = "z = "
    }).show();
}

int main() {
    //test_slider_heatmap();
    //viz_test();
    //auto test_pair = test_toeplitz_identity_kernel();
    //std::cout << test_pair.second;
    //test_toeplitz_multiplication();
    //test_cartesian_coords_gives_unity_kernel();
    //test_tensor_array_operator();


    test_toeplitz_multiplication();
    //hasty::viz::test_tensor_viz();

    return 0;
}
