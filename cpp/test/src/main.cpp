#include <configure_file_settings.hpp>

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_fft_mod;
import hasty_generic_value_mod;
import hasty_server_mod;
import hasty_io_mod;
import hasty_server_mod;
import hasty_viz_mod;

void server_test() {
    std::cout << "Starting server..." << std::endl;

    hasty::GenericValueBank bank;
    hasty::CommandRegistry registry;

    // function_id 0: element-wise tensor add
    registry.register_command(0, "add",
        [](const std::string&, std::vector<hasty::GenericValue> inputs)
            -> std::pair<std::string, std::vector<hasty::GenericValue>>
        {
            if (inputs.size() != 2 || !inputs[0].is_tensor() || !inputs[1].is_tensor())
                throw std::runtime_error("add: requires 2 tensor inputs");
            return std::make_pair(std::string(""), std::vector<hasty::GenericValue>{hasty::GenericValue(inputs[0].as_tensor().add(inputs[1].as_tensor()))});
        });

    auto handle = start_grpc_server(bank, registry, "0.0.0.0:50051");
    //auto handle = start_server(bank, registry, "unix:///tmp/hasty.sock");

    // Signal readiness to any waiting test runner
    std::cout << "READY" << std::endl;

    handle.wait();
}

void test_toeplitz_multiplication_3D(hasty::ArrayRef<hasty::i64> im_size, double rtol = 1e-5, double atol = 1e-3)
{
    using namespace hasty;
    using namespace hasty::fft;

    Device cuda0(eDeviceType::CUDA, 0);

    Tensor input = rand({1, im_size[0], im_size[1], im_size[2]}, TensorOptions(cuda0, eScalarType::ComplexFloat));


    i64 NZ      = input.size(1);
    i64 NY      = input.size(2);
    i64 NX      = input.size(3);
    
    bool cartesion_coords = false;
    Tensor coords;
    if (cartesion_coords) {
        coords = create_cartesian_coords<3>({NZ, NY, NX}, cuda0);
    } else {
        coords = rand({3, 10000}, TensorOptions(cuda0, eScalarType::Float));
        coords.mul_(Scalar{2*3.141592f});
        coords.add_(Scalar{-3.141592f});
    }
    
    
    i64 npts = coords.size(1);
    
    Tensor weights = ones({npts}, TensorOptions(cuda0, eScalarType::ComplexFloat));


    Tensor kernel = create_toeplitz_kernel_standard(coords, weights, {NZ, NY, NX});


    Tensor output_toep = zeros({1, NZ, NY, NX}, TensorOptions(cuda0, eScalarType::ComplexFloat));
    Tensor output_nufft = zeros_like(output_toep);

    {
        Tensor intermediate_output = zeros({1, npts}, TensorOptions(cuda0, eScalarType::ComplexFloat));
        {
            NufftOptions<cuda_t, f32, UTN> opts;
            NufftPlan<cuda_t, f32, 3, UTN> plan({NZ, NY, NX}, opts);
            plan.setpts(coords);
            plan.execute(input, intermediate_output);

        }
        intermediate_output.mul_(Scalar{1.0f / static_cast<float>(NZ * NY * NX)});  // scale for unnormalized FFT
        {
            NufftOptions<cuda_t, f32, NTU> opts;
            NufftPlan<cuda_t, f32, 3, NTU> plan({NZ, NY, NX}, opts);
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

    output_toep = output_toep.view({NZ, NY, NX}).cpu();
    output_nufft = output_nufft.view({NZ, NY, NX}).cpu();

    bool real_allclose = allclose(output_toep.real(), output_nufft.real(), rtol, atol);
    bool imag_allclose = allclose(output_toep.imag(), output_nufft.imag(), rtol, atol);

    if (!real_allclose || !imag_allclose) {
        std::cout << "Test: Toeplitz Multiplication (3D): FAILED\n";
    } else {
        std::cout << "Test: Toeplitz Multiplication (3D): PASSED\n";
    }

}

void test_toeplitz_multiplication_2D(hasty::ArrayRef<hasty::i64> im_size, double rtol = 1e-5, double atol = 1e-3)
{
    using namespace hasty;
    using namespace hasty::fft;

    Device cuda0(eDeviceType::CUDA, 0);

    Tensor input = rand({1, im_size[0], im_size[1]}, TensorOptions(cuda0, eScalarType::ComplexFloat));
    input = input.to(TensorOptions(cuda0, eScalarType::ComplexFloat)).contiguous();

    i64 NY = input.size(1);
    i64 NX = input.size(2);

    bool cartesian_coords = false;
    Tensor coords;

    if (cartesian_coords) {
        coords = create_cartesian_coords<2>({NY, NX}, cuda0);
    } else {
        coords = rand({2, 10000}, TensorOptions(cuda0, eScalarType::Float));
        coords.mul_(Scalar{2 * 3.141592f});
        coords.add_(Scalar{-3.141592f});
    }

    i64 npts = coords.size(1);

    Tensor weights = ones({npts}, TensorOptions(cuda0, eScalarType::ComplexFloat));

    Tensor kernel = create_toeplitz_kernel_standard(coords, weights, {NY, NX});

    Tensor output_toep = zeros({1, NY, NX}, TensorOptions(cuda0, eScalarType::ComplexFloat));
    Tensor output_nufft = zeros_like(output_toep);

    // --- NUFFT reference ---
    {
        Tensor intermediate_output = zeros({1, npts}, TensorOptions(cuda0, eScalarType::ComplexFloat));

        {
            NufftOptions<cuda_t, f32, UTN> opts;
            NufftPlan<cuda_t, f32, 2, UTN> plan({NY, NX}, opts);
            plan.setpts(coords);
            plan.execute(input, intermediate_output);
        }

        intermediate_output.mul_(Scalar{1.0f / static_cast<float>(NY * NX)});

        {
            NufftOptions<cuda_t, f32, NTU> opts;
            NufftPlan<cuda_t, f32, 2, NTU> plan({NY, NX}, opts);
            plan.setpts(coords);
            plan.execute(intermediate_output, output_nufft);
        }
    }

    // --- Toeplitz ---
    toeplitz_multiplication(
        input, output_toep, kernel,
        std::nullopt, std::nullopt, std::nullopt,
        ToeplitzMultType::NONE,
        ToeplitzMultType::NONE,
        ToeplitzMultType::NONE,
        ToeplitzMultType::NONE,
        ToeplitzMultType::NONE,
        ToeplitzAccumulateType::NONE
    );

    bool real_allclose = allclose(output_toep.real(), output_nufft.real(), rtol, atol);
    bool imag_allclose = allclose(output_toep.imag(), output_nufft.imag(), rtol, atol);

    if (!real_allclose || !imag_allclose) {
        std::cout << "Test: Toeplitz Multiplication (2D): FAILED\n";
    } else {
        std::cout << "Test: Toeplitz Multiplication (2D): PASSED\n";
    }
}

void test_toeplitz_multiplication_1D(hasty::ArrayRef<hasty::i64> im_size, double rtol = 1e-5, double atol = 1e-3)
{
    using namespace hasty;
    using namespace hasty::fft;

    Device cuda0(eDeviceType::CUDA, 0);

    Tensor input = rand({1, im_size[0]}, TensorOptions(cuda0, eScalarType::ComplexFloat));

    i64 NX = input.size(1);

    bool cartesian_coords = false;
    Tensor coords;

    if (cartesian_coords) {
        coords = create_cartesian_coords<1>({NX}, cuda0);
    } else {
        coords = rand({1, 10000}, TensorOptions(cuda0, eScalarType::Float));
        coords.mul_(Scalar{2 * 3.141592f});
        coords.add_(Scalar{-3.141592f});
    }

    i64 npts = coords.size(1);

    Tensor weights = ones({npts}, TensorOptions(cuda0, eScalarType::ComplexFloat));

    Tensor kernel = create_toeplitz_kernel_standard(coords, weights, {NX});

    Tensor output_toep = zeros({1, NX}, TensorOptions(cuda0, eScalarType::ComplexFloat));
    Tensor output_nufft = zeros_like(output_toep);

    // --- NUFFT reference ---
    {
        Tensor intermediate_output = zeros({1, npts}, TensorOptions(cuda0, eScalarType::ComplexFloat));

        {
            NufftOptions<cuda_t, f32, UTN> opts;
            NufftPlan<cuda_t, f32, 1, UTN> plan({NX}, opts);
            plan.setpts(coords);
            plan.execute(input, intermediate_output);
        }

        intermediate_output.mul_(Scalar{1.0f / static_cast<float>(NX)});

        {
            NufftOptions<cuda_t, f32, NTU> opts;
            NufftPlan<cuda_t, f32, 1, NTU> plan({NX}, opts);
            plan.setpts(coords);
            plan.execute(intermediate_output, output_nufft);
        }
    }

    // --- Toeplitz ---
    toeplitz_multiplication(
        input, output_toep, kernel,
        std::nullopt, std::nullopt, std::nullopt,
        ToeplitzMultType::NONE,
        ToeplitzMultType::NONE,
        ToeplitzMultType::NONE,
        ToeplitzMultType::NONE,
        ToeplitzMultType::NONE,
        ToeplitzAccumulateType::NONE
    );

    output_toep = output_toep.view({NX}).cpu();
    output_nufft = output_nufft.view({NX}).cpu();

    

}

void test_toeplitz_multiplication()
{
    test_toeplitz_multiplication_1D({128}, 1e-5);
    test_toeplitz_multiplication_2D({128, 128}, 1e-5);
    test_toeplitz_multiplication_3D({128, 128, 128}, 1e-5);

    test_toeplitz_multiplication_1D({303}, 1e-5);
    test_toeplitz_multiplication_2D({303, 384}, 1e-5);
    test_toeplitz_multiplication_3D({64, 128, 303}, 1e-5);
}

void test_toeplitz_multiplication_2D_visual()
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

        //input = rand({384, 303}, TensorOptions(eScalarType::Float));

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
        coords = create_cartesian_coords<2>({NY, NX}, cuda0);
    } else {
        coords = rand({2, 10000}, TensorOptions(cuda0, eScalarType::Float));
        coords.mul_(Scalar{2*3.141592f});
        coords.add_(Scalar{-3.141592f});
    }
    
    
    i64 npts = coords.size(1);
    
    Tensor weights = ones({npts}, TensorOptions(cuda0, eScalarType::ComplexFloat));


    //Tensor kernel = create_toeplitz_kernel(coords, weights, {NY, NX}, true);
    Tensor kernel = create_toeplitz_kernel(coords, weights, {NY, NX}, true);

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
        auto temp_input = input.to(eScalarType::ComplexDouble).contiguous();
        auto coords_temp = coords.to(eScalarType::Double).contiguous();
        Tensor intermediate_output = zeros({1, npts}, TensorOptions(cuda0, eScalarType::ComplexDouble));
        {
            NufftOptions<cuda_t, f64, UTN> opts;
            NufftPlan<cuda_t, f64, 2, UTN> plan({NY, NX}, opts);
            plan.setpts(coords_temp);
            plan.execute(temp_input, intermediate_output);
        }
        intermediate_output.mul_(Scalar{1.0f / static_cast<double>(NY * NX)});  // scale for unnormalized FFT
        auto output_temp = zeros({1, NY, NX}, TensorOptions(cuda0, eScalarType::ComplexDouble));
        {
            NufftOptions<cuda_t, f64, NTU> opts;
            NufftPlan<cuda_t, f64, 2, NTU> plan({NY, NX}, opts);
            plan.setpts(coords_temp);
            plan.execute(intermediate_output, output_temp);
        }
        output_nufft = output_temp.to(eScalarType::ComplexFloat);
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

    {
        auto diff = output_toep.sub(output_nufft).view({NY, NX});
        auto diff_real = diff.real().contiguous();
        auto diff_imag = diff.imag().contiguous();
    
        auto diff_real_rel = diff_real.abs().div_(output_nufft.view({NY, NX}).real().abs().add(1e-8)).cpu();
        auto diff_imag_rel = diff_imag.abs().div_(output_nufft.view({NY, NX}).imag().abs().add(1e-8)).cpu();

        diff_real = diff_real.cpu();
        diff_imag = diff_imag.cpu();

        viz::default_heatmap(viz::DefaultHeatmapOptions<2, 2>{
            .z = Arr{
                    Arr{diff_real.spanning_view(), diff_imag.spanning_view()},
                    Arr{diff_real_rel.spanning_view(), diff_imag_rel.spanning_view()}
                },
            .titles = Arr{
                Arr<std::string,2>{"Difference Real Part", "Difference Imaginary Part"},
                Arr<std::string,2>{"Relative Difference Real Part", "Relative Difference Imaginary Part"}
            }
        }).show();

    }

    output_toep = output_toep.view({NY, NX});
    output_nufft = output_nufft.view({NY, NX});

    auto output_toep_real = output_toep.real().cpu().contiguous();
    auto output_toep_imag = output_toep.imag().cpu().contiguous();

    auto output_nufft_real = output_nufft.real().cpu().contiguous();
    auto output_nufft_imag = output_nufft.imag().cpu().contiguous();

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

    /*
    {
        auto straight_output = zeros({2*NY, 2*NX}, TensorOptions(cuda0, eScalarType::ComplexFloat));
        straight_output[Slice(0, NY), Slice(0, NX)] = input.squeeze(0);
        straight_output = fftn(straight_output);
        straight_output.mul_(Scalar{1.0f / static_cast<float>(4 * NY * NX)});  // scale for unnormalized FFT
        straight_output.mul_(kernel);
        straight_output = ifftn(straight_output);
        straight_output = straight_output[Slice(0, NY), Slice(0, NX)];

        auto straight_diff = straight_output.sub(output_nufft);
        auto straight_diff_real = straight_diff.real().cpu().contiguous();
        auto straight_diff_imag = straight_diff.imag().cpu().contiguous();

        viz::default_heatmap(viz::DefaultHeatmapOptions<1, 2>{
            .z = {{straight_diff_real.spanning_view(), straight_diff_imag.spanning_view()}},
            .titles = {{"Straight Difference Real Part", "Straight Difference Imaginary Part"}}
        }).show();

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
    */


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


    //test_toeplitz_multiplication();
    test_toeplitz_multiplication_2D_visual();
    //hasty::viz::test_tensor_viz();

    return 0;
}
