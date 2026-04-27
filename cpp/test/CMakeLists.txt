cmake_minimum_required(VERSION 4.2.0 FATAL_ERROR)

project(HastyTest VERSION 0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(HastyCompute REQUIRED)

# <================== HastyTest ==================>
add_executable(HastyTest "src/main.cpp")
target_compile_features(HastyTest PRIVATE cxx_std_23 INTERFACE cxx_std_23)
set_target_properties(HastyTest PROPERTIES
	CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)
target_link_libraries(HastyTest PRIVATE HastyCompute::HastyCompute)
install(TARGETS HastyTest RUNTIME DESTINATION bin)

# <================== Hasty Data Test ==================>
add_executable(HastyDataTest "src/hasty_data_test.cpp")
target_compile_features(HastyDataTest PRIVATE cxx_std_23 INTERFACE cxx_std_23)
set_target_properties(HastyDataTest PROPERTIES
	CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)
target_link_libraries(HastyDataTest PRIVATE HastyCompute::HastyCompute)
install(TARGETS HastyDataTest RUNTIME DESTINATION bin)

# <================== Linalg (SVD) Test ==================>
add_executable(HastyLinalgTest "src/linalg_test.cpp")
target_compile_features(HastyLinalgTest PRIVATE cxx_std_23 INTERFACE cxx_std_23)
set_target_properties(HastyLinalgTest PROPERTIES
	CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)
target_link_libraries(HastyLinalgTest PRIVATE HastyCompute::HastyCompute)
install(TARGETS HastyLinalgTest RUNTIME DESTINATION bin)

# <================== VkFFT Test ==================>
add_executable(HastyVkFFTTest "src/vkfft_test.cpp")
target_compile_features(HastyVkFFTTest PRIVATE cxx_std_23 INTERFACE cxx_std_23)
set_target_properties(HastyVkFFTTest PROPERTIES
	CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)
find_package(CUDAToolkit REQUIRED)
target_include_directories(HastyVkFFTTest PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/../lib/external/vkFFT"
	"${CUDAToolkit_INCLUDE_DIRS}"
)
target_link_libraries(HastyVkFFTTest PRIVATE
	CUDA::cudart CUDA::cuda_driver CUDA::nvrtc CUDA::cufft)
install(TARGETS HastyVkFFTTest RUNTIME DESTINATION bin)
