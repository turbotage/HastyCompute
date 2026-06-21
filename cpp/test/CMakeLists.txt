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

# <================== Hasty Server Test =================>
add_executable(HastyServerTest "src/hasty_server.cpp")
target_compile_features(HastyServerTest PRIVATE cxx_std_23 INTERFACE cxx_std_23)
set_target_properties(HastyServerTest PROPERTIES
	CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)
target_link_libraries(HastyServerTest PRIVATE HastyCompute::HastyCompute)
add_custom_command(TARGET HastyServerTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/../../certs/cert.pem"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../certs/key.pem"
        "$<TARGET_FILE_DIR:HastyServerTest>"
    COMMENT "Copying TLS cert and key to binary directory"
)
install(TARGETS HastyServerTest RUNTIME DESTINATION bin)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../certs/cert.pem"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../certs/key.pem"
    DESTINATION bin
)

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

# <================== Butterfly Factorization Test ==================>
add_executable(HastyButterflyTest "src/butterfly_test.cpp")
target_compile_features(HastyButterflyTest PRIVATE cxx_std_23 INTERFACE cxx_std_23)
set_target_properties(HastyButterflyTest PROPERTIES
	CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)
target_link_libraries(HastyButterflyTest PRIVATE HastyCompute::HastyCompute)
install(TARGETS HastyButterflyTest RUNTIME DESTINATION bin)

# <================== Non-Fourier Interpolator Test ==================>
add_executable(HastyNonFourierInterpTest "src/non_fourier_interp_test.cpp")
target_compile_features(HastyNonFourierInterpTest PRIVATE cxx_std_23 INTERFACE cxx_std_23)
set_target_properties(HastyNonFourierInterpTest PROPERTIES
	CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_SCAN_FOR_MODULES ON CXX_MODULE_STD ON)
target_link_libraries(HastyNonFourierInterpTest PRIVATE HastyCompute::HastyCompute)
install(TARGETS HastyNonFourierInterpTest RUNTIME DESTINATION bin)

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

# Generate LD_LIBRARY_PATH wrapper scripts for all test executables.
# dlopen from within system shared libs (e.g. cuDNN) does not inherit the
# binary's RUNPATH, so install/lib must be set via LD_LIBRARY_PATH.
foreach(_test HastyNonFourierInterpTest HastyVkFFTTest)
    set(_wrapper "${CMAKE_CURRENT_BINARY_DIR}/${_test}.sh")
    file(GENERATE OUTPUT "${_wrapper}" CONTENT
"#!/usr/bin/env bash
SCRIPT_DIR=\"\$(cd \"\$(dirname \"\${BASH_SOURCE[0]}\")\" && pwd)\"
INSTALL_LIB=\"\${SCRIPT_DIR}/../install/lib\"
exec env LD_LIBRARY_PATH=\"\${INSTALL_LIB}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\" \
    \"\${SCRIPT_DIR}/${_test}\" \"\$@\"
"
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
    )
endforeach()
