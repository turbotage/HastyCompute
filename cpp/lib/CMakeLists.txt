cmake_minimum_required(VERSION 4.3.0 FATAL_ERROR)

#set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b" CACHE STRING "" FORCE)

project(HastyCompute VERSION 0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)

#message("Hello ${pybind11_DIR}")
# Add the unsupported compiler flag explicitly
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)

add_library(HastyCompute SHARED)

set(CMAKE_COLOR_DIAGNOSTICS ON)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	add_compile_options(
		-ftemplate-backtrace-limit=5
		-fdiagnostics-show-template-tree
		-fcolor-diagnostics
		-fdiagnostics-format=vi
	)
endif()

target_compile_features(HastyCompute 
	PRIVATE cxx_std_23
	INTERFACE cxx_std_23)

set_target_properties(HastyCompute PROPERTIES
	CXX_STANDARD 23
	CXX_STANDARD_REQUIRED ON
	CXX_SCAN_FOR_MODULES ON
	CXX_MODULE_STD ON
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	target_compile_options(HastyCompute PRIVATE -ftime-trace)
endif()

#target_link_libraries(HastyCuCompute PRIVATE c++ c++abi)

include(cmake/CPM.cmake)
include(FetchContent)

# LIBRARY CPP SOURCES
target_sources(HastyCompute
	PRIVATE
		"src/wrapper/vkfft.cpp"
		"src/fft/toeplitz_creation.cpp"
		"src/fft/toeplitz_mult.cpp"
		"src/fft/toeplitz_transform.cpp"
)

# LIBRARY CPP MODULE SOURCES
target_sources(HastyCompute
    PUBLIC
        FILE_SET cxx_modules_public TYPE CXX_MODULES FILES
			"src/fft/fft_mod.cppm"
			"src/fft/nufft.cppm"
			"src/fft/toeplitz.cppm"
			"src/generic_value/generic_value_mod.cppm"
			"src/nvrtc/nvrtc_mod.cppm"
			"src/tensor/tensor_background_impl.cppm"
			"src/tensor/tensor_background.cppm"
			"src/tensor/tensor_concepts.cppm"
			"src/tensor/tensor_external.cppm"
			"src/tensor/tensor_mod.cppm"
			"src/tensor/tensor_scalar.cppm"
            "src/tensor/tensor_tensor.cppm"
			"src/threading/thread_stream.cppm"
			"src/threading/storage_threadpool.cppm"
			"src/threading/threading_mod.cppm"
			"src/util/util_alias.cppm"
			"src/util/util_containers.cppm"
            "src/util/util_meta.cppm"
            "src/util/util_mod.cppm"
            "src/util/util_span.cppm"
			"src/util/util_str.cppm"
            "src/util/util_typing.cppm"
			"src/viz/viz_mod.cppm"
			"src/wrapper/torch_wrapper_mod.cppm"
			"src/wrapper/vkfft_mod.cppm"
)

# LIB INTERFACE CPP SOURCES
#target_sources(HastyCuCompute
#	PRIVATE
#		""
#)

# LIB INTERFACE CPP MODULE SOURCES
#target_sources(HastyCompute
#    PUBLIC
#        FILE_SET cxx_modules_private TYPE CXX_MODULES FILES
#			"public/threading/thread_stream.cppm"
#			"public/threading/threading.cppm"
#			""
#            "interface/public/interface.cppm"
#            "interface/viz.cppm"
#)

set(DB_PRODUCTION_MODE ON)
FetchContent_Declare(
  battery-embed
  GIT_REPOSITORY https://github.com/batterycenter/embed.git
  GIT_TAG        v1.2.19
)
FetchContent_MakeAvailable(battery-embed)

b_embed(HastyCompute "src/fft/kernels/toeplitz_load_1D.cu")
b_embed(HastyCompute "src/fft/kernels/toeplitz_load_2D.cu")
b_embed(HastyCompute "src/fft/kernels/toeplitz_load_3D.cu")

get_target_property(_incs HastyCompute INTERFACE_INCLUDE_DIRECTORIES)
list(REMOVE_ITEM _incs "${EMBED_BINARY_DIR}/autogen/hastycompute/include")
set_target_properties(HastyCompute PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_incs}")

find_package(CUDAToolkit REQUIRED)


target_include_directories(HastyCompute
	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/vkFFT>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_definitions(HastyCompute PRIVATE VKFFT_BACKEND=1)

target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:CUDA::toolkit>)
target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:CUDA::nvrtc>)
target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:CUDA::cudart>)
target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:CUDA::cufft>)
target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:CUDA::cuda_driver>)

get_target_property(NVRTC_PATH CUDA::nvrtc LOCATION)
get_target_property(CUDART_PATH CUDA::cudart LOCATION)

get_filename_component(NVRTC_FILENAME "${NVRTC_PATH}" NAME)
get_filename_component(CUDART_FILENAME "${CUDART_PATH}" NAME)

add_custom_command(TARGET HastyCompute POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${NVRTC_PATH}" "$<TARGET_FILE_DIR:HastyCompute>/${NVRTC_FILENAME}"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${CUDART_PATH}" "$<TARGET_FILE_DIR:HastyCompute>/${CUDART_FILENAME}"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		$<TARGET_FILE:finufft> "$<TARGET_FILE_DIR:HastyCompute>/"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		$<TARGET_FILE:cufinufft> "$<TARGET_FILE_DIR:HastyCompute>/"
)

# this gives access to cufinufft
set(FINUFFT_USE_CUDA ON CACHE BOOL "" FORCE)
# this disables finufft leaving only cufinufft available
set(FINUFFT_USE_CPU ON CACHE BOOL "" FORCE)
set(FINUFFT_USE_DUCC0 ON CACHE BOOL "" FORCE)
set(FINUFFT_STATIC_LINKING OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  finufft
  GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git
  GIT_TAG        v2.5.0 #master #v2.5.0
)

FetchContent_MakeAvailable(finufft)

# finufft/cufinufft are shared libraries that carry OpenMP as a PUBLIC
# dependency (both INTERFACE_COMPILE_OPTIONS and via OpenMP::OpenMP_CXX
# in INTERFACE_LINK_LIBRARIES).  That causes -fopenmp to bleed into
# HastyCompute's own C++ module compilations, which breaks the `std`
# module BMI because it was compiled without that flag.
#
# HastyCompute wraps finufft completely — no consumer of HastyCompute
# ever calls finufft headers directly, so they need neither the compile
# flag nor the link-time OpenMP target.  The .so files handle OpenMP
# entirely at runtime through their own ELF NEEDED entries.
foreach(_tgt finufft cufinufft)
    if(TARGET ${_tgt})
        set_property(TARGET ${_tgt} PROPERTY INTERFACE_COMPILE_OPTIONS "")
        get_target_property(_iface_libs ${_tgt} INTERFACE_LINK_LIBRARIES)
        if(_iface_libs)
            list(FILTER _iface_libs EXCLUDE REGEX "OpenMP|[Oo]penmp|omp")
            set_property(TARGET ${_tgt} PROPERTY INTERFACE_LINK_LIBRARIES "${_iface_libs}")
        endif()
        set_target_properties(${_tgt} PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
endforeach()

target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:finufft>)
target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:cufinufft>)

set_target_properties(cufinufft PROPERTIES EXPORT_EXCLUDE TRUE)
set_target_properties(finufft PROPERTIES EXPORT_EXCLUDE TRUE)


FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        12.0.0
) # 10.2.1
FetchContent_MakeAvailable(fmt)

target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:fmt::fmt>)

set_target_properties(fmt PROPERTIES EXPORT_EXCLUDE TRUE)



FetchContent_Declare(
	plotlypp
	GIT_REPOSITORY https://github.com/jimmyorourke/plotlypp.git
	GIT_TAG main
)
FetchContent_MakeAvailable(plotlypp)

target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:plotlypp::plotlypp>)





find_package(Torch REQUIRED)
target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:${TORCH_LIBRARIES}>)

find_package(HighFive CONFIG REQUIRED)
target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:HighFive>)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET HastyCompute
					 POST_BUILD
					 COMMAND ${CMAKE_COMMAND} -E copy_if_different
					 ${TORCH_DLLS}
					 $<TARGET_FILE_DIR:HastyCompute>)
endif (MSVC)


find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:Python3::Python>)
target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:pybind11::module>)

set_target_properties(HastyCompute PROPERTIES
		 INTERPROCEDURAL_OPTIMIZATION ON
	 )

target_compile_definitions(HastyCompute PUBLIC _GLIBCXX_USE_CXX11_ABI=1)



set_target_properties(HastyCompute PROPERTIES INSTALL_RPATH "$ORIGIN")


#install(FILES cpp/corelib/interface/include/interface.hpp DESTINATION include/HastyComputes)

set(HastyCompute_INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/interface/include>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/HastyCompute>
)

# Don't add external dependencies to PUBLIC includes - keep them PRIVATE
target_include_directories(HastyCompute 
    PUBLIC
        ${HastyCompute_INCLUDE_DIRS}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated_includes>
        ${pybind11_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)

set(MODULE_CACHE_RELATIVE_PATH "module_cache")
set(TENSOR_CACHE_RELATIVE_PATH "tensor_cache")
set(DATA_RELATIVE_PATH "data")
configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/include/configure_file_settings.hpp.in"
	"${CMAKE_CURRENT_BINARY_DIR}/generated_includes/configure_file_settings.hpp"
)

# Apply to target
target_include_directories(HastyCompute 
	PUBLIC
    	${HastyCompute_INCLUDE_DIRS}
	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated_includes>
)



################ INSTALL ####################
include(GNUInstallDirs)

message(STATUS "HASTY_DATA_FOLDER: ${HASTY_DATA_FOLDER}")
if (NOT DEFINED HASTY_DATA_FOLDER OR HASTY_DATA_FOLDER STREQUAL "")
	message(STATUS "HASTY_DATA_FOLDER not set, using default path.")
	# This default path assumes both HastyCompute and HastyData are cloned in the same parent directory
	set(HASTY_DATA_FOLDER ${CMAKE_SOURCE_DIR}"../../../HastyData")
endif()

# Copy models folder to build directory (for Debug/Release builds)
add_custom_command(TARGET HastyCompute POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		${HASTY_DATA_FOLDER}/models
		${CMAKE_BINARY_DIR}/data/models
	COMMENT "Copying wavelet models to build directory"
)

install(TARGETS HastyCompute EXPORT HastyComputeTargets
    RUNTIME_DEPENDENCIES
        # System libs the target machine must always provide
        PRE_EXCLUDE_REGEXES
            ".*libstdc\\+\\+.*" ".*libgcc.*"
            ".*libc\\.so.*" ".*libc\\+\\+.*" ".*ld-linux.*"
            ".*libdl\\.so.*" ".*libm\\.so.*" ".*libpthread\\.so.*" ".*librt\\.so.*"
            ".*libcuda\\.so.*"    # GPU driver — never bundle this
        POST_EXCLUDE_REGEXES
            ".*libstdc\\+\\+.*" ".*libgcc.*"
            ".*libc\\.so.*" ".*libc\\+\\+.*" ".*ld-linux.*"
            ".*libdl\\.so.*" ".*libm\\.so.*" ".*libpthread\\.so.*" ".*librt\\.so.*"
            ".*libcuda\\.so.*"
        # Tell CMake where to find libs that live outside standard system paths
        DIRECTORIES
            "$<TARGET_FILE_DIR:finufft>"
            "$<TARGET_FILE_DIR:cufinufft>"
            "${TORCH_INSTALL_PREFIX}/lib"
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
    FILE_SET HEADERS DESTINATION include/HastyCompute
    FILE_SET cxx_modules_public DESTINATION include/HastyCompute/modules
)

# Install the export set
install(EXPORT HastyComputeTargets
	FILE HastyComputeTargets.cmake
	NAMESPACE HastyCompute::
	DESTINATION lib/cmake/HastyCompute
)

install(FILES "${NVRTC_PATH}" "${CUDART_PATH}" DESTINATION lib)

# Install models folder at top level
install(DIRECTORY ${HASTY_DATA_FOLDER}/models
	DESTINATION .
	FILES_MATCHING PATTERN "*.pt"
	PATTERN "*.gitkeep" EXCLUDE
)

# Install the include files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
	DESTINATION include/HastyCompute
)

install(FILES $<TARGET_FILE:finufft> $<TARGET_FILE:cufinufft> DESTINATION lib)

# Install dependency headers needed by installed module files
install(DIRECTORY ${finufft_SOURCE_DIR}/include/ DESTINATION include)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/vkFFT/
	DESTINATION include/HastyCompute
	FILES_MATCHING PATTERN "*.h"
)

# Install the CMake package configuration file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/HastyComputeConfigVersion.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY AnyNewerVersion
)

configure_file(HastyComputeConfig.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/HastyComputeConfig.cmake"
	@ONLY
)

install(FILES
	"${CMAKE_CURRENT_BINARY_DIR}/HastyComputeConfig.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/HastyComputeConfigVersion.cmake"
	DESTINATION lib/cmake/HastyCompute
)

install(CODE "file(REMOVE_RECURSE \"${CMAKE_INSTALL_PREFIX}/share/finufft/examples\")")