cmake_minimum_required(VERSION 4.3.0 FATAL_ERROR)

project(HastyCompute VERSION 0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)

# NVCC with GCC host may exceed NVCC's tested host compiler version range
add_compile_options($<$<COMPILE_LANG_AND_ID:CUDA,NVCC>:-allow-unsupported-compiler>)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-ftemplate-backtrace-limit=5>
        $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-show-template-tree>
        $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>
        $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-format=vi>
    )
endif()

# CMAKE_EXE_LINKER_FLAGS from CommonPresets (e.g. -fuse-ld=lld-22) is Clang-only
# syntax that GCC-15 (NVCC host compiler) rejects during CUDA ABI detection.
# Clear the global flags here; lld is applied per-target below using CMAKE_LINKER.
set(CMAKE_EXE_LINKER_FLAGS    "" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "" CACHE STRING "" FORCE)

# ── Primary target ────────────────────────────────────────────────────────────
add_library(HastyCompute SHARED)

target_compile_features(HastyCompute PRIVATE cxx_std_23 INTERFACE cxx_std_23)

set_target_properties(HastyCompute PROPERTIES
    CXX_STANDARD          23
    CXX_STANDARD_REQUIRED ON
    CXX_SCAN_FOR_MODULES  ON
    CXX_MODULE_STD        ON
    INTERPROCEDURAL_OPTIMIZATION ON
    INSTALL_RPATH         "$ORIGIN"
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(HastyCompute PRIVATE -ftime-trace)
endif()

# Apply lld to HastyCompute C++ link steps only, derived from CMAKE_LINKER (preset).
# LINK_LANGUAGE:CXX guard prevents the flag from reaching GCC-15 CUDA host links.
if(CMAKE_LINKER)
    get_filename_component(_linker_name "${CMAKE_LINKER}" NAME)
    target_link_options(HastyCompute PRIVATE
        $<$<LINK_LANGUAGE:CXX>:-fuse-ld=${_linker_name}>
    )
endif()

target_compile_definitions(HastyCompute
    PRIVATE  VKFFT_BACKEND=1
    PUBLIC   _GLIBCXX_USE_CXX11_ABI=1
)

target_include_directories(HastyCompute PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/vkFFT>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# ── Sources ───────────────────────────────────────────────────────────────────
target_sources(HastyCompute
    PRIVATE
        "src/fft/toeplitz_creation.cpp"
        "src/fft/toeplitz_mult.cpp"
        "src/fft/toeplitz_transform.cpp"
        "src/io/io_hdf5_impl.cpp"
        "src/linalg/linalg_impl.cpp"
        "src/server/cmd_registry_impl.cpp"
        "src/server/generic_value_bank_impl.cpp"
        "src/server/grpc_server_impl.cpp"
        "src/server/server_mod_impl.cpp"
        "src/viz/viz.cpp"
        "src/wrapper/vkfft.cpp"
)

target_sources(HastyCompute
    PUBLIC FILE_SET cxx_modules_public TYPE CXX_MODULES FILES
        "src/fft/dft.cppm"
        "src/fft/fft_mod.cppm"
        "src/fft/nufft.cppm"
        "src/fft/toeplitz.cppm"
        "src/generic_value/generic_value_mod.cppm"
        "src/generic_value/generic_value_slice.cppm"
        "src/io/io_blosc.cppm"
        "src/io/io_hdf5.cppm"
        "src/io/io_mod.cppm"
        "src/io/io_nifti.cppm"
        "src/linalg/linalg_mod.cppm"
        "src/mri/forward.cppm"
        "src/mri/mri_mod.cppm"
        "src/mri/off_fourier_interpolators.cppm"
        "src/nvrtc/nvrtc_mod.cppm"
        "src/python/nifti_and_registration.cppm"
        "src/python/python_mod.cppm"
        "src/server/cmd_registry.cppm"
        "src/server/generic_value_bank.cppm"
        "src/server/grpc_server.cppm"
        "src/server/http_server.cppm"
        "src/server/server_mod.cppm"
        "src/tensor/tensor_background_impl.cppm"
        "src/tensor/tensor_background.cppm"
        "src/tensor/tensor_comprep.cppm"
        "src/tensor/tensor_concepts.cppm"
        "src/tensor/tensor_external_linalg.cppm"
        "src/tensor/tensor_external_math.cppm"
        "src/tensor/tensor_external.cppm"
        "src/tensor/tensor_json.cppm"
        "src/tensor/tensor_masking.cppm"
        "src/tensor/tensor_mod.cppm"
        "src/tensor/tensor_scalar.cppm"
        "src/tensor/tensor_tensor.cppm"
        "src/threading/storage_threadpool.cppm"
        "src/threading/thread_stream.cppm"
        "src/threading/threading_mod.cppm"
        "src/util/util_alias.cppm"
        "src/util/util_containers.cppm"
        "src/util/util_idx.cppm"
        "src/util/util_meta.cppm"
        "src/util/util_mod.cppm"
        "src/util/util_span.cppm"
        "src/util/util_str.cppm"
        "src/util/util_stream.cppm"
        "src/util/util_typing.cppm"
        "src/viz/viz_mod.cppm"
        "src/wrapper/torch_wrapper_mod.cppm"
        "src/wrapper/vkfft_mod.cppm"
)

# ── External dependencies ─────────────────────────────────────────────────────
# FetchContent files (lowercase names) configure at cmake-configure time.
# ExternalXxx files are true ExternalProjects that build before the lib itself.
# Order matters: ExternalGRPC before ExternalLibTorch (LibTorch reuses gRPC's
# protobuf), and finufft before ExternalLibTorch (finufft enables CUDA language).
include(FetchContent)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
cmake_host_system_information(RESULT _CPU_THREADS QUERY NUMBER_OF_LOGICAL_CORES)

include(cmake/battery_embed.cmake)
include(cmake/cuda.cmake)
include(cmake/finufft.cmake)       # enables CUDA language via FetchContent
include(cmake/fmt.cmake)
include(cmake/plotlypp.cmake)
include(cmake/json.cmake)
include(cmake/blosc2.cmake)
include(cmake/httplib.cmake)
include(cmake/ExternalGRPC.cmake)  # builds gRPC + installs protobuf; sets GRPC_INSTALL_DIR
include(cmake/ExternalLibTorch.cmake)  # reuses GRPC_PROTOBUF_DIR; depends on grpc_external
include(cmake/hdf5.cmake)
include(cmake/nifti.cmake)
include(cmake/ExternalPETSc.cmake)
include(cmake/venv.cmake)

# ── Public include dirs ───────────────────────────────────────────────────────
target_include_directories(HastyCompute
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/interface/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated_includes>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/HastyCompute>
    PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)

# ── Install ───────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT DEFINED HASTY_DATA_FOLDER OR HASTY_DATA_FOLDER STREQUAL "")
    set(HASTY_DATA_FOLDER "${CMAKE_SOURCE_DIR}/../../../HastyData")
endif()

set(HASTY_SCRIPTS_DIR "${CMAKE_SOURCE_DIR}/../../python/scripts")

# Debug builds bake absolute build-tree paths; release bakes install-relative paths.
# C++ resolves relative paths at runtime via exe_dir() (see python_mod.cppm).
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR NOT CMAKE_BUILD_TYPE)
    set(HASTY_VENV_PYTHON "${CMAKE_BINARY_DIR}/.venv/bin/python")
    # HASTY_SCRIPTS_DIR already points to source tree — correct for debug
else()
    set(HASTY_VENV_PYTHON "../.venv/bin/python")
    set(HASTY_SCRIPTS_DIR "../python/scripts")
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/configure_file_settings.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated_includes/configure_file_settings.hpp"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated_includes/configure_file_settings.hpp"
    DESTINATION include/HastyCompute
)

add_custom_command(TARGET HastyCompute POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${HASTY_DATA_FOLDER}/models ${CMAKE_BINARY_DIR}/data/models
    COMMENT "Copying wavelet models to build directory"
)

# Locate MKL runtime library directory (needed by LibTorch at runtime).
# Searches: MKLROOT env var, Intel oneAPI standard path, conda prefix.
find_path(_mkl_lib_dir
    NAMES libmkl_core.so
    HINTS
        "$ENV{MKLROOT}/lib"
        "/opt/intel/oneapi/mkl/latest/lib"
        "/opt/intel/oneapi/mkl/2025.3/lib"
        "/opt/intel/oneapi/mkl/2025.2/lib"
        "/opt/intel/oneapi/mkl/2025.1/lib"
        "/opt/intel/oneapi/mkl/2024.2/lib"
        "$ENV{CONDA_PREFIX}/lib"
        "/usr/lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
)
if(_mkl_lib_dir)
    message(STATUS "MKL lib dir: ${_mkl_lib_dir}")
else()
    message(WARNING "MKL lib dir not found — libmkl_*.so files may be absent from install")
endif()

install(TARGETS HastyCompute EXPORT HastyComputeTargets
    RUNTIME_DEPENDENCIES
        PRE_EXCLUDE_REGEXES
            ".*libstdc\\+\\+.*" ".*libgcc.*" ".*libc\\.so.*" ".*libc\\+\\+.*"
            ".*ld-linux.*" ".*libdl\\.so.*" ".*libm\\.so.*"
            ".*libpthread\\.so.*" ".*librt\\.so.*" ".*libcuda\\.so.*"
        POST_EXCLUDE_REGEXES
            ".*libstdc\\+\\+.*" ".*libgcc.*" ".*libc\\.so.*" ".*libc\\+\\+.*"
            ".*ld-linux.*" ".*libdl\\.so.*" ".*libm\\.so.*"
            ".*libpthread\\.so.*" ".*librt\\.so.*" ".*libcuda\\.so.*"
        DIRECTORIES
            "$<TARGET_FILE_DIR:finufft>"
            "$<TARGET_FILE_DIR:cufinufft>"
            "${TORCH_INSTALL_PREFIX}/lib"
            "${GRPC_INSTALL_DIR}/lib"
            $<$<BOOL:${_mkl_lib_dir}>:${_mkl_lib_dir}>
    LIBRARY  DESTINATION lib
    ARCHIVE  DESTINATION lib
    RUNTIME  DESTINATION bin
    INCLUDES DESTINATION include
    FILE_SET HEADERS             DESTINATION include/HastyCompute
    FILE_SET cxx_modules_public  DESTINATION include/HastyCompute/modules
)

install(EXPORT HastyComputeTargets
    FILE      HastyComputeTargets.cmake
    NAMESPACE HastyCompute::
    DESTINATION lib/cmake/HastyCompute
)

install(FILES "${NVRTC_PATH}" "${CUDART_PATH}" DESTINATION lib)

# MKL VML dispatch libs are dlopen'd at runtime — not in the link graph,
# so RUNTIME_DEPENDENCIES misses them. Copy all libmkl_vml_* explicitly.
if(_mkl_lib_dir)
    install(CODE "
        file(GLOB _mkl_runtime_libs LIST_DIRECTORIES false
            \"${_mkl_lib_dir}/libmkl_*.so*\")
        foreach(_lib IN LISTS _mkl_runtime_libs)
            file(INSTALL \"\${_lib}\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/lib\"
                FOLLOW_SYMLINK_CHAIN)
        endforeach()
    ")
endif()

# Patch libtorch_cuda.so in the DESTINATION after every install.
# RUNTIME_DEPENDENCIES copies libtorch_cuda.so from the ExternalProject install
# dir to CMAKE_INSTALL_PREFIX/lib on every cmake --install run.  The source is
# patched by ExternalProject's INSTALL_COMMAND (runs once), but the destination
# must be re-patched here so the deployed copy always has sr3std mangling.
install(CODE "
    set(_so \"\${CMAKE_INSTALL_PREFIX}/lib/libtorch_cuda.so\")
    if(EXISTS \"\${_so}\")
        execute_process(
            COMMAND ${Python3_EXECUTABLE}
                \"${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_libtorch_dynsym.py\"
                \"\${_so}\"
            RESULT_VARIABLE _r
        )
        if(_r)
            message(WARNING \"libtorch_cuda.so ABI patch failed (exit \${_r})\")
        endif()
    endif()
")

install(DIRECTORY ${HASTY_DATA_FOLDER}/models
    DESTINATION .
    FILES_MATCHING PATTERN "*.pt"
    PATTERN "*.gitkeep" EXCLUDE
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include/HastyCompute)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/vkFFT/
    DESTINATION include/HastyCompute FILES_MATCHING PATTERN "*.h"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HastyComputeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(HastyComputeConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/HastyComputeConfig.cmake"
    INSTALL_DESTINATION lib/cmake/HastyCompute
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/HastyComputeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/HastyComputeConfigVersion.cmake"
    DESTINATION lib/cmake/HastyCompute
)

install(CODE "file(REMOVE_RECURSE \"${CMAKE_INSTALL_PREFIX}/share/finufft/examples\")")

# CUDA library compatibility symlinks.
# PyTorch built against CUDA 13 runtime but lazy-loads libcublasLt.so.12 by name.
# Create .so and .so.12 symlinks pointing to the actual .so.13 installed file.
install(CODE "
    file(GLOB _cublaslt \"\${CMAKE_INSTALL_PREFIX}/lib/libcublasLt.so.*.*.*\")
    list(GET _cublaslt 0 _clt)
    if(_clt)
        get_filename_component(_clt_name \"\${_clt}\" NAME)
        foreach(_link libcublasLt.so libcublasLt.so.12)
            execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
                \"\${_clt_name}\"
                \"\${CMAKE_INSTALL_PREFIX}/lib/\${_link}\")
        endforeach()
    endif()
")

install(DIRECTORY "${CMAKE_SOURCE_DIR}/../../python/scripts/"
    DESTINATION "python/scripts"
    FILES_MATCHING PATTERN "*.py"
)
