# cpp/lib/cmake/ExternalLibTorch.cmake
#
# Builds PyTorch C++ libraries from source as an ExternalProject.
# Included by cpp/lib/CMakeLists.txt, which means all cmake variables
# (CMAKE_CXX_COMPILER, CMAKE_CUDA_COMPILER, CMAKE_LINKER, etc.) come directly
# from the lib preset via CommonPresets.json — no hardcoded paths here.
#
# Compiler split (critical for ABI correctness):
#   CMAKE_CXX_COMPILER  = Clang  (same as HastyCompute — avoids pure-GCC runtime issues)
#   CMAKE_CUDA_COMPILER = NVCC   (PyTorch is full of NVCC-isms; Clang-as-CUDA fails)
#   CMAKE_CUDA_HOST_COMPILER = GCC-15  (NVCC uses this for host code in .cu files)
#
# Why this split fixes the ntsr4__T0 ABI mismatch:
#   NVCC's cudafe++ renames std → __T0 in .cu template SFINAE params.
#   With Clang as host, that becomes ntsr4__T0 in libtorch_cuda.so.
#   With GCC as host, GCC mangles __T0 consistently as ntsr3std.
#   libtorch_cpu.so (Clang) and libtorch_cuda.so (NVCC+GCC) both emit ntsr3std.
#   HastyCompute (Clang) also emits ntsr3std → everything links.
#
# After include() the following are set for use in CMakeLists.txt:
#   TORCH_LIBRARIES        — IMPORTED target names for target_link_libraries
#   TORCH_INCLUDE_DIRS     — include paths (for PRIVATE compile includes)
#   TORCH_INSTALL_PREFIX   — root of the install tree (for configure_package_config_file)

include(ExternalProject)

set(_TORCH_GIT_TAG "v2.11.0")

set(LIBTORCH_SRC_DIR     "${CMAKE_CURRENT_BINARY_DIR}/_deps/libtorch-src")
set(LIBTORCH_BUILD_DIR   "${CMAKE_CURRENT_BINARY_DIR}/_deps/libtorch-build")
set(LIBTORCH_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/libtorch-install")

# CUDA compiler for LibTorch is always NVCC (not the project's Clang CUDA compiler).
set(_torch_nvcc "${CUDAToolkit_ROOT}/bin/nvcc")
# NVCC host compiler — taken from CMAKE_CUDA_HOST_COMPILER (set in the preset).
# This is the GCC that handles cudafe++ output; changing the preset value here
# automatically propagates to LibTorch.
set(_torch_cuda_host_cxx "${CMAKE_CUDA_HOST_COMPILER}")

# Convert CMAKE_CUDA_ARCHITECTURES integer list (e.g. "89;90") to PyTorch's
# dot-notation list (e.g. "8.9 9.0"). cmake uses "89", PyTorch expects "8.9".
set(_torch_arch_list "")
foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(LENGTH "${_arch}" _len)
    math(EXPR _minor_start "${_len} - 1")
    string(SUBSTRING "${_arch}" 0 ${_minor_start} _major)
    string(SUBSTRING "${_arch}" ${_minor_start} 1 _minor)
    list(APPEND _torch_arch_list "${_major}.${_minor}")
endforeach()
string(REPLACE ";" " " _torch_cuda_arch_list "${_torch_arch_list}")

# Derive CUDA root from CUDAToolkit_ROOT (set in the preset).
if(DEFINED CUDAToolkit_ROOT AND NOT CUDAToolkit_ROOT STREQUAL "")
    set(_torch_cuda_root "${CUDAToolkit_ROOT}")
else()
    get_filename_component(_torch_cuda_bin "${_torch_nvcc}" DIRECTORY)
    get_filename_component(_torch_cuda_root "${_torch_cuda_bin}" DIRECTORY)
endif()

# PyTorch's core C++ internals conditionally include python_headers.h even with
# BUILD_PYTHON=OFF.  Find Python3 dev headers and forward them so the build
# doesn't fail with 'Python.h file not found'.
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 with Development component required to build LibTorch "
        "(needed for Python.h even when BUILD_PYTHON=OFF)")
endif()

ExternalProject_Add(libtorch_external
    DEPENDS grpc_external   # uses grpc_external's installed protobuf

    GIT_REPOSITORY       "https://github.com/pytorch/pytorch.git"
    GIT_TAG              "${_TORCH_GIT_TAG}"
    GIT_SUBMODULES_RECURSE TRUE
    GIT_PROGRESS         TRUE
    GIT_SHALLOW          FALSE
    UPDATE_DISCONNECTED  TRUE

    SOURCE_DIR  "${LIBTORCH_SRC_DIR}"
    BINARY_DIR  "${LIBTORCH_BUILD_DIR}"
    INSTALL_DIR "${LIBTORCH_INSTALL_DIR}"

    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${LIBTORCH_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release

        # C/C++ compiler: Clang with GCC-15 sysroot, same as HastyCompute.
        # This avoids ABI/runtime issues that arise with pure-GCC LibTorch + Clang app.
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        "-DCMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN=${CMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN}"
        "-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=${CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN}"
        # No linker flags: GCC-15 (NVCC host) rejects -fuse-ld=lld-22 and
        # full-path -fuse-ld forms. Let LibTorch use default linker (ld.bfd).
        # CMAKE_LINKER not forwarded — NVCC device-link uses its own rules.

        # CUDA compiler: NVCC (not Clang). PyTorch adds -Xfatbin and other NVCC-only
        # flags; Clang as CUDA compiler rejects them. NVCC handles them natively.
        -DCMAKE_CUDA_COMPILER=${_torch_nvcc}
        # Host compiler for NVCC's .cu compilation: GCC-15.
        # cudafe++ renames template params, but GCC mangles them consistently as ntsr3std.
        -DCMAKE_CUDA_HOST_COMPILER=${_torch_cuda_host_cxx}
        # GCC-15 may exceed NVCC's tested host compiler range; allow it.
        "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
        "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
        "-DTORCH_CUDA_ARCH_LIST=${_torch_cuda_arch_list}"
        -DCUDAToolkit_ROOT=${_torch_cuda_root}

        # Use the protobuf already built by grpc_external — one protobuf, no duplication.
        -DUSE_SYSTEM_PROTOBUF=ON
        "-DProtobuf_DIR=${GRPC_PROTOBUF_DIR}"

        # Python headers needed for internal C++ dispatch even with BUILD_PYTHON=OFF
        "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
        "-DPython3_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}"
        "-DPython3_LIBRARIES=${Python3_LIBRARIES}"
        "-DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}"

        # Build only the C++ library — no Python, no tests, no benchmarks
        -DBUILD_PYTHON=OFF
        -DUSE_NUMPY=OFF
        -DBUILD_TEST=OFF
        -DBUILD_SHARED_LIBS=ON
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        # clang-22 in C++23 mode flags __COUNTER__ as a C2y extension; suppress it
        # so third-party headers inside PyTorch don't fail with -Werror
        "-DCMAKE_CXX_FLAGS=-Wno-c2y-extensions"

        # Disable unused features to reduce build time
        -DUSE_DISTRIBUTED=OFF
        -DUSE_MPI=OFF
        -DUSE_KINETO=OFF
        -DUSE_FBGEMM=OFF
        -DUSE_NNPACK=OFF
        -DUSE_QNNPACK=OFF
        -DUSE_XNNPACK=OFF
        -DUSE_PYTORCH_QNNPACK=OFF
        -DBUILD_CAFFE2=ON
        -DUSE_OPENMP=ON
        -DUSE_CUDA=ON

    BUILD_COMMAND
        ${CMAKE_COMMAND} --build "${LIBTORCH_BUILD_DIR}" --parallel ${_CPU_THREADS}

    # Strip LibTorch's bundled protobuf headers from the install tree.
    INSTALL_COMMAND
        ${CMAKE_COMMAND} --install "${LIBTORCH_BUILD_DIR}" --prefix "${LIBTORCH_INSTALL_DIR}"
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${LIBTORCH_INSTALL_DIR}/include/google"

    BUILD_BYPRODUCTS
        "${LIBTORCH_INSTALL_DIR}/lib/libtorch.so"
        "${LIBTORCH_INSTALL_DIR}/lib/libtorch_cpu.so"
        "${LIBTORCH_INSTALL_DIR}/lib/libtorch_cuda.so"
        "${LIBTORCH_INSTALL_DIR}/lib/libc10.so"
        "${LIBTORCH_INSTALL_DIR}/lib/libc10_cuda.so"

    STAMP_DIR  "${CMAKE_CURRENT_BINARY_DIR}/_deps/libtorch-stamp"
    USES_TERMINAL_BUILD  TRUE
    LOG_CONFIGURE        TRUE
    LOG_INSTALL          TRUE
    LOG_OUTPUT_ON_FAILURE TRUE
)

# ── IMPORTED targets ──────────────────────────────────────────────────────────
# cmake 4.x validates INTERFACE_INCLUDE_DIRECTORIES at configure time.
# Pre-creating the install directories is the documented ExternalProject pattern.
file(MAKE_DIRECTORY "${LIBTORCH_INSTALL_DIR}/include")
file(MAKE_DIRECTORY "${LIBTORCH_INSTALL_DIR}/include/torch/csrc/api/include")

foreach(_lib torch_cpu torch_cuda c10 c10_cuda)
    add_library(Torch::${_lib} SHARED IMPORTED GLOBAL)
    add_dependencies(Torch::${_lib} libtorch_external)
    set_target_properties(Torch::${_lib} PROPERTIES
        IMPORTED_LOCATION "${LIBTORCH_INSTALL_DIR}/lib/lib${_lib}.so"
    )
endforeach()

add_library(Torch::torch SHARED IMPORTED GLOBAL)
add_dependencies(Torch::torch libtorch_external)
set_target_properties(Torch::torch PROPERTIES
    IMPORTED_LOCATION             "${LIBTORCH_INSTALL_DIR}/lib/libtorch.so"
    INTERFACE_INCLUDE_DIRECTORIES
        "${LIBTORCH_INSTALL_DIR}/include;${LIBTORCH_INSTALL_DIR}/include/torch/csrc/api/include"
    INTERFACE_LINK_LIBRARIES
        "Torch::torch_cpu;Torch::torch_cuda;Torch::c10;Torch::c10_cuda"
    INTERFACE_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=1"
)

# Variables used by CMakeLists.txt (include dirs, RPATH DIRECTORIES, config file)
set(TORCH_LIBRARIES      "Torch::torch"                               CACHE INTERNAL "")
set(TORCH_INCLUDE_DIRS
    "${LIBTORCH_INSTALL_DIR}/include"
    "${LIBTORCH_INSTALL_DIR}/include/torch/csrc/api/include"          CACHE INTERNAL "")
set(TORCH_INSTALL_PREFIX "${LIBTORCH_INSTALL_DIR}"                    CACHE INTERNAL "")

target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:${TORCH_LIBRARIES}>)

# Torch .so files are copied into the install tree by HastyCompute's
# RUNTIME_DEPENDENCIES — no explicit install(TARGETS) needed here.
