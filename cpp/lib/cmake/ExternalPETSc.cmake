# cpp/lib/cmake/ExternalPETSc.cmake
#
# Builds PETSc + SLEPc as ExternalProjects.
# Included by cpp/lib/CMakeLists.txt, so cmake variables
# (CMAKE_CUDA_COMPILER, Python3_EXECUTABLE, etc.) come from the lib preset.
# No hardcoded paths — change the toolchain in CommonPresets.json only.
#
# Exported IMPORTED targets:
#   PETSc::petsc
#   SLEPc::slepc

include(ExternalProject)

set(_PETSC_VERSION "v3.25.0")
set(_SLEPC_VERSION "v3.25.0")
set(_PETSC_ARCH    "arch-complex-cuda-opt")

set(PETSC_SRC_DIR     "${CMAKE_CURRENT_BINARY_DIR}/_deps/petsc-src")
set(PETSC_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/petsc-install")
set(SLEPC_SRC_DIR     "${CMAKE_CURRENT_BINARY_DIR}/_deps/slepc-src")
set(SLEPC_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/slepc-install")

# Derive CUDA root from CMAKE_CUDA_COMPILER (set by the lib preset)
get_filename_component(_psc_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(_psc_cuda_root "${_psc_cuda_bin}" DIRECTORY)

ExternalProject_Add(petsc_external
    GIT_REPOSITORY  "https://gitlab.com/petsc/petsc.git"
    GIT_TAG         "${_PETSC_VERSION}"
    GIT_SHALLOW     TRUE
    GIT_PROGRESS    TRUE

    SOURCE_DIR      "${PETSC_SRC_DIR}"
    BUILD_IN_SOURCE TRUE
    UPDATE_COMMAND  ""

    CONFIGURE_COMMAND
        ${CMAKE_COMMAND} -E env
            "PATH=${_psc_cuda_root}/bin:$ENV{PATH}"
            "PETSC_DIR=${PETSC_SRC_DIR}"
            "PETSC_ARCH=${_PETSC_ARCH}"
        "${Python3_EXECUTABLE}" "${PETSC_SRC_DIR}/configure"
            "--PETSC_ARCH=${_PETSC_ARCH}"
            "--prefix=${PETSC_INSTALL_DIR}"
            "--with-scalar-type=complex"
            "--with-precision=double"
            "--with-cuda=1"
            "--with-cuda-include=${_psc_cuda_root}/include"
            "--with-cuda-lib=-L${_psc_cuda_root}/lib64 -lcudart -lcufft -lcublas -lcusparse -lcusolver -lcurand -L${_psc_cuda_root}/lib64/stubs -lcuda -lnvidia-ml"
            "--with-mpi=0"
            "--with-fc=0"
            "--download-f2cblaslapack=1"
            "--with-quadmath=0"
            "--with-debugging=0"
            "--with-shared-libraries=1"
            "COPTFLAGS=-O3"
            "CXXOPTFLAGS=-O3"

    BUILD_COMMAND
        ${CMAKE_COMMAND} -E env
            "PATH=${_psc_cuda_root}/bin:$ENV{PATH}"
            "PETSC_DIR=${PETSC_SRC_DIR}"
            "PETSC_ARCH=${_PETSC_ARCH}"
        make -j${_CPU_THREADS} all

    INSTALL_COMMAND
        ${CMAKE_COMMAND} -E env
            "PATH=${_psc_cuda_root}/bin:$ENV{PATH}"
            "PETSC_DIR=${PETSC_SRC_DIR}"
            "PETSC_ARCH=${_PETSC_ARCH}"
        make install
        COMMAND sed -i "/PETSC_HAVE_REAL___FLOAT128/d"
            "${PETSC_INSTALL_DIR}/include/petscconf.h"

    BUILD_BYPRODUCTS
        "${PETSC_INSTALL_DIR}/lib/libpetsc.so"
        "${PETSC_INSTALL_DIR}/include/petsc.h"

    STAMP_DIR  "${CMAKE_CURRENT_BINARY_DIR}/_deps/petsc-stamp"
    USES_TERMINAL_BUILD  TRUE
    LOG_CONFIGURE TRUE LOG_INSTALL TRUE LOG_OUTPUT_ON_FAILURE TRUE
)

ExternalProject_Add(slepc_external
    DEPENDS petsc_external

    GIT_REPOSITORY  "https://gitlab.com/slepc/slepc.git"
    GIT_TAG         "${_SLEPC_VERSION}"
    GIT_SHALLOW     TRUE
    GIT_PROGRESS    TRUE

    SOURCE_DIR      "${SLEPC_SRC_DIR}"
    BUILD_IN_SOURCE TRUE
    UPDATE_COMMAND  ""

    CONFIGURE_COMMAND
        ${CMAKE_COMMAND} -E env
            "SLEPC_DIR=${SLEPC_SRC_DIR}"
            "PETSC_DIR=${PETSC_INSTALL_DIR}"
        "${Python3_EXECUTABLE}" "${SLEPC_SRC_DIR}/configure"
            "--prefix=${SLEPC_INSTALL_DIR}"

    BUILD_COMMAND
        ${CMAKE_COMMAND} -E env
            "PATH=${_psc_cuda_root}/bin:$ENV{PATH}"
            "SLEPC_DIR=${SLEPC_SRC_DIR}"
            "PETSC_DIR=${PETSC_INSTALL_DIR}"
        make -j${_CPU_THREADS}

    INSTALL_COMMAND
        ${CMAKE_COMMAND} -E env
            "PATH=${_psc_cuda_root}/bin:$ENV{PATH}"
            "SLEPC_DIR=${SLEPC_SRC_DIR}"
            "PETSC_DIR=${PETSC_INSTALL_DIR}"
        make install

    BUILD_BYPRODUCTS
        "${SLEPC_INSTALL_DIR}/lib/libslepc.so"
        "${SLEPC_INSTALL_DIR}/include/slepceps.h"

    STAMP_DIR  "${CMAKE_CURRENT_BINARY_DIR}/_deps/slepc-stamp"
    USES_TERMINAL_BUILD  TRUE
    LOG_CONFIGURE TRUE LOG_INSTALL TRUE LOG_OUTPUT_ON_FAILURE TRUE
)

# ── IMPORTED targets ──────────────────────────────────────────────────────────
file(MAKE_DIRECTORY "${PETSC_INSTALL_DIR}/include")
file(MAKE_DIRECTORY "${SLEPC_INSTALL_DIR}/include")

add_library(PETSc::petsc SHARED IMPORTED GLOBAL)
add_dependencies(PETSc::petsc petsc_external)
set_target_properties(PETSc::petsc PROPERTIES
    IMPORTED_LOCATION             "${PETSC_INSTALL_DIR}/lib/libpetsc.so"
    INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INSTALL_DIR}/include"
    INTERFACE_COMPILE_DEFINITIONS "PETSC_SKIP_REAL___FLOAT128"
)

add_library(SLEPc::slepc SHARED IMPORTED GLOBAL)
add_dependencies(SLEPc::slepc slepc_external)
set_target_properties(SLEPc::slepc PROPERTIES
    IMPORTED_LOCATION             "${SLEPC_INSTALL_DIR}/lib/libslepc.so"
    INTERFACE_INCLUDE_DIRECTORIES "${SLEPC_INSTALL_DIR}/include"
    INTERFACE_LINK_LIBRARIES      "PETSc::petsc"
)

target_link_libraries(HastyCompute PRIVATE
    $<BUILD_INTERFACE:PETSc::petsc>
    $<BUILD_INTERFACE:SLEPc::slepc>
)
