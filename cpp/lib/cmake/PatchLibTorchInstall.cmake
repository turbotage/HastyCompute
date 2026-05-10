# PatchLibTorchInstall.cmake
# Called from ExternalLibTorch.cmake INSTALL_COMMAND after libtorch installs.
# Patches libtorch_cuda.so: renames ntsr4__T0 → ntsr3std in .dynsym/.dynstr
# and all related ELF sections (.dynamic, .gnu.version_r, .gnu.version_d).
#
# Required variable (passed via -D):
#   LIBTORCH_INSTALL_DIR  — root of the libtorch install tree

cmake_minimum_required(VERSION 3.21)

set(_cuda_so "${LIBTORCH_INSTALL_DIR}/lib/libtorch_cuda.so")
set(_script  "${CMAKE_CURRENT_LIST_DIR}/patch_libtorch_dynsym.py")

if(NOT EXISTS "${_cuda_so}")
    message(FATAL_ERROR "PatchLibTorchInstall: ${_cuda_so} not found")
endif()

find_program(_python NAMES python3 python REQUIRED)

execute_process(
    COMMAND "${_python}" "${_script}" "${_cuda_so}"
    RESULT_VARIABLE _r
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE  _err
)

message(STATUS "${_out}")
if(_r)
    message(FATAL_ERROR "PatchLibTorchInstall: patch_libtorch_dynsym.py failed (${_r})\n${_err}")
endif()
