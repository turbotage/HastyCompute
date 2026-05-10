# PatchLibTorch.cmake — applied via ExternalProject PATCH_COMMAND before configure.
# Placeholder: cuda.cmake patch removed (Clang is the correct NVCC host with C++20).

# ── utils.cmake patch ─────────────────────────────────────────────────────────
# Problem: utils.cmake loops private_compile_options and wraps each with
# -Xcompiler for CUDA targets. When CXX=Clang, the list contains Clang-only
# flags (e.g. -Wmove). The existing GNU host filter skips only -Wextra-semi and
# -Wunused-private-field; all other Clang flags reach GCC → build error.
# Fix: extend the filter with all remaining Clang-only warning flags.

set(_utils_cmake "${LIBTORCH_SRC_DIR}/cmake/public/utils.cmake")

file(READ "${_utils_cmake}" _content)

set(_old [[        if("${option}" STREQUAL "-Wunused-private-field")
          continue()
        endif()
      endif()
      target_compile_options(${libname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${option}>)]])

set(_new [[        if("${option}" STREQUAL "-Wunused-private-field")
          continue()
        endif()
        # HastyCompute patch: skip additional Clang-only flags GCC doesn't know.
        # continue() exits the innermost loop; IN_LIST avoids nested foreach.
        set(_gcc_incompatible_flags
            -Wmove -Wreorder-ctor -Wunused-lambda-capture
            -Wno-unknown-warning-option
            -Werror=inconsistent-missing-override
            -Werror=inconsistent-missing-destructor-override
            -Werror=macro-redefined
            -Werror=deprecated-copy-with-dtor)
        if("${option}" IN_LIST _gcc_incompatible_flags)
          continue()
        endif()
      endif()
      target_compile_options(${libname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${option}>)]])

string(REPLACE "${_old}" "${_new}" _content "${_content}")
file(WRITE "${_utils_cmake}" "${_content}")

message(STATUS "PatchLibTorch: patched cmake/public/utils.cmake (GNU host Clang-flag filter)")
