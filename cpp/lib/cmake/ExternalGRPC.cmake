# ExternalGRPC.cmake — builds gRPC from source as an ExternalProject.
#
# gRPC manages its own compatible protobuf version internally (module provider)
# and installs it alongside itself.  ExternalLibTorch.cmake then points LibTorch
# at this same installed protobuf via USE_SYSTEM_PROTOBUF=ON, so there is only
# ONE protobuf compilation shared by both.
#
# All compiler/linker variables flow from the lib preset via cmake variables —
# changing the toolchain in CommonPresets.json propagates here automatically.
#
# After include() the following IMPORTED targets are available:
#   protobuf::libprotobuf  — shared protobuf library
#   protobuf::protoc       — protoc executable
#   grpc++                 — shared gRPC C++ library
#   grpc_cpp_plugin        — protoc plugin executable
#
# GRPC_INSTALL_DIR and GRPC_PROTOBUF_DIR are set as CACHE INTERNAL for use
# by ExternalLibTorch.cmake (included after this file).

include(ExternalProject)

set(_grpc_src     "${CMAKE_CURRENT_BINARY_DIR}/_deps/grpc-src")
set(_grpc_build   "${CMAKE_CURRENT_BINARY_DIR}/_deps/grpc-build")
set(_grpc_install "${CMAKE_CURRENT_BINARY_DIR}/_deps/grpc-install")

ExternalProject_Add(grpc_external
    GIT_REPOSITORY       "https://github.com/grpc/grpc.git"
    GIT_TAG              v1.80.0
    GIT_SUBMODULES_RECURSE TRUE
    GIT_SHALLOW          FALSE
    GIT_PROGRESS         TRUE
    UPDATE_DISCONNECTED  TRUE

    SOURCE_DIR  "${_grpc_src}"
    BINARY_DIR  "${_grpc_build}"
    INSTALL_DIR "${_grpc_install}"

    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${_grpc_install}
        -DCMAKE_BUILD_TYPE=Release

        # Compiler/toolchain — all from the lib preset, no hardcoded paths
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        "-DCMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN=${CMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN}"
        "-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=${CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN}"
        -DCMAKE_LINKER=${CMAKE_LINKER}
        "-DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}"
        "-DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"

        # Build shared libs so transitive deps (abseil, re2 …) are encoded in
        # ELF NEEDED entries and we don't have to list them individually.
        -DBUILD_SHARED_LIBS=ON
        "-DCMAKE_INSTALL_RPATH=${_grpc_install}/lib"
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON

        # gRPC manages its own compatible protobuf (module) and installs it.
        # LibTorch will then use USE_SYSTEM_PROTOBUF pointing here.
        -DgRPC_PROTOBUF_PROVIDER=module
        -DgRPC_ABSL_PROVIDER=module
        -DgRPC_RE2_PROVIDER=module
        -DgRPC_SSL_PROVIDER=package
        -DgRPC_ZLIB_PROVIDER=package
        -DgRPC_BUILD_TESTS=OFF
        -DgRPC_INSTALL=ON
        -Dprotobuf_INSTALL=ON
        -Dutf8_range_ENABLE_INSTALL=ON
        "-DCMAKE_JOB_POOLS=link=2"
        -DCMAKE_JOB_POOL_LINK=link

    BUILD_COMMAND ${CMAKE_COMMAND} --build "${_grpc_build}" --parallel ${_CPU_THREADS}

    # Explicit --prefix overrides whatever CMAKE_INSTALL_PREFIX was baked into
    # cmake_install.cmake at configure time (e.g. /usr/local from a stale cache).
    INSTALL_COMMAND ${CMAKE_COMMAND} --install "${_grpc_build}" --prefix "${_grpc_install}"

    BUILD_BYPRODUCTS
        "${_grpc_install}/lib/libgrpc++.so"
        "${_grpc_install}/lib/libgrpc.so"
        "${_grpc_install}/lib/libprotobuf.so"
        "${_grpc_install}/bin/grpc_cpp_plugin"
        "${_grpc_install}/bin/protoc"

    STAMP_DIR  "${CMAKE_CURRENT_BINARY_DIR}/_deps/grpc-stamp"
    USES_TERMINAL_BUILD  TRUE
    LOG_CONFIGURE TRUE LOG_INSTALL TRUE LOG_OUTPUT_ON_FAILURE TRUE
)

# Expose install paths for ExternalLibTorch (included after this file)
set(GRPC_INSTALL_DIR  "${_grpc_install}" CACHE INTERNAL "")
set(GRPC_PROTOBUF_DIR "${_grpc_install}/lib/cmake/protobuf" CACHE INTERNAL "")

# ── IMPORTED targets ──────────────────────────────────────────────────────────
# cmake 4.x validates INTERFACE_INCLUDE_DIRECTORIES at configure time even for
# generator expressions.  Pre-creating the directories is the documented
# ExternalProject pattern (explicitly mentioned in cmake's ExternalProject docs).
file(MAKE_DIRECTORY "${_grpc_install}/include")

add_library(protobuf::libprotobuf SHARED IMPORTED GLOBAL)
add_dependencies(protobuf::libprotobuf grpc_external)
set_target_properties(protobuf::libprotobuf PROPERTIES
    IMPORTED_LOCATION             "${_grpc_install}/lib/libprotobuf.so"
    INTERFACE_INCLUDE_DIRECTORIES "${_grpc_install}/include"
)

add_executable(protobuf::protoc IMPORTED GLOBAL)
add_dependencies(protobuf::protoc grpc_external)
set_property(TARGET protobuf::protoc PROPERTY
    IMPORTED_LOCATION "${_grpc_install}/bin/protoc"
)

add_library(grpc++ SHARED IMPORTED GLOBAL)
add_dependencies(grpc++ grpc_external)
set_target_properties(grpc++ PROPERTIES
    IMPORTED_LOCATION             "${_grpc_install}/lib/libgrpc++.so"
    INTERFACE_INCLUDE_DIRECTORIES "${_grpc_install}/include"
    INTERFACE_LINK_LIBRARIES      "protobuf::libprotobuf"
    INTERFACE_LINK_DIRECTORIES    "${_grpc_install}/lib"
)

add_executable(grpc_cpp_plugin IMPORTED GLOBAL)
add_dependencies(grpc_cpp_plugin grpc_external)
set_property(TARGET grpc_cpp_plugin PROPERTY
    IMPORTED_LOCATION "${_grpc_install}/bin/grpc_cpp_plugin"
)

# ── Proto code generation ─────────────────────────────────────────────────────
# Use the installed protoc/grpc_cpp_plugin by absolute path.
# The BUILD_BYPRODUCTS paths are listed as DEPENDS so the custom command
# runs only after grpc_external finishes.
set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/server/protos")
set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_protos")
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

set(_protoc      "${_grpc_install}/bin/protoc")
set(_grpc_plugin "${_grpc_install}/bin/grpc_cpp_plugin")

file(GLOB PROTO_FILES "${PROTO_SRC_DIR}/*.proto")
foreach(proto_file ${PROTO_FILES})
    get_filename_component(proto_name ${proto_file} NAME_WE)
    set(proto_src "${PROTO_GEN_DIR}/${proto_name}.pb.cc")
    set(proto_hdr "${PROTO_GEN_DIR}/${proto_name}.pb.h")
    set(grpc_src  "${PROTO_GEN_DIR}/${proto_name}.grpc.pb.cc")
    set(grpc_hdr  "${PROTO_GEN_DIR}/${proto_name}.grpc.pb.h")
    add_custom_command(
        OUTPUT  ${proto_src} ${proto_hdr} ${grpc_src} ${grpc_hdr}
        COMMAND "${_protoc}"
        ARGS    --proto_path=${PROTO_SRC_DIR}
                --cpp_out=${PROTO_GEN_DIR}
                --grpc_out=${PROTO_GEN_DIR}
                --plugin=protoc-gen-grpc=${_grpc_plugin}
                ${proto_file}
        DEPENDS ${proto_file} "${_protoc}" "${_grpc_plugin}"
        COMMENT "Generating C++ from ${proto_file}"
    )
    list(APPEND PROTO_SRCS ${proto_src} ${grpc_src})
endforeach()

add_library(hasty_server_protos STATIC ${PROTO_SRCS})
set_target_properties(hasty_server_protos PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(hasty_server_protos PUBLIC ${PROTO_GEN_DIR})
target_link_libraries(hasty_server_protos PUBLIC protobuf::libprotobuf grpc++)
target_link_directories(hasty_server_protos PUBLIC
    "$<BUILD_INTERFACE:${_grpc_install}/lib>")

target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:hasty_server_protos>)
# Add gRPC and proto include dirs directly so C++ module BMI scanning sees them.
# Transitive includes through linked targets are not always forwarded to the
# module-scanning step; explicit PRIVATE include is the reliable path.
target_include_directories(HastyCompute PRIVATE
    ${PROTO_GEN_DIR}
    "${_grpc_install}/include"
)
