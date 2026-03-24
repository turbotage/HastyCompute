cmake_minimum_required(VERSION 4.3.0 FATAL_ERROR)

project(HastyServer VERSION 0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT TARGET HastyCompute)
    find_package(HastyCompute REQUIRED)
endif()

add_library(HastyServer SHARED)

target_sources(HastyServer
    PUBLIC
        FILE_SET CXX_MODULES FILES src/server.cppm
    PRIVATE
        src/server_impl.cpp
)

target_compile_features(HastyServer
    PRIVATE cxx_std_23
    INTERFACE cxx_std_23
)

set_target_properties(HastyServer PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_SCAN_FOR_MODULES ON
    CXX_MODULE_STD ON
)

# ---------------------------------------------------------------------------
# gRPC via FetchContent — force static so HastyServer.so is self-contained
# ---------------------------------------------------------------------------

include(FetchContent)

set(BUILD_SHARED_LIBS           OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL             OFF CACHE BOOL "" FORCE)
set(utf8_range_ENABLE_INSTALL    OFF CACHE BOOL "" FORCE)
set(gRPC_PROTOBUF_PROVIDER      "module" CACHE STRING "" FORCE)
set(gRPC_ABSL_PROVIDER          "module" CACHE STRING "" FORCE)
set(gRPC_RE2_PROVIDER           "module" CACHE STRING "" FORCE)
set(gRPC_SSL_PROVIDER           "package" CACHE STRING "" FORCE)
set(gRPC_ZLIB_PROVIDER          "package" CACHE STRING "" FORCE)
set(gRPC_BUILD_TESTS             OFF CACHE BOOL "" FORCE)
set(gRPC_INSTALL                 OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    grpc
    GIT_REPOSITORY https://github.com/grpc/grpc.git
    GIT_TAG v1.76.0
)
FetchContent_MakeAvailable(grpc)

# ---------------------------------------------------------------------------
# Proto generation
# ---------------------------------------------------------------------------

set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/protos")
set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_protos")
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

file(GLOB PROTO_FILES "${PROTO_SRC_DIR}/*.proto")

foreach(proto_file ${PROTO_FILES})
    get_filename_component(proto_name ${proto_file} NAME_WE)
    set(proto_src  "${PROTO_GEN_DIR}/${proto_name}.pb.cc")
    set(proto_hdr  "${PROTO_GEN_DIR}/${proto_name}.pb.h")
    set(grpc_src   "${PROTO_GEN_DIR}/${proto_name}.grpc.pb.cc")
    set(grpc_hdr   "${PROTO_GEN_DIR}/${proto_name}.grpc.pb.h")

    add_custom_command(
        OUTPUT ${proto_src} ${proto_hdr} ${grpc_src} ${grpc_hdr}
        COMMAND protobuf::protoc
        ARGS --proto_path=${PROTO_SRC_DIR}
             --cpp_out=${PROTO_GEN_DIR}
             --grpc_out=${PROTO_GEN_DIR}
             --plugin=protoc-gen-grpc=$<TARGET_FILE:grpc_cpp_plugin>
             ${proto_file}
        DEPENDS ${proto_file} protobuf::protoc grpc_cpp_plugin
        COMMENT "Generating C++ sources from ${proto_file}"
    )

    list(APPEND PROTO_SRCS ${proto_src} ${grpc_src})
endforeach()

add_library(server_protos STATIC ${PROTO_SRCS})
set_target_properties(server_protos PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(server_protos PUBLIC ${PROTO_GEN_DIR})
target_link_libraries(server_protos PUBLIC protobuf::libprotobuf grpc++)

target_link_libraries(HastyServer PRIVATE $<BUILD_INTERFACE:server_protos>)
target_link_libraries(HastyServer PRIVATE HastyCompute::HastyCompute)
target_include_directories(HastyServer PRIVATE ${PROTO_GEN_DIR})
add_dependencies(HastyServer server_protos)

# ---------------------------------------------------------------------------
# CMake package export
# ---------------------------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS HastyServer
    EXPORT HastyServerTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FILE_SET CXX_MODULES DESTINATION include/HastyServer/modules
)

install(EXPORT HastyServerTargets
    FILE HastyServerTargets.cmake
    NAMESPACE HastyServer::
    DESTINATION lib/cmake/HastyServer
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/HastyServerConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/HastyServerConfig.cmake
    INSTALL_DESTINATION lib/cmake/HastyServer
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/HastyServerConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/HastyServerConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/HastyServerConfigVersion.cmake
    DESTINATION lib/cmake/HastyServer
)
