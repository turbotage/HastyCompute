cmake_minimum_required(VERSION 4.2.0)

project(HastyComputeRoot VERSION 0.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(ExternalProject)

# Parse which targets to build (defaults to all)
if(NOT DEFINED HASTY_BUILD_TARGETS)
	set(HASTY_BUILD_TARGETS "lib;test;server")
endif()

# Convert to list
string(REPLACE ";" "|" HASTY_BUILD_TARGETS_STR "${HASTY_BUILD_TARGETS}")
string(REPLACE "|" ";" HASTY_BUILD_LIST "${HASTY_BUILD_TARGETS_STR}")

# Check what to build
if("lib" IN_LIST HASTY_BUILD_LIST)
	set(HASTY_BUILD_LIB ON)
endif()
if("test" IN_LIST HASTY_BUILD_LIST)
	set(HASTY_BUILD_TEST ON)
endif()
if("server" IN_LIST HASTY_BUILD_LIST)
	set(HASTY_BUILD_SERVER ON)
endif()


# Determine which preset to use based on CMAKE_BUILD_TYPE
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(LIB_PRESET "HastyComputeLib-ClangDebug-Linux")
	set(TEST_PRESET "HastyComputeTest-ClangDebug-Linux")
	set(SERVER_PRESET "HastyComputeServer-ClangDebug-Linux")
else()
	set(LIB_PRESET "HastyComputeLib-ClangRelease-Linux")
	set(TEST_PRESET "HastyComputeTest-ClangRelease-Linux")
	set(SERVER_PRESET "HastyComputeServer-ClangRelease-Linux")
endif()

if (HASTY_BUILD_LIB)
	ExternalProject_Add(
		HastyCompute
		SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib
		BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib-build
		CONFIGURE_COMMAND ${CMAKE_COMMAND} --preset ${LIB_PRESET} -S ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib -B ${CMAKE_CURRENT_BINARY_DIR}/lib-build
		BUILD_COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/lib-build
		INSTALL_COMMAND ${CMAKE_COMMAND} --install ${CMAKE_CURRENT_BINARY_DIR}/lib-build --prefix ${CMAKE_CURRENT_BINARY_DIR}/install
		BUILD_ALWAYS 1
	)
endif()

if (HASTY_BUILD_SERVER)
	ExternalProject_Add(
		HastyServer
		SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/server
		BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/server-build
		CONFIGURE_COMMAND ${CMAKE_COMMAND} --preset ${SERVER_PRESET} -DCMAKE_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/install -S ${CMAKE_CURRENT_SOURCE_DIR}/cpp/server -B ${CMAKE_CURRENT_BINARY_DIR}/server-build
		BUILD_COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/server-build
		BUILD_ALWAYS 1
		INSTALL_COMMAND ${CMAKE_COMMAND} --install ${CMAKE_CURRENT_BINARY_DIR}/server-build --prefix ${CMAKE_CURRENT_BINARY_DIR}/install
		DEPENDS HastyCompute
	)
endif()

if (HASTY_BUILD_TEST)
	set(_test_deps HastyCompute)
	if (HASTY_BUILD_SERVER)
		list(APPEND _test_deps HastyServer)
	endif()
	ExternalProject_Add(
		HastyTest
		SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/test
		BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test-build
		CONFIGURE_COMMAND ${CMAKE_COMMAND} --preset ${TEST_PRESET} -DCMAKE_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/install -S ${CMAKE_CURRENT_SOURCE_DIR}/cpp/test -B ${CMAKE_CURRENT_BINARY_DIR}/test-build
		BUILD_COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/test-build
		BUILD_ALWAYS 1
		INSTALL_COMMAND ""
		DEPENDS ${_test_deps}
	)
endif()


# Merge compile_commands.json from all subprojects
set(MERGE_DEPS "")
if(HASTY_BUILD_LIB)
    list(APPEND MERGE_DEPS HastyCompute)
endif()
if(HASTY_BUILD_TEST)
    list(APPEND MERGE_DEPS HastyTest)
endif()
if(HASTY_BUILD_SERVER)
    list(APPEND MERGE_DEPS HastyServer)
endif()

add_custom_target(merge_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Merging compile_commands.json..."
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/merge_compile_commands.py
        ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${MERGE_DEPS}
    COMMENT "Merging compile_commands.json from subprojects"
)

# Cleans own code build artifacts and forces full reconfigure+rebuild,
# but preserves fetched/compiled dependencies (_deps) so grpc/protobuf/
# abseil/finufft etc. do NOT need to be rebuilt.
add_custom_target(clean-cache
    # Remove cmake caches and build-system files (NOT _deps)
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/lib-build/CMakeCache.txt
        ${CMAKE_CURRENT_BINARY_DIR}/server-build/CMakeCache.txt
        ${CMAKE_CURRENT_BINARY_DIR}/test-build/CMakeCache.txt
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/lib-build/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/server-build/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/test-build/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles
    # Remove ExternalProject stamps so configure/build steps re-run
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/HastyCompute-prefix/src/HastyCompute-stamp
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/HastyServer-prefix/src/HastyServer-stamp
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/HastyTest-prefix/src/HastyTest-stamp
    # Remove installed own libs and cmake configs so they are freshly reinstalled
    # (runtime dep libs like libtorch/libcuda etc. are intentionally preserved)
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/install/lib/libHastyCompute.so
        ${CMAKE_CURRENT_BINARY_DIR}/install/lib/libHastyServer.so
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/install/lib/cmake
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/install/include
    # Ensure build dirs exist so ExternalProject configure can cd into them
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/lib-build
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/server-build
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/test-build
    COMMAND ${CMAKE_COMMAND} -E echo "Clean done — _deps and runtime libs preserved"
    COMMENT "Cleaning own code; fetched dependencies preserved"
)

# Nuclear option: removes everything including fetched dependencies and stamps.
# Next build will re-download and recompile grpc/protobuf/abseil/etc.
add_custom_target(deep-clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/lib-build
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/server-build
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/test-build
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/install
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/HastyCompute-prefix
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/HastyServer-prefix
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/HastyTest-prefix
    COMMAND ${CMAKE_COMMAND} -E echo "Deep clean complete"
    COMMENT "Removing all build dirs, install, and stamps (full rebuild required)"
)
