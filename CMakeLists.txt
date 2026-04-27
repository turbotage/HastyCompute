cmake_minimum_required(VERSION 4.2.0)

project(HastyComputeRoot VERSION 0.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(ExternalProject)

# Parse which targets to build (defaults to lib+test; server is part of lib)
if(NOT DEFINED HASTY_BUILD_TARGETS)
	set(HASTY_BUILD_TARGETS "lib;test")
endif()
string(REPLACE ";" "|" _tmp "${HASTY_BUILD_TARGETS}")
string(REPLACE "|" ";" HASTY_BUILD_LIST "${_tmp}")

if("lib"  IN_LIST HASTY_BUILD_LIST)
    set(HASTY_BUILD_LIB  ON)
endif()
if("test" IN_LIST HASTY_BUILD_LIST)
    set(HASTY_BUILD_TEST ON)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(LIB_PRESET  "HastyComputeLib-ClangDebug-Linux")
	set(TEST_PRESET "HastyComputeTest-ClangDebug-Linux")
else()
	set(LIB_PRESET  "HastyComputeLib-ClangRelWithDebInfo-Linux")
	set(TEST_PRESET "HastyComputeTest-ClangRelWithDebInfo-Linux")
endif()

if (HASTY_BUILD_LIB)
	ExternalProject_Add(
		HastyCompute
		SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib
		BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib-build
		CONFIGURE_COMMAND
			${CMAKE_COMMAND} --preset ${LIB_PRESET}
			-S ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib
			-B ${CMAKE_CURRENT_BINARY_DIR}/lib-build
		BUILD_COMMAND   ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/lib-build
		INSTALL_COMMAND ${CMAKE_COMMAND} --install ${CMAKE_CURRENT_BINARY_DIR}/lib-build
			--prefix ${CMAKE_CURRENT_BINARY_DIR}/install
		BUILD_ALWAYS 1
		USES_TERMINAL_BUILD TRUE
	)
endif()

if (HASTY_BUILD_TEST)
	ExternalProject_Add(
		HastyTest
		SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/test
		BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test-build
		CONFIGURE_COMMAND
			${CMAKE_COMMAND} --preset ${TEST_PRESET}
			-DCMAKE_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/install
			-S ${CMAKE_CURRENT_SOURCE_DIR}/cpp/test
			-B ${CMAKE_CURRENT_BINARY_DIR}/test-build
		BUILD_COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/test-build
		BUILD_ALWAYS 1
		INSTALL_COMMAND ""
		DEPENDS HastyCompute
		USES_TERMINAL_BUILD TRUE
	)
endif()

# ── Utilities ─────────────────────────────────────────────────────────────────
set(MERGE_DEPS "")
if(HASTY_BUILD_LIB) 
    list(APPEND MERGE_DEPS HastyCompute) 
endif()
if(HASTY_BUILD_TEST) 
    list(APPEND MERGE_DEPS HastyTest)
endif()

add_custom_target(merge_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Merging compile_commands.json..."
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/merge_compile_commands.py
        ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${MERGE_DEPS}
    COMMENT "Merging compile_commands.json from subprojects"
)

add_custom_target(reconfigure-lib
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/HastyCompute-prefix/src/HastyCompute-stamp/HastyCompute-configure
        ${CMAKE_CURRENT_BINARY_DIR}/HastyCompute-prefix/src/HastyCompute-stamp/HastyCompute-build
        ${CMAKE_CURRENT_BINARY_DIR}/HastyCompute-prefix/src/HastyCompute-stamp/HastyCompute-install
        ${CMAKE_CURRENT_BINARY_DIR}/HastyCompute-prefix/src/HastyCompute-stamp/HastyCompute-done
        ${CMAKE_CURRENT_BINARY_DIR}/HastyTest-prefix/src/HastyTest-stamp/HastyTest-configure
    COMMENT "Clearing lib+test configure stamps"
)

add_custom_target(clean-cache
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/lib-build/CMakeCache.txt
        ${CMAKE_CURRENT_BINARY_DIR}/test-build/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/lib-build/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/test-build/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/HastyCompute-prefix/src/HastyCompute-stamp
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_CURRENT_BINARY_DIR}/HastyTest-prefix/src/HastyTest-stamp
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/install/lib/libHastyCompute.so
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/install/lib/cmake
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/install/include
    COMMAND ${CMAKE_COMMAND} -E make_directory   ${CMAKE_CURRENT_BINARY_DIR}/lib-build
    COMMAND ${CMAKE_COMMAND} -E make_directory   ${CMAKE_CURRENT_BINARY_DIR}/test-build
    COMMENT "Cleaning own code; _deps preserved"
)

add_custom_target(deep-clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/lib-build
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/test-build
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/install
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/HastyCompute-prefix
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/HastyTest-prefix
    COMMENT "Removing all build dirs, install, and stamps (full rebuild required)"
)
