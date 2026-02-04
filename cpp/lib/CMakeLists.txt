cmake_minimum_required(VERSION 4.2.0 FATAL_ERROR)

#set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")

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
#target_sources(HastyCuCompute
#	PRIVATE
#		""
#)

# LIBRARY CPP MODULE SOURCES
target_sources(HastyCompute
    PRIVATE 
        FILE_SET cxx_modules_private TYPE CXX_MODULES FILES
			"src/tensor/tensor_background_impl.cppm"
			"src/tensor/tensor_background.cppm"
			"src/tensor/tensor_scalar.cppm"
            "src/tensor/tensor_tensor.cppm"
			"src/util/util_alias.cppm"
            "src/util/util_meta.cppm"
            "src/util/util_span.cppm"
			"src/util/util_str.cppm"
            "src/util/util_typing.cppm"
            "src/util/util.cppm"
            "src/wrapper/torch_wrapper.cppm"
)

# LIB INTERFACE CPP SOURCES
#target_sources(HastyCuCompute
#	PRIVATE
#		""
#)

# LIB INTERFACE CPP MODULE SOURCES
target_sources(HastyCompute
    PUBLIC 
        FILE_SET cxx_modules_public TYPE CXX_MODULES FILES
            "interface/interface.cppm"
)

set(DB_PRODUCTION_MODE ON)
FetchContent_Declare(
  battery-embed
  GIT_REPOSITORY https://github.com/batterycenter/embed.git
  GIT_TAG        v1.2.19
)
FetchContent_MakeAvailable(battery-embed)

b_embed(HastyCompute "src/fft/kernels/toeplitz_load_2D.cu")
b_embed(HastyCompute "src/fft/kernels/toeplitz_load_3D.cu")

get_target_property(_incs HastyCompute INTERFACE_INCLUDE_DIRECTORIES)
list(REMOVE_ITEM _incs "${EMBED_BINARY_DIR}/autogen/hastycompute/include")
set_target_properties(HastyCompute PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_incs}")

find_package(CUDAToolkit REQUIRED)


target_include_directories(HastyCompute
	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/vkFFT>
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
)

# this gives access to cufinufft
set(FINUFFT_USE_CUDA ON CACHE BOOL "" FORCE)
# this disables finufft leaving only cufinufft available
set(FINUFFT_USE_CPU ON CACHE BOOL "" FORCE)
set(FINUFFT_USE_DUCC0 ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  finufft
  GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git
  GIT_TAG        master #v2.4.1
)

FetchContent_MakeAvailable(finufft)

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
		 CXX_VISIBILITY_PRESET "hidden"
		 VISIBILITY_INLINES_HIDDEN ON
	 )

target_compile_definitions(HastyCompute PUBLIC _GLIBCXX_USE_CXX11_ABI=1)



set_target_properties(HastyCompute PROPERTIES INSTALL_RPATH "$ORIGIN")


#install(FILES cpp/corelib/interface/include/interface.hpp DESTINATION include/HastyCompute)

set(HastyCompute_INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/interface/include>
    $<INSTALL_INTERFACE:include>
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
        ${Python_INCLUDE_DIRS}
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
        #PRE_EXCLUDE_REGEXES ".*libstdc\+\+.*" ".*libgcc.*" ".*libc.so.*" ".*libc\+\+.*"
        #POST_EXCLUDE_REGEXES ".*libstdc\+\+.*" ".*libgcc.*" ".*libc.so.*" ".*libc\+\+.*"
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