cmake_minimum_required(VERSION 3.28)

set(VCPKG_FEATURE_FLAGS "versions")

project("HastyCompute" VERSION 0.1)

include(FetchContent)
#include(cmake/CPM.cmake)

#set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "2182bf5c-ef0d-489a-91da-49dbc3090d2a")
#if (MSVC)
#	set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
#endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)


set(BUILD_SHARED_LIBS ON)

option(BUILD_CUDA OFF)

message("CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE}")
message("CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")
message("BUILD_CUDA = ${BUILD_CUDA}")


set(BACKEND_HEADERS
	
)

add_library(HastyComputeLib SHARED

	"lib/tests/tests1.cpp"
	"lib/tests/tests2.cpp"

)

if (MSVC)
	#set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
	set_target_properties(HastyComputeLib PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_sources(HastyComputeLib 
	PUBLIC FILE_SET CXX_MODULES FILES
		"lib/torch_util.ixx"
		"lib/torch_util.cpp"

		"lib/device.ixx"
		"lib/hasty.ixx"
		"lib/hasty_compute.ixx"
		"lib/hasty_util.ixx"
		"lib/metadata.ixx"
		"lib/hdf5.ixx"
		"lib/hdf5.cpp"
	
		"lib/threading/thread_pool.ixx"

		"lib/expr/parser/token.ixx" 
		"lib/expr/parser/lexer.ixx" 
		"lib/expr/parser/defaultexp.ixx"
		"lib/expr/parser/shunter.ixx"
		"lib/expr/algebra/token_algebra.ixx"
		"lib/expr/algebra/unary/neg.ixx"
		"lib/expr/algebra/unary/trig.ixx"
		"lib/expr/algebra/unary/unary.ixx"
		"lib/expr/algebra/binary/add.ixx"
		"lib/expr/algebra/binary/div.ixx"
		"lib/expr/algebra/binary/mul.ixx"
		"lib/expr/algebra/binary/pow.ixx"
		"lib/expr/algebra/binary/sub.ixx"
		"lib/expr/expr.ixx" 
		"lib/expr/expr.cpp"
		"lib/expr/symbolic.ixx"

		"lib/linalg/permute.ixx"
		"lib/linalg/solver.ixx"

		"lib/nlsq/nlsq_symbolic.ixx"
		"lib/nlsq/nlsq.ixx"

		"lib/fft/dct.ixx"
		"lib/fft/dct.cpp"
		"lib/fft/fft.ixx"
		"lib/fft/fft.cpp"
		"lib/fft/nufft.ixx"
		"lib/fft/nufft.cpp"

		"lib/mri/sense.ixx"
		"lib/mri/sense.cpp"

		"lib/op/vec.ixx"
		"lib/op/vec.cpp"
		"lib/op/op.ixx"
		"lib/op/op.cpp"
		"lib/op/opalgebra.ixx"
		"lib/op/opalgebra.cpp"
		"lib/op/opalgs.ixx"
		"lib/op/opalgs.cpp"
		"lib/op/mriop.ixx"
		"lib/op/mriop.cpp"
		"lib/op/proxop.ixx"
		"lib/op/proxop.cpp"
		"lib/op/fftop.ixx"
		"lib/op/fftop.cpp"
		"lib/op/block.ixx"
		"lib/op/svt.ixx"
		"lib/op/svt.cpp"

		"lib/mri/precond.ixx"
		"lib/mri/precond.cpp"

		"lib/mri/batch_sense.ixx"
		"lib/mri/batch_sense.cpp"

		"lib/mri/flowmri.ixx"

)

target_sources(HastyComputeLib PUBLIC 
	$<INSTALL_INTERFACE:lib/hasty.hpp>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/lib/hasty.hpp>)

get_target_property(HASTYLIB_PUBLIC_HEADERS HastyComputeLib INTERFACE_SOURCES)
set_target_properties(HastyComputeLib PROPERTIES PUBLIC_HEADER "${HASTYLIB_PUBLIC_HEADERS}")



# Linking CUDA
if (${BUILD_CUDA})
	find_package(CUDAToolkit REQUIRED)
	message("CUDA Build chosen - Found CUDAToolkit = ${CUDAToolkit_FOUND}")
	if (${CUDAToolkit_FOUND})
		message("CUDA Toolkit Version = ${CUDAToolkit_VERSION}")
	endif()

	if (NOT TARGET CUDA::nvtx3)
		message(FATAL_ERROR "Failed to find nvToolsExt here")
	endif()

	target_link_libraries(HastyComputeLib PUBLIC CUDA::toolkit)
	target_link_libraries(HastyComputeLib PUBLIC CUDA::nvrtc)
	target_link_libraries(HastyComputeLib PUBLIC CUDA::cudart)
	#target_link_libraries(HastyComputeLib PUBLIC CUDA::nvToolsExt)

	message("Linked CUDA Libs")
endif()


message("Toolchain file = ${CMAKE_TOOLCHAIN_FILE}")
message("BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message("Prefix = ${CMAKE_PREFIX_PATH}")


if (${BUILD_CUDA})
	set(USE_CUDA ON)
	#set(USE_CUDNN ON)
	#set(CAFFE2_USE_CUDNN TRUE)
endif()
find_package(Torch REQUIRED)
target_link_libraries(HastyComputeLib PUBLIC ${TORCH_LIBRARIES})


# Linking fiNUFFT
find_package(finufft REQUIRED)
target_link_libraries(HastyComputeLib PUBLIC FINUFFT::finufft)

if (${BUILD_CUDA})
	target_link_libraries(HastyComputeLib PUBLIC FINUFFT::cufinufft)
endif()

# Linking SymEngine
find_package(SymEngine CONFIG REQUIRED)
target_link_libraries(HastyComputeLib PUBLIC symengine)

find_package(HighFive CONFIG REQUIRED)
target_link_libraries(HastyComputeLib PUBLIC HighFive)


#target_compile_features(HastyComputeLib PUBLIC cxx_std_23)
#set_property(TARGET HastyComputeLib PROPERTY CXX_STANDARD 23)

if (MSVC)
  file(GLOB finufft_DLLS "${finufft_INSTALL_PREFIX}/../../../bin/*.dll")
  #message("finufft DLLS: ${finufft_DLLS}")
  add_custom_command(TARGET HastyComputeLib
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${finufft_DLLS}
                     $<TARGET_FILE_DIR:HastyComputeLib>)
endif()

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET HastyComputeLib
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:HastyComputeLib>)
endif()

# Tests

add_executable(HastyTest "tests/test.cpp")
target_link_libraries(HastyTest PUBLIC HastyComputeLib)
#target_compile_features(HastyTest PRIVATE cxx_std_23)




# Python Interface
#find_package(PythonLibs REQUIRED)
find_package(Python3 3.11 EXACT COMPONENTS Interpreter Development REQUIRED)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

add_library(HastyPyInterface SHARED 
	"python/cpp/py_batched_sense.cpp"
	"python/cpp/py_interface.cpp"
	"python/cpp/py_sense.cpp"
	"python/cpp/py_nufft.cpp"
	"python/cpp/py_svt.cpp"
)

#pybind11_add_module(HastyPyInterface SHARED "python/cpp/py_interface.cpp")
target_link_libraries(HastyPyInterface PUBLIC HastyComputeLib)
target_link_libraries(HastyPyInterface PUBLIC Python3::Python)
target_link_libraries(HastyPyInterface PUBLIC ${Python3_LIBRARY_RELEASE})
target_include_directories(HastyPyInterface PUBLIC ${Python3_INCLUDE_DIRS})
set_target_properties(HastyPyInterface PROPERTIES LINKER_LANGUAGE CXX)


set(HASTY_BUILD_PY_TESTS ON)
if (${HASTY_BUILD_PY_TESTS})
	find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")
	# Tests of the Python Interface
	add_executable(HastyPyTest "tests/py_test.cpp")
	target_link_libraries(HastyPyTest PUBLIC HastyComputeLib)
	target_link_libraries(HastyPyTest PUBLIC HastyPyInterface)
	target_include_directories(HastyPyTest PRIVATE ${MATPLOTLIB_CPP_INCLUDE_DIRS})
	set_target_properties(HastyPyTest PROPERTIES LINKER_LANGUAGE CXX)
endif()



set(HASTY_BUILD_VIZ ON)
if (${HASTY_BUILD_VIZ})
	find_package(glfw3 REQUIRED)
	find_package(Vulkan REQUIRED)
	find_package(imgui CONFIG REQUIRED)
	find_package(implot CONFIG REQUIRED)
	find_package(glm CONFIG REQUIRED)
	find_package(ktx CONFIG REQUIRED)
	find_package(unofficial-skia CONFIG REQUIRED)
	
	add_executable(HastyViz 
		"viz/viz.cpp" 
		"viz/skia.cpp"
		"viz/vizapp.cpp" 
		"viz/orthoslicer.cpp"
		"viz/slicer.cpp"

		"viz/base/VulkanDevice.cpp"
		"viz/base/VulkanBuffer.cpp"
		"viz/base/VulkanTools.cpp"
	)

	target_link_libraries(HastyViz PUBLIC HastyComputeLib)
	target_link_libraries(HastyViz PUBLIC Vulkan::Vulkan)
	target_link_libraries(HastyViz PUBLIC imgui::imgui)
	target_link_libraries(HastyViz PUBLIC implot::implot)
	target_link_libraries(HastyViz PUBLIC glm::glm)
	target_link_libraries(HastyViz PUBLIC KTX::ktx)
	target_link_libraries(HastyViz PUBLIC unofficial::skia::skia)
	target_link_libraries(HastyViz PUBLIC glfw)

	target_include_directories(HastyViz PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
	add_custom_command(
		TARGET HastyViz
		POST_BUILD
		COMMAND
			${CMAKE_COMMAND} -E copy_directory
			${CMAKE_SOURCE_DIR}/viz/shaders $<TARGET_FILE_DIR:HastyViz>/shaders)
	add_custom_command(
		TARGET HastyViz
		POST_BUILD
		COMMAND
			${CMAKE_COMMAND} -E copy_directory
			${CMAKE_SOURCE_DIR}/viz/fonts $<TARGET_FILE_DIR:HastyViz>/fonts)

endif()







#[[


install(TARGETS HastyComputeLib
	EXPORT hasty-targets
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	CXX_MODULES_BMI DESTINATION lib
	PUBLIC_HEADER DESTINATION include
	INCLUDES DESTINATION include
)

install(TARGETS HastyComputeLib
	COMPONENT HastyComputeLib
	RUNTIME_DEPENDENCIES
		PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
		POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
		DIRECTORIES 
			$<TARGET_FILE_DIR:HastyComputeLib>
			$ENV{AF_PATH}/lib
)

install(TARGETS HastyPyInterface 
	EXPORT hasty-targets
	RUNTIME DESTINATION bin
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	INCLUDES DESTINATION include
)

install(TARGETS HastyPyInterface 
	COMPONENT HastyPyInterface
	RUNTIME_DEPENDENCIES 
		PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
		POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
		DIRECTORIES 
			$<TARGET_FILE_DIR:HastyPyInterface>
			$ENV{PYTHON_PATH}
)

include(CMakePackageConfigHelpers)

# Creates FooConfigVersion.cmake in the build/Foo directory:
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/hasty/hastyConfigVersion.cmake"
  VERSION       ${hasty_VERSION}
  COMPATIBILITY AnyNewerVersion)


# Export the targets (change namespace appropriately):
export(EXPORT    hasty-targets
       FILE      "${CMAKE_CURRENT_BINARY_DIR}/hasty/hastyTargets.cmake"
       NAMESPACE HASTY::)

# Copy the FooConfig.cmake to the build/Foo directory:
configure_file(hastyConfig.cmake
               "${CMAKE_CURRENT_BINARY_DIR}/hasty/hastyConfig.cmake"
               COPYONLY)

set(ConfigPackageLocation lib/cmake/hasty)
install(EXPORT      hasty-targets
        FILE        hastyTargets.cmake
        NAMESPACE   HASTY::
        DESTINATION ${ConfigPackageLocation})

# This also installs relative to CMAKE_INSTALL_PREFIX:
install(
  FILES       hastyConfig.cmake
              "${CMAKE_CURRENT_BINARY_DIR}/hasty/hastyConfigVersion.cmake"
  DESTINATION ${ConfigPackageLocation})

]]

