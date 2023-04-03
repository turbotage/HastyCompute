cmake_minimum_required(VERSION 3.26)

project("HastyCompute")


set(CMAKE_VERBOSE_MAKEFILE ON)

set(BUILD_CUDA TRUE)

if (BUILD_CUDA)
	find_package(CUDAToolkit REQUIRED)
	message("CUDA Build chosen - Found CUDAToolkit = ${CUDAToolkit_FOUND}")
	if (${CUDAToolkit_FOUND})
		message("CUDA Toolkit Version = ${CUDAToolkit_VERSION}")
	endif()
endif()

if (BUILD_CUDA)
	set(BACKEND_SOURCES 
		"lib/hasty_cu.ixx"
		"lib/linalg/permute_cu.ixx"
		"lib/linalg/solver_cu.ixx"
	)
endif()



set(STL_AS_MODULES false)
if (${STL_AS_MODULES})
	add_compile_definitions("STL_AS_MODULES")
endif()

if(${STL_AS_MODULES})
	if (MSVC)
		#set(TARGET HastyComputeLib PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
		set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

		#if (MSVC)
		# Set STL Sources for module linking
		#set(PART_STL_SOURCES $ENV{VCToolsInstallDir})
		#cmake_path(CONVERT ${PART_STL_SOURCES} TO_CMAKE_PATH_LIST PART_STL_SOURCES)
		#set(STL_SOURCES 
		#	"${PART_STL_SOURCES}modules/std.ixx"
		#	"${PART_STL_SOURCES}modules/std.compat.ixx"
		#)
		#unset(PART_STL_SOURCES)
		#set(PART_STL_SOURCES $ENV{USERPROFILE})
		#cmake_path(CONVERT ${PART_STL_SOURCES} TO_CMAKE_PATH_LIST PART_STL_SOURCES)
		#set(STL_SOURCES
		#	"${PART_STL_SOURCES}/source/repos/STLModules/std.obj"
		#	"${PART_STL_SOURCES}/source/repos/STLModules/std.compat.obj"
		#)
		#unset(PART_STL_SOURCES)
		#message("STL Sources Found at: "${STL_SOURCES})
		#else()
		#endif()

	endif()
endif()



add_library(HastyComputeLib
	${STL_SOURCES}

	"lib/hasty_util.ixx"
	"lib/hasty_compute.ixx"

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

	${BACKEND_SOURCES}
)


if (BUILD_CUDA)
	target_link_libraries(HastyComputeLib PUBLIC CUDA::toolkit)
	target_link_libraries(HastyComputeLib PUBLIC CUDA::nvrtc)
	target_link_libraries(HastyComputeLib PRIVATE CUDA::cudart)
	target_link_libraries(HastyComputeLib PRIVATE CUDA::nvtx3)

	message("Linked CUDA Libs")
endif()

find_package(SymEngine CONFIG REQUIRED)
target_link_libraries(HastyComputeLib PUBLIC symengine)

find_package(ArrayFire)
target_link_libraries(HastyComputeLib PUBLIC ArrayFire::af)

set(CAFFE2_USE_CUDNN true)
find_package(Torch REQUIRED)
target_link_libraries(HastyComputeLib PUBLIC ${TORCH_LIBRARIES})


#set_property(TARGET HastyComputeLib PROPERTY CXX_STANDARD 23)
target_compile_features(HastyComputeLib PRIVATE cxx_std_23)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET HastyComputeLib
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:HastyComputeLib>)
endif (MSVC)








add_executable(HastyTest "tests/test.ixx")
target_link_libraries(HastyTest HastyComputeLib)

#set_property(TARGET HastyTest PROPERTY CXX_STANDARD 23)
target_compile_features(HastyTest PRIVATE cxx_std_23)

