cmake_minimum_required(VERSION 3.26)

project("HastyCompute")

set(BUILD_CUDA TRUE)

if (BUILD_CUDA)
	find_package(CUDAToolkit REQUIRED)
	message("CUDA Build chosen - Found CUDAToolkit = ${CUDAToolkit_FOUND}")
	if (${CUDAToolkit_FOUND})
		message("CUDA Toolkit Version = ${CUDAToolkit_VERSION}")
	endif()
endif()

find_package(ArrayFire)
find_package(SymEngine CONFIG REQUIRED)

if (BUILD_CUDA)
	set(BACKEND_SOURCES 
		"lib/hasty_cu.ixx"
		"lib/linalg/permute_cu.ixx"
		"lib/linalg/solver_cu.ixx"
	)
else()
	set(BACKEND_SOURCES
	"")
endif()

if (MSVC)
# Set STL Sources for module linking
set(PART_STL_SOURCES $ENV{VCToolsInstallDir})
cmake_path(CONVERT ${PART_STL_SOURCES} TO_CMAKE_PATH_LIST PART_STL_SOURCES)
set(STL_SOURCES "${PART_STL_SOURCES}modules/std.ixx")
unset(PART_STL_SOURCES)
message("STL Sources Found at: "${STL_SOURCES})
else()
set(STL_SOURCES "")
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

#if (BUILD_CUDA)
	#target_link_libraries(HastyComputeLib PUBLIC ArrayFire::afcuda)
#else()
	#target_link_libraries(HastyComputeLib PUBLIC ArrayFire::afopencl)
#endif()
target_link_libraries(HastyComputeLib PUBLIC ArrayFire::af)

target_link_libraries(HastyComputeLib PUBLIC symengine)

if (BUILD_CUDA)
	target_link_libraries(HastyComputeLib PUBLIC CUDA::toolkit)
	target_link_libraries(HastyComputeLib PUBLIC CUDA::nvrtc)
	target_link_libraries(HastyComputeLib PRIVATE CUDA::cudart)

	message("Linked CUDA Libs")
endif()

#set_property(TARGET HastyComputeLib PROPERTY CXX_STANDARD 23)
target_compile_features(HastyComputeLib PRIVATE cxx_std_23)


add_executable(HastyTest "tests/test.ixx")
target_link_libraries(HastyTest HastyComputeLib)


#set_property(TARGET HastyTest PROPERTY CXX_STANDARD 23)
target_compile_features(HastyTest PRIVATE cxx_std_23)



if (MSVC)
	#set(TARGET HastyComputeLib PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
