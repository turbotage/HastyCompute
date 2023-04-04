cmake_minimum_required(VERSION 3.25)

project("HastyCompute")


set(CMAKE_VERBOSE_MAKEFILE ON)

set(BUILD_CUDA TRUE)


if (${BUILD_CUDA})
	enable_language(CUDA)
	set(BACKEND_SOURCES 
		"lib/hasty_cu.ixx"
		"lib/linalg/permute_cu.ixx"
		"lib/linalg/solver_cu.ixx"
	)
endif()



set(STL_AS_MODULES FALSE)
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



#add_library(HastyComputeLib
add_executable(HastyComputeLib
	"tests/test.ixx"

	${STL_SOURCES}

	"lib/metadata.ixx"

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

	"lib/expr/symbolic.ixx"

	"lib/nlsq/nlsq_symbolic.ixx"

	${BACKEND_SOURCES}
)

# Linking CUDA
if (${BUILD_CUDA})
	set(CMAKE_CUDA_ARCHITECTURES 90)

	find_package(CUDAToolkit REQUIRED)
	message("CUDA Build chosen - Found CUDAToolkit = ${CUDAToolkit_FOUND}")
	if (${CUDAToolkit_FOUND})
		message("CUDA Toolkit Version = ${CUDAToolkit_VERSION}")
	endif()

	target_link_libraries(HastyComputeLib PUBLIC CUDA::toolkit)
	target_link_libraries(HastyComputeLib PUBLIC CUDA::nvrtc)
	target_link_libraries(HastyComputeLib PUBLIC CUDA::cudart)

	message("Linked CUDA Libs")
endif()

# Linking SymEngine
find_package(SymEngine CONFIG REQUIRED)
target_link_libraries(HastyComputeLib PUBLIC symengine)

# Linking ArrayFire
find_package(ArrayFire)
target_link_libraries(HastyComputeLib PUBLIC ArrayFire::af)

if (${BUILD_CUDA})
	set(CAFFE2_USE_CUDNN TRUE)
endif()
find_package(Torch REQUIRED)
target_link_libraries(HastyComputeLib PUBLIC ${TORCH_LIBRARIES})

# Linking fiNUFFT
if (${BUILD_CUDA})
	set(FINUFFT_USE_CUDA ON)
endif()
add_subdirectory("deps/finufft")
target_link_libraries(HastyComputeLib PUBLIC finufft)
if (${BUILD_CUDA})
	target_link_libraries(HastyComputeLib PUBLIC cufinufft)
endif()

#set_property(TARGET HastyComputeLib PROPERTY CXX_STANDARD 23)
target_compile_features(HastyComputeLib PRIVATE cxx_std_23)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET HastyComputeLib
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:HastyComputeLib>)
endif()








#add_executable(HastyTest "tests/test.ixx")
#target_link_libraries(HastyTest HastyComputeLib)
#target_compile_features(HastyTest PRIVATE cxx_std_23)

