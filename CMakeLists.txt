cmake_minimum_required(VERSION 3.24)

project("HastyCompute")



set(BUILD_SHARED_LIBS ON)
if (MSVC)
	#set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
#set(CMAKE_VERBOSE_MAKEFILE ON)

option(BUILD_CUDA OFF)

message("CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE}")
message("CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")
message("BUILD_CUDA = ${BUILD_CUDA}")

if (${BUILD_CUDA})
	set(BACKEND_SOURCES
		"lib/hasty_cu.ixx"
		"lib/linalg/permute_cu.ixx"
		"lib/linalg/solver_cu.ixx"
		"lib/fft/nufft_cu.cpp"
		"lib/cs/sense_cu.cpp"
		"lib/cs/llr_cu.cpp"
	)
	set(BACKEND_HEADERS
		"lib/fft/nufft_cu.hpp"
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



set(TEST_SOURCES
	"lib/tests/tests1.cpp"
)

add_library(HastyComputeLib SHARED
	${STL_SOURCES}

	"lib/metadata.ixx"
	"lib/hasty_util.ixx"
	"lib/hasty_compute.ixx"
	"lib/torch_util.cpp"

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

	${TEST_SOURCES}
)

# Linking CUDA
if (${BUILD_CUDA})
	find_package(CUDAToolkit REQUIRED)
	message("CUDA Build chosen - Found CUDAToolkit = ${CUDAToolkit_FOUND}")
	if (${CUDAToolkit_FOUND})
		message("CUDA Toolkit Version = ${CUDAToolkit_VERSION}")
	endif()

	target_link_libraries(HastyComputeLib PUBLIC CUDA::toolkit)
	target_link_libraries(HastyComputeLib PUBLIC CUDA::nvrtc)
	target_link_libraries(HastyComputeLib PUBLIC CUDA::cudart)
	target_link_libraries(HastyComputeLib PUBLIC CUDA::nvToolsExt)

	message("Linked CUDA Libs")
endif()


message("Toolchain file = ${CMAKE_TOOLCHAIN_FILE}")
message("BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message("Prefix = ${CMAKE_PREFIX_PATH}")

# Linking ArrayFire
find_package(ArrayFire REQUIRED)
target_include_directories(HastyComputeLib PUBLIC ${ArrayFire_INCLUDE_DIRS})
target_link_libraries(HastyComputeLib PUBLIC ArrayFire::af)
set_target_properties(${ArrayFire_CUDA_LIBRARIES} PROPERTIES
	MAP_IMPORTED_CONFIG_DEBUG Release
	MAP_IMPORTED_CONFIG_RELEASE Release
	MAP_IMPORTED_CONFIG_MINSIZEREL Release
)



if (${BUILD_CUDA})
	set(USE_CUDA ON)
	set(USE_CUDNN ON)
	set(CAFFE2_USE_CUDNN TRUE)
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



#set_property(TARGET HastyComputeLib PROPERTY CXX_STANDARD 23)
target_compile_features(HastyComputeLib PUBLIC cxx_std_20)
set_property(TARGET HastyComputeLib PROPERTY CXX_STANDARD 20)

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

set_target_properties(HastyComputeLib PROPERTIES PUBLIC_HEADER "lib/hasty.hpp")
install(TARGETS HastyComputeLib
	LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
	ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
	PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/include")












add_executable(HastyTest "tests/test.ixx")
target_link_libraries(HastyTest HastyComputeLib)
target_compile_features(HastyTest PRIVATE cxx_std_23)

