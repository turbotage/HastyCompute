cmake_minimum_required(VERSION 4.2.0 FATAL_ERROR)

#set(CMAKE_VERBOSE_MAKEFILE ON)

#set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
set(CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN "/opt/gcc-15")

project(HastyTest VERSION 0.1 LANGUAGES CXX)

add_executable(HastyTest "src/main.cpp")

target_compile_features(HastyTest 
	PRIVATE cxx_std_23
	INTERFACE cxx_std_23
)

set_target_properties(HastyTest PROPERTIES
	CXX_STANDARD 23
	CXX_STANDARD_REQUIRED ON
	CXX_SCAN_FOR_MODULES ON
	CXX_MODULE_STD ON
)

find_package(HastyCompute REQUIRED)
target_link_libraries(HastyTest PRIVATE
	HastyCompute::HastyCompute
)
