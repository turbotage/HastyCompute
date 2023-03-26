cmake_minimum_required(VERSION 3.24)

project("HastyCompute")


find_package(ArrayFire)

add_library(HastyComputeLib
	"lib/hasty_compute.ixx"
	"lib/linalg/permute.ixx"
	"lib/linalg/solver.ixx"
)

set_property(TARGET HastyComputeLib PROPERTY CXX_STANDARD 20)

 
target_link_libraries(HastyComputeLib ArrayFire::af)

add_executable(HastyTest "tests/test.ixx")
target_link_libraries(HastyTest HastyComputeLib)
set_property(TARGET HastyTest PROPERTY CXX_STANDARD 20)
