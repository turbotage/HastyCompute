cmake_minimum_required(VERSION 4.2.0 FATAL_ERROR)

#set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")

set(CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN "/opt/gcc-15")

project(HastyTest VERSION 0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
find_package(HastyServer REQUIRED)
target_link_libraries(HastyTest PRIVATE
	HastyCompute::HastyCompute
	HastyServer::HastyServer
)
# lld doesn't resolve transitive shared-lib deps; link directly against the
# LibTorch libs that HastyCompute installed alongside itself.
foreach(_lib c10 torch torch_cpu c10_cuda torch_cuda)
    find_library(_found_lib ${_lib})
    if(_found_lib)
        target_link_libraries(HastyTest PRIVATE ${_found_lib})
    endif()
    unset(_found_lib CACHE)
endforeach()

install(TARGETS HastyTest RUNTIME DESTINATION bin)