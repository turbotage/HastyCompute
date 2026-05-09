set(HDF5_BUILD_TESTS           OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_TOOLS           OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_EXAMPLES        OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_CPP_LIB         OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_HL_LIB          OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_Z_LIB_SUPPORT  OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_PLUGIN_SUPPORT OFF CACHE BOOL "" FORCE)
# Force shared libs ON before FetchContent so hdf5-shared target is always created.
# Some earlier FetchContent deps (e.g. blosc2/zlib-ng) leave BUILD_SHARED_LIBS in a
# state that defeats HDF5's option() default of ON.
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    hdf5
    GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
    GIT_TAG        hdf5_1.14.6
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(hdf5)

set(HIGHFIVE_USE_BOOST  OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_FIND_HDF5  OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_USE_HDF5   OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    HighFive
    GIT_REPOSITORY https://github.com/highfive-devs/highfive.git
    GIT_TAG        v3.3.0
    GIT_SHALLOW    TRUE
    GIT_CONFIG     "credential.helper=" "core.askPass="
)
FetchContent_MakeAvailable(HighFive)

# Wire HDF5 directly into HastyCompute (not via HighFive's INTERFACE) so the
# export checker doesn't cascade into hdf5's sub-dep graph.
target_include_directories(HastyCompute PRIVATE
    $<BUILD_INTERFACE:${hdf5_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${hdf5_BINARY_DIR}/src>
)
target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:HighFive>)
target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:hdf5-shared>)

install(TARGETS hdf5-shared
    EXPORT   HastyComputeTargets
    LIBRARY  DESTINATION lib
    RUNTIME  DESTINATION bin
)
# HDF5 source tree has the C API; build tree has the generated H5pubconf.h.
install(DIRECTORY ${hdf5_SOURCE_DIR}/src/ DESTINATION include FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${hdf5_BINARY_DIR}/src/ DESTINATION include FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${highfive_SOURCE_DIR}/include/ DESTINATION include)
