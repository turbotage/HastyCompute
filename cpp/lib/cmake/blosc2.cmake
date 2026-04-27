set(BLOSC_INSTALL    OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS    OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS      OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    blosc2
    GIT_REPOSITORY https://github.com/Blosc/c-blosc2.git
    GIT_TAG v2.23.1
)
FetchContent_MakeAvailable(blosc2)

# Same interface-strip treatment: lz4/zstd/zlib are IMPORTED and safe to keep;
# strip any FetchContent sub-deps to prevent export-checker cascade.
if(TARGET blosc2_shared)
    set_property(TARGET blosc2_shared PROPERTY INTERFACE_COMPILE_OPTIONS "")
    get_target_property(_iface blosc2_shared INTERFACE_LINK_LIBRARIES)
    if(_iface)
        set(_clean "")
        foreach(_lib IN LISTS _iface)
            if(TARGET "${_lib}")
                get_target_property(_imp "${_lib}" IMPORTED)
                if(_imp)
                    list(APPEND _clean "${_lib}")
                endif()
            else()
                list(APPEND _clean "${_lib}")
            endif()
        endforeach()
        set_property(TARGET blosc2_shared PROPERTY INTERFACE_LINK_LIBRARIES "${_clean}")
    endif()
    set_target_properties(blosc2_shared PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

target_link_libraries(HastyCompute PUBLIC blosc2_shared)

install(TARGETS blosc2_shared
    EXPORT   HastyComputeTargets
    LIBRARY  DESTINATION lib
    RUNTIME  DESTINATION bin
    INCLUDES DESTINATION include
)
install(DIRECTORY ${blosc2_SOURCE_DIR}/include/ DESTINATION include)
