FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG        12.0.0
)
FetchContent_MakeAvailable(fmt)

# Same interface-strip treatment as finufft to prevent cascade into fmt's
# FetchContent sub-dep graph when cmake 4.x walks the export set.
if(TARGET fmt)
    set_property(TARGET fmt PROPERTY INTERFACE_COMPILE_OPTIONS "")
    get_target_property(_iface fmt INTERFACE_LINK_LIBRARIES)
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
        set_property(TARGET fmt PROPERTY INTERFACE_LINK_LIBRARIES "${_clean}")
    endif()
    set_target_properties(fmt PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

target_link_libraries(HastyCompute PUBLIC fmt::fmt)

install(TARGETS fmt
    EXPORT   HastyComputeTargets
    LIBRARY  DESTINATION lib
    ARCHIVE  DESTINATION lib
    RUNTIME  DESTINATION bin
    INCLUDES DESTINATION include
)
install(DIRECTORY ${fmt_SOURCE_DIR}/include/ DESTINATION include)
