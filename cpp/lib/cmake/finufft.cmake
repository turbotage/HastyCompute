set(FINUFFT_USE_CUDA        ON  CACHE BOOL "" FORCE)
set(FINUFFT_USE_CPU         ON  CACHE BOOL "" FORCE)
set(FINUFFT_USE_DUCC0       ON  CACHE BOOL "" FORCE)
set(FINUFFT_STATIC_LINKING  OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    finufft
    GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git
    GIT_TAG        v2.5.1
)
FetchContent_MakeAvailable(finufft)

# finufft/cufinufft carry OpenMP as a PUBLIC dep which bleeds -fopenmp into
# HastyCompute's C++ module BMI compilations, breaking the std module.
# Strip all non-IMPORTED interface deps and compile options — the .so files
# carry their own runtime deps via ELF NEEDED entries.
foreach(_tgt finufft cufinufft)
    if(TARGET ${_tgt})
        set_property(TARGET ${_tgt} PROPERTY INTERFACE_COMPILE_OPTIONS "")

        get_target_property(_iface ${_tgt} INTERFACE_LINK_LIBRARIES)
        if(_iface)
            set(_clean "")
            foreach(_lib IN LISTS _iface)
                if("${_lib}" MATCHES "[Oo]pen[Mm][Pp]|omp")
                    continue()
                endif()
                if(TARGET "${_lib}")
                    get_target_property(_imp "${_lib}" IMPORTED)
                    if(_imp)
                        list(APPEND _clean "${_lib}")
                    endif()
                else()
                    list(APPEND _clean "${_lib}")
                endif()
            endforeach()
            set_property(TARGET ${_tgt} PROPERTY INTERFACE_LINK_LIBRARIES "${_clean}")
        endif()
        set_target_properties(${_tgt} PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
endforeach()

target_link_libraries(HastyCompute PUBLIC finufft cufinufft)

# Stage finufft .so files next to libHastyCompute.so
add_custom_command(TARGET HastyCompute POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:finufft>   "$<TARGET_FILE_DIR:HastyCompute>/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:cufinufft> "$<TARGET_FILE_DIR:HastyCompute>/"
)

install(TARGETS finufft cufinufft
    EXPORT   HastyComputeTargets
    LIBRARY  DESTINATION lib
    RUNTIME  DESTINATION bin
    INCLUDES DESTINATION include
)
install(DIRECTORY ${finufft_SOURCE_DIR}/include/ DESTINATION include)
