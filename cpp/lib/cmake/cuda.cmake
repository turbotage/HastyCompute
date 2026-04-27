find_package(CUDAToolkit REQUIRED)

foreach(_tgt toolkit nvrtc cudart cufft cuda_driver)
    if(TARGET CUDA::${_tgt})
        target_link_libraries(HastyCompute PRIVATE CUDA::${_tgt})
    endif()
endforeach()

# Stage CUDA runtime .so files next to libHastyCompute.so so the install tree
# is self-contained.  Variables NVRTC_PATH / CUDART_PATH are used by install().
if(TARGET CUDA::nvrtc OR TARGET CUDA::cudart)
    if(TARGET CUDA::nvrtc)
        get_target_property(NVRTC_PATH CUDA::nvrtc LOCATION)
        get_filename_component(NVRTC_FILENAME "${NVRTC_PATH}" NAME)
        add_custom_command(TARGET HastyCompute POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${NVRTC_PATH}" "$<TARGET_FILE_DIR:HastyCompute>/${NVRTC_FILENAME}"
        )
    endif()
    if(TARGET CUDA::cudart)
        get_target_property(CUDART_PATH CUDA::cudart LOCATION)
        get_filename_component(CUDART_FILENAME "${CUDART_PATH}" NAME)
        add_custom_command(TARGET HastyCompute POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CUDART_PATH}" "$<TARGET_FILE_DIR:HastyCompute>/${CUDART_FILENAME}"
        )
    endif()
endif()
