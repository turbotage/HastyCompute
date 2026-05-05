set(NIFTI_BUILD_TESTING   OFF CACHE BOOL "" FORCE)
set(NIFTI_INSTALL_NO_DOCS ON  CACHE BOOL "" FORCE)
set(USE_ITK_ZLIB          OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    nifti_clib
    GIT_REPOSITORY https://github.com/NIFTI-Imaging/nifti_clib.git
    GIT_TAG        v3.0.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nifti_clib)

target_link_libraries(HastyCompute PRIVATE niftiio)

install(TARGETS niftiio znz
    EXPORT   HastyComputeTargets
    LIBRARY  DESTINATION lib
    RUNTIME  DESTINATION bin
)
install(DIRECTORY ${nifti_clib_SOURCE_DIR}/niftilib/ DESTINATION include FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${nifti_clib_SOURCE_DIR}/znzlib/   DESTINATION include FILES_MATCHING PATTERN "*.h")
