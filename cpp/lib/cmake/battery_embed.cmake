set(DB_PRODUCTION_MODE ON)
FetchContent_Declare(
    battery-embed
    GIT_REPOSITORY https://github.com/batterycenter/embed.git
    GIT_TAG        v1.2.19
)
FetchContent_MakeAvailable(battery-embed)

b_embed(HastyCompute "src/fft/kernels/toeplitz_load_1D.cu")
b_embed(HastyCompute "src/fft/kernels/toeplitz_load_2D.cu")
b_embed(HastyCompute "src/fft/kernels/toeplitz_load_3D.cu")

# battery-embed injects its autogen include dir into HastyCompute's
# INTERFACE_INCLUDE_DIRECTORIES — strip it so consumers don't see internal paths.
get_target_property(_incs HastyCompute INTERFACE_INCLUDE_DIRECTORIES)
list(REMOVE_ITEM _incs "${EMBED_BINARY_DIR}/autogen/hastycompute/include")
set_target_properties(HastyCompute PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_incs}")
