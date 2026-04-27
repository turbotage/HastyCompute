FetchContent_Declare(
    plotlypp
    GIT_REPOSITORY https://github.com/jimmyorourke/plotlypp.git
    GIT_TAG main
)
FetchContent_MakeAvailable(plotlypp)

# Header-only — keep PRIVATE/BUILD_INTERFACE so it never appears in the
# export checker's walk; headers reach consumers via HastyCompute's install.
target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:plotlypp::plotlypp>)

install(DIRECTORY ${plotlypp_SOURCE_DIR}/include/ DESTINATION include)
