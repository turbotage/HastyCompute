# cpp-httplib — header-only HTTP/1.1 server+client, no TLS deps.
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE   OFF CACHE BOOL "" FORCE)
set(HTTPLIB_COMPILE                  OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.7
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(httplib)

target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:httplib::httplib>)
install(FILES ${httplib_SOURCE_DIR}/httplib.h DESTINATION include)
