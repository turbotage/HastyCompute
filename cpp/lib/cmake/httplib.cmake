# cpp-httplib — header-only HTTP/1.1 server+client, TLS via system OpenSSL.
find_package(OpenSSL REQUIRED)
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE ON  CACHE BOOL "" FORCE)
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
target_link_libraries(HastyCompute PUBLIC OpenSSL::SSL OpenSSL::Crypto)
install(FILES ${httplib_SOURCE_DIR}/httplib.h DESTINATION include)
