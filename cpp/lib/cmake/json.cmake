FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Header-only — same rationale as plotlypp.
target_link_libraries(HastyCompute PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)

install(DIRECTORY ${json_SOURCE_DIR}/include/ DESTINATION include)
