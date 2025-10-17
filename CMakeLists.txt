cmake_minimum_required(VERSION 3.10)
project(text_detector LANGUAGES C)

set(CMAKE_C_STANDARD 11)
file(GLOB SOURCE_FILES "${CMAKE_SOURCE_DIR}/src/*.c")

add_executable(mobileone
    ${SOURCE_FILES}
)

target_include_directories(mobileone
    PRIVATE
    include
)

target_compile_options(mobileone
    PRIVATE
    -Os
    -fno-builtin
)

target_link_options(mobileone
    PRIVATE
    -nodefaultlibs
    -flto
)

target_link_libraries(mobileone
    PRIVATE
    c
)
