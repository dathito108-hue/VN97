cmake_minimum_required(VERSION 3.16)
project(vn97_native LANGUAGES CXX)

set(VN97_PACKED_TERNARY_SOURCES
    src/packed_ternary.cpp
)
set(VN97_RECURRENT_SOURCES
    src/recurrent.cpp
)

set(VN97_ARM64 FALSE)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" VN97_SYSTEM_PROCESSOR_LOWER)
if(VN97_SYSTEM_PROCESSOR_LOWER MATCHES "^(aarch64|arm64)$")
    set(VN97_ARM64 TRUE)
endif()
if(DEFINED CMAKE_ANDROID_ARCH_ABI AND CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    set(VN97_ARM64 TRUE)
endif()
if(APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    set(VN97_ARM64 TRUE)
endif()

if(VN97_ARM64)
    list(APPEND VN97_PACKED_TERNARY_SOURCES src/packed_ternary_neon.cpp)
    list(APPEND VN97_RECURRENT_SOURCES src/recurrent_neon.cpp)
endif()

add_library(vn97_packed_ternary STATIC ${VN97_PACKED_TERNARY_SOURCES})
target_include_directories(vn97_packed_ternary PUBLIC include)
target_compile_features(vn97_packed_ternary PUBLIC cxx_std_17)

add_library(vn97_recurrent STATIC ${VN97_RECURRENT_SOURCES})
target_include_directories(vn97_recurrent PUBLIC include)
target_compile_features(vn97_recurrent PUBLIC cxx_std_17)

if(VN97_ARM64)
    target_compile_definitions(vn97_packed_ternary PRIVATE VN97_HAS_ARM64_NEON=1)
    target_compile_definitions(vn97_recurrent PRIVATE VN97_HAS_ARM64_NEON=1)
endif()

option(VN97_BUILD_TESTS "Build VN97 native tests" ON)
if(VN97_BUILD_TESTS)
    enable_testing()

    add_executable(vn97_packed_ternary_test tests/packed_ternary_test.cpp)
    target_link_libraries(vn97_packed_ternary_test PRIVATE vn97_packed_ternary)
    add_test(NAME vn97_packed_ternary_test COMMAND vn97_packed_ternary_test)

    add_executable(vn97_recurrent_test tests/recurrent_test.cpp)
    target_link_libraries(vn97_recurrent_test PRIVATE vn97_recurrent)
    add_test(NAME vn97_recurrent_test COMMAND vn97_recurrent_test)

    add_library(vn97_recurrent_test_bridge SHARED ${VN97_RECURRENT_SOURCES})
    target_include_directories(vn97_recurrent_test_bridge PRIVATE include)
    target_compile_features(vn97_recurrent_test_bridge PRIVATE cxx_std_17)
    if(VN97_ARM64)
        target_compile_definitions(
            vn97_recurrent_test_bridge PRIVATE VN97_HAS_ARM64_NEON=1)
    endif()

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(
        NAME vn97_recurrent_python_equivalence
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../src"
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/recurrent_python_equivalence.py
                $<TARGET_FILE:vn97_recurrent_test_bridge>
    )
endif()
