cmake_minimum_required(VERSION 3.22.1)
project(vn97_android_runtime LANGUAGES CXX)

get_filename_component(
    VN97_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../../.."
    ABSOLUTE
)

set(VN97_BUILD_TESTS OFF CACHE BOOL "Build VN97 native tests" FORCE)
add_subdirectory(
    "${VN97_ROOT}/native"
    "${CMAKE_BINARY_DIR}/vn97-native"
)

add_library(vn97_jni SHARED vn97_jni.cpp vn97_model_jni.cpp vn97_generation_jni.cpp)
target_compile_features(vn97_jni PRIVATE cxx_std_17)
target_compile_options(vn97_jni PRIVATE -Wall -Wextra -Werror)
target_include_directories(vn97_jni PRIVATE "${VN97_ROOT}/native/include")
target_link_libraries(vn97_jni PRIVATE vn97_runtime vn97_model_image vn97_generation)
