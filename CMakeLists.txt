# CMake 最低版本号要求
cmake_minimum_required (VERSION 4.3.3)

# 项目信息
project (Llama2 VERSION 1.0 LANGUAGES CXX CUDA)

# 查找 CUDA Toolkit，提供 CUDA::cudart 等 imported targets
find_package(CUDAToolkit REQUIRED)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 包含公共头文件目录（让所有子目标都能找到）
include_directories(${CMAKE_SOURCE_DIR}/include)

# 启用测试
enable_testing()

# 添加库子目录
add_subdirectory(src)
add_subdirectory(test)