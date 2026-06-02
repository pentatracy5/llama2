# CMake 最低版本号要求
cmake_minimum_required (VERSION 4.3.3)

# 项目信息
project (Demo)

# 指定生成目标
add_executable(Demo main.cpp MathFunctions.cpp)

# 把当前源码路径加入头文件搜索路径
target_include_directories(Demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})