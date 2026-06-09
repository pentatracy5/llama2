# src/CMakeLists.txt
# 添加 core 子模块
add_subdirectory(core)

# 主可执行文件
add_executable(Llama2 main.cpp)

# 链接 core 库
# PRIVATE 表示 Llama2 自己用 core，但依赖 Llama2 的目标不需要知道 core
target_link_libraries(Llama2 PRIVATE core)
