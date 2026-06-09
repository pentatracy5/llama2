# src/core/CMakeLists.txt
# 收集 core 目录下所有 .cpp 和 .cu 源文件
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.cu")

# 构建 core 静态库
add_library(core STATIC ${CORE_SOURCES})

# 设置 CUDA 分离编译（如果有 .cu 文件跨目标调用需要）
set_target_properties(core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 链接 CUDA 运行时（Timer 等模块依赖 cudaEvent/cudaStream）
target_link_libraries(core PUBLIC CUDA::cudart)
