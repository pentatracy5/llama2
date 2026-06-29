# src/kernel/CMakeLists.txt
# 收集 kernel 目录下所有 .cpp 和 .cu 源文件
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.cu")

# 构建 kernel 静态库
add_library(kernel STATIC ${KERNEL_SOURCES})

# 设置 CUDA 分离编译（如果有 .cu 文件跨目标调用需要）
set_target_properties(kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 链接 CUDA 运行时和 cuBLAS
target_link_libraries(kernel PUBLIC CUDA::cudart CUDA::cublas)
