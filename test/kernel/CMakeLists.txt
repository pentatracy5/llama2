# 构建 kernel 单元测试可执行文件
add_executable(test_kernel test_embedding.cu)

# 链接 kernel 库和 Google Test 库
target_link_libraries(test_kernel PRIVATE kernel GTest::gtest_main)

set_target_properties(test_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 注册测试，让 ctest 可以自动发现
include(GoogleTest)
gtest_discover_tests(test_kernel)
