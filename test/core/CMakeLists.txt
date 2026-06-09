# 构建 Tensor 单元测试可执行文件
add_executable(test_tensor test_tensor.cu)

# 链接 Google Test 库（gtest_main 会自动提供 main 函数）
target_link_libraries(test_tensor PRIVATE GTest::gtest_main)

# 注册测试，让 ctest 可以自动发现
include(GoogleTest)
gtest_discover_tests(test_tensor)
