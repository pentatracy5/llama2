# test/core/CMakeLists.txt
# 递归收集 test/core 目录下所有 .cpp 和 .cu 源文件
file(GLOB_RECURSE TEST_CORE_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.cu")

# 构建 core 单元测试可执行文件
add_executable(test_core ${TEST_CORE_SOURCES})

# 链接 Google Test 库（gtest_main 会自动提供 main 函数）
target_link_libraries(test_core PRIVATE GTest::gtest_main)

# 注册测试，让 ctest 可以自动发现
include(GoogleTest)
gtest_discover_tests(test_core)
