add_executable(work4_tests src/test_matrix_mul.cu)
target_link_libraries(work4_tests PRIVATE work4_core GTest::gtest_main Eigen3::Eigen)

include(GoogleTest)
gtest_discover_tests(work4_tests)