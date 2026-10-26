add_executable(work3_tests src/test_matrix_mul.cu)
target_link_libraries(work3_tests PRIVATE
    work3_core
    Eigen3::Eigen
    GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(work3_tests)