add_executable(run_tests
    src/test_vector_add.cu
)

target_link_libraries(run_tests PRIVATE
    hsys_core
    Eigen3::Eigen
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(run_tests)