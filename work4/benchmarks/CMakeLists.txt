add_executable(work4_benchmarks src/bench_matrix_mul.cu)
target_link_libraries(work4_benchmarks PRIVATE work4_core benchmark::benchmark Eigen3::Eigen)