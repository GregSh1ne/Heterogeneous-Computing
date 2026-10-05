add_executable(work2_benchmarks src/bench_matrix_mul.cu)
target_link_libraries(work2_benchmarks PRIVATE work2_core benchmark::benchmark Eigen3::Eigen)