add_executable(work3_benchmarks src/bench_matrix_mul.cu)
target_link_libraries(work3_benchmarks PRIVATE
    work3_core
    benchmark::benchmark
)