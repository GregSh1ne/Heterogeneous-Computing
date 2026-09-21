add_executable(run_benchmarks
    src/bench_vector_add.cu
)

target_link_libraries(run_benchmarks PRIVATE
    hsys_core
    Eigen3::Eigen
    benchmark::benchmark
)