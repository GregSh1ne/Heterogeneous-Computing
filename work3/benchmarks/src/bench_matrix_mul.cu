#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

#include "cuda_check.cuh"
#include "matmul_strategy.cuh"
#include "matrix.cuh"

// Бенчмарк блочного умножения с Shared Memory
static void BM_CUDA_Matmul_SharedMemory(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));

  Matrix<float> a(n, n);
  Matrix<float> b(n, n);
  Matrix<float> c(n, n);

  std::vector<float> init_data(n * n, 1.0f);
  a.data().copy_from_host(init_data.data());
  b.data().copy_from_host(init_data.data());

  // Устанавливаем стратегию Shared Memory
  Matrix<float>::SetStrategy(std::make_shared<SharedMemoryMatmulStrategy<float>>());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (auto _ : state) {
    CUDA_CHECK(cudaEventRecord(start, 0));

    c = a * b;

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    state.SetIterationTime(elapsed_ms / 1000.0f);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  state.SetItemsProcessed(
      static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(2 * n * n * n));
}

// Бенчмарк наивного умножения
static void BM_CUDA_Matmul_Naive(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));

  Matrix<float> a(n, n);
  Matrix<float> b(n, n);
  Matrix<float> c(n, n);

  std::vector<float> init_data(n * n, 1.0f);
  a.data().copy_from_host(init_data.data());
  b.data().copy_from_host(init_data.data());

  // Переключаем стратегию на Naive
  Matrix<float>::SetStrategy(std::make_shared<NaiveMatmulStrategy<float>>());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (auto _ : state) {
    CUDA_CHECK(cudaEventRecord(start, 0));

    c = a * b;

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    state.SetIterationTime(elapsed_ms / 1000.0f);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  state.SetItemsProcessed(
      static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(2 * n * n * n));
}

#define BENCH_ARGS ->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)

BENCHMARK(BM_CUDA_Matmul_SharedMemory)
BENCH_ARGS->UseManualTime()->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CUDA_Matmul_Naive)
BENCH_ARGS->UseManualTime()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();