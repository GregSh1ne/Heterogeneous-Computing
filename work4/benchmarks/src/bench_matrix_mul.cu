#include <benchmark/benchmark.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "cuda_check.cuh"
#include "matmul_strategy.cuh"
#include "matrix.cuh"

namespace {

void FillMatrixHost(Matrix<__half>& d_mat, float value) {
  std::vector<__half> h_buf(d_mat.size(), __float2half(value));
  d_mat.data().copy_from_host(h_buf.data());
}

}  // namespace

// Предыдущая реализация operator* (Shared Memory из Работы №3)
static void BM_CUDA_Matmul_SharedMemory(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));

  Matrix<__half>::SetStrategy(
      std::make_shared<SharedMemoryMatmulStrategy<__half>>());

  Matrix<__half> a(n, n);
  Matrix<__half> b(n, n);
  FillMatrixHost(a, 0.5f);
  FillMatrixHost(b, 0.5f);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Прогрев
  Matrix<__half> warmup = a * b;
  (void)warmup;

  for (auto _ : state) {
    CUDA_CHECK(cudaEventRecord(start));
    Matrix<__half> c = a * b;
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
    benchmark::DoNotOptimize(c);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

// Новая реализация operator* (CUDA WMMA Tensor Cores, Работа №4)
static void BM_CUDA_Matmul_WMMA(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));

  Matrix<__half>::SetStrategy(std::make_shared<WmmaMatmulStrategy<__half>>());

  Matrix<__half> a(n, n);
  Matrix<__half> b(n, n);
  FillMatrixHost(a, 0.5f);
  FillMatrixHost(b, 0.5f);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Прогрев
  Matrix<__half> warmup = a * b;
  (void)warmup;

  for (auto _ : state) {
    CUDA_CHECK(cudaEventRecord(start));
    Matrix<__half> c = a * b;
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
    benchmark::DoNotOptimize(c);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

BENCHMARK(BM_CUDA_Matmul_SharedMemory)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CUDA_Matmul_WMMA)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();