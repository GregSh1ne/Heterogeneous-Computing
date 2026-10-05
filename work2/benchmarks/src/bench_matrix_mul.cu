#include <Eigen/Dense>
#include <algorithm>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <vector>

#include "cuda_check.cuh"
#include "matrix.cuh"

using RowMajorMatrixXf
    = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static void BM_CUDA_MatMul_Naive(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));

  Matrix<float> a(n, n);
  Matrix<float> b(n, n);
  Matrix<float> c(n, n);

  std::vector<float> host_data(n * n, 1.0f);
  a.data().copy_from_host(host_data.data());
  b.data().copy_from_host(host_data.data());

  constexpr dim3 block_dim(16, 16);
  const dim3 grid_dim(static_cast<unsigned int>((n + block_dim.x - 1) / block_dim.x),
      static_cast<unsigned int>((n + block_dim.y - 1) / block_dim.y));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (auto _ : state) {
    CUDA_CHECK(cudaEventRecord(start, 0));

    kernel_matmul_naive<float><<<grid_dim, block_dim>>>(a.view(), b.view(), c.view());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    const float safe_ms = std::max(elapsed_ms, 0.0001f);
    state.SetIterationTime(safe_ms / 1000.0f);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 2 * n * n * n);
}

// CPU Benchmark на базе Eigen
static void BM_Eigen_MatMul(benchmark::State& state) {
  const auto n = static_cast<Eigen::Index>(state.range(0));

  RowMajorMatrixXf a = RowMajorMatrixXf::Random();
  RowMajorMatrixXf b = RowMajorMatrixXf::Random();
  RowMajorMatrixXf c(n, n);

  for (auto _ : state) {
    // noalias() исключает создание промежуточных матриц в динамической памяти
    c.noalias() = a * b;
    benchmark::DoNotOptimize(c.data());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 2 * n * n * n);
}

#define MATMUL_ARGS ->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)

BENCHMARK(BM_CUDA_MatMul_Naive)
MATMUL_ARGS->UseManualTime()->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Eigen_MatMul)
MATMUL_ARGS->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();