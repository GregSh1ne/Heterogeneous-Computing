#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <vector>

#include "cuda_check.cuh"
#include "vector.cuh"

static void BM_CUDA_VectorAdd(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));

  Vector<float> a(n);
  Vector<float> b(n);
  Vector<float> c(n);

  std::vector<float> init_data(n, 1.0f);
  a.data().copy_from_host(init_data.data());
  b.data().copy_from_host(init_data.data());

  constexpr unsigned int block_size = 256;
  const unsigned int grid_size
      = static_cast<unsigned int>((n + block_size - 1) / block_size);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (auto _ : state) {
    CUDA_CHECK(cudaEventRecord(start, 0));

    kernel_vecadd<float><<<grid_size, block_size>>>(a.view(), b.view(), c.view());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    state.SetIterationTime(elapsed_ms / 1000.0f);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(n) * 3 * sizeof(float));
  state.SetItemsProcessed(
      static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_Eigen_VectorAdd(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));

  Eigen::VectorXf a = Eigen::VectorXf::Constant(static_cast<Eigen::Index>(n), 1.0f);
  Eigen::VectorXf b = Eigen::VectorXf::Constant(static_cast<Eigen::Index>(n), 1.0f);
  Eigen::VectorXf c(n);

  for (auto _ : state) {
    c = a + b;
    benchmark::DoNotOptimize(c.data());
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(n) * 3 * sizeof(float));
  state.SetItemsProcessed(
      static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

#define BENCH_ARGS                                                                       \
  ->Arg(8)->Arg(64)->Arg(512)->Arg(4096)->Arg(32768)->Arg(262144)->Arg(2097152)->Arg(    \
      16777216)

BENCHMARK(BM_CUDA_VectorAdd)
BENCH_ARGS->UseManualTime()->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Eigen_VectorAdd)
BENCH_ARGS->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();