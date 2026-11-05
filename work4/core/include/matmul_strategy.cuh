#ifndef WORK4_CORE_MATMUL_STRATEGY_CUH_
#define WORK4_CORE_MATMUL_STRATEGY_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <type_traits>

#include "cuda_check.cuh"
#include "kernel_matmul_naive.cuh"
#include "kernel_matmul_shmem.cuh"
#include "kernel_matmul_wmma.cuh"
#include "matrix_view.cuh"

template <typename AtomT>
class IMatmulStrategy {
 public:
  virtual ~IMatmulStrategy() = default;
  virtual void Multiply(const MatrixView<AtomT>& a, const MatrixView<AtomT>& b,
                        MatrixView<AtomT>& c) const = 0;
};

// Стратегия 1: Наивное умножение (Работа №2)
template <typename AtomT>
class NaiveMatmulStrategy : public IMatmulStrategy<AtomT> {
 public:
  void Multiply(const MatrixView<AtomT>& a, const MatrixView<AtomT>& b,
                MatrixView<AtomT>& c) const override {
    constexpr dim3 kBlockDim(16, 16);
    const dim3 kGridDim(
        static_cast<unsigned int>((c.ncols() + kBlockDim.x - 1) / kBlockDim.x),
        static_cast<unsigned int>((c.nrows() + kBlockDim.y - 1) / kBlockDim.y));

    kernel_matmul_naive<AtomT><<<kGridDim, kBlockDim>>>(a, b, c);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
  }
};

// Стратегия 2: Блочное умножение через Shared Memory (Работа №3)
template <typename AtomT>
class SharedMemoryMatmulStrategy : public IMatmulStrategy<AtomT> {
 public:
  void Multiply(const MatrixView<AtomT>& a, const MatrixView<AtomT>& b,
                MatrixView<AtomT>& c) const override {
    constexpr dim3 kBlockDim(16, 16);
    const dim3 kGridDim(
        static_cast<unsigned int>((c.ncols() + kBlockDim.x - 1) / kBlockDim.x),
        static_cast<unsigned int>((c.nrows() + kBlockDim.y - 1) / kBlockDim.y));

    kernel_matmul_shmem<AtomT, 16><<<kGridDim, kBlockDim>>>(a, b, c);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
  }
};

// Стратегия 3: Умножение на тензорных ядрах через CUDA WMMA (Работа №4)
template <typename AtomT>
class WmmaMatmulStrategy : public IMatmulStrategy<AtomT> {
 public:
  void Multiply(const MatrixView<AtomT>& a, const MatrixView<AtomT>& b,
                MatrixView<AtomT>& c) const override {
    static_assert(std::is_same_v<AtomT, __half>,
                  "WmmaMatmulStrategy requires __half element type");

    if (a.nrows() % wmma_config::kWmmaM != 0 ||
        a.ncols() % wmma_config::kWmmaK != 0 ||
        b.ncols() % wmma_config::kWmmaN != 0) {
      throw std::invalid_argument(
          "Matrix dimensions must be multiples of 16 for WMMA Tensor Cores");
    }

    constexpr dim3 kBlockDim(wmma_config::kThreadsPerBlock);
    const dim3 kGridDim(
        static_cast<unsigned int>(
            (c.ncols() + wmma_config::kBlockItemsX - 1) /
            wmma_config::kBlockItemsX),
        static_cast<unsigned int>(
            (c.nrows() + wmma_config::kBlockItemsY - 1) /
            wmma_config::kBlockItemsY));

    kernel_matmul_wmma<AtomT><<<kGridDim, kBlockDim>>>(a, b, c);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
  }
};

#endif  // WORK4_CORE_MATMUL_STRATEGY_CUH_