#pragma once

#include "cuda_check.cuh"
#include "kernel_matmul_naive.cuh"
#include "kernel_matmul_shmem.cuh"
#include "matrix_view.cuh"
#include <cuda_runtime.h>

template <typename AtomT>
class IMatmulStrategy {
 public:
  virtual ~IMatmulStrategy() = default;
  virtual void Multiply(
      const MatrixView<AtomT>& a, const MatrixView<AtomT>& b, MatrixView<AtomT>& c) const
      = 0;
};

// Стратегия умножения на базе наивного ядра
template <typename AtomT>
class NaiveMatmulStrategy : public IMatmulStrategy<AtomT> {
 public:
  void Multiply(const MatrixView<AtomT>& a,
      const MatrixView<AtomT>& b,
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

// Стратегия блочного умножения с разделяемой памятью
template <typename AtomT>
class SharedMemoryMatmulStrategy : public IMatmulStrategy<AtomT> {
 public:
  void Multiply(const MatrixView<AtomT>& a,
      const MatrixView<AtomT>& b,
      MatrixView<AtomT>& c) const override {
    constexpr dim3 kBlockDim(16, 16);
    const dim3 kGridDim(
        static_cast<unsigned int>((c.ncols() + kBlockDim.x - 1) / kBlockDim.x),
        static_cast<unsigned int>((c.nrows() + kBlockDim.y - 1) / kBlockDim.y));

    kernel_matmul_shmem<AtomT, 16><<<kGridDim, kBlockDim>>>(a, b, c);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
  }
};  // WORK3_CORE_MATMUL_STRATEGY_CUH_