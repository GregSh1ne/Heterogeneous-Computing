#ifndef WORK4_CORE_KERNEL_MATMUL_NAIVE_CUH_
#define WORK4_CORE_KERNEL_MATMUL_NAIVE_CUH_

#include <cuda_runtime.h>

#include <cstddef>

#include "matrix_view.cuh"

template <typename AtomT>
__global__ void kernel_matmul_naive(MatrixView<AtomT> a, MatrixView<AtomT> b,
                                    MatrixView<AtomT> c) {
  const std::size_t col =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

  if (row < c.nrows() && col < c.ncols()) {
    AtomT sum = static_cast<AtomT>(0.0f);
    const std::size_t k_len = a.ncols();
    for (std::size_t p = 0; p < k_len; ++p) {
      sum = sum + a(row, p) * b(p, col);
    }
    c(row, col) = sum;
  }
}

#endif  // WORK4_CORE_KERNEL_MATMUL_NAIVE_CUH_