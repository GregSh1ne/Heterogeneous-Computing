#ifndef WORK3_CORE_KERNEL_MATMUL_NAIVE_CUH_
#define WORK3_CORE_KERNEL_MATMUL_NAIVE_CUH_

#include "matrix_view.cuh"
#include <cuda_runtime.h>

template <typename AtomT>
__global__ void kernel_matmul_naive(
    MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) {
  const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < c.nrows() && col < c.ncols()) {
    AtomT sum = static_cast<AtomT>(0);
    const std::size_t k_dim = a.ncols();
    for (std::size_t k = 0; k < k_dim; ++k) {
      sum += a(row, k) * b(k, col);
    }
    c(row, col) = sum;
  }
}

#endif  // WORK3_CORE_KERNEL_MATMUL_NAIVE_CUH_