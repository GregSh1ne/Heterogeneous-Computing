#pragma once

#include "matrix_view.cuh"
#include <cuda_runtime.h>

template <typename AtomT>
__global__ void kernel_matmul_naive(
    MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (row < c.nrows() && col < c.ncols()) {
    AtomT sum = static_cast<AtomT>(0);
    const std::size_t k_len = a.ncols();  // или b.nrows()

#pragma unroll 4
    for (std::size_t p = 0; p < k_len; ++p) {
      sum += a(row, p) * b(p, col);
    }

    c(row, col) = sum;
  }
}