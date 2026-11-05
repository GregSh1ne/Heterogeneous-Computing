#ifndef WORK4_CORE_KERNEL_MATMUL_SHMEM_CUH_
#define WORK4_CORE_KERNEL_MATMUL_SHMEM_CUH_

#include <cuda_runtime.h>

#include <cstddef>

#include "matrix_view.cuh"

template <typename AtomT, unsigned int kTileSize = 16>
__global__ void kernel_matmul_shmem(MatrixView<AtomT> a, MatrixView<AtomT> b,
                                    MatrixView<AtomT> c) {
  __shared__ AtomT s_a[kTileSize][kTileSize];
  __shared__ AtomT s_b[kTileSize][kTileSize];

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  const std::size_t row =
      static_cast<std::size_t>(blockIdx.y) * kTileSize + ty;
  const std::size_t col =
      static_cast<std::size_t>(blockIdx.x) * kTileSize + tx;

  AtomT sum = static_cast<AtomT>(0.0f);
  const std::size_t k_len = a.ncols();
  const std::size_t num_tiles = (k_len + kTileSize - 1) / kTileSize;

  for (std::size_t t = 0; t < num_tiles; ++t) {
    const std::size_t a_col = t * kTileSize + tx;
    if (row < a.nrows() && a_col < a.ncols()) {
      s_a[ty][tx] = a(row, a_col);
    } else {
      s_a[ty][tx] = static_cast<AtomT>(0.0f);
    }

    const std::size_t b_row = t * kTileSize + ty;
    if (b_row < b.nrows() && col < b.ncols()) {
      s_b[ty][tx] = b(b_row, col);
    } else {
      s_b[ty][tx] = static_cast<AtomT>(0.0f);
    }

    __syncthreads();

    for (unsigned int p = 0; p < kTileSize; ++p) {
      sum = sum + s_a[ty][p] * s_b[p][tx];
    }

    __syncthreads();
  }

  if (row < c.nrows() && col < c.ncols()) {
    c(row, col) = sum;
  }
}

#endif  // WORK4_CORE_KERNEL_MATMUL_SHMEM_CUH_