#pragma once

#include "matrix_view.cuh"
#include <cuda_runtime.h>

template <typename AtomT, unsigned int kTileSize = 16>
__global__ void kernel_matmul_shmem(
    MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) {
  // Разделяемая память для тайлов матриц A и B
  __shared__ AtomT s_a[kTileSize][kTileSize];
  __shared__ AtomT s_b[kTileSize][kTileSize];

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  // Глобальные координаты результирующего элемента в матрице C
  const unsigned int row = blockIdx.y * kTileSize + ty;
  const unsigned int col = blockIdx.x * kTileSize + tx;

  AtomT sum = static_cast<AtomT>(0);

  // Количество тайлов вдоль внутреннего измерения k
  const unsigned int k_dim = static_cast<unsigned int>(a.ncols());
  const unsigned int num_tiles = (k_dim + kTileSize - 1) / kTileSize;

  for (unsigned int t = 0; t < num_tiles; ++t) {
    // 1. Кооперативная загрузка тайла матрицы A в разделяемую память
    const unsigned int a_col = t * kTileSize + tx;
    if (row < a.nrows() && a_col < a.ncols()) {
      s_a[ty][tx] = a(row, a_col);
    } else {
      s_a[ty][tx] = static_cast<AtomT>(0);
    }

    // 2. Кооперативная загрузка тайла матрицы B в разделяемую память
    const unsigned int b_row = t * kTileSize + ty;
    if (b_row < b.nrows() && col < b.ncols()) {
      s_b[ty][tx] = b(b_row, col);
    } else {
      s_b[ty][tx] = static_cast<AtomT>(0);
    }
    // Барьерная синхронизация: ждём завершения загрузки тайла ВСЕМИ нитями блока
    __syncthreads();

    // 3. Вычисление скалярного произведения в быстром кэше
    for (unsigned int k = 0; k < kTileSize; ++k) {
      sum += s_a[ty][k] * s_b[k][tx];
    }
    // Барьерная синхронизация: исключаем перезапись тайла до завершения расчетов всеми
    // нитями
    __syncthreads();
  }

  // Запись накопленной суммы в результирующую матрицу C
  if (row < c.nrows() && col < c.ncols()) {
    c(row, col) = sum;
  }
}  // WORK3_CORE_KERNEL_MATMUL_SHMEM_CUH_