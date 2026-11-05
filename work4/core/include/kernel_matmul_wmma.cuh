#ifndef WORK4_CORE_KERNEL_MATMUL_WMMA_CUH_
#define WORK4_CORE_KERNEL_MATMUL_WMMA_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>

#include "matrix_view.cuh"

namespace wmma_config {
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

constexpr int kWarpSize = 32;
constexpr int kBlockWarpsX = 4;  // 4 варпа по горизонтали (Warp 0, 2, 4, 6)
constexpr int kBlockWarpsY = 2;  // 2 варпа по вертикали (Warp 0, 1)
constexpr int kBlockWarps = kBlockWarpsX * kBlockWarpsY;  // 8 варпов в блоке
constexpr int kThreadsPerBlock = kBlockWarps * kWarpSize;  // 256 нитей

constexpr int kBlockItemsX = kBlockWarpsX * kWmmaN;  // 64 столбца на блок
constexpr int kBlockItemsY = kBlockWarpsY * kWmmaM;  // 32 строки на блок
constexpr int kBlockItemsK = kWmmaK;                 // шаг K = 16
}  // namespace wmma_config

template <typename AtomT = __half>
__global__ void kernel_matmul_wmma(MatrixView<AtomT> a, MatrixView<AtomT> b,
                                   MatrixView<AtomT> c) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  using namespace wmma_config;

  // Индекс варпа внутри блока (0..7) согласно схеме Рис. 1:
  // Чётные и нечётные варпы образуют пары по вертикали (Warp0/Warp1 и т.д.)
  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const int warp_row_in_block = warp_id % kBlockWarpsY;
  const int warp_col_in_block = warp_id / kBlockWarpsY;

  // Координаты верхнего левого угла тайла 16x16 в матрице C
  const std::size_t c_row =
      (static_cast<std::size_t>(blockIdx.y) * kBlockWarpsY +
       warp_row_in_block) *
      kWmmaM;
  const std::size_t c_col =
      (static_cast<std::size_t>(blockIdx.x) * kBlockWarpsX +
       warp_col_in_block) *
      kWmmaN;

  if (c_row >= c.nrows() || c_col >= c.ncols()) {
    return;
  }

  // Объявление фрагментов в регистровом файле (RF)
  wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half,
                 wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, __half> acc_frag;

  wmma::fill_fragment(acc_frag, __float2half(0.0f));

  const std::size_t k_len = a.ncols();
  const std::size_t ld_a = a.ncols();
  const std::size_t ld_b = b.ncols();
  const std::size_t ld_c = c.ncols();

  for (std::size_t k_step = 0; k_step < k_len; k_step += kBlockItemsK) {
    const __half* a_tile_ptr = a.data() + c_row * ld_a + k_step;
    const __half* b_tile_ptr = b.data() + k_step * ld_b + c_col;

    wmma::load_matrix_sync(a_frag, a_tile_ptr, static_cast<unsigned>(ld_a));
    wmma::load_matrix_sync(b_frag, b_tile_ptr, static_cast<unsigned>(ld_b));
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  __half* c_tile_ptr = c.data() + c_row * ld_c + c_col;
  wmma::store_matrix_sync(c_tile_ptr, acc_frag, static_cast<unsigned>(ld_c),
                          wmma::mem_row_major);
#endif
}

#endif  // WORK4_CORE_KERNEL_MATMUL_WMMA_CUH_