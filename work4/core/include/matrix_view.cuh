#ifndef WORK4_CORE_MATRIX_VIEW_CUH_
#define WORK4_CORE_MATRIX_VIEW_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

template <typename AtomT>
class MatrixView {
 private:
  AtomT* data_{nullptr};
  std::size_t nrows_{0};
  std::size_t ncols_{0};

 public:
  __host__ __device__ MatrixView() noexcept {}

  __host__ __device__ MatrixView(AtomT* data, std::size_t nrows,
                                 std::size_t ncols) noexcept
      : data_(data), nrows_(nrows), ncols_(ncols) {}

  __host__ __device__ [[nodiscard]] std::size_t size() const noexcept {
    return nrows_ * ncols_;
  }

  __host__ __device__ [[nodiscard]] std::size_t nrows() const noexcept {
    return nrows_;
  }

  __host__ __device__ [[nodiscard]] std::size_t ncols() const noexcept {
    return ncols_;
  }

  __host__ __device__ [[nodiscard]] AtomT* data() noexcept { return data_; }

  __host__ __device__ [[nodiscard]] const AtomT* data() const noexcept {
    return data_;
  }

  __device__ [[nodiscard]] AtomT& operator()(std::size_t i,
                                             std::size_t j) noexcept {
    return data_[i * ncols_ + j];
  }

  __device__ [[nodiscard]] const AtomT& operator()(
      std::size_t i, std::size_t j) const noexcept {
    return data_[i * ncols_ + j];
  }
};

static_assert(std::is_trivially_copyable_v<MatrixView<float>>,
              "MatrixView<float> must be trivially copyable");
static_assert(std::is_trivially_copyable_v<MatrixView<__half>>,
              "MatrixView<__half> must be trivially copyable");

#endif  // WORK4_CORE_MATRIX_VIEW_CUH_