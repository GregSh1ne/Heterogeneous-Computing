#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>

template <typename AtomT>
class MatrixView {
 private:
  AtomT* data_ = nullptr;
  std::size_t nrows_ = 0;
  std::size_t ncols_ = 0;

 public:
  __host__ __device__ MatrixView() noexcept = default;

  __host__ __device__ MatrixView(
      AtomT* data, std::size_t nrows, std::size_t ncols) noexcept
      : data_(data)
      , nrows_(nrows)
      , ncols_(ncols) {}

  __host__ __device__ [[nodiscard]] AtomT* data() noexcept {
    return data_;
  }

  __host__ __device__ [[nodiscard]] const AtomT* data() const noexcept {
    return data_;
  }

  __host__ __device__ [[nodiscard]] std::size_t nrows() const noexcept {
    return nrows_;
  }

  __host__ __device__ [[nodiscard]] std::size_t ncols() const noexcept {
    return ncols_;
  }

  __host__ __device__ [[nodiscard]] std::size_t size() const noexcept {
    return nrows_ * ncols_;
  }

  __host__ __device__ AtomT& operator()(std::size_t row, std::size_t col) noexcept {
    return data_[row * ncols_ + col];
  }

  __host__ __device__ const AtomT& operator()(
      std::size_t row, std::size_t col) const noexcept {
    return data_[row * ncols_ + col];
  }

  __host__ __device__ AtomT& operator[](std::size_t idx) noexcept {
    return data_[idx];
  }

  __host__ __device__ const AtomT& operator[](std::size_t idx) const noexcept {
    return data_[idx];
  }
};

static_assert(std::is_trivially_copyable_v<MatrixView<float>>,
    "MatrixView must be trivially copyable to pass by value into CUDA kernels");

// WORK3_CORE_MATRIX_VIEW_CUH_