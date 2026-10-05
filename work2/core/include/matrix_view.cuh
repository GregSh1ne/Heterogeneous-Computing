#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>

template <typename AtomT>
class MatrixView {
 private:
  AtomT* data_{nullptr};
  std::size_t nrows_{0};
  std::size_t ncols_{0};

 public:
  __host__ __device__ MatrixView() noexcept = default;

  __host__ __device__ MatrixView(
      AtomT* data, std::size_t nrows, std::size_t ncols) noexcept
      : data_(data)
      , nrows_(nrows)
      , ncols_(ncols) {}

  __host__ __device__ [[nodiscard]] std::size_t size() const noexcept {
    return nrows_ * ncols_;
  }

  __host__ __device__ [[nodiscard]] std::size_t nrows() const noexcept {
    return nrows_;
  }

  __host__ __device__ [[nodiscard]] std::size_t ncols() const noexcept {
    return ncols_;
  }

  __host__ __device__ [[nodiscard]] AtomT* data() noexcept {
    return data_;
  }

  __host__ __device__ [[nodiscard]] const AtomT* data() const noexcept {
    return data_;
  }

  // Линейный доступ по индексу
  __host__ __device__ [[nodiscard]] AtomT& operator[](std::size_t n) noexcept {
    return data_[n];
  }

  __host__ __device__ [[nodiscard]] const AtomT& operator[](
      std::size_t n) const noexcept {
    return data_[n];
  }

  // Двумерный доступ row-major: data_[i * ncols_ + j]
  __host__ __device__ [[nodiscard]] AtomT& operator()(
      std::size_t i, std::size_t j) noexcept {
    return data_[i * ncols_ + j];
  }

  __host__ __device__ [[nodiscard]] const AtomT& operator()(
      std::size_t i, std::size_t j) const noexcept {
    return data_[i * ncols_ + j];
  }
};

static_assert(std::is_trivially_copyable_v<MatrixView<float>>,
    "MatrixView must be trivially copyable to pass by value into CUDA kernels!");