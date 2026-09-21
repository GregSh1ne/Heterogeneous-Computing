#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>

template <typename AtomT>
class VectorView {
 private:
  AtomT* data_{nullptr};
  std::size_t size_{0};

 public:
  __host__ __device__ VectorView() = default;

  __host__ __device__ VectorView(AtomT* data, std::size_t size)
      : data_(data)
      , size_(size) {}

  __host__ __device__ [[nodiscard]] std::size_t size() const noexcept {
    return size_;
  }

  __host__ __device__ [[nodiscard]] AtomT* data() noexcept {
    return data_;
  }

  __host__ __device__ [[nodiscard]] const AtomT* data() const noexcept {
    return data_;
  }

  __host__ __device__ AtomT& operator[](std::size_t n) noexcept {
    return data_[n];
  }

  __host__ __device__ const AtomT& operator[](std::size_t n) const noexcept {
    return data_[n];
  }

  __host__ __device__ AtomT& operator()(std::size_t i) noexcept {
    return data_[i];
  }

  __host__ __device__ const AtomT& operator()(std::size_t i) const noexcept {
    return data_[i];
  }
};

static_assert(std::is_trivially_copyable_v<VectorView<float>>,
    "VectorView must be trivially copyable for passing to CUDA kernels by value!");