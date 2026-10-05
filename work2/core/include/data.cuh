#pragma once

#include "cuda_check.cuh"
#include <cstddef>
#include <utility>

template <typename AtomT>
class Data {
 private:
  std::size_t size_{0};
  AtomT* data_{nullptr};

 public:
  explicit Data(std::size_t size)
      : size_(size)
      , data_(nullptr) {
    if (size_ > 0)
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(AtomT)));
  }

  ~Data() {
    if (data_ != nullptr) cudaFree(data_);
  }

  Data(const Data& other)
      : size_(other.size_)
      , data_(nullptr) {
    if (size_ > 0) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(AtomT)));
      CUDA_CHECK(cudaMemcpy(
          data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice));
    }
  }

  Data(Data&& other) noexcept
      : size_(other.size_)
      , data_(other.data_) {
    other.size_ = 0;
    other.data_ = nullptr;
  }

  Data& operator=(const Data& other) {
    if (this != &other) {
      if (size_ != other.size_) {
        if (data_ != nullptr) {
          CUDA_CHECK(cudaFree(data_));
          data_ = nullptr;
        }
        size_ = other.size_;
        if (size_ > 0)
          CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(AtomT)));
      }
      if (size_ > 0)
        CUDA_CHECK(cudaMemcpy(
            data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice));
    }
    return *this;
  }

  Data& operator=(Data&& other) noexcept {
    if (this != &other) return *this;

    if (data_ != nullptr) cudaFree(data_);
    size_ = other.size_;
    data_ = other.data_;
    other.size_ = 0;
    other.data_ = nullptr;
  }

  [[nodiscard]] AtomT* data() noexcept {
    return data_;
  }

  [[nodiscard]] const AtomT* data() const noexcept {
    return data_;
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return size_;
  }

  void copy_to_host(AtomT* host_ptr) const {
    if (size_ > 0 && data_ != nullptr && host_ptr != nullptr) {
      CUDA_CHECK(
          cudaMemcpy(host_ptr, data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToHost));
    }
  }

  void copy_from_host(const AtomT* host_ptr) {
    if (size_ > 0 && data_ != nullptr && host_ptr != nullptr) {
      CUDA_CHECK(
          cudaMemcpy(data_, host_ptr, size_ * sizeof(AtomT), cudaMemcpyHostToDevice));
    }
  }
};