#pragma once

#include "data.cuh"
#include "kernel_vecadd.cuh"
#include "vector_view.cuh"
#include <memory>
#include <stdexcept>

template <typename AtomT>
class Vector {
 private:
  std::shared_ptr<Data<AtomT>> data_;
  VectorView<AtomT> view_;

 public:
  explicit Vector(std::size_t size)
      : data_(std::make_shared<Data<AtomT>>(size))
      , view_(data_->data(), size) {}

  Vector(const Vector& other)
      : data_(other.data_)
      , view_(other.view_) {}

  Vector(Vector&& other) noexcept
      : data_(std::move(other.data_))
      , view_(other.view_) {
    other.view_ = VectorView<AtomT>(nullptr, 0);
  }

  Vector& operator=(const Vector& other) {
    if (this != &other) {
      data_ = other.data_;
      view_ = other.view_;
    }
    return *this;
  }

  Vector& operator=(Vector&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      view_ = other.view_;
      other.view_ = VectorView<AtomT>(nullptr, 0);
    }
    return *this;
  }

  ~Vector() = default;

  [[nodiscard]] std::size_t size() const noexcept {
    return view_.size();
  }

  [[nodiscard]] Data<AtomT>& data() noexcept {
    return *data_;
  }

  [[nodiscard]] const Data<AtomT>& data() const noexcept {
    return *data_;
  }

  [[nodiscard]] VectorView<AtomT>& view() noexcept {
    return view_;
  }

  [[nodiscard]] const VectorView<AtomT>& view() const noexcept {
    return view_;
  }

  friend Vector<AtomT> operator+(const Vector<AtomT>& lhs, const Vector<AtomT>& rhs) {
    if (lhs.size() != rhs.size()) {
      throw std::invalid_argument("Vectors must have the same size for addition!");
    }

    const std::size_t n = lhs.size();
    Vector<AtomT> result(n);

    if (n == 0) {
      return result;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size
        = static_cast<unsigned int>((n + block_size - 1) / block_size);

    kernel_vecadd<AtomT>
        <<<grid_size, block_size>>>(lhs.view(), rhs.view(), result.view());
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    return result;
  }
};