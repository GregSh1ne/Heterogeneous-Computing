#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

#include "cuda_check.cuh"
#include "data.cuh"
#include "kernel_matmul_naive.cuh"
#include "matrix_view.cuh"

template <typename AtomT>
class Matrix {
 private:
  std::shared_ptr<Data<AtomT>> data_;
  MatrixView<AtomT> view_;

 public:
  Matrix(std::size_t nrows, std::size_t ncols)
      : data_(std::make_shared<Data<AtomT>>(nrows * ncols))
      , view_(data_->data(), nrows, ncols) {}

  [[nodiscard]] std::size_t size() const noexcept {
    return view_.size();
  }

  [[nodiscard]] std::size_t nrows() const noexcept {
    return view_.nrows();
  }

  [[nodiscard]] std::size_t ncols() const noexcept {
    return view_.ncols();
  }

  [[nodiscard]] Data<AtomT>& data() noexcept {
    return *data_;
  }

  [[nodiscard]] const Data<AtomT>& data() const noexcept {
    return *data_;
  }

  [[nodiscard]] MatrixView<AtomT>& view() noexcept {
    return view_;
  }

  [[nodiscard]] const MatrixView<AtomT>& view() const noexcept {
    return view_;
  }

  // Перегрузка оператора умножения матриц C = A * B
  friend Matrix<AtomT> operator*(const Matrix<AtomT>& a, const Matrix<AtomT>& b) {
    if (a.ncols() != b.nrows()) {
      throw std::invalid_argument(
          "Matrix inner dimensions must match for multiplication: "
          "A.ncols ("
          + std::to_string(a.ncols()) + ") != B.nrows (" + std::to_string(b.nrows())
          + ")");
    }

    const std::size_t m = a.nrows();
    const std::size_t n = b.ncols();
    Matrix<AtomT> c(m, n);

    if (m == 0 || n == 0 || a.ncols() == 0) {
      return c;
    }

    constexpr dim3 block_dim(16, 16);
    const dim3 grid_dim(static_cast<unsigned int>((n + block_dim.x - 1) / block_dim.x),
        static_cast<unsigned int>((m + block_dim.y - 1) / block_dim.y));

    kernel_matmul_naive<AtomT><<<grid_dim, block_dim>>>(a.view(), b.view(), c.view());
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    return c;
  }
};