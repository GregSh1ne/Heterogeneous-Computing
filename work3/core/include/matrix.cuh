#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "data.cuh"
#include "matmul_strategy.cuh"
#include "matrix_view.cuh"

template <typename AtomT>
class Matrix {
 private:
  std::shared_ptr<Data<AtomT>> data_;
  MatrixView<AtomT> view_;

  // Статическая ссылка на текущую стратегию вычислений
  inline static std::shared_ptr<IMatmulStrategy<AtomT>> default_strategy_
      = std::make_shared<SharedMemoryMatmulStrategy<AtomT>>();

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

  // Метод переключения стратегии
  static void SetStrategy(std::shared_ptr<IMatmulStrategy<AtomT>> strategy) {
    default_strategy_ = std::move(strategy);
  }

  static std::shared_ptr<IMatmulStrategy<AtomT>> GetStrategy() {
    return default_strategy_;
  }

  friend Matrix<AtomT> operator*(const Matrix<AtomT>& a, const Matrix<AtomT>& b) {
    if (a.ncols() != b.nrows()) {
      throw std::invalid_argument(
          "Matrix inner dimensions must match for multiplication: A.ncols ("
          + std::to_string(a.ncols()) + ") != B.nrows (" + std::to_string(b.nrows())
          + ")");
    }

    const std::size_t m = a.nrows();
    const std::size_t n = b.ncols();
    Matrix<AtomT> c(m, n);

    if (m == 0 || n == 0 || a.ncols() == 0) {
      return c;
    }

    default_strategy_->Multiply(a.view(), b.view(), c.view());
    return c;
  }
};  // WORK3_CORE_MATRIX_CUH_