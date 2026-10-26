#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <random>
#include <tuple>
#include <vector>

#include "matrix.cuh"

using RowMajorMatrixXf
    = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class MatrixMulParameterizedTest
    : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t>> {
};

TEST_P(MatrixMulParameterizedTest, CompareSharedMemoryWithEigen) {
  const auto [m, k, n] = GetParam();

  Matrix<float> a_gpu(m, k);
  Matrix<float> b_gpu(k, n);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> a_host(m * k);
  std::vector<float> b_host(k * n);

  for (auto& val : a_host) val = dist(rng);
  for (auto& val : b_host) val = dist(rng);

  a_gpu.data().copy_from_host(a_host.data());
  b_gpu.data().copy_from_host(b_host.data());

  // Вычисление произведения на GPU с использованием operator* (Shared Memory)
  Matrix<float> c_gpu = a_gpu * b_gpu;

  std::vector<float> c_gpu_host(m * n);
  c_gpu.data().copy_to_host(c_gpu_host.data());

  // Вычисление эталона на CPU с помощью Eigen
  Eigen::Map<const RowMajorMatrixXf> a_eigen(a_host.data(), m, k);
  Eigen::Map<const RowMajorMatrixXf> b_eigen(b_host.data(), k, n);
  RowMajorMatrixXf c_eigen = a_eigen * b_eigen;

  Eigen::Map<const RowMajorMatrixXf> c_cuda_map(c_gpu_host.data(), m, n);

  // Сверка с абсолютной погрешностью 1e-5f
  EXPECT_TRUE(c_eigen.isApprox(c_cuda_map, 1e-5f));
}

TEST(MatrixMulTest, DimensionMismatchThrows) {
  Matrix<float> a(10, 20);
  Matrix<float> b(21, 30);
  EXPECT_THROW({ (void)(a * b); }, std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationSuite,
    MatrixMulParameterizedTest,
    ::testing::Combine(::testing::Values(1, 2, 3, 127, 128, 129, 512),
        ::testing::Values(1, 2, 3, 127, 128, 129, 512),
        ::testing::Values(1, 2, 3, 127, 128, 129, 512)));