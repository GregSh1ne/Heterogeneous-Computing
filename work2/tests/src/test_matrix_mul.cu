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

TEST_P(MatrixMulParameterizedTest, MatchesEigenResult) {
  const auto [m, k, n] = GetParam();

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> host_a(m * k);
  std::vector<float> host_b(k * n);
  for (auto& val : host_a) val = dist(gen);
  for (auto& val : host_b) val = dist(gen);

  Matrix<float> gpu_a(m, k);
  Matrix<float> gpu_b(k, n);
  gpu_a.data().copy_from_host(host_a.data());
  gpu_b.data().copy_from_host(host_b.data());

  Matrix<float> gpu_c = gpu_a * gpu_b;

  std::vector<float> host_c(m * n);
  gpu_c.data().copy_to_host(host_c.data());

  Eigen::Map<const RowMajorMatrixXf> eigen_a(host_a.data(), m, k);
  Eigen::Map<const RowMajorMatrixXf> eigen_b(host_b.data(), k, n);
  RowMajorMatrixXf eigen_c = eigen_a * eigen_b;

  Eigen::Map<const RowMajorMatrixXf> result_gpu(host_c.data(), m, n);

  // Верификация через isApprox с точностью 1e-5f
  // Если норма разности близка к нулю, isApprox проверяет абсолютную точность
  const float prec = 1e-5f;
  bool is_correct = eigen_c.isApprox(result_gpu, prec);
  if (!is_correct) {
    // Дополнительная проверка абсолютной погрешности для больших размерностей
    float max_diff = (eigen_c - result_gpu).cwiseAbs().maxCoeff();
    EXPECT_LE(max_diff, prec * std::max(1.0f, eigen_c.cwiseAbs().maxCoeff()));
  } else {
    EXPECT_TRUE(is_correct);
  }
}

INSTANTIATE_TEST_SUITE_P(MatrixMulBoundaryCases,
    MatrixMulParameterizedTest,
    ::testing::Combine(::testing::Values(1, 2, 3, 127, 128, 129, 512),
        ::testing::Values(1, 2, 3, 127, 128, 129, 512),
        ::testing::Values(1, 2, 3, 127, 128, 129, 512)));

// Тест на проверку выброса исключения при несогласованных размерах
TEST(MatrixMulExceptions, DimensionMismatchThrows) {
  Matrix<float> a(10, 20);
  Matrix<float> b(25, 30);
  EXPECT_THROW([[maybe_unused]] auto c = a * b, std::invalid_argument);
}