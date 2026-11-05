#define EIGEN_DONT_VECTORIZE
#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cstddef>
#include <random>
#include <tuple>

#include "cuda_check.cuh"
#include "matmul_strategy.cuh"
#include "matrix.cuh"

using EigenHalfMatrix
    = Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenFloatMatrix
    = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static_assert(sizeof(Eigen::half) == sizeof(__half),
    "Eigen::half and CUDA __half must have identical size (2 bytes)");

class WmmaMatmulParameterizedTest
    : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t>> {
};

TEST_P(WmmaMatmulParameterizedTest, MatchesEigenHalfReference) {
  const auto [m, k, n] = GetParam();

  Matrix<__half>::SetStrategy(std::make_shared<WmmaMatmulStrategy<__half>>());

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  EigenHalfMatrix h_a(m, k);
  EigenHalfMatrix h_b(k, n);
  EigenHalfMatrix h_c(m, n);

  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      h_a(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j))
          = Eigen::half(dist(gen));
    }
  }

  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      h_b(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j))
          = Eigen::half(dist(gen));
    }
  }

  Matrix<__half> d_a(m, k);
  Matrix<__half> d_b(k, n);
  d_a.data().copy_from_host(reinterpret_cast<const __half*>(h_a.data()));
  d_b.data().copy_from_host(reinterpret_cast<const __half*>(h_b.data()));

  Matrix<__half> d_c = d_a * d_b;
  d_c.data().copy_to_host(reinterpret_cast<__half*>(h_c.data()));

  const EigenHalfMatrix expected_half
      = (h_a.cast<float>() * h_b.cast<float>()).cast<Eigen::half>();
  const Eigen::MatrixXf expected = expected_half.cast<float>();
  const Eigen::MatrixXf actual = h_c.cast<float>();

  EXPECT_TRUE(actual.isApprox(expected, 1e-2f));
}

INSTANTIATE_TEST_SUITE_P(WmmaDimensions,
    WmmaMatmulParameterizedTest,
    ::testing::Combine(::testing::Values(16, 32, 64, 128, 256, 512),
        ::testing::Values(16, 32, 64, 128, 256, 512),
        ::testing::Values(16, 32, 64, 128, 256, 512)));

TEST(WmmaMatmulExceptionTest, ThrowsOnDimensionMismatch) {
  Matrix<__half> a(16, 32);
  Matrix<__half> b(16, 16);
  EXPECT_THROW(
      {
        Matrix<__half> c = a * b;
        (void)c;
      },
      std::invalid_argument);
}

TEST(WmmaMatmulExceptionTest, ThrowsOnUnalignedDimensions) {
  Matrix<__half>::SetStrategy(std::make_shared<WmmaMatmulStrategy<__half>>());
  Matrix<__half> a(15, 16);
  Matrix<__half> b(16, 16);
  EXPECT_THROW(
      {
        Matrix<__half> c = a * b;
        (void)c;
      },
      std::invalid_argument);
}