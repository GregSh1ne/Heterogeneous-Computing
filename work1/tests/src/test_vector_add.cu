#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "vector.cuh"

class VectorAddParameterizedTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VectorAddParameterizedTest, MatchesEigenVectorXf) {
  const std::size_t n = GetParam();

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

  Eigen::VectorXf eigen_a(n);
  Eigen::VectorXf eigen_b(n);

  for (std::size_t i = 0; i < n; ++i) {
    eigen_a[static_cast<Eigen::Index>(i)] = dist(rng);
    eigen_b[static_cast<Eigen::Index>(i)] = dist(rng);
  }

  Eigen::VectorXf eigen_expected = eigen_a + eigen_b;

  Vector<float> gpu_a(n);
  Vector<float> gpu_b(n);

  gpu_a.data().copy_from_host(eigen_a.data());
  gpu_b.data().copy_from_host(eigen_b.data());

  Vector<float> gpu_res = gpu_a + gpu_b;

  std::vector<float> host_result(n);
  gpu_res.data().copy_to_host(host_result.data());

  Eigen::Map<Eigen::VectorXf> eigen_gpu_map(
      host_result.data(), static_cast<Eigen::Index>(n));

  EXPECT_TRUE(eigen_gpu_map.isApprox(eigen_expected, 1e-6f))
      << "Mismatch found for vector size N = " << n;
}

INSTANTIATE_TEST_SUITE_P(VectorSizes,
    VectorAddParameterizedTest,
    ::testing::Values(1, 2, 3, 127, 128, 129, 512, 1024, 1029));

TEST(VectorEdgeCases, HandlesZeroSize) {
  Vector<float> a(0);
  Vector<float> b(0);
  EXPECT_NO_THROW({
    Vector<float> c = a + b;
    EXPECT_EQ(c.size(), 0);
  });
}

TEST(VectorEdgeCases, ThrowsOnDimensionMismatch) {
  Vector<float> a(128);
  Vector<float> b(256);
  EXPECT_THROW((void)(a + b), std::invalid_argument);
}