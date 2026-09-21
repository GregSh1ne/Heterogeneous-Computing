#pragma once

#include "vector_view.cuh"

template <typename AtomT>
__global__ void kernel_vecadd(
    VectorView<AtomT> a, VectorView<AtomT> b, VectorView<AtomT> c) {
  const std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < c.size()) {
    c[idx] = a[idx] + b[idx];
  }
}