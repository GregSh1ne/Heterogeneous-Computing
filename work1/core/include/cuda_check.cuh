#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    cudaError_t err = (call);                                                            \
    if (err != cudaSuccess) {                                                            \
      throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)     \
                               + " at " + __FILE__ + ":" + std::to_string(__LINE__));    \
    }                                                                                    \
  }                                                                                      \
  while (0)

#define CUDA_CHECK_LAST_ERROR()                                                          \
  do {                                                                                   \
    cudaError_t err = cudaGetLastError();                                                \
    if (err != cudaSuccess) {                                                            \
      throw std::runtime_error(std::string("CUDA Kernel Error: ")                        \
                               + cudaGetErrorString(err) + " at " + __FILE__ + ":"       \
                               + std::to_string(__LINE__));                              \
    }                                                                                    \
  }                                                                                      \
  while (0)