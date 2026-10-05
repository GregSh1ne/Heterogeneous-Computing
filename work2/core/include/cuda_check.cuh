#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    cudaError_t status_ = (call);                                                        \
    if (status_ != cudaSuccess) {                                                        \
      throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(status_) \
                               + " at " + __FILE__ + ":" + std::to_string(__LINE__));    \
    }                                                                                    \
  }                                                                                      \
  while (0)

#define CUDA_CHECK_LAST_ERROR()                                                          \
  do {                                                                                   \
    cudaError_t status_ = cudaGetLastError();                                            \
    if (status_ != cudaSuccess) {                                                        \
      throw std::runtime_error(std::string("CUDA Kernel Error: ")                        \
                               + cudaGetErrorString(status_) + " at " + __FILE__ + ":"   \
                               + std::to_string(__LINE__));                              \
    }                                                                                    \
  }                                                                                      \
  while (0)