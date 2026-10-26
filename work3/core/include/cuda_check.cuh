#ifndef WORK3_CORE_CUDA_CHECK_CUH_
#define WORK3_CORE_CUDA_CHECK_CUH_

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    const cudaError_t status = (call);                                                   \
    if (status != cudaSuccess) {                                                         \
      throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(status)  \
                               + " at " + __FILE__ + ":" + std::to_string(__LINE__));    \
    }                                                                                    \
  }                                                                                      \
  while (0)

#define CUDA_CHECK_LAST_ERROR()                                                          \
  do {                                                                                   \
    const cudaError_t status = cudaGetLastError();                                       \
    if (status != cudaSuccess) {                                                         \
      throw std::runtime_error(std::string("CUDA Kernel Error: ")                        \
                               + cudaGetErrorString(status) + " at " + __FILE__ + ":"    \
                               + std::to_string(__LINE__));                              \
    }                                                                                    \
  }                                                                                      \
  while (0)

#endif  // WORK3_CORE_CUDA_CHECK_CUH_