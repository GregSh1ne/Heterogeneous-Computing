add_library(hsys_core STATIC
    src/vector.cu
)

target_include_directories(hsys_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(hsys_core PUBLIC
    CUDA::cudart
)

set_target_properties(hsys_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)