add_executable(shared_memory shared_memory.cu)
target_link_libraries(shared_memory PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(shared_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()



add_executable(register_outer_product_float4 register_outer_product_float4.cu)
target_link_libraries(register_outer_product_float4 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(register_outer_product_float4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
