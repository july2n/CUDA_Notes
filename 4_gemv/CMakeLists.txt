add_executable(sgemv sgemv.cu)
target_link_libraries(sgemv PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sgemv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

