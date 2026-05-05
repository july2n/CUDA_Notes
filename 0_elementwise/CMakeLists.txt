add_executable(add add.cu)
target_link_libraries(add PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

