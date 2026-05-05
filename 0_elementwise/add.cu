#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL(a,b) ((a+b-1)/(b))     // 向上整除

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

__global__ void elementwise_add_float32x4_pack_kernel(float *a, float *b, float *c, int N)
{
    // 使用float4等向量化访存方式 凑齐128字节
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if ((idx + 3) < N)
    {
        float4 temp_a = FLOAT4(a[idx]);
        float4 temp_b = FLOAT4(b[idx]);
        float4 temp_c;

        temp_c.x = temp_a.x + temp_b.x;
        temp_c.y = temp_a.y + temp_b.y;
        temp_c.z = temp_a.z + temp_b.z;
        temp_c.w = temp_a.w + temp_b.w;

        FLOAT4(c[idx]) = temp_c;
    }
    else if (idx < N)
    {
        for (int i = 0; (idx + i) < N; i++)
        {
            c[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}

__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c, int N)
{
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if ((idx + 7) < N)
    {
        // temporary register(memory), .local space in ptx, addressable
        half pack_a[8], pack_b[8], pack_c[8]; // 8x16 bits=128 bits.
        // reinterpret as float4 and load 128 bits in 1 memory issue.
        LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
        LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]); // load 128 bits

#pragma unroll
        for (int i = 0; i < 8; i += 2)
        {
            // __hadd2 for half2 x 4
            HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
        }
        // reinterpret as float4 and store 128 bits in 1 memory issue.
        LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
    }
    else if (idx < N)
    {
        for (int i = 0; (idx + i) < N; i++)
        {
            c[idx + i] = __hadd(a[idx + i], b[idx + i]);
        }
    }
}

int main()
{
    constexpr int N = 7;
    float *a_h = (float *)malloc(N * sizeof(float));
    float *b_h = (float *)malloc(N * sizeof(float));
    float *c_h = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        a_h[i] = i;
        b_h[i] = N - 1 - i;
    }

    float *a_d = nullptr;
    float *b_d = nullptr;
    float *c_d = nullptr;
    cudaCheck(cudaMalloc((void **)&a_d, N * sizeof(float)));
    cudaCheck(cudaMalloc((void **)&b_d, N * sizeof(float)));
    cudaCheck(cudaMalloc((void **)&c_d, N * sizeof(float)));
    cudaCheck(cudaMemcpy(a_d, a_h, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(b_d, b_h, N * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 1024;
    int grid_size = CEIL(CEIL(N, 4), 1024);
    elementwise_add_float32x4_pack_kernel<<<grid_size, block_size>>>(a_d, b_d, c_d, N);

    cudaCheck(cudaMemcpy(c_h, c_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("a_h:\n");
    for (int i = 0; i < N; i++)
    {
        if (i == N - 1)
            printf("%f\n", a_h[i]);
        else
            printf("%f ", a_h[i]);
    }
    printf("b_h:\n");
    for (int i = 0; i < N; i++)
    {
        if (i == N - 1)
            printf("%f\n", b_h[i]);
        else
            printf("%f ", b_h[i]);
    }
    printf("c_h:\n");
    for (int i = 0; i < N; i++)
    {
        if (i == N - 1)
            printf("%f\n", c_h[i]);
        else
            printf("%f ", c_h[i]);
    }
    return 0;
}