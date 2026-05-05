#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.cuh"

void host_reduce(float* x, const int N, float* sum) {
    *sum = 0.0f;
    for (int i = 0; i < N; i++) {
        *sum += x[i];
    }
}

__global__ void device_shared_memory(float* d_x, float* d_y, const int N) {
    const int tid = threadIdx.x;
    const int n = blockIdx.x * blockDim.x + tid;
    extern __shared__ float s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(d_y, s_y[0]);
    }
}

template <const int BLOCK_SIZE>
void call_reduce_shared_memory(float* d_x, float* d_y, float* h_y, const int N) {
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    *h_y = 0.0f;
    cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice);
    device_shared_memory<<<grid_size, block_size, sizeof(float) * BLOCK_SIZE>>>(d_x, d_y, N);
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

__global__ void device_warp_shuffle(const float* __restrict__ input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;

    float local_sum = 0.0f;
    if (idx < N) {
        float4 tmp_x = FLOAT4(input[idx]);
        local_sum += tmp_x.x + tmp_x.y + tmp_x.z + tmp_x.w;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    __shared__ float s_mem[32];
    int warpId = tid / 32;
    int laneId = tid % 32;

    if (laneId == 0) {
        s_mem[warpId] = local_sum;
    }
    __syncthreads();

    if (warpId == 0) {
        float block_sum = (laneId < (blockDim.x / 32)) ? s_mem[laneId] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
        }
        if (laneId == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

template <const int BLOCK_SIZE>
void call_reduce_warp_shuffle(float* d_x, float* d_y, float* h_y, const int N) {
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE * 4);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    *h_y = 0.0f;
    cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice);
    device_warp_shuffle<<<grid_size, block_size>>>(d_x, d_y, N);
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

int main() {
    size_t N = 100000000;
    constexpr size_t BLOCK_SIZE = 128;
    const int repeat_times = 10;

    float *h_nums = (float *)malloc(sizeof(float) * N);
    float *h_sum = (float *)malloc(sizeof(float));
    randomize_matrix(h_nums, N);

    float total_time_h = TIME_RECORD(repeat_times, ([&]{host_reduce(h_nums, N, h_sum);}));
    printf("[reduce_host]: sum = %f, total_time = %f ms\n", *h_sum, total_time_h / repeat_times);

    float *d_nums, *d_sum;
    cudaMalloc((void **)&d_nums, sizeof(float) * N);
    cudaMalloc((void **)&d_sum, sizeof(float));
    float *h_result = (float *)malloc(sizeof(float));

    // shared memory version
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_sm = TIME_RECORD(repeat_times, ([&]{call_reduce_shared_memory<BLOCK_SIZE>(d_nums, d_sum, h_result, N);}));
    printf("[reduce_shared_memory]: sum = %f, total_time = %f ms\n", *h_result, total_time_sm / repeat_times);

    // warp shuffle version
    *h_result = 0.0f;
    cudaMemcpy(d_sum, h_result, sizeof(float), cudaMemcpyHostToDevice);
    float total_time_ws = TIME_RECORD(repeat_times, ([&]{call_reduce_warp_shuffle<BLOCK_SIZE>(d_nums, d_sum, h_result, N);}));
    printf("[reduce_warp_shuffle]: sum = %f, total_time = %f ms\n", *h_result, total_time_ws / repeat_times);

    free(h_nums);
    free(h_sum);
    free(h_result);
    cudaFree(d_nums);
    cudaFree(d_sum);
    return 0;
}