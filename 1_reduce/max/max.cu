#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include "utils.cuh"

void max_cpu(float* input, float* output, int N) {
    *output =  *(std::max_element(input, input + N));  // 计算输入数组的最大值
}

// 输入 (256, 768)
// 输出 (256, 1)
// BLOCK_SIZE = 768 (注意：768 必须是 32 的倍数，这里刚好是 24 个 warp)
__global__ void max_kernel(float* input, float* output, int total_elements) {
    __shared__ float s_mem[32]; // 足够容纳最多 32 个 warp 的结果
    
    int row = blockIdx.x; 
    int tid = threadIdx.x;
    int idx = row * blockDim.x + tid; // 当前线程处理的全局索引
    
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;

    // 1. 求每个 Warp 内的最大值
    // 初始化：越界位置填最小浮点数
    float val = (idx < total_elements) ? input[idx] : -3.40282e+38f; 

    // Warp Shuffle 归约
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }

    // 2. 将 Warp 结果写入共享内存
    if (laneId == 0) {
        s_mem[warpId] = val;
    }

    // 重要：必须同步，确保所有 warp 都写完了 s_mem
    __syncthreads();

    // 3. 用第一个 Warp 汇总共享内存中的结果
    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        // 读取刚才各 warp 存入的值，超出范围的填极小值
        val = (laneId < warpNum) ? s_mem[laneId] : -3.40282e+38f;

        // 再次进行 Warp 内部归约
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }

        // 4. 最终结果写入该行对应的 output 位置
        if (laneId == 0) {
            output[row] = val;
        }
    }
}

int main() {
    const int ROWS = 256;
    const int COLS = 768;
    size_t N = ROWS * COLS;
    constexpr size_t BLOCK_SIZE = 768; // 每行 768 个元素，由一个 block 处理
    const int repeat_times = 10;

    float* input  = (float*)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        input[i] = i;
    }
    float* output_ref = (float*)malloc(ROWS * sizeof(float));
    for (int r = 0; r < ROWS; r++) {
        max_cpu(input + r * COLS, &output_ref[r], COLS);
    }
    float total_time_h = TIME_RECORD(repeat_times, ([&]{
        for (int r = 0; r < ROWS; r++) {
            max_cpu(input + r * COLS, &output_ref[r], COLS);
        }
    }));
    printf("[max_cpu]: total_time_h = %f ms\n", total_time_h / repeat_times);

    float* output = (float*)malloc(ROWS * sizeof(float));
    float* input_device  = nullptr;
    float* output_device = nullptr;

    cudaMalloc(&input_device, N * sizeof(float));
    cudaMalloc(&output_device, ROWS * sizeof(float));
    cudaMemcpy(input_device, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // max: 每个 block 处理一行，256 个 block
    int block_size = BLOCK_SIZE;
    int grid_size  = ROWS;
    float total_time_1 = TIME_RECORD(repeat_times, ([&]{max_kernel<<<grid_size, block_size>>>(input_device, output_device, N);}));
    printf("[max_kernel]: total_time_1 = %f ms\n", total_time_1 / repeat_times);
    cudaMemcpy(output, output_device, ROWS * sizeof(float), cudaMemcpyDeviceToHost);

    for (int r = 0; r < ROWS; r++) {
        printf("row %d: output = %f, output_ref = %f\n", r, output[r], output_ref[r]);
    }

    free(input);
    free(output);
    free(output_ref);
    cudaFree(input_device);
    cudaFree(output_device);
    return 0;
}