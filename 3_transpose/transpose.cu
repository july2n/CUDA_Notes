#include <stdio.h>
#include <stdlib.h>
#include "utils.cuh"

void host_transpose(float *input, int M, int N, float *output)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            output[i * M + j] = input[j * N + i];
        }
    }
}

// 使用共享内存中转，合并读取+写入，但是存在 bank conflict
template <const int TILE_DIM>
__global__ void device_shared_memory(const float *input, float *output, int M, int N)
{
    __shared__ float shared[TILE_DIM][TILE_DIM];
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;

    const int x1 = bx + threadIdx.x;
    const int y1 = by + threadIdx.y;

    if (y1 < M && x1 < N)
    {
        shared[threadIdx.y][threadIdx.x] = input[y1 * N + x1];
    }
    __syncthreads();

    const int x2 = by + threadIdx.x;
    const int y2 = bx + threadIdx.y;
    if (y2 < N && x2 < M)
    {
        // 合并写入，但是存在bank冲突：
        // 可以看出，同一个warp中的32个线程（连续的32个threaIdx.x值）
        // 将对应共享内存中跨度为32的数据，也就说，这32个线程恰好访问
        // 同一个bank中的32个数据，这将导致32路bank冲突
        output[y2 * M + x2] = shared[threadIdx.x][threadIdx.y];
    }
}

// 使用共享内存中转，合并读取+写入，对共享内存做padding，解决bank conflict
template <const int TILE_DIM>
__global__ void device_padding(const float *input, float *output, int M, int N)
{
    __shared__ float shared[TILE_DIM][TILE_DIM + 1];
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;
    const int x1 = bx + threadIdx.x;
    const int y1 = by + threadIdx.y;
    if(y1 < M && x1 < N)
    {
        shared[threadIdx.y][threadIdx.x] = input[y1 * N + x1];
    }
    __syncthreads();

    const int x2 = by + threadIdx.x;
    const int y2 = bx + threadIdx.y;
    if(y2 < N && x2 < M)
    {
        // 通过做padding后，同一个warp中的32个线程（连续的32个threaIdx.x值）
        // 将对应共享内存中跨度为33的数据
        // 如果第一个线程访问第一个bank中的第一层
        // 那么第二个线程访问第二个bank中的第二层
        // 以此类推，32个线程访问32个不同bank，不存在bank冲突
        output[y2 * M + x2] = shared[threadIdx.x][threadIdx.y];
    }
}

// 使用共享内存中转，合并读取+写入，使用swizzling解决bank conflict
template <const int TILE_DIM>
__global__ void device_swizzling(const float *input, float *output, int M, int N)
{
    __shared__ float shared[TILE_DIM][TILE_DIM];
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;

    const int x1 = bx + threadIdx.x;
    const int y1 = by + threadIdx.y;

    if (y1 < M && x1 < N)
    {
        shared[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[y1 * N + x1];
    }
    __syncthreads();

    const int x2 = by + threadIdx.x;
    const int y2 = bx + threadIdx.y;
    if (y2 < N && x2 < M)
    {
        // swizzling主要利用了异或运算的以下两个性质来规避bank conflict：
        // 1. 运算的封闭性  2. x1^y!=x2^y当且仅当x1!=x2
        // 举例：
        // 第一行的访存位置由0,0,0,0...变为0,1,2,3...
        // 第二行的访存位置由1,1,1,1...变为1,0,3,2...
        // 第三行的访存位置由2,2,2,2...变为2,3,0,1...
        // 第四行的访存位置由3,3,3,3...变为3,2,1,0...
        // 这样既能保证充分利用shared memory的空间（由于性质1和2）
        // 又能保证warp中的各个线程不会访问同一bank（由于性质2）
        output[y2 * M + x2] = shared[threadIdx.x][threadIdx.x ^ threadIdx.y];
    }
}

int main()
{
    // 输入是M行N列，转置后是N行M列
    size_t M = 12800;
    size_t N = 1280;
    constexpr size_t BLOCK_SIZE = 32;
    const int repeat_times = 10;

    // --------------------host 端计算一遍转置, 输出的结果用于后续验证---------------------- //
    float *h_matrix = (float *)malloc(sizeof(float) * M * N);
    float *h_matrix_tr_ref = (float *)malloc(sizeof(float) * N * M);
    randomize_matrix(h_matrix, M * N);
    host_transpose(h_matrix, M, N, h_matrix_tr_ref);
    // printf("init_matrix:\n");
    // print_matrix(h_matrix, M, N);
    // printf("host_transpose:\n");
    // print_matrix(h_matrix_tr_ref, N, M);

    float *d_matrix;
    cudaMalloc((void **)&d_matrix, sizeof(float) * M * N);
    cudaMemcpy(d_matrix, h_matrix, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    free(h_matrix);

    // --------------------------------call device_shared_memory--------------------------------- //
    float *d_output1;
    cudaMalloc((void **)&d_output1, sizeof(float) * N * M);    // device输出内存
    float *h_output1 = (float *)malloc(sizeof(float) * N * M); // host内存, 用于保存device输出的结果

    dim3 block_size1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size1(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE)); // 根据input的形状(M行N列)进行切块
    float total_time1 = TIME_RECORD(repeat_times, ([&]
                                                   { device_shared_memory<BLOCK_SIZE><<<grid_size1, block_size1>>>(d_matrix, d_output1, M, N); }));
    cudaMemcpy(h_output1, d_output1, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output1, h_matrix_tr_ref, M * N);
    printf("[device_shared_memory] Average time: (%f) ms\n", total_time1 / repeat_times);

    cudaFree(d_output1);
    free(h_output1);

    // --------------------------------call device_padding--------------------------------- //
    float *d_output2;
    cudaMalloc((void **)&d_output2, sizeof(float) * N * M);    // device输出内存
    float *h_output2 = (float *)malloc(sizeof(float) * N * M); // host内存, 用于保存device输出的结果

    dim3 block_size2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size2(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE)); // 根据input的形状(M行N列)进行切块
    float total_time2 = TIME_RECORD(repeat_times, ([&]
                                                   { device_padding<BLOCK_SIZE><<<grid_size2, block_size2>>>(d_matrix, d_output2, M, N); }));
    cudaMemcpy(h_output2, d_output2, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output2, h_matrix_tr_ref, M * N);
    printf("[device_padding] Average time: (%f) ms\n", total_time2 / repeat_times);

    cudaFree(d_output2);
    free(h_output2);

    // --------------------------------call device_swizzling--------------------------------- //
    float *d_output3;
    cudaMalloc((void **)&d_output3, sizeof(float) * N * M);    // device输出内存
    float *h_output3 = (float *)malloc(sizeof(float) * N * M); // host内存, 用于保存device输出的结果

    dim3 block_size3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size3(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE)); // 根据input的形状(M行N列)进行切块
    float total_time3 = TIME_RECORD(repeat_times, ([&]
                                                   { device_swizzling<BLOCK_SIZE><<<grid_size3, block_size3>>>(d_matrix, d_output3, M, N); }));
    cudaMemcpy(h_output3, d_output3, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output3, h_matrix_tr_ref, M * N);
    printf("[device_swizzling] Average time: (%f) ms\n", total_time3 / repeat_times);

    cudaFree(d_output3);
    free(h_output3);

    // ---------------------------------------------------------------------------------- //
    free(h_matrix_tr_ref);

    return 0;
}