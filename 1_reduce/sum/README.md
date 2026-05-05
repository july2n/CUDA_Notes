# CUDA Reduce Sum 性能优化

本项目对比两种 reduce sum 实现：**shared memory 版本** vs **warp shuffle + float4 版本**。

## Kernel 1: Shared Memory + 线程同步

```cpp
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
```

### 原理
- 每个 block 将数据从 global memory 拷贝到 shared memory
- 树形归约：每轮减半，需要 `log2(BLOCK_SIZE)` 次 `__syncthreads()`


## Kernel 2: Warp Shuffle + Float4

```cpp
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
```

### 改进点 1: Warp Shuffle 避免线程同步

**传统 shared memory 归约**需要多次 `__syncthreads()`：
```
Step 1: __syncthreads() → 所有 warp 同步
Step 2: __syncthreads() → 所有 warp 同步
...
```

**Warp shuffle 指令** `__shfl_down_sync` 在同一 warp 内硬件级同步：
- 5 次半减（16→8→4→2→1）全部在寄存器上完成
- **无需任何显式同步**

```cpp
// warp 内归约：0 次 syncthreads
for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
}
```

### 改进点 2: 向量化访存 Float4

每个线程一次处理 4 个 float：

```cpp
// ❌ 4 次内存访问
local_sum += input[idx];
local_sum += input[idx + 1];
local_sum += input[idx + 2];
local_sum += input[idx + 3];

// ✅ 1 次内存访问（128 位）
float4 tmp = FLOAT4(input[idx]);
local_sum = tmp.x + tmp.y + tmp.z + tmp.w;
```

### 改进点 3: 线程组织方式

**每个线程处理 4 个元素**，grid 尺寸变为 `CEIL(N, BLOCK_SIZE * 4)`：

```cpp
int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
```


## 性能对比

| 特性 | Shared Memory | Warp Shuffle + Float4 |
|------|----------------|----------------------|
| 内存访问次数 | N 次 | N/4 次 |
| `__syncthreads()` 调用次数 | log2(BLOCK_SIZE) 次 | 1 次 |
| 代码复杂度 | 简单 | 中等 |

