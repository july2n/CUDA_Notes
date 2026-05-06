# Elementwise Add with Vectorized Memory Access

## 代码功能

向量加法 `c = a + b`，演示 CUDA 中利用向量化访存提升带宽性能。

## 核心知识点

### 1. 向量化访存 (Vectorized Memory Access)

GPU 显存访问以 **128 字节 cache line** 为基本单元。散列访问（每个线程独立访问 4 字节）无法有效利用总线带宽。

向量化访存的核心思想：**多个相邻元素合并为一次内存访问**。以 float 数据为例：

| 类型 | 元素大小 | 一次访存元素数 | 总字节数 |
|------|----------|----------------|----------|
| float  | 4 字节  | 4              | 16 字节  |
| half   | 2 字节  | 8              | 16 字节  |

通过 `float4`、`half2` 等向量类型，编译器可生成合并的内存访问指令，一次读写 128 位（16 字节）。

```cuda
float4 temp_a = FLOAT4(a[idx]);  // 一次读取 16 字节，而非 4 次 4 字节
```

### 2. 向量化访存实现原理

使用 `reinterpret_cast` 将指针强制转换为向量类型指针，再解引用：

```cuda
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
```

这要求：
- 数据地址 **128 位对齐**（16 字节边界）
- 连续访问的元素数量是向量类型的整数倍

### 3. Thread 与 Element 的映射

每个线程处理多个元素，线程索引需要乘以每线程处理元素数：

```cuda
int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;  // float4: 每线程4元素
int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);  // half8: 每线程8元素
```

### 4. 边界处理

向量化后一个线程处理多个元素，需检查是否会越界：

```cuda
if ((idx + 3) < N) {  // float4 处理 4 个元素
    // 向量化访问
} else if (idx < N) {
    // 尾部元素，逐个处理
}
```

## 不同数据类型的向量化方案

### Float32 (float4)

float 占 4 字节，`float4` 包含 4 个 float，共 16 字节（128 位）：

```cuda
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void elementwise_add_float32x4_pack_kernel(float *a, float *b, float *c, int N)
{
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
    // 边界处理...
}
```

### Float16 (half) - 使用 half2 和 float4 组合

half 占 2 字节，直接用 `half2` 只能一次访存 4 字节（32 位），效率不高。**利用 float4 作为 128 位操作的临时寄存器**，每次处理 8 个 half：

```cuda
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c, int N)
{
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if ((idx + 7) < N)
    {
        half pack_a[8], pack_b[8], pack_c[8];  // 8 x 2字节 = 16字节 = 128位

        // 用 float4 一次性加载 128 位
        LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
        LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

        // 使用 __hadd2 同时处理两个 half
        #pragma unroll
        for (int i = 0; i < 8; i += 2)
        {
            HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
        }

        // 用 float4 一次性存储 128 位
        LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
    }
    // 边界处理...
}
```

### 通用宏说明

| 宏定义 | 作用 | 适用场景 |
|--------|------|----------|
| `FLOAT4(value)` | 将变量按 float4 向量类型解释 | float 数据类型的向量化读写 |
| `HALF2(value)` | 将变量按 half2 向量类型解释 | half 数据类型的向量化计算 |
| `LDST128BITS(value)` | 按 float4（128位）解释变量 | 任意 128 位一次性读写 |
| `CEIL(a, b)` | 向上整除 | 计算 grid 大小等 |

### 向量化数据类型选择

```
内存对齐要求: 起始地址必须是类型大小的整数倍
  float4 要求: 16 字节对齐
  half2  要求: 4  字节对齐
  float2 要求: 8  字节对齐

选择依据:
  - 数据类型大小决定每次访存的元素数量
  - 确保总字节数达到 128 位以充分利用 memory transaction
```
