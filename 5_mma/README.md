# WMMA / MMA 学习笔记

本目录展示从 **WMMA (Warp-level Matrix Multiply Accumulate)** API 到 **原始 MMA PTX 指令** 的演进过程，理解 Tensor Core 的底层工作模式。

## 目录结构

```
5_mma/
├── CMakeLists.txt
├── common/                     # 测试框架
├── hgemm_v1_mma_m16n8k16_naive_kernel.cu  # 完整 GEMM (m16n8k16 MMA)
├── v1_simple_wmma.cu           # V1: 基础 WMMA (直接全局内存)
├── v2_shared_memory_wmma.cu    # V2: 添加共享内存
├── v3_shared_memory_wmma_padding.cu  # V3: 共享内存 Padding
├── v4_shared_memory_mma.cu     # V4: 原始 MMA PTX 指令
└── v5_shared_memory_mma_swizzle.cu  # V5: Swizzle 优化
```

## 版本演进

| 版本 | 文件 | 关键特性 |
|------|------|----------|
| V1 | `v1_simple_wmma.cu` | `wmma::load_matrix_sync` / `wmma::mma_sync` / `wmma::store_matrix_sync`，直接从全局内存加载 |
| V2 | `v2_shared_memory_wmma.cu` | 添加 `__shared__` 共享内存，减少全局内存访问 |
| V3 | `v3_shared_memory_wmma_padding.cu` | `smem_a[16][16+8]` 添加 Padding，避免 `ldmatrix` 的 bank conflict |
| V4 | `v4_shared_memory_mma.cu` | 使用原始 PTX `mma.sync.aligned.m16n8k16` 替代 wmma API，`ldmatrix.sync.aligned.x4` 加载 |
| V5 | `v5_shared_memory_mma_swizzle.cu` | Swizzle 地址变换优化 shared memory 访问模式 |

## 核心概念

### WMMA API

```cpp
using namespace nvcuda::wmma;

// 定义 fragment (16×16×16 MMA)
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
fragment<accumulator, 16, 16, 16, half> c_frag;

// 加载 → 计算 → 存储
load_matrix_sync(a_frag, A, 16);
load_matrix_sync(b_frag, B, 16);
fill_fragment(c_frag, 0.0f);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(C, c_frag, 16, mem_row_major);
```

### MMA PTX 指令

```cpp
// ldmatrix 加载 (一次加载 4 个 8-byte, 共 16×16 half)
asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
    : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)
    : "l"(__cvta_generic_to_shared(addr)));

// mma.sync 指令 (m16n8k16)
asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
             "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
             : "=r"(RD0), "=r"(RD1)
             : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3),
               "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1));
```

### Swizzle 优化

利用异或运算规避 bank conflict：

```cpp
template <uint32_t S, uint32_t B, uint32_t M>
__device__ __forceinline__ uint32_t swizzle(uint32_t addr) {
    constexpr auto Bmask = ((1 << B) - 1) << M;
    return ((addr >> S) & Bmask) ^ addr;
}


## 补充文件

### `hgemm_v1_mma_m16n8k16_naive_kernel.cu`

完整的 GEMM 实现示例，展示：
- 多 tile K 循环
- 共享内存双缓冲 (A/B)
- Warp-level MMA 计算
- 结果写回全局内存

尺寸: M=512, N=2048, K=1024