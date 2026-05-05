# CUDA Notes

CUDA 编程学习笔记，涵盖 CUDA Core、Tensor Core 与 CUTLASS 三大板块。

## 项目结构

```
CUDA_Notes/
├── CMakeLists.txt              # CMake 配置
├── 0_elementwise/              # 向量加法：向量化访存
├── 1_reduce/                   # 规约：shared memory 与 warp shuffle
├── 2_sgemm/                    # 矩阵乘法：shared memory 版
├── 3_transpose/                # 矩阵转置 (TODO)
├── 4_gemv/                     # 矩阵向量乘法 (TODO)
├── 5_mma/                      # Tensor Core MMA 操作 (TODO)
└── 6_cutlass/                  # CUTLASS 库使用 (TODO)
```

## 核心概念

### CUDA Core (流式多处理器)

- **Streaming Multiprocessor (SM)**: GPU 基本执行单元
- **Thread Block**: 共享 shared memory 的线程组
- **Warp**: 32 个线程为一组，SIMT 执行
- **Register**: 每个线程的私有寄存器
- **Shared Memory**: block 内线程共享，bandwidth 是 global memory 的 ~10x

### Tensor Core

- **WMMA (Warp-level Matrix Multiply Accumulate)**: 在 CUDA Core 上模拟 Tensor Core 访存模式
- 支持 `mma.sync.m16n8k8.f16.f16.f16.f16` 等指令
- 比 CUDA Core 的矩阵乘法吞吐量大 8-16x

### CUTLASS

NVIDIA 官方的高性能 GEMM 库模板，基于 C++ 模板和 CUDA 抽象，封装了最优的 WMMA/Tensor Core 实现。

## 目录详解

### 0_elementwise - 向量化访存 (CUDA Core)

学习向量化访存提升带宽利用率。GPU 显存访问以 128 字节 cache line 为基本单元，散列访问无法充分利用带宽。通过 `float4`、`half2` 等向量类型合并内存访问。

**核心知识点:**
- 向量化访存原理
- `reinterpret_cast` 实现向量类型指针
- 128 位对齐要求
- float4 / half2 混合使用

**代码文件:** `add.cu`

---

### 1_reduce - 规约操作 (CUDA Core)

学习并行规约的两种经典实现：shared memory 多次同步 和 warp shuffle 单指令多数据。

**核心知识点:**
- `__syncthreads()` 线程同步
- 规约树的两路、三路展开
- `__shfl_down_sync` warp 内数据交换
- `atomicAdd` 多 block 结果合并

**代码文件:**
- `sum/` - 求和规约
- `max/` - 最大值规约

---

### 2_sgemm - 单精度矩阵乘法 (CUDA Core)

从零实现 FP32 矩阵乘法 `C = α * A * B + β * C`，重点学习 tiling 与 shared memory 优化。

**核心知识点:**
- 矩阵分块 (tiling) 策略
- 寄存器版外积计算
- shared memory 存储 A/B tile
- bank conflict 避免

**代码文件:**
- `shared_memory.cu` - 基于 shared memory 的矩阵乘法
- `register_outer_product_float4.cu` - 寄存器外积优化版

---

### 3_transpose - 矩阵转置 (TODO)

转置是 memory-bound 操作的典型代表，将学习 coalesced access 与 bank conflict 优化。

---

### 4_gemv - 矩阵向量乘法 (TODO)

学习 memory-bound 场景下的访存优化。

---

### 5_mma - Tensor Core WMMA (Tensor Core)

学习使用 `nvcuda::wmma` API 实现 warp-level 矩阵乘法，理解 Tensor Core 的工作模式。

**核心知识点:**
- `wmma::load_matrix_sync` / `wmma::store_matrix_sync`
- `wmma::mma_sync` 指令
- fragment 概念与 layout (row_major / col_major)
- Tensor Core 的 16x16x16 mma 结构

---

### 6_cutlass - CUTLASS 库 (CUTLASS)

学习使用 NVIDIA CUTLASS 库实现高性能矩阵乘法。

**核心知识点:**
- CUTLASS 模板结构 (`kernel`, `threadblock`, `warp`)
- `cutlass::gemm::device::Gemm` 接口
- 自定义 kernel 与 epilogue

---

## 编译与运行

### 编译

```bash
cd CUDA_Notes
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 运行

各子目录为独立可执行文件，运行前确保处于 `build/` 目录：

```bash
# 向量加法
./0_elementwise/add

# 规约
./1_reduce/sum
./1_reduce/max

# 矩阵乘法
./2_sgemm/shared_memory
./2_sgemm/register_outer_product_float4
```


## 学习路线

```
0_elementwise (向量化访存基础)
         ↓
1_reduce (并行规约与同步)
         ↓
2_sgemm (矩阵乘法 tiling 与 shared memory)
         ↓
5_mma (Tensor Core warp-level mma)
         ↓
6_cutlass (生产级高性能 GEMM)
```

## 参考资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
