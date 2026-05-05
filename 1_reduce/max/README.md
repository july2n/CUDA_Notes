# CUDA Warp Shuffle 归约：最大值计算

## 1. 问题描述

- **输入**: `(256, 768)` 的二维矩阵
- **输出**: `(256, 1)` 每行元素的最大值
- **目标**: 利用 Warp Shuffle 实现高效的行级归约

## 2. Grid-Block-Warp-Thread 层级映射

```
Grid: 256 blocks         → 每个 block 处理一行
  └─ Block 0..255
       └─ Block_Size = 768 threads
            └─ warpSize = 32
                 └─ 24 warps per block (768 / 32 = 24)
```

| 层级 | 数量 | 作用 |
|------|------|------|
| Grid | 256 | 并行处理 256 行 |
| Block | 768 threads | 每行一个 block，处理 768 元素 |
| Warp | 32 threads | CUDA 基本执行单元，并行归约 |
| Thread | - | 加载数据 + Warp Shuffle 参与归约 |


## 3. Warp Shuffle 归约：避免线程同步等待

### 3.1 传统共享内存归约的问题

```
Thread 0 ──┐
Thread 1 ──┼──> 共享内存写入 ──> __syncthreads() ──> 共享内存读取 ──> ...
   ...     │
Thread N  ─┘
```

- 每次归约迭代都需要 `__syncthreads()` 同步
- 同步操作有硬件开销，阻塞所有线程直到全体完成

### 3.2 Warp Shuffle 归约的优势

```cpp
for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
}
```

- **无需同步**: `__shfl_down_sync` 是同一 Warp 内的逻辑移寄存器交换，硬件保证原子性
- **低延迟**: 数据通过互联网络直接传递，不经过共享内存
- **高效率**: 每轮迭代使有效数据量翻倍，log₂(32)=5 次迭代完成 32 元素归约

### 3.3 两阶段归约流程

```
阶段1: Warp 内部归约 (每个 Warp 输出 1 个结果)
  Thread 0~31: val = max(val, shfl_down(val, 16))
  Thread 0~31: val = max(val, shfl_down(val, 8))
  ...
  Thread 0:    val = max(val, shfl_down(val, 1)) → 得到 Warp 内最大值

阶段2: 第一个 Warp 汇总所有 Warp 结果
  Warp 0 的 lane 0~23 读取 s_mem[0]~s_mem[23]
  Warp 0 内再次执行 5 次 Shuffle 归约 → 最终最大值
```

## 4. 预期输出

每行输出 `output` 和 `output_ref` 对比，验证 CUDA 结果正确性。