# 矩阵转置

## 如何优化**全局内存**的访问？
1. **尽量合并访问**，即连续的线程读取连续的内存，且尽量让访问的全局内存的首地址是32字节（一次数据传输处理的数据量）的倍数；
2. 如果不能同时合并读取和写入，则应该**尽量做到合并写入**。

## 如何利用**共享内存**优化矩阵转置？

### 问题背景
矩阵转置的特殊性：读操作和写操作的访问模式是**transpose（转置）关系**，无法同时做到合并读和合并写。

### 解决方案一：使用共享内存中转
利用共享内存中转，读操作和写操作都是合并的，但是**存在 bank conflict**。

```cpp
__shared__ float shared[TILE_DIM][TILE_DIM];
// 读操作：合并访问
shared[threadIdx.y][threadIdx.x] = input[y1 * N + x1];
// 写操作：合并访问，但存在bank冲突
output[y2 * M + x2] = shared[threadIdx.x][threadIdx.y];
```

**Bank Conflict 原因**：同一个 warp 中的 32 个线程（连续的 threadIdx.x 值）访问共享内存中跨度为 32 的数据，这 32 个线程恰好访问同一个 bank 中的 32 个数据，导致 32 路 bank 冲突。

### 解决方案二：对共享内存做 padding
通过对共享内存做 padding（每行多一个元素），解决 bank conflict。

```cpp
__shared__ float shared[TILE_DIM][TILE_DIM + 1];
```

**原理**：做 padding 后，同一个 warp 中的 32 个线程访问共享内存中跨度为 33 的数据。如果第一个线程访问第一个 bank 中的第一层，第二个线程访问第二个 bank 中的第二层，以此类推，32 个线程访问 32 个不同 bank，不存在 bank 冲突。

### 解决方案三：使用 swizzling
使用 swizzling（异或运算）解决 bank conflict，不需要对共享内存做 padding。

```cpp
// 写入时swizzling
shared[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[y1 * N + x1];
// 读出时swizzling
output[y2 * M + x2] = shared[threadIdx.x][threadIdx.x ^ threadIdx.y];
```

**swizzling 利用了异或运算的两个性质**：
1. **运算的封闭性**：保证充分利用 shared memory 的空间
2. **x1^y != x2^y 当且仅当 x1 != x2**：保证 warp 中的各个线程不会访问同一 bank

**变换示例**：
| 原位置 | 变换后 |
|--------|--------|
| (0,0),(0,0),(0,0)... | (0,0),(0,1),(0,2),(0,3)... |
| (1,1),(1,1),(1,1)... | (1,1),(1,0),(1,3),(1,2)... |
| (2,2),(2,2),(2,2)... | (2,2),(2,3),(2,0),(2,1)... |
| (3,3),(3,3),(3,3)... | (3,3),(3,2),(3,1),(3,0)... |

## 代码实现
- `host_transpose`：CPU 基准实现
- `device_shared_memory`：共享内存版本（存在 bank conflict）
- `device_padding`：共享内存 + padding 版本
- `device_swizzling`：共享内存 + swizzling 版本
