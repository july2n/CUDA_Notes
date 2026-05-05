# CUTE GEMM 实现

本目录包含使用 NVIDIA CUTE (CUDA Template Engine) 库实现的矩阵乘法 (GEMM) 优化版本。

## 目录结构

```
7_cute/
├── CMakeLists.txt      # CMake 构建配置
├── utils.h             # 测试工具: 性能测试、精度验证
├── v1_native_gemm.cu    # V1: 基础版本 (无共享内存)
├── v2_shared_memory.cu # V2: 添加共享内存优化
├── v3_epilogue.cu      # V3: 添加 Epilogue (寄存器→共享内存→全局)
└── v4_multistage.cu    # V4: 多级流水线 (K-stage 预取)
```

## 版本说明

### V1: Native GEMM (`v1_native_gemm.cu`)
- 直接从全局内存加载数据到寄存器
- 使用 `local_tile` 将矩阵划分为 block tile
- 通过 `TiledMMA` 和 `ThrMMA` 将 tile 分配给线程
- 数据路径: Global → Register → MMA → Register → Global

### V2: Shared Memory (`v2_shared_memory.cu`)
- 引入共享内存作为缓存
- 使用 `cp_async` 异步拷贝加速 global→shared
- 使用 `Swizzle` 布局优化 shared memory 访问模式
- 数据路径: Global → Shared → Register → MMA → Register → Global

### V3: Epilogue (`v3_epilogue.cu`)
- 结果通过共享内存写回 (而非直接寄存器→全局)
- 支持更大的指令宽度 (wider instructions)
- 使用 `make_tiled_copy_C` 实现 R2S 和 S2G 两阶段拷贝
- 数据路径: MMA → Register → Shared → Register → Global

### V4: Multi-Stage (`v4_multistage.cu`)
- K方向 4-stage 流水线: 计算与访存重叠
- 预取 `kStage-1` 个 tile 到共享内存
- 双缓冲技术避免共享内存-bank冲突
- 最大化 GPU 内存带宽利用率

## 共同特性

| 特性 | 说明 |
|------|------|
| 数据类型 | `cute::half_t` (FP16) |
| MMA 操作 | `SM80_16x8x16_F16F16F16F16_TN` |
| Block Size | BM=128, BN=256, BK=32 |
| Shared Layout | Swizzle<3,3,3> 优化 bank conflict |
| Copy 策略 | G2S: `SM80_CP_ASYNC_CACHEGLOBAL`, S2R: `SM75_U32x4_LDSM_N` |

## 构建

```bash
cd build/7_cute
cmake ..
make
```

生成的可执行文件:
- `v1_native_gemm`
- `v2_shared_memory`
- `v3_epilogue`
- `v4_multistage`

## 测试方法

每个版本运行后会:
1. 计算前5个矩阵的精度误差 (vs cuBLAS)
2. 测试64种不同尺寸 (256×256 到 16384×16384) 的性能

输出格式:
```
M N K = xxxxxx xxxxxx xxxxxx, Time = min avg max s, AVG Performance = xxxx Gflops
```