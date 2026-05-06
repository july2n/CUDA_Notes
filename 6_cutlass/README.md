# CUTLASS 学习笔记

本目录基于 [CUTLASS 官方例程](https://github.com/NVIDIA/cutlass/tree/master/examples) 构建，展示从 CUTLASS 2.x 到 3.x 各阶段的 GEMM 实现，是理解 CUTLASS 编程模型的最小可执行目录。

## 目录结构

```
6_cutlass/
├── CMakeLists.txt          # 构建配置
├── common/                 # 共享头文件
├── v1_print_half.cu        # half_t 类型打印（环境验证）
├── v2_turing_tensorop_gemm.cu  # CUTLASS 2.x — Turing TensorCore GEMM
├── v3_gemm_bias_relu.cu    # CUTLASS 2.x — GEMM + Bias + ReLU
├── v4_hopper_collective_builder.cu  # CUTLASS 3.x — CollectiveBuilder (Hopper)
└── gemm_softmax/           # GEMM + Softmax（自定义 EpilogueVisitor）
    ├── CMakeLists.txt
    ├── gemm_softmax.cu
    ├── gemm_with_softmax.h
    └── gemm_with_epilogue_visitor.h
```

## 各版本详解

### v1 — 编译环境验证

**源文件**: `v1_print_half.cu`

验证 CUTLASS `half_t`（即 `__half`）类型是否可正常构造和打印。

```cpp
cutlass::half_t x = 2.25_hf;
std::cout << x << std::endl;  // 输出: 2.25
```

### v2 — CUTLASS 2.x 风格：Turing TensorCore GEMM

**源文件**: `v2_turing_tensorop_gemm.cu`
**笔记**: [初学CUTLASS 2.x — 以 Turing Tensor Core GEMM 为例](https://zhuanlan.zhihu.com/p/2032866203585209114)

CUTLASS 2.x 的核心是 `cutlass::gemm::device::Gemm` 模板，将 GEMM 内核拆解为若干可配置的组件：

```
D = alpha * A * B + beta * C
```

| 组件 | 描述 |
|------|------|
| `ElementInputA/B` | 输入矩阵元素类型（int8_t） |
| `ElementOutput` | 输出矩阵元素类型（int32_t） |
| `LayoutInputA/B` | 矩阵内存布局（RowMajor/ColumnMajor） |
| `ShapeMMAThreadBlock` | ThreadBlock tile 大小（128×256×64） |
| `ShapeMMAWarp` | Warp tile 大小（64×64×64） |
| `ShapeMMAOp` | MMA operation tile 大小（8×8×16） |
| `EpilogueOp` | Epilogue 计算（LinearCombination: alpha*X+beta*C） |
| `SwizzleThreadBlock` | ThreadBlock 调度策略 |
| `NumStages` | MMA pipeline 阶段数（2） |

内核层次结构：

```
ThreadBlock Tile (128×256×64)
  └── Warp Tile (64×64×64)
       └── MMA Op Tile (8×8×16)
```

### v3 — CUTLASS 2.x 风格：GEMM + Bias + ReLU

**源文件**: `v3_gemm_bias_relu.cu`

在 v2 的基础上，将 Epilogue 从 `LinearCombination` 替换为 `LinearCombinationRelu`，在 GEMM 完成后对每个输出元素执行 `max(0, x)`（ReLU）：

```cpp
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue
>;
```

### v4 — CUTLASS 3.x：CollectiveBuilder（Hopper 架构）

**源文件**: `v4_hopper_collective_builder.cu`
**笔记**: [初学CUTLASS 3.x — collective_builder及配置调优](https://zhuanlan.zhihu.com/p/2033131870070757357)

CUTLASS 3.x 引入了 `CollectiveBuilder`，它将 GEMM 分解为 **Producer**（数据加载）和 **Consumer**（计算）两个阶段，采用双 pipeline（`Math` + ` prologue/epilogue`）更好地隐藏内存访问延迟：

```
Producer:  global → register → shared memory
             ↑                          ↓
Consumer:  register ← MMA ← shared memory ← register
```

CUTLASS 3.x 的主要变化：
- 使用 `cutlass::gemm::collective::CollectiveBuilder` 替代 `cutlass::gemm::device::Gemm`
- Warp-level MMA 直接操作 register，无需 shared memory 中转（`cluster` 支持）
- `Mainloop` 包含 Producer + MMA，共享内存用于 Producer↔Consumer 通信
- `Epilogue` 负责 Scale/Bias/Activation 等后处理


### gemm_softmax — 自定义 EpilogueVisitor（Softmax）

**目录**: `gemm_softmax/`

本例展示如何在 CUTLASS 中实现自定义 epilogue。以 **GEMM + Softmax** 为例：

```
Y = softmax(X · W + b)  其中 X·W 是一个 GEMM 操作
```

实现方式：
- `gemm_with_softmax.h`: 基于 `cutlass::epilogue::thread::Visitor` 模式，在 GEMM 的 epilogue 阶段直接追加 softmax 计算，而非事后处理
- `gemm_with_epilogue_visitor.h`: 通用 `EpilogueVisitor` 模式，支持链式后处理（Scale、Add、Bias、Softmax 等）

这种设计的优势在于：softmax 的 reduce 操作可以融入 GEMM 的 epilogue 阶段，减少额外的 kernel  launch开销和全局内存访问。

## CUTLASS 版本演进

| 版本 | 典型 API | 架构支持 | 核心抽象 / 特性 |
|------|----------|----------|------------------|
| CUTLASS 2.x | `gemm::device::Gemm` (Legacy) | Volta/Turing/Ampere | 基于 Iterator 的分层平铺；Epilogue Fusion；Thread-level 映射 |
| CUTLASS 3.x (High-level) | `CollectiveBuilder` / `gemm::kernel::GemmUniversal` | Ampere/Hopper/Blackwell | 声明式编程；将 Mainloop 与 Epilogue 解耦；自动生成最优的 Collective 算子 |
| CuTe (3.x+ 底层) | `cute::Layout`, `cute::Tensor`, `TiledMMA` | Volta (SM70) 及以上 | Layout 映射（Strided Indexing）；Hierarchical Tiling；将硬件指令（TMA, WGMMA）抽象为 Atom |

## 常见问题

**Q: 编译报错 "cannot find cutlass"**
A: 修改 `CMakeLists.txt` 中的 `CUTLASS_PATH` 和 `CUTLASS_UTIL_PATH` 为本机 CUTLASS 安装路径。

**Q: Hopper kernel 在非 Hopper 架构上运行失败**
A: `v4` 需要 Sm90 (Hopper) 或更高架构。Turing/Ampere 用户请从 v2/v3 开始。

**Q: 如何选择 tile size**
A: 小 tile 适合小矩阵，大 tile 适合大矩阵。具体数值参考 CUTLASS 官方 `test/` 目录中各架构的 benchmark 结果。
