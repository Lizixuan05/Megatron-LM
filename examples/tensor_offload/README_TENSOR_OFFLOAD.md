# Tensor Offload 技术实现指南

## 目录
1. [概述](#概述)
2. [实现原理](#实现原理)
3. [使用方法](#使用方法)
4. [代码解析](#代码解析)
5. [性能分析](#性能分析)
6. [常见问题](#常见问题)

## 概述

Tensor Offload 是一种内存优化技术，通过将暂时不用的模型参数从GPU内存转移到CPU内存，在需要时再加载回来，从而减少GPU显存占用。这个实现是为了学习和实践目的，展示了如何在Megatron-LM中集成这样的功能,性能较差

### 核心思想

```
训练流程：
1. Layer N forward 前   → 将 Layer N 的参数从 CPU 预取到 GPU
2. Layer N forward      → 使用 GPU 上的参数进行计算
3. Layer N backward 后  → 将 Layer N 的参数 offload 回 CPU
```

## 实现原理

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                   TensorOffloadManager                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Layer Registry                                      │  │
│  │  layer_id → [param1, param2, ...]                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Operations                                          │  │
│  │  • prefetch_layer()  : CPU → GPU                    │  │
│  │  • offload_layer()   : GPU → CPU                    │  │
│  │  • prefetch_next_layers() : 预取多层                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
           ↓                           ↑
    ┌──────────────┐          ┌──────────────┐
    │ forward hook │          │backward hook │
    │  (prefetch)  │          │  (offload)   │
    └──────────────┘          └──────────────┘
           ↓                           ↑
    ┌─────────────────────────────────────┐
    │      TransformerLayer               │
    │  forward() → backward()             │
    └─────────────────────────────────────┘
```

### 关键组件

#### 1. TensorOffloadManager
位置: `megatron/core/tensor_offload.py`

主要方法：
- `register_layer(layer_id, module)`: 注册需要管理的层
- `prefetch_layer(layer_id)`: 将参数加载到GPU
- `offload_layer(layer_id)`: 将参数转移到CPU
- `on_forward_pre_hook(layer_id)`: forward前的hook
- `on_backward_post_hook(layer_id)`: backward后的hook

#### 2. TransformerLayer Hooks
位置: `megatron/core/transformer/transformer_layer.py`

修改点：
- `__init__`: 注册layer到offload manager
- `forward`: 添加pre/post hooks

#### 3. 训练循环集成
位置: `megatron/training/training.py`

- 在 `get_model()` 中初始化offload manager
- 在 `train()` 结束时打印统计信息

#### 4. 命令行参数
位置: `megatron/training/arguments.py`

新增参数：
- `--enable-tensor-offload`: 启用tensor offload
- `--tensor-offload-pin-memory`: 使用pinned memory
- `--tensor-offload-num-prefetch-layers`: 预取层数

## 使用方法

### 基本用法

```bash
# 基本命令（在你原有的训练命令基础上添加以下参数）
python pretrain_gpt.py \
    --enable-tensor-offload \
    --tensor-offload-pin-memory \
    --tensor-offload-num-prefetch-layers 2 \
    [其他训练参数...]
```

### 完整示例

```bash
#!/bin/bash

# 设置基本路径
MEGATRON_PATH=/data/home/scyb226/lzx/Megatron-LM
DATA_PATH=/path/to/your/data
CHECKPOINT_PATH=/path/to/checkpoint

# GPT-3 125M 模型配置
python ${MEGATRON_PATH}/pretrain_gpt.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --save ${CHECKPOINT_PATH} \
    --load ${CHECKPOINT_PATH} \
    --data-path ${DATA_PATH} \
    --vocab-file ${DATA_PATH}/vocab.json \
    --merge-file ${DATA_PATH}/merges.txt \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --checkpoint-activations \
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --fp16 \
    --enable-tensor-offload \
    --tensor-offload-pin-memory \
    --tensor-offload-num-prefetch-layers 2
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-tensor-offload` | flag | False | 是否启用tensor offload |
| `--tensor-offload-pin-memory` | flag | True | 是否使用pinned memory（加速传输） |
| `--tensor-offload-num-prefetch-layers` | int | 1 | 提前预取几层（隐藏延迟） |

## 代码解析

### 1. 核心数据结构

```python
class TensorOffloadManager:
    def __init__(self):
        # 存储每层的参数列表
        self.layer_params: Dict[int, List[torch.nn.Parameter]] = OrderedDict()
        
        # 跟踪参数当前在CPU还是GPU
        self.layer_param_devices: Dict[int, List[str]] = {}
        
        # 用于异步传输的CUDA stream
        self.prefetch_stream = torch.cuda.Stream()
        
        # 统计信息
        self.stats = {
            'total_offloaded_bytes': 0,
            'total_prefetched_bytes': 0,
            'num_offload_ops': 0,
            'num_prefetch_ops': 0,
        }
```

### 2. 参数注册

当模型创建时，每个TransformerLayer会注册自己：

```python
# 在 TransformerLayer.__init__ 中
from megatron.core.tensor_offload import get_offload_manager
offload_manager = get_offload_manager()
if offload_manager is not None:
    offload_manager.register_layer(self.layer_number, self)
```

### 3. Forward Hook

在layer forward之前，确保参数在GPU上：

```python
def forward(self, *args, **kwargs):
    # Pre-hook: 预取参数到GPU
    offload_manager = get_offload_manager()
    if offload_manager is not None:
        offload_manager.on_forward_pre_hook(self.layer_number)
    
    # 正常的forward计算
    hidden_states, context = self._forward_attention(*args, **kwargs)
    output = self._forward_mlp(hidden_states, ...)
    
    # Post-hook: 注册backward hook
    if offload_manager is not None:
        if output.requires_grad:
            def backward_hook(grad):
                offload_manager.on_backward_post_hook(self.layer_number)
                return grad
            output.register_hook(backward_hook)
    
    return output, context
```

### 4. 参数移动

CPU到GPU (prefetch):

```python
def prefetch_layer(self, layer_id: int):
    params = self.layer_params[layer_id]
    for param in params:
        if param.data.device.type == 'cpu':
            # 创建GPU tensor
            gpu_tensor = torch.empty(
                param.size(),
                dtype=param.dtype,
                device=torch.cuda.current_device()
            )
            # 异步复制
            with torch.cuda.stream(self.prefetch_stream):
                gpu_tensor.copy_(param.data, non_blocking=True)
            # 更新参数指针
            param.data = gpu_tensor
```

GPU到CPU (offload):

```python
def offload_layer(self, layer_id: int):
    params = self.layer_params[layer_id]
    for param in params:
        if param.data.device.type == 'cuda':
            # 创建CPU tensor (使用pinned memory)
            cpu_tensor = torch.empty(
                param.size(),
                dtype=param.dtype,
                device='cpu',
                pin_memory=self.pin_memory
            )
            # 复制数据
            cpu_tensor.copy_(param.data)
            # 更新参数指针
            param.data = cpu_tensor
```

### 5. 预取优化

为了隐藏传输延迟，提前预取接下来的几层：

```python
def prefetch_next_layers(self, current_layer_id: int):
    layer_ids = list(self.layer_params.keys())
    current_idx = layer_ids.index(current_layer_id)
    
    # 异步预取接下来的num_prefetch_layers层
    for i in range(1, self.num_prefetch_layers + 1):
        next_idx = current_idx + i
        if next_idx < len(layer_ids):
            next_layer_id = layer_ids[next_idx]
            self.prefetch_layer(next_layer_id, async_op=True)
```

## 性能分析

### 内存使用

假设一个12层的GPT模型，每层参数约100MB：

**不使用offload:**
```
GPU内存 = 12层 × 100MB = 1200MB
CPU内存 = 0MB
```

**使用offload (prefetch=1):**
```
GPU内存 = 2层 × 100MB = 200MB  (当前层 + 预取1层)
CPU内存 = 10层 × 100MB = 1000MB
显存节约 = 1000MB (83%)
```

### 时间开销

关键因素：
1. **PCIe带宽**: 通常为 16GB/s (PCIe 3.0 x16)
2. **参数大小**: 100MB 的层需要约 6ms 传输时间
3. **预取机制**: 可以隐藏大部分延迟

**理想情况** (预取完全隐藏延迟):
```
额外时间 ≈ 0ms
```

**最坏情况** (没有预取):
```
每层额外时间 = 2 × 传输时间 = 2 × 6ms = 12ms
总额外时间 = 12层 × 12ms = 144ms per iteration
```

### 实测建议

1. **小模型** (< 1B参数): offload可能得不偿失
2. **中等模型** (1B-10B): 需要权衡内存和速度
3. **大模型** (> 10B): 当显存不够时很有用

## 常见问题

### Q1: 为什么训练变慢了？

**A:** Tensor offload会增加CPU-GPU通信时间。优化方法：
- 增加 `--tensor-offload-num-prefetch-layers` 来预取更多层
- 确保使用了 `--tensor-offload-pin-memory`
- 检查PCIe带宽是否足够

### Q2: 如何验证offload是否工作？

**A:** 查看训练日志中的统计信息：
```
============================================================
Tensor Offload Statistics:
  Total offloaded: 12000.00 MB (1000 operations)
  Total prefetched: 12000.00 MB (1000 operations)
============================================================
```

### Q3: 可以offload更多东西吗？

**A:** 可以！当前实现只offload了参数。你可以扩展来offload：
- 优化器状态 (momentum, variance)
- 激活值 (activation checkpointing的补充)
- 梯度

### Q4: 如何调试offload相关问题？

**A:** 在代码中设置日志级别：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

然后查看详细的offload操作日志。

### Q5: Pipeline并行是否兼容？

**A:** 兼容！每个pipeline stage会独立管理自己的层。但要注意：
- pipeline bubble时间可能增加
- 可能需要调整 `num_prefetch_layers`

## 进阶扩展

### 1. 选择性Offload

只offload某些特定的层：

```python
def register_layer(self, layer_id: int, module: nn.Module):
    # 只offload偶数层
    if layer_id % 2 == 0:
        # ... 注册逻辑
```

### 2. Offload优化器状态

在 `TensorOffloadManager` 中添加：

```python
def offload_optimizer_state(self, optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            # 将momentum、variance等移到CPU
            for key in state:
                if isinstance(state[key], torch.Tensor):
                    state[key] = state[key].cpu()
```

### 3. 自适应预取

根据实际传输时间动态调整预取层数：

```python
def adapt_prefetch_layers(self):
    # 测量传输时间
    transfer_time = self.measure_transfer_time()
    compute_time = self.measure_compute_time()
    
    # 调整预取层数
    if transfer_time > compute_time:
        self.num_prefetch_layers += 1
    elif transfer_time < compute_time / 2:
        self.num_prefetch_layers = max(1, self.num_prefetch_layers - 1)
```

### 4. GPU内存监控

添加自动内存监控：

```python
def should_offload(self):
    # 检查GPU内存使用率
    free_memory = torch.cuda.mem_get_info()[0]
    total_memory = torch.cuda.mem_get_info()[1]
    usage_ratio = 1 - (free_memory / total_memory)
    
    # 如果使用率超过80%，启用offload
    return usage_ratio > 0.8
```

**作者注**: 这个实现主要用于作者自学实践的目的。在生产环境中，建议使用经过充分测试的框架如DeepSpeed、FSDP等。

