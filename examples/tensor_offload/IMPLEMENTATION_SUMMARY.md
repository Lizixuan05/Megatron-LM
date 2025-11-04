# Tensor Offload 架构说明

## 🏗️ 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                            │
│                    (megatron/training/training.py)              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ get_model(): 初始化                                       │  │
│  │   └─> initialize_offload_manager()                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│  ┌────────────────────────▼──────────────────────────────────┐ │
│  │ train(): 训练循环                                         │ │
│  │   └─> train_step()                                        │ │
│  │        └─> forward_backward_func()                        │ │
│  │             └─> model.forward()                           │ │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TransformerLayer                             │
│           (megatron/core/transformer/transformer_layer.py)      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ __init__():                                               │  │
│  │   └─> offload_manager.register_layer(layer_id, self)     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│  ┌────────────────────────▼──────────────────────────────────┐ │
│  │ forward():                                                │ │
│  │   1. offload_manager.on_forward_pre_hook(layer_id)       │ │
│  │   2. 执行forward计算                                      │ │
│  │   3. offload_manager.on_forward_post_hook(layer_id)      │ │
│  │   4. 注册backward_hook                                    │ │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TensorOffloadManager                           │
│                (megatron/core/tensor_offload.py)                │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Data Structures:                                          │  │
│  │   • layer_params: {layer_id → [param1, param2, ...]}    │  │
│  │   • layer_param_devices: {layer_id → ['cpu'/'cuda']}    │  │
│  │   • prefetch_stream: torch.cuda.Stream()                 │  │
│  │   • stats: 统计信息                                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Core Methods:                                             │  │
│  │                                                           │  │
│  │  register_layer(layer_id, module)                        │  │
│  │    └─> 收集layer的所有参数                                │  │
│  │                                                           │  │
│  │  on_forward_pre_hook(layer_id)                           │  │
│  │    ├─> prefetch_layer(layer_id)        [CPU → GPU]      │  │
│  │    └─> prefetch_next_layers(layer_id)  [异步预取]        │  │
│  │                                                           │  │
│  │  on_backward_post_hook(layer_id)                         │  │
│  │    └─> offload_layer(layer_id)         [GPU → CPU]      │  │
│  │                                                           │  │
│  │  prefetch_layer(layer_id)                                │  │
│  │    └─> 遍历参数，创建GPU tensor，异步copy                │  │
│  │                                                           │  │
│  │  offload_layer(layer_id)                                 │  │
│  │    └─> 遍历参数，创建CPU tensor (pinned)，copy           │  │
│  │                                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                    ▲                     ▼
                    │                     │
              ┌─────┴─────┐         ┌────┴─────┐
              │ CPU内存   │         │ GPU内存  │
              │ (大容量)  │         │ (有限)   │
              └───────────┘         └──────────┘
```

## 🔄 数据流详解

### 1. 初始化阶段

```
程序启动
   │
   ├─> parse_args()                       # 解析命令行参数
   │     └─> --enable-tensor-offload
   │
   ├─> get_model()                        # 创建模型
   │     │
   │     ├─> model_provider_func()        # 构建模型
   │     │     └─> TransformerLayer.__init__()
   │     │           └─> offload_manager.register_layer()
   │     │
   │     └─> initialize_offload_manager() # 初始化管理器
   │           └─> 创建TensorOffloadManager实例
   │
   └─> 所有layer注册完成
```

### 2. Forward Pass

```
Input Data (batch)
   │
   ├─> Layer 0
   │   │
   │   ├─> [Pre-Hook]  offload_manager.on_forward_pre_hook(0)
   │   │                 ├─> prefetch_layer(0)           CPU → GPU
   │   │                 └─> prefetch_next_layers(0)     预取Layer 1, 2
   │   │
   │   ├─> [Forward]    self_attention + mlp
   │   │                 (使用GPU上的参数)
   │   │
   │   └─> [Post-Hook]  offload_manager.on_forward_post_hook(0)
   │                     └─> 注册backward hook
   │
   ├─> Layer 1
   │   │
   │   ├─> [Pre-Hook]  prefetch_layer(1)           (已预取，直接用)
   │   │                prefetch_next_layers(1)     预取Layer 2, 3
   │   │
   │   ├─> [Forward]   计算
   │   │
   │   └─> [Post-Hook] 注册backward hook
   │
   └─> ... (后续layer类似)
```

### 3. Backward Pass

```
Loss Gradient
   │
   ├─> Layer N (最后一层)
   │   │
   │   ├─> [Backward]  计算梯度
   │   │
   │   └─> [Hook]      offload_manager.on_backward_post_hook(N)
   │                    └─> offload_layer(N)        GPU → CPU
   │
   ├─> Layer N-1
   │   │
   │   ├─> [Backward]  计算梯度
   │   │
   │   └─> [Hook]      offload_layer(N-1)          GPU → CPU
   │
   └─> ... (反向传播)
```

## 📦 参数状态转换

```
┌─────────────┐
│   创建模型  │  所有参数在GPU
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 初始offload │  可选：立即offload所有参数到CPU
└──────┬──────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│                 训练循环                                │
│                                                         │
│  Layer i:                                               │
│    CPU状态 ──prefetch──> GPU状态 ──forward──>          │
│                                      │                  │
│                                      ▼                  │
│                                  计算完成               │
│                                      │                  │
│                                      ▼                  │
│                                  backward               │
│                                      │                  │
│                                      ▼                  │
│    CPU状态 <──offload─── GPU状态                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```


# Tensor Offload 实现总结

## 📋 实现概览

本实现在Megatron-LM中添加了完整的tensor offload功能，用于学习和实践目的。

## 📦 文件清单

### 1. 新增文件 (6个)

#### 核心实现
| 文件路径 | 大小 | 说明 |
|---------|------|------|
| `megatron/core/tensor_offload.py` | ~400行 | TensorOffloadManager核心类 |

#### 文档和示例
| 文件路径 | 大小 | 说明 |
|---------|------|------|
| `examples/tensor_offload/README_TENSOR_OFFLOAD.md` | ~800行 | 详细技术文档 |
| `examples/tensor_offload/IMPLEMENTATION_SUMMARY.md` | 本文件 | 实现总结 |
| `examples/tensor_offload/simple_test.py` | ~400行 | 独立功能测试 |

### 2. 修改的文件 (3个)

| 文件路径 | 修改位置 | 修改内容 |
|---------|----------|----------|
| `megatron/core/transformer/transformer_layer.py` | `__init__`, `forward` | 添加offload hooks |
| `megatron/training/arguments.py` | `_add_experimental_args` | 添加3个命令行参数 |
| `megatron/training/training.py` | `get_model`, `train` | 初始化和统计打印 |

## 🔧 核心组件详解

### 1. TensorOffloadManager

**文件**: `megatron/core/tensor_offload.py`

**核心数据结构**:
```python
class TensorOffloadManager:
    layer_params: Dict[int, List[Parameter]]  # 参数注册表
    layer_param_devices: Dict[int, List[str]] # 设备跟踪
    prefetch_stream: torch.cuda.Stream        # 异步传输
    stats: Dict                               # 统计信息
```

**核心方法**:
- `register_layer(layer_id, module)`: 注册层
- `prefetch_layer(layer_id)`: CPU → GPU
- `offload_layer(layer_id)`: GPU → CPU
- `prefetch_next_layers(current)`: 预取优化
- `on_forward_pre_hook(layer_id)`: Forward前hook
- `on_backward_post_hook(layer_id)`: Backward后hook
- `print_stats()`: 统计信息

**关键特性**:
- ✅ 支持异步传输
- ✅ 使用pinned memory加速
- ✅ 预取机制隐藏延迟
- ✅ 完整的统计信息

### 2. TransformerLayer集成

**文件**: `megatron/core/transformer/transformer_layer.py`

**修改点1**: 在 `__init__` 中注册layer
```python
# 第408-414行
from megatron.core.tensor_offload import get_offload_manager
offload_manager = get_offload_manager()
if offload_manager is not None:
    offload_manager.register_layer(self.layer_number, self)
```

**修改点2**: 在 `forward` 中添加hooks
```python
# 第430-463行
def forward(self, *args, **kwargs):
    # Pre-hook
    offload_manager = get_offload_manager()
    if offload_manager is not None:
        offload_manager.on_forward_pre_hook(self.layer_number)
    
    # Forward计算
    hidden_states, context = self._forward_attention(*args, **kwargs)
    output = self._forward_mlp(hidden_states, ...)
    
    # Post-hook + Backward hook注册
    if offload_manager is not None:
        offload_manager.on_forward_post_hook(self.layer_number)
        if output.requires_grad:
            def backward_hook(grad):
                offload_manager.on_backward_post_hook(self.layer_number)
                return grad
            output.register_hook(backward_hook)
    
    return output, context
```

### 3. 命令行参数

**文件**: `megatron/training/arguments.py`

**新增参数** (第3257-3264行):
```python
def _add_experimental_args(parser):
    # ... 现有参数 ...
    
    # Tensor Offload参数
    group.add_argument('--enable-tensor-offload', 
                      action='store_true',
                      help='Enable tensor offload to CPU')
    group.add_argument('--tensor-offload-pin-memory', 
                      action='store_true', default=True,
                      help='Use pinned memory for faster transfers')
    group.add_argument('--tensor-offload-num-prefetch-layers', 
                      type=int, default=1,
                      help='Number of layers to prefetch in advance')
```

### 4. 训练循环集成

**文件**: `megatron/training/training.py`

**修改点1**: 在 `get_model` 中初始化 (第943-957行)
```python
def get_model(...):
    # ... 模型创建 ...
    
    # 初始化offload manager
    if hasattr(args, 'enable_tensor_offload') and args.enable_tensor_offload:
        from megatron.core.tensor_offload import initialize_offload_manager
        offload_manager = initialize_offload_manager(
            enabled=True,
            pin_memory=args.tensor_offload_pin_memory,
            num_prefetch_layers=args.tensor_offload_num_prefetch_layers,
        )
        print_rank_0("Tensor Offload Manager initialized!")
```

**修改点2**: 在 `train` 结束时打印统计 (第2514-2519行)
```python
def train(...):
    # ... 训练循环 ...
    
    # 打印统计信息
    from megatron.core.tensor_offload import get_offload_manager
    offload_manager = get_offload_manager()
    if offload_manager is not None:
        offload_manager.print_stats()
```

