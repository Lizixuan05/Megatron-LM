# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Tensor Offload Manager for Megatron-LM
======================================

这个模块实现了一个简单但功能完整的tensor offload机制。
主要目的是学习实践，展示如何在训练过程中管理GPU/CPU内存。

核心思想：
1. 在forward前将参数从CPU预取到GPU（使用独立的H2D stream和事件机制）
2. 在forward后将参数offload回CPU（使用独立的D2H stream，可选FWD后立即释放策略）
3. 使用异步操作和事件来隐藏传输延迟，避免全局同步阻塞
4. 支持per-layer事件等待，实现精确的同步控制
5. 支持FWD后立即释放策略，最小化峰值显存占用

关键特性：
- 双流机制：独立的H2D和D2H stream，实现真正的异步传输
- 事件驱动：per-layer ready事件，计算流仅等待所需层的数据就绪
- 可选FWD后释放：支持在FWD后立即offload，BWD前JIT预取的策略
- 设备感知：记录每层的真实目标设备，支持PP/TP分片场景
"""

import logging
from typing import Dict, List, Optional, Set
import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)


class TensorOffloadManager:
    """
    管理模型参数在CPU和GPU之间的offload。
    
    这是一个改进的实现，支持双流、事件机制和可选的FWD后释放策略。
    
    工作流程：
    1. 注册需要管理的层（记录真实目标设备）
    2. 在forward前调用prefetch将参数异步加载到GPU（使用H2D stream和事件）
    3. 计算流通过wait_event等待层数据就绪，避免全局同步
    4. 在forward后可选立即offload（release_after_fwd=True时）
    5. 在backward前JIT预取（如果FWD后已释放）
    6. 在backward后offload参数回CPU（使用D2H stream）
    
    注意：
    - 使用torch.no_grad()保护所有搬运操作，避免污染计算图
    - 优先使用事件机制而非全局synchronize()
    - 支持PP/TP分片，每个rank只管理自己的层
    """
    
    def __init__(
        self, 
        enabled: bool = True,
        offload_optimizer_states: bool = True,
        pin_memory: bool = True,
        num_prefetch_layers: int = 1,
        release_after_fwd: bool = True,
        bucket_mb: int = 0,
    ):
        """
        初始化TensorOffloadManager。
        
        Args:
            enabled: 是否启用offload
            offload_optimizer_states: 是否也offload优化器状态（更高级的功能）
            pin_memory: 是否使用pinned memory加速传输
            num_prefetch_layers: 提前预取几层的参数（用于隐藏延迟）
            release_after_fwd: 若为True，则在FWD后立刻D2H回写，BWD前再JIT预取（峰值显存最低）
            bucket_mb: 预留分桶大小（MB），用于后续分桶实现
        """
        self.enabled = enabled
        self.offload_optimizer_states = offload_optimizer_states
        self.pin_memory = pin_memory
        self.num_prefetch_layers = num_prefetch_layers
        self.release_after_fwd = release_after_fwd
        self.bucket_mb = bucket_mb
        
        # 存储层和其参数的映射
        self.layer_params: Dict[int, List[torch.nn.Parameter]] = OrderedDict()
        self.layer_param_devices: Dict[int, List[str]] = {}  # 跟踪参数当前位置
        
        # 两条独立 stream：H2D 预取、D2H 回写
        self.h2d_stream = torch.cuda.Stream() if enabled else None
        self.d2h_stream = torch.cuda.Stream() if enabled else None
        
        # 每层一个 H2D 完成事件
        self.layer_ready_events: Dict[int, torch.cuda.Event] = {}
        
        # 记录每层"FWD 后是否已释放"，用于 BWD 前是否要再取回
        self.layer_released_after_fwd: Dict[int, bool] = {}
        
        # 每层注册时的目标 device（避免用 current_device 误判）
        self.layer_target_device: Dict[int, torch.device] = {}
        
        # 统计信息
        self.stats = {
            'total_offloaded_bytes': 0,
            'total_prefetched_bytes': 0,
            'num_offload_ops': 0,
            'num_prefetch_ops': 0,
        }
        
        # 当前正在使用的层
        self.current_layer_id: Optional[int] = None
        
        # 跟踪已注册的层，用于初始offload
        self.initial_offload_done = False
        
        logger.info(f"TensorOffloadManager initialized: enabled={enabled}, "
                   f"pin_memory={pin_memory}, num_prefetch_layers={num_prefetch_layers}, "
                   f"release_after_fwd={release_after_fwd}, bucket_mb={bucket_mb}")
    
    def register_layer(self, layer_id: int, module: nn.Module):
        """
        注册一个需要管理的层。
        
        Args:
            layer_id: 层的唯一标识符（通常是层号）
            module: 要管理的模块
        """
        if not self.enabled:
            return
        
        # 收集该层的所有参数
        params = [p for p in module.parameters() if p.requires_grad]
        
        self.layer_params[layer_id] = params
        
        # 记录该层自然隶属的设备（第一枚参数的 device）
        dev = next((p.device for p in params), torch.device('cuda', torch.cuda.current_device()))
        self.layer_target_device[layer_id] = dev
        
        self.layer_param_devices[layer_id] = ['cuda' if p.is_cuda else 'cpu' for p in params]
        self.layer_released_after_fwd[layer_id] = False
        
        total_size = sum(p.numel() * p.element_size() for p in params)
        logger.info(f"Registered layer {layer_id} on {dev} with {len(params)} params "
                   f"({total_size / 1024**2:.2f} MB)")
    
    def initial_offload_all_layers(self):
        """
        初始化时将所有已注册的层offload到CPU。
        这应该在模型初始化完成后、训练开始前调用。
        """
        if not self.enabled or self.initial_offload_done:
            return
        
        logger.info("Performing initial offload of all layers to CPU...")
        total_offloaded = 0
        for layer_id in self.layer_params.keys():
            # 使用同步操作确保完成
            self.offload_layer(layer_id, async_op=False, empty_cache=False)
            total_offloaded += sum(
                p.numel() * p.element_size() 
                for p in self.layer_params[layer_id]
            )
        
        # 最后统一清空缓存
        if total_offloaded > 0:
            torch.cuda.empty_cache()
            logger.info(f"Initial offload completed: {total_offloaded / 1024**2:.2f} MB offloaded to CPU")
        
        self.initial_offload_done = True
    
    def offload_layer(self, layer_id: int, async_op: bool = True, empty_cache: bool = False):
        """
        将指定层的参数offload到CPU。
        
        Args:
            layer_id: 要offload的层ID
            async_op: 是否使用异步操作
            empty_cache: 是否在offload后清空GPU缓存（用于真正释放内存）
        """
        if not self.enabled or layer_id not in self.layer_params:
            return
        
        params = self.layer_params[layer_id]
        devices = self.layer_param_devices[layer_id]
        
        total_bytes = 0
        stream = self.d2h_stream if (async_op and self.d2h_stream is not None) else torch.cuda.current_stream()
        
        with torch.no_grad():
            for i, (param, devflag) in enumerate(zip(params, devices)):
                if devflag == 'cuda':  # 只offload在GPU上的参数
                    src = param.data  # GPU tensor
                    
                    # 用 pinned CPU tensor 做落地
                    cpu_tensor = torch.empty(
                        param.size(), 
                        dtype=param.dtype, 
                        device='cpu',
                        pin_memory=self.pin_memory
                    )
                    
                    if async_op and self.d2h_stream is not None:
                        with torch.cuda.stream(stream):
                            cpu_tensor.copy_(src, non_blocking=True)
                    else:
                        cpu_tensor.copy_(src)
                    
                    # 重新绑定 data（no_grad 保护，避免构图污染）
                    param.data = cpu_tensor
                    
                    # 显式删除旧的 GPU 张量引用，利于 GC
                    del src
                    
                    # 更新设备状态
                    devices[i] = 'cpu'
                    
                    total_bytes += param.numel() * param.element_size()
        
        # 注意：不做全局同步；是否 empty_cache 交给外层定期处理
        if empty_cache and total_bytes > 0:
            torch.cuda.empty_cache()
        
        self.stats['total_offloaded_bytes'] += total_bytes
        self.stats['num_offload_ops'] += 1
        
        if total_bytes > 0:
            logger.info(f"Offloaded layer {layer_id}: {total_bytes / 1024**2:.2f} MB")
    
    def prefetch_layer(self, layer_id: int, async_op: bool = True):
        """
        将指定层的参数预取到GPU。
        
        Args:
            layer_id: 要预取的层ID
            async_op: 是否使用异步操作
        """
        if not self.enabled or layer_id not in self.layer_params:
            return
        
        params = self.layer_params[layer_id]
        devices = self.layer_param_devices[layer_id]
        tgt = self.layer_target_device[layer_id]
        
        total_bytes = 0
        if async_op and self.h2d_stream is not None:
            stream = self.h2d_stream
        else:
            stream = torch.cuda.current_stream()
        
        with torch.no_grad():
            for i, (param, devflag) in enumerate(zip(params, devices)):
                if devflag == 'cpu':  # 只预取在CPU上的参数
                    # 用与参数同 dtype，在该层目标 device 上建空张量
                    gpu_tensor = torch.empty(param.size(), dtype=param.dtype, device=tgt)
                    
                    # 从 CPU (最好是 pinned) 拷贝到 GPU
                    if async_op and self.h2d_stream is not None:
                        with torch.cuda.stream(stream):
                            gpu_tensor.copy_(param.data, non_blocking=True)
                    else:
                        gpu_tensor.copy_(param.data)
                    
                    # 重新绑定 storage（注意：在 no_grad 内）
                    param.data = gpu_tensor
                    
                    # 更新设备状态
                    devices[i] = 'cuda'
                    
                    total_bytes += param.numel() * param.element_size()
        
        # H2D 完成后在该层记录事件，供 compute stream 精确等待
        # 只有当实际有数据传输时才记录事件
        if async_op and self.h2d_stream is not None and total_bytes > 0:
            evt = torch.cuda.Event()
            evt.record(self.h2d_stream)
            self.layer_ready_events[layer_id] = evt
        
        self.stats['total_prefetched_bytes'] += total_bytes
        self.stats['num_prefetch_ops'] += 1
        
        logger.debug(f"Prefetched layer {layer_id}: {total_bytes / 1024**2:.2f} MB")
    
    def prefetch_next_layers(self, current_layer_id: int):
        """
        预取接下来几层的参数（用于隐藏传输延迟）。
        
        Args:
            current_layer_id: 当前层ID
        """
        if not self.enabled:
            return
        
        # 预取接下来的num_prefetch_layers层
        layer_ids = list(self.layer_params.keys())
        try:
            current_idx = layer_ids.index(current_layer_id)
            for i in range(1, self.num_prefetch_layers + 1):
                next_idx = current_idx + i
                if next_idx < len(layer_ids):
                    next_layer_id = layer_ids[next_idx]
                    self.prefetch_layer(next_layer_id, async_op=True)
        except ValueError:
            pass
    
    def synchronize(self):
        """同步所有异步操作（仅在必要时使用，优先使用事件机制）。"""
        if self.enabled:
            if self.h2d_stream is not None:
                self.h2d_stream.synchronize()
            if self.d2h_stream is not None:
                self.d2h_stream.synchronize()
    
    def on_forward_pre_hook(self, layer_id: int):
        """
        在layer forward前调用的hook。
        
        Args:
            layer_id: 即将执行forward的层ID
        """
        if not self.enabled:
            return
        
        # 若层参数在 CPU，则异步预取
        self.prefetch_layer(layer_id, async_op=True)
        
        # 仅让当前 compute stream 等待本层的 ready 事件
        evt = self.layer_ready_events.get(layer_id, None)
        if evt is not None:
            torch.cuda.current_stream().wait_event(evt)
        
        # 前瞻预取后续层（可隐藏下一层 H2D）
        self.prefetch_next_layers(layer_id)
        
        self.current_layer_id = layer_id
    
    def on_forward_post_hook(self, layer_id: int):
        """
        在layer forward后调用的hook。
        
        Args:
            layer_id: 刚完成forward的层ID
        """
        if not self.enabled:
            return
        
        if self.release_after_fwd:
            # 立刻异步回写（减少峰值显存），BWD 前会再 JIT 拉回
            self.offload_layer(layer_id, async_op=True, empty_cache=False)
            self.layer_released_after_fwd[layer_id] = True
    
    def on_backward_pre_hook(self, layer_id: int):
        """
        在layer backward前调用的hook。
        
        Args:
            layer_id: 即将执行backward的层ID
        """
        if not self.enabled:
            return
        
        # 若 FWD 后曾释放，则在 BWD 前 JIT 取回
        if self.layer_released_after_fwd.get(layer_id, False):
            self.prefetch_layer(layer_id, async_op=True)
            evt = self.layer_ready_events.get(layer_id, None)
            if evt is not None:
                torch.cuda.current_stream().wait_event(evt)
            self.layer_released_after_fwd[layer_id] = False
    
    def on_backward_post_hook(self, layer_id: int):
        """
        在layer backward后调用的hook。
        
        Args:
            layer_id: 刚完成backward的层ID
        """
        if not self.enabled:
            return
        
        # Backward完成后，offload这一层（异步，并清空缓存以释放内存）
        # 注意：empty_cache=True 会导致性能下降，但能真正释放GPU内存
        # 可以根据需要调整，或者定期调用 empty_cache
        self.offload_layer(layer_id, async_op=True, empty_cache=False)
    
    def get_stats(self) -> Dict:
        """获取统计信息。"""
        stats = self.stats.copy()
        stats['total_offloaded_mb'] = stats['total_offloaded_bytes'] / 1024**2
        stats['total_prefetched_mb'] = stats['total_prefetched_bytes'] / 1024**2
        return stats
    
    def print_stats(self):
        """打印统计信息。"""
        stats = self.get_stats()
        logger.info("=" * 60)
        logger.info("Tensor Offload Statistics:")
        logger.info(f"  Total offloaded: {stats['total_offloaded_mb']:.2f} MB "
                   f"({stats['num_offload_ops']} operations)")
        logger.info(f"  Total prefetched: {stats['total_prefetched_mb']:.2f} MB "
                   f"({stats['num_prefetch_ops']} operations)")
        logger.info("=" * 60)


# 全局offload manager实例
_global_offload_manager: Optional[TensorOffloadManager] = None


def get_offload_manager() -> Optional[TensorOffloadManager]:
    """获取全局offload manager。"""
    return _global_offload_manager


def initialize_offload_manager(
    enabled: bool = True,
    offload_optimizer_states: bool = False,
    pin_memory: bool = True,
    num_prefetch_layers: int = 1,
    release_after_fwd: bool = False,
    bucket_mb: int = 0,
) -> TensorOffloadManager:
    """
    初始化全局offload manager。
    
    Args:
        enabled: 是否启用offload
        offload_optimizer_states: 是否也offload优化器状态
        pin_memory: 是否使用pinned memory
        num_prefetch_layers: 预取层数
        release_after_fwd: 若为True，则在FWD后立刻D2H回写，BWD前再JIT预取
        bucket_mb: 预留分桶大小（MB），用于后续分桶实现
        
    Returns:
        TensorOffloadManager实例
    """
    global _global_offload_manager
    _global_offload_manager = TensorOffloadManager(
        enabled=enabled,
        offload_optimizer_states=offload_optimizer_states,
        pin_memory=pin_memory,
        num_prefetch_layers=num_prefetch_layers,
        release_after_fwd=release_after_fwd,
        bucket_mb=bucket_mb,
    )
    return _global_offload_manager

