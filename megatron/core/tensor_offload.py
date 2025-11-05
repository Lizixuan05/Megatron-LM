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
        offload_optimizer_states: bool = False,
        pin_memory: bool = True,
        num_prefetch_layers: int = 1,
        release_after_fwd: bool = False,
        bucket_mb: int = 0,
    ):
        """
        初始化TensorOffloadManager。
        
        Args:
            enabled: 是否启用offload
            offload_optimizer_states: 是否也offload优化器状态
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
        
        # 每层 D2H 完成事件（供后续 H2D 等待）
        self.layer_d2h_done_events: Dict[int, torch.cuda.Event] = {}
        
        # 记录每层"FWD 后是否已释放"，用于 BWD 前是否要再取回
        self.layer_released_after_fwd: Dict[int, bool] = {}
        
        # H2D 源张量生命周期保护：暂存 CPU 张量引用，直到 H2D 完成
        self._pending_h2d_src: Dict[int, List[torch.Tensor]] = {}
        
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
        
        # 用于跟踪每层的 saved_tensors_hooks 上下文
        self._layer_saved_ctx: Dict[int, object] = {}
        
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
        
        # 记录该层的目标设备（应该是GPU设备，即使参数当前在CPU上）
        # 对于tensor offload场景，参数最终要在GPU上计算，所以target应该是CUDA
        # 检查参数当前是否在GPU上，如果是则使用该设备；否则使用当前CUDA设备
        first_cuda_param_device = next((p.device for p in params if p.is_cuda), None)
        if first_cuda_param_device is not None:
            dev = first_cuda_param_device
        else:
            # 所有参数都在CPU上，使用当前的CUDA设备作为目标
            dev = torch.device('cuda', torch.cuda.current_device())
        self.layer_target_device[layer_id] = dev
        
        self.layer_param_devices[layer_id] = ['cuda' if p.is_cuda else 'cpu' for p in params]
        self.layer_released_after_fwd[layer_id] = False
        
        total_size = sum(p.numel() * p.element_size() for p in params)
        actual_device = next((p.device for p in params), dev)
        logger.info(f"Registered layer {layer_id} (current: {actual_device}, target: {dev}) with {len(params)} params "
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
            # 同步 D2H，保证真的落回 CPU
            self.offload_layer(layer_id, async_op=False, empty_cache=False)
            # 标记该层"已释放"，后续首次需要会 JIT 取回
            self.layer_released_after_fwd[layer_id] = True
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
        
        注意：同时处理 model_param 和 main_param（用于混合精度优化器）
        
        Args:
            layer_id: 要offload的层ID
            async_op: 是否使用异步操作
            empty_cache: 是否在offload后清空GPU缓存（用于真正释放内存）
        """
        if not self.enabled or layer_id not in self.layer_params:
            return
        
        # 清理已完成的 H2D 源张量引用
        self._cleanup_completed_h2d_sources()
        
        params = self.layer_params[layer_id]
        devices = self.layer_param_devices[layer_id]
        
        total_bytes = 0
        stream = self.d2h_stream if (async_op and self.d2h_stream is not None) else torch.cuda.current_stream()
        
        # ========== CRITICAL FIX: Stream Synchronization ==========
        # 在异步offload之前，必须确保default stream上的所有操作都已完成
        # 特别是optimizer.step()中的参数更新操作
        # 否则可能会拷贝到未更新的参数值，导致下一次迭代使用错误的参数
        if async_op and self.d2h_stream is not None:
            # 创建一个event来标记default stream的当前位置
            sync_event = torch.cuda.Event()
            sync_event.record(torch.cuda.current_stream())
            # 让d2h_stream等待这个event，确保之前的所有操作都已完成
            self.d2h_stream.wait_event(sync_event)
        # ==========================================================
        
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
                        #  关键：保护源 GPU 张量生命周期，防止 D2H 未完成时被回收
                        src.record_stream(self.d2h_stream)
                    else:
                        cpu_tensor.copy_(src)
                    
                    # 重新绑定 data（no_grad 保护，避免构图污染）
                    param.data = cpu_tensor
                    
                    # 显式删除旧的 GPU 张量引用，利于 GC
                    del src
                    
                    # === 关键：同步处理 main_param（混合精度优化器） ===
                    # 如果参数有 main_param 属性（fp32 副本），也需要 offload
                    if hasattr(param, 'main_param') and param.main_param is not None:
                        if param.main_param.is_cuda:
                            main_src = param.main_param.data
                            main_cpu_tensor = torch.empty(
                                param.main_param.size(),
                                dtype=param.main_param.dtype,
                                device='cpu',
                                pin_memory=self.pin_memory
                            )
                            if async_op and self.d2h_stream is not None:
                                with torch.cuda.stream(stream):
                                    main_cpu_tensor.copy_(main_src, non_blocking=True)
                                #  保护 main_param 源 GPU 张量生命周期
                                main_src.record_stream(self.d2h_stream)
                            else:
                                main_cpu_tensor.copy_(main_src)
                            
                            param.main_param.data = main_cpu_tensor
                            del main_src
                            total_bytes += param.main_param.numel() * param.main_param.element_size()
                    # ===================================================
                    
                    # 更新设备状态
                    devices[i] = 'cpu'
                    
                    total_bytes += param.numel() * param.element_size()
        
        # 关键：记录该层 D2H 完成事件（在 d2h_stream 上）
        if async_op and self.d2h_stream is not None and total_bytes > 0:
            d2h_evt = torch.cuda.Event()
            d2h_evt.record(self.d2h_stream)
            self.layer_d2h_done_events[layer_id] = d2h_evt
        
        # 注意：不做全局同步；是否 empty_cache 交给外层定期处理
        if empty_cache and total_bytes > 0:
            torch.cuda.empty_cache()
        
        self.stats['total_offloaded_bytes'] += total_bytes
        self.stats['num_offload_ops'] += 1
        
    
    def prefetch_layer(self, layer_id: int, async_op: bool = True):
        """
        将指定层的参数预取到GPU。
        
        注意：同时处理 model_param 和 main_param（用于混合精度优化器）
        
        Args:
            layer_id: 要预取的层ID
            async_op: 是否使用异步操作
        """
        if not self.enabled or layer_id not in self.layer_params:
            return
        
        # 清理已完成的 H2D 源张量引用
        self._cleanup_completed_h2d_sources()
        
        params = self.layer_params[layer_id]
        devices = self.layer_param_devices[layer_id]
        tgt = self.layer_target_device[layer_id]
        
        # DEBUG: 添加日志
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            logger.debug(f"[Rank {rank}] prefetch_layer {layer_id} start")
        
        #  如果这一层存在未完成的 D2H，必须在 H2D 前等待它
        pending_d2h = self.layer_d2h_done_events.get(layer_id, None)
        if pending_d2h is not None:
            if async_op and self.h2d_stream is not None:
                self.h2d_stream.wait_event(pending_d2h)
            else:
                # 同步路径：确保 d2h_stream 已经结束
                if self.d2h_stream is not None:
                    self.d2h_stream.synchronize()
            # 一旦我们保证了 D2H 完成，就可以清理这个事件
            self.layer_d2h_done_events.pop(layer_id, None)
        
        total_bytes = 0
        if async_op and self.h2d_stream is not None:
            stream = self.h2d_stream
        else:
            stream = torch.cuda.current_stream()
        
        # H2D 源张量（CPU）生命周期保护：暂存强引用列表
        cpu_src_refs = []
        
        with torch.no_grad():
            for i, (param, devflag) in enumerate(zip(params, devices)):
                if devflag == 'cpu':  # 只预取在CPU上的参数
                    # 用与参数同 dtype，在该层目标 device 上建空张量
                    gpu_tensor = torch.empty(param.size(), dtype=param.dtype, device=tgt)
                    
                    # 关键：暂存 CPU 源张量引用，防止 H2D 未完成时被释放
                    cpu_src = param.data
                    if async_op and self.h2d_stream is not None:
                        cpu_src_refs.append(cpu_src)
                    
                    # 从 CPU (最好是 pinned) 拷贝到 GPU
                    if async_op and self.h2d_stream is not None:
                        with torch.cuda.stream(stream):
                            gpu_tensor.copy_(cpu_src, non_blocking=True)
                    else:
                        gpu_tensor.copy_(cpu_src)
                    
                    # 重新绑定 storage（注意：在 no_grad 内）
                    param.data = gpu_tensor
                    
                    # === 关键：同步处理 main_param（混合精度优化器） ===
                    # 如果参数有 main_param 属性（fp32 副本），也需要预取
                    if hasattr(param, 'main_param') and param.main_param is not None:
                        if not param.main_param.is_cuda:
                            main_gpu_tensor = torch.empty(
                                param.main_param.size(),
                                dtype=param.main_param.dtype,
                                device=tgt
                            )
                            # ★ 暂存 main_param 的 CPU 源张量引用
                            main_cpu_src = param.main_param.data
                            if async_op and self.h2d_stream is not None:
                                cpu_src_refs.append(main_cpu_src)
                                with torch.cuda.stream(stream):
                                    main_gpu_tensor.copy_(main_cpu_src, non_blocking=True)
                            else:
                                main_gpu_tensor.copy_(main_cpu_src)
                            
                            param.main_param.data = main_gpu_tensor
                            total_bytes += param.main_param.numel() * param.main_param.element_size()
                    # ===================================================
                    
                    # 更新设备状态
                    devices[i] = 'cuda'
                    
                    total_bytes += param.numel() * param.element_size()
        
        # H2D 完成后在该层记录事件，供 compute stream 精确等待
        # 只有当实际有数据传输时才记录事件
        if async_op and self.h2d_stream is not None and total_bytes > 0:
            evt = torch.cuda.Event()
            evt.record(self.h2d_stream)
            self.layer_ready_events[layer_id] = evt
            
            #  保存 CPU 源张量引用，防止 H2D 未完成时被释放
            if cpu_src_refs:
                self._pending_h2d_src[layer_id] = cpu_src_refs
        
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
    
    def _cleanup_completed_h2d_sources(self):
        """
        清理已完成 H2D 的 CPU 源张量引用。
        通过查询事件状态，释放不再需要的引用。
        """
        if not self.enabled:
            return
        
        layers_to_remove = []
        for layer_id, evt in self.layer_ready_events.items():
            # 如果 H2D 事件已完成，可以安全释放 CPU 源张量引用
            if evt.query():
                layers_to_remove.append(layer_id)
        
        for layer_id in layers_to_remove:
            self._pending_h2d_src.pop(layer_id, None)
    
    def ensure_on_device(self, layer_id: int):
        """
        确保该层参数已在 GPU。若在 CPU，则异步预取并在当前计算流等待事件。
        无论是正向、反向还是重算，这个入口都安全可重入。
        """
        if not self.enabled or layer_id not in self.layer_params:
            return
        
        # 清理已完成的 H2D 源张量引用
        self._cleanup_completed_h2d_sources()
        
        devices = self.layer_param_devices[layer_id]
        # 快速路径：都在 cuda
        if all(flag == 'cuda' for flag in devices):
            return
        
        # 触发异步预取
        self.prefetch_layer(layer_id, async_op=True)
        evt = self.layer_ready_events.get(layer_id, None)
        if evt is not None:
            # 在当前（很关键：autograd/重算所在）计算流上等待
            torch.cuda.current_stream().wait_event(evt)
    
    def _saved_tensors_pack(self, t: torch.Tensor):
        """保存张量时的 pack hook，我们不动保存的张量，直接原样返回，避免引入额外拷贝/同步。"""
        return t
    
    def _enter_saved_hooks(self, layer_id: int):
        """
        在该层 forward 入口处进入 saved_tensors_hooks 作用域。
        只要这个作用域还在，凡是在该层 forward 里被保存到 ctx 的张量，
        其在 backward 解包前都会先触发 _saved_tensors_unpack(layer_id)。
        """
        from torch.autograd.graph import saved_tensors_hooks
        
        def _unpack(t: torch.Tensor):
            # 反向真正需要这些 saved tensors 的那一刻，确保权重已在 GPU
            self.ensure_on_device(layer_id)
            return t
        
        # 用 per-layer 的上下文，前置 __enter__，并在 forward_post 再 __exit__
        ctx = saved_tensors_hooks(self._saved_tensors_pack, _unpack)
        token = ctx.__enter__()
        # 记录起来，post 时 __exit__
        self._layer_saved_ctx[layer_id] = ctx
    
    def _exit_saved_hooks(self, layer_id: int):
        """离开 saved_tensors_hooks 作用域。"""
        ctx = self._layer_saved_ctx.pop(layer_id, None)
        if ctx is not None:
            ctx.__exit__(None, None, None)
    
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
        
        # 进入 saved_tensors_hooks 作用域（关键：覆盖 checkpoint 重算等场景）
        self._enter_saved_hooks(layer_id)
        
        # 若层参数在 CPU，则异步预取 + 事件
        self.ensure_on_device(layer_id)
        
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
        
        # 离开 saved_tensors_hooks 作用域
        self._exit_saved_hooks(layer_id)
        
        if self.release_after_fwd:
            # 立即异步 D2H，减少峰值显存
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
        
        注意：
            不在这里 offload！因为优化器的 prepare_grads() 和参数更新需要参数在 GPU。
            应该在 optimizer.step() 完成后调用 on_optimizer_step_post() 来统一 offload。
        """
        if not self.enabled:
            return
        
        # 不做任何操作，等待 optimizer.step() 完成后再 offload
        # 在新的架构中，saved_tensors_hooks 的 unpack 会自动处理 BWD 前的预取
        pass
    
    def on_optimizer_step_pre(self):
        """
        在optimizer.step()之前调用，确保所有参数都在GPU上。
        
        这是关键的一步：在Pipeline Parallel中，由于调度的复杂性，
        某些层可能在forward后被offload了，但它们的backward还没有执行
        （或者saved_tensors_hooks的unpack还没有触发）。
        
        optimizer.step()需要所有参数都在GPU上进行梯度处理，
        因此我们需要在这里确保所有参数都已预取回GPU。
        
        Example:
            # 在训练循环中
            loss.backward()
            offload_manager.on_optimizer_step_pre()  # 确保所有参数在GPU
            optimizer.step()
            offload_manager.on_optimizer_step_post()  # 再次offload
        """
        if not self.enabled:
            return
        
        logger.debug("Ensuring all parameters are on GPU before optimizer step...")
        
        # 确保所有层的参数都在GPU上
        for layer_id in self.layer_params.keys():
            self.ensure_on_device(layer_id)
        
        # 同步H2D stream，确保所有预取都完成
        if self.h2d_stream is not None:
            self.h2d_stream.synchronize()
        
        logger.debug("All parameters ready on GPU for optimizer step")
    
    def on_optimizer_step_post(self):
        """
        在optimizer.step()完成后调用，将所有层的参数offload到CPU。
        
        这个方法应该在训练循环中，optimizer.step()之后、下一次forward之前调用。
        它会将所有当前在GPU上的参数offload到CPU，为下一次迭代腾出显存。
        
        Example:
            # 在训练循环中
            loss.backward()
            offload_manager.on_optimizer_step_pre()  # 确保所有参数在GPU
            optimizer.step()
            offload_manager.on_optimizer_step_post()  # 在这里统一offload
        """
        if not self.enabled:
            return
        
        # 将所有层的参数offload到CPU
        for layer_id in self.layer_params.keys():
            self.offload_layer(layer_id, async_op=True, empty_cache=False)
            self.layer_released_after_fwd[layer_id] = True
        
        # 可选：同步D2H stream，确保offload完成
        if self.d2h_stream is not None:
            self.d2h_stream.synchronize()
        
        # 清理所有已完成的 H2D 源张量引用（optimizer step 后，所有 H2D 应该都已完成）
        self._pending_h2d_src.clear()
        
        # 定期清空GPU缓存以真正释放内存
        torch.cuda.empty_cache()
        
        logger.debug("Offloaded all layers after optimizer step")
    
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


def attach_offload_hooks_to_layer(layer: nn.Module, layer_id: int, mgr: TensorOffloadManager):
    """
    将offload hooks附加到指定层上。
    
    这个函数会：
    1. 在manager中注册该层
    2. 挂载forward pre/post hooks以在正确时机管理参数传输
    
    Args:
        layer: 要管理的层模块
        layer_id: 层的唯一标识符
        mgr: TensorOffloadManager实例
        
    Returns:
        Tuple of hook handles (pre_hook_handle, post_hook_handle)
        
    Example:
        mgr = initialize_offload_manager(enabled=True, release_after_fwd=True)
        handles = []
        for lid, block in enumerate(model.transformer.layers):
            handles += attach_offload_hooks_to_layer(block, lid, mgr)
        mgr.initial_offload_all_layers()
    """
    # 注册层
    mgr.register_layer(layer_id, layer)
    
    # 用 module hooks 在 forward 进/出时调用 manager 钩子
    def _pre(_module, _inputs):
        mgr.on_forward_pre_hook(layer_id)
        return None
    
    def _post(_module, _inputs, _outputs):
        mgr.on_forward_post_hook(layer_id)
        return None
    
    h1 = layer.register_forward_pre_hook(_pre, with_kwargs=False)
    h2 = layer.register_forward_hook(_post, with_kwargs=False)
    
    return (h1, h2)

