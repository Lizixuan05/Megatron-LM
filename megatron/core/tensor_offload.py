# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import logging
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from collections import OrderedDict

# 在文件顶部其他 import 之后添加
try:
    from torch.utils.checkpoint import checkpoint as _ckpt
except Exception:
    _ckpt = None  # 没有就禁用重计算

logger = logging.getLogger(__name__)

class TensorOffloadManager:
    """
    Megatron-LM 按层参数 Offload 管理器（PP 友好）。
    - PP 默认：BWD 后 offload，下一次用到前 JIT 预取。
    - 双流 + 事件，严格保证 H2D/D2H 与计算流的拓扑顺序。
    - 支持（可选）优化器状态 offload（教学/实验用途）。
    """

    def __init__(
        self,
        enabled: bool = True,
        offload_optimizer_states: bool = False,
        pin_memory: bool = True,
        num_prefetch_layers: int = 1,
        release_after_fwd: bool = False,
        bucket_mb: int = 0,
        # [NEW] 更贴合 PP 的默认策略与性能/稳定性开关
        pp_mode: bool = True,
        copy_chunk_mb: int = 64,
        include_buffers: bool = False,
        offload_gradients: bool = True,  # 新增：梯度 offload
        aggressive_release: bool = True,  # 新增：更积极的显存释放
        recompute_enabled: bool = True,
        recompute_policy: str = "auto",   # "always" 或 "auto"
        recompute_min_params_mb: float = 64,  # auto 时的阈值（按层参数规模）
        recompute_use_reentrant: Optional[bool] = False,  # PyTorch 2.x 建议 False
    ):
        self.enabled = enabled
        self.offload_optimizer_states = offload_optimizer_states
        self.pin_memory = pin_memory
        self.num_prefetch_layers = max(0, int(num_prefetch_layers))
        self.release_after_fwd = bool(release_after_fwd)
        self.bucket_mb = bucket_mb
        self.pp_mode = pp_mode
        self.copy_chunk_bytes = max(1, int(copy_chunk_mb)) * 1024 * 1024
        self.include_buffers = include_buffers
        self.offload_gradients = offload_gradients
        self.aggressive_release = aggressive_release
        
        # 重计算相关配置
        self.recompute_enabled = recompute_enabled and (_ckpt is not None)
        self.recompute_policy = recompute_policy
        self.recompute_min_params_bytes = int(recompute_min_params_mb * 1024 * 1024)
        self.recompute_use_reentrant = recompute_use_reentrant
        if recompute_enabled and _ckpt is None:
            logger.warning("torch.utils.checkpoint 不可用，已禁用重计算。")

        # PP 下强制关闭 FWD 后立刻 offload（1F1B/交错 1F1B 窗口极短）
        if self.pp_mode and self.release_after_fwd:
            logger.warning("Pipeline 模式下不建议 FWD 后立刻 offload —— 已自动禁用。")
            self.release_after_fwd = False

        # 层 -> 参数
        self.layer_params: Dict[int, List[torch.nn.Parameter]] = OrderedDict()
        # （可选）层 -> 持久 buffer（如 BN 的运行均值/方差）
        self.layer_buffers: Dict[int, List[torch.Tensor]] = OrderedDict()
        # 跟踪每个 param 当前落在 'cuda' 还是 'cpu'
        self.layer_param_devices: Dict[int, List[str]] = {}
        # 跟踪每个梯度当前落在 'cuda' 还是 'cpu'
        self.layer_grad_devices: Dict[int, List[str]] = {}
        # 目标设备（用于确保在正确 device 上分配）
        self.layer_target_device: Dict[int, torch.device] = {}
        # 每层参数规模（用于 auto 重计算策略）
        self.layer_param_bytes: Dict[int, int] = {}

        # 共享参数去重映射（param_id -> 主属层）
        self.param_owner: Dict[int, int] = {}

        # 独立 H2D / D2H stream（单设备场景足够；多设备时目标 device guard）
        self.h2d_stream = torch.cuda.Stream() if enabled else None
        self.d2h_stream = torch.cuda.Stream() if enabled else None

        # 事件池（复用事件对象减少分配）
        self._event_pool: List[torch.cuda.Event] = []
        self.layer_ready_events: Dict[int, torch.cuda.Event] = {}      # H2D 完成
        self.layer_d2h_done_events: Dict[int, torch.cuda.Event] = {}   # D2H 完成

        # 记录"FWD 后是否释放过"
        self.layer_released_after_fwd: Dict[int, bool] = {}

        # 保护 H2D 源 CPU 张量生命周期
        self._pending_h2d_src: Dict[int, List[torch.Tensor]] = {}

        # CPU pinned memory 缓存池（减少反复分配）
        self._cpu_tensor_cache: Dict[Tuple[torch.Size, torch.dtype], List[torch.Tensor]] = {}
        self._max_cache_size = 100  # 最多缓存100个张量

        # 统计
        self.stats = {
            'total_offloaded_bytes': 0,
            'total_prefetched_bytes': 0,
            'num_offload_ops': 0,
            'num_prefetch_ops': 0,
            'total_grad_offloaded_bytes': 0,
            'num_cache_hits': 0,
        }

        # 优化器（可选）
        self._optimizer = None

        # 初始 offload 标记
        self.initial_offload_done = False

        logger.info(
            f"TensorOffloadManager initialized: enabled={enabled}, pin_memory={pin_memory}, "
            f"num_prefetch_layers={self.num_prefetch_layers}, pp_mode={pp_mode}, "
            f"release_after_fwd={self.release_after_fwd}, copy_chunk_mb={copy_chunk_mb}, "
            f"offload_optimizer_states={offload_optimizer_states}, offload_gradients={offload_gradients}, "
            f"aggressive_release={aggressive_release}, "
            f"recompute_enabled={self.recompute_enabled}, recompute_policy={self.recompute_policy}, "
            f"recompute_min_params_mb={recompute_min_params_mb}, recompute_use_reentrant={self.recompute_use_reentrant}"
        )

    # [NEW] attach 优化器用于教学用 offload states
    def attach_optimizer(self, optimizer: torch.optim.Optimizer):
        self._optimizer = optimizer
        if self.offload_optimizer_states:
            logger.info("Optimizer attached. Will offload optimizer states to CPU (experimental).")

    def register_layer(self, layer_id: int, module: nn.Module):
        """注册层以进行 offload 管理。"""
        if not self.enabled:
            return

        params = [p for p in module.parameters() if p.requires_grad]
        if self.include_buffers:
            bufs = [b for b in module.buffers() if getattr(b, 'persistent', True)]
        else:
            bufs = []

        # 记录 owner，避免共享参数重复搬运
        for p in params:
            pid = id(p)
            if pid not in self.param_owner:
                self.param_owner[pid] = layer_id

        self.layer_params[layer_id] = params
        self.layer_buffers[layer_id] = bufs

        # 确定目标设备
        first_cuda_param_device = next((p.device for p in params if p.is_cuda), None)
        dev = first_cuda_param_device if first_cuda_param_device is not None \
              else torch.device('cuda', torch.cuda.current_device())
        self.layer_target_device[layer_id] = dev

        self.layer_param_devices[layer_id] = ['cuda' if p.is_cuda else 'cpu' for p in params]
        self.layer_grad_devices[layer_id] = ['none' for _ in params]  # 初始化梯度状态
        self.layer_released_after_fwd[layer_id] = False

        total_size = sum(p.numel() * p.element_size() for p in params) if params else 0
        self.layer_param_bytes[layer_id] = int(total_size)
        logger.info(
            f"Registered layer {layer_id} (target: {dev}) with {len(params)} params "
            f"({total_size / 1024**2:.2f} MB) + {len(bufs)} buffers"
        )

    def initial_offload_all_layers(self):
        """
        仅在：
          1) 模型构建/优化器状态初始化完成后；
          2) 首次迭代之前；
        调用一次。将参数统一落到 CPU，配合 JIT 预取启动训练。
        性能优化：使用异步操作，最后统一同步。
        """
        if not self.enabled or self.initial_offload_done:
            return
        logger.info("Initial offload all registered layers to CPU...")
        # 性能优化：使用异步操作，批量 offload
        for layer_id in self.layer_params.keys():
            self.offload_layer(layer_id, async_op=True, empty_cache=False)
            self.layer_released_after_fwd[layer_id] = True
        # 等待所有异步操作完成
        if self.d2h_stream is not None:
            self.d2h_stream.synchronize()
        torch.cuda.empty_cache()
        self.initial_offload_done = True
        logger.info("Initial offload completed.")

    # --------------------- 内部工具 --------------------- #
    def _get_event(self) -> torch.cuda.Event:
        """从事件池获取或创建新事件。"""
        if self._event_pool:
            return self._event_pool.pop()
        return torch.cuda.Event()
    
    def _release_event(self, event: torch.cuda.Event):
        """将事件归还到事件池。"""
        if len(self._event_pool) < 50:  # 限制事件池大小
            self._event_pool.append(event)
    
    def _get_cpu_tensor(self, size: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """从缓存池获取或创建 CPU pinned tensor。"""
        if not self.pin_memory:
            return torch.empty(size, dtype=dtype, device='cpu')
        
        key = (size, dtype)
        cache_list = self._cpu_tensor_cache.get(key, [])
        
        if cache_list:
            self.stats['num_cache_hits'] += 1
            return cache_list.pop()
        
        return torch.empty(size, dtype=dtype, device='cpu', pin_memory=True)
    
    def _release_cpu_tensor(self, tensor: torch.Tensor):
        """将 CPU tensor 归还到缓存池。"""
        if not self.pin_memory or not tensor.is_pinned():
            return
        
        key = (tensor.size(), tensor.dtype)
        cache_list = self._cpu_tensor_cache.setdefault(key, [])
        
        # 限制每个 key 的缓存数量
        if len(cache_list) < 10:
            cache_list.append(tensor)
    
    def _clear_tensor_cache(self):
        """清空张量缓存池，释放内存。"""
        total_cleared = sum(len(v) for v in self._cpu_tensor_cache.values())
        self._cpu_tensor_cache.clear()
        if total_cleared > 0:
            logger.debug(f"Cleared {total_cleared} cached CPU tensors")
    
    def _chunked_copy(self, dst: torch.Tensor, src: torch.Tensor, stream: Optional[torch.cuda.Stream], async_op: bool):
        """按 MB 分块拷贝，降低长拷贝对流的占用。性能优化：避免重复创建 stream context。"""
        if dst.numel() == 0:
            return
        view_dst = dst.view(-1)
        view_src = src.view(-1)
        elem = src.element_size()
        chunk_elems = max(1, self.copy_chunk_bytes // elem)
        n = view_dst.numel()
        
        # 性能优化：将 stream context 提升到循环外
        if async_op and stream is not None:
            with torch.cuda.stream(stream):
                off = 0
                while off < n:
                    end = min(off + chunk_elems, n)
                    view_dst[off:end].copy_(view_src[off:end], non_blocking=True)
                    off = end
        else:
            off = 0
            while off < n:
                end = min(off + chunk_elems, n)
                view_dst[off:end].copy_(view_src[off:end])
                off = end

    def _cleanup_completed_h2d_sources(self):
        """清理已完成的 H2D 源张量。性能优化：减少不必要的遍历。"""
        if not self.enabled or not self._pending_h2d_src:
            return
        layers_to_remove = []
        for lid in list(self.layer_ready_events.keys()):
            evt = self.layer_ready_events.get(lid)
            if evt and lid in self._pending_h2d_src and evt.query():
                layers_to_remove.append(lid)
        for lid in layers_to_remove:
            self._pending_h2d_src.pop(lid, None)
            # 归还事件到事件池
            evt = self.layer_ready_events.pop(lid, None)
            if evt:
                self._release_event(evt)

    # --------------------- D2H & H2D 核心 --------------------- #
    def offload_layer(self, layer_id: int, async_op: bool = True, empty_cache: bool = False):
        """将层参数从 GPU offload 到 CPU。性能优化：使用缓存池，减少内存分配。"""
        if not self.enabled or layer_id not in self.layer_params:
            return

        params = self.layer_params[layer_id]
        if not params:  # 空参数列表保护
            return
            
        devices = self.layer_param_devices[layer_id]

        total_bytes = 0
        # 让 d2h_stream 等待 default/compute 流位置（确保 optimizer.step() 已写完）
        stream = self.d2h_stream if (async_op and self.d2h_stream is not None) else None
        use_async = async_op and stream is not None
        
        if use_async:
            sync_event = self._get_event()
            sync_event.record(torch.cuda.current_stream())
            stream.wait_event(sync_event)
            self._release_event(sync_event)

        with torch.no_grad():
            for i, (param, devflag) in enumerate(zip(params, devices)):
                # 共享参数：只由 owner 层负责搬运，其他层跳过
                if self.param_owner.get(id(param), layer_id) != layer_id:
                    continue
                if devflag == 'cuda':
                    src = param.data  # GPU tensor
                    # 使用缓存池获取 CPU tensor
                    cpu_tensor = self._get_cpu_tensor(param.size(), param.dtype)
                    self._chunked_copy(cpu_tensor, src, stream, use_async)
                    # 错误修复：record_stream 应该在赋值前调用
                    if use_async:
                        src.record_stream(stream)
                    param.data = cpu_tensor
                    del src
                    devices[i] = 'cpu'
                    total_bytes += param.numel() * param.element_size()

                    # main_param（若存在）
                    if hasattr(param, 'main_param') and param.main_param is not None and param.main_param.is_cuda:
                        main_src = param.main_param.data
                        main_cpu = self._get_cpu_tensor(param.main_param.size(), param.main_param.dtype)
                        self._chunked_copy(main_cpu, main_src, stream, use_async)
                        if use_async:
                            main_src.record_stream(stream)
                        param.main_param.data = main_cpu
                        del main_src
                        total_bytes += param.main_param.numel() * param.main_param.element_size()

        if use_async and total_bytes > 0:
            # 归还旧事件到池中
            old_evt = self.layer_d2h_done_events.get(layer_id)
            if old_evt:
                self._release_event(old_evt)
            # 获取新事件
            d2h_evt = self._get_event()
            d2h_evt.record(stream)
            self.layer_d2h_done_events[layer_id] = d2h_evt

        if (empty_cache or self.aggressive_release) and total_bytes > 0:
            torch.cuda.empty_cache()

        self.stats['total_offloaded_bytes'] += total_bytes
        self.stats['num_offload_ops'] += 1

    def prefetch_layer(self, layer_id: int, async_op: bool = True):
        """预取层参数从 CPU 到 GPU。性能优化：使用事件池，及时释放旧 CPU 张量。"""
        if not self.enabled or layer_id not in self.layer_params:
            return

        params = self.layer_params[layer_id]
        if not params:  # 空参数列表保护
            return
            
        devices = self.layer_param_devices[layer_id]
        tgt = self.layer_target_device[layer_id]

        # 若有未完成 D2H，H2D 需等待
        pending = self.layer_d2h_done_events.get(layer_id, None)
        if pending is not None:
            if async_op and self.h2d_stream is not None:
                self.h2d_stream.wait_event(pending)
            else:
                if self.d2h_stream is not None:
                    self.d2h_stream.synchronize()
            # 归还事件到池
            self._release_event(self.layer_d2h_done_events.pop(layer_id))

        total_bytes = 0
        stream = self.h2d_stream if (async_op and self.h2d_stream is not None) else None
        use_async = async_op and stream is not None
        cpu_src_refs: List[torch.Tensor] = []
        old_cpu_tensors: List[torch.Tensor] = []

        with torch.no_grad(), torch.cuda.device(tgt):
            for i, (param, devflag) in enumerate(zip(params, devices)):
                if self.param_owner.get(id(param), layer_id) != layer_id:
                    continue
                if devflag == 'cpu':
                    old_cpu = param.data
                    old_cpu_tensors.append(old_cpu)
                    
                    gpu_tensor = torch.empty(param.size(), dtype=param.dtype, device=tgt)
                    cpu_src = param.data
                    if use_async:
                        cpu_src_refs.append(cpu_src)
                    self._chunked_copy(gpu_tensor, cpu_src, stream, use_async)
                    param.data = gpu_tensor
                    devices[i] = 'cuda'
                    total_bytes += param.numel() * param.element_size()

                    if hasattr(param, 'main_param') and param.main_param is not None and (not param.main_param.is_cuda):
                        old_main_cpu = param.main_param.data
                        old_cpu_tensors.append(old_main_cpu)
                        
                        main_gpu = torch.empty_like(param.main_param, device=tgt)
                        main_cpu_src = param.main_param.data
                        if use_async:
                            cpu_src_refs.append(main_cpu_src)
                        self._chunked_copy(main_gpu, main_cpu_src, stream, use_async)
                        param.main_param.data = main_gpu
                        total_bytes += param.main_param.numel() * param.main_param.element_size()

        if use_async and total_bytes > 0:
            # 归还旧事件到池
            old_evt = self.layer_ready_events.get(layer_id)
            if old_evt:
                self._release_event(old_evt)
            # 获取新事件
            evt = self._get_event()
            evt.record(stream)
            self.layer_ready_events[layer_id] = evt
            if cpu_src_refs:
                self._pending_h2d_src[layer_id] = cpu_src_refs

        # 将旧的 CPU 张量归还到缓存池（如果不再需要）
        if not use_async:
            for old_cpu in old_cpu_tensors:
                self._release_cpu_tensor(old_cpu)

        self.stats['total_prefetched_bytes'] += total_bytes
        self.stats['num_prefetch_ops'] += 1

    # --------------------- 梯度管理 --------------------- #
    def offload_layer_gradients(self, layer_id: int, async_op: bool = True):
        """将层梯度从 GPU offload 到 CPU。"""
        if not self.enabled or not self.offload_gradients:
            return
        if layer_id not in self.layer_params:
            return

        params = self.layer_params[layer_id]
        if not params:
            return
            
        grad_devices = self.layer_grad_devices[layer_id]
        stream = self.d2h_stream if (async_op and self.d2h_stream is not None) else None
        use_async = async_op and stream is not None
        
        if use_async:
            sync_event = self._get_event()
            sync_event.record(torch.cuda.current_stream())
            stream.wait_event(sync_event)
            self._release_event(sync_event)

        total_bytes = 0
        with torch.no_grad():
            for i, param in enumerate(params):
                if self.param_owner.get(id(param), layer_id) != layer_id:
                    continue
                if param.grad is not None and param.grad.is_cuda:
                    grad_src = param.grad.data
                    cpu_grad = self._get_cpu_tensor(param.grad.size(), param.grad.dtype)
                    self._chunked_copy(cpu_grad, grad_src, stream, use_async)
                    if use_async:
                        grad_src.record_stream(stream)
                    param.grad.data = cpu_grad
                    grad_devices[i] = 'cpu'
                    total_bytes += param.grad.numel() * param.grad.element_size()
                    # 立即释放GPU梯度内存
                    del grad_src
        
        if self.aggressive_release and total_bytes > 0:
            torch.cuda.empty_cache()
        
        self.stats['total_grad_offloaded_bytes'] += total_bytes
    
    def prefetch_layer_gradients(self, layer_id: int, async_op: bool = True):
        """预取层梯度从 CPU 到 GPU（用于 optimizer step）。"""
        if not self.enabled or not self.offload_gradients:
            return
        if layer_id not in self.layer_params:
            return

        params = self.layer_params[layer_id]
        if not params:
            return
            
        grad_devices = self.layer_grad_devices[layer_id]
        tgt = self.layer_target_device[layer_id]
        stream = self.h2d_stream if (async_op and self.h2d_stream is not None) else None
        use_async = async_op and stream is not None

        with torch.no_grad(), torch.cuda.device(tgt):
            for i, param in enumerate(params):
                if self.param_owner.get(id(param), layer_id) != layer_id:
                    continue
                if grad_devices[i] == 'cpu' and param.grad is not None and not param.grad.is_cuda:
                    old_cpu_grad = param.grad.data
                    gpu_grad = torch.empty(param.grad.size(), dtype=param.grad.dtype, device=tgt)
                    self._chunked_copy(gpu_grad, param.grad.data, stream, use_async)
                    param.grad.data = gpu_grad
                    grad_devices[i] = 'cuda'
                    # 归还旧的CPU梯度到缓存池
                    if not use_async:
                        self._release_cpu_tensor(old_cpu_grad)

    # --------------------- 预取编排 --------------------- #
    def prefetch_next_layers(self, current_layer_id: int):
        """预取后续层。性能优化：避免重复转换，只预取需要的层。"""
        if not self.enabled or self.num_prefetch_layers <= 0:
            return
        layer_ids = list(self.layer_params.keys())
        try:
            cur = layer_ids.index(current_layer_id)
            for i in range(1, self.num_prefetch_layers + 1):
                nxt = cur + i
                if nxt < len(layer_ids):
                    next_layer_id = layer_ids[nxt]
                    # 性能优化：如果已经在 GPU，跳过预取
                    devices = self.layer_param_devices[next_layer_id]
                    if not all(flag == 'cuda' for flag in devices):
                        self.prefetch_layer(next_layer_id, async_op=True)
        except ValueError:
            pass

    def ensure_on_device(self, layer_id: int):
        """确保层参数在 GPU 上。性能优化：快速路径判断。"""
        if not self.enabled or layer_id not in self.layer_params:
            return
        devices = self.layer_param_devices[layer_id]
        # 快速路径：如果已经全部在 GPU，直接返回
        if all(flag == 'cuda' for flag in devices):
            return
        self.prefetch_layer(layer_id, async_op=True)
        evt = self.layer_ready_events.get(layer_id, None)
        if evt is not None:
            torch.cuda.current_stream().wait_event(evt)

    # --------------------- saved_tensors hooks --------------------- #
    def _saved_pack(self, t: torch.Tensor):
        """保存张量的打包函数（当前直接返回）。"""
        return t

    def _enter_saved_hooks(self, layer_id: int):
        """进入 saved_tensors hooks 上下文。"""
        from torch.autograd.graph import saved_tensors_hooks
        def _unpack(t: torch.Tensor):
            # 在反向传播时确保张量在正确的设备上
            self.ensure_on_device(layer_id)
            return t
        ctx = saved_tensors_hooks(self._saved_pack, _unpack)
        ctx.__enter__()
        # 以 layer_id 存储，forward_post 再退出
        if not hasattr(self, "_layer_saved_ctx"):
            self._layer_saved_ctx = {}
        self._layer_saved_ctx[layer_id] = ctx

    def _exit_saved_hooks(self, layer_id: int):
        """退出 saved_tensors hooks 上下文。"""
        ctx = getattr(self, "_layer_saved_ctx", {}).pop(layer_id, None)
        if ctx is not None:
            ctx.__exit__(None, None, None)

    # --------------------- 重计算：策略与包装 --------------------- #
    def _should_recompute(self, layer_id: int) -> bool:
        """按策略决定该层是否进行重计算。"""
        if not self.recompute_enabled:
            return False
        if self.recompute_policy == "always":
            return True
        if self.recompute_policy == "auto":
            size = self.layer_param_bytes.get(layer_id, 0)
            return size >= self.recompute_min_params_bytes
        return False

    def _patch_forward_with_recompute(self, module: nn.Module, layer_id: int):
        """
        用重计算包装该 module.forward：
        - 初次前向：确保参数在 GPU、预取后续层；进入 saved_tensors hooks；
                    然后用 checkpoint 执行一次 forward。
        - 反向阶段触发的二次前向（重计算）：同样会在函数入口 ensure_on_device()，保证与 offload 协同。
        """
        if getattr(module, "_offload_prev_forward", None) is not None:
            return  # 已经打过补丁
        orig_forward = module.forward
        self.register_layer(layer_id, module)  # 确保已注册

        def _run_fwd(*a, **kw):
            # 该函数会在"初次前向"和"重计算时的二次前向"都会被调用
            # —— 在两种路径下都确保参数在 GPU。
            self.ensure_on_device(layer_id)
            # 在重计算的前向里，同样使用 saved_tensors_hooks，确保中间保存/解包时能 JIT 取回参数
            self._enter_saved_hooks(layer_id)
            try:
                return orig_forward(*a, **kw)
            finally:
                # 这里退出 hooks：不管是初次前向还是重算前向都对称关闭
                self._exit_saved_hooks(layer_id)

        def _wrapped_forward(*args, **kwargs):
            # 初次前向路径（只有这条路径会进入该 wrapper；重算时由 _ckpt 调回 _run_fwd）
            if not self.enabled:
                return orig_forward(*args, **kwargs)
            # 初次前向：保障当前层参数在设备，并预取后续层（减少重算/下层等待）
            self.ensure_on_device(layer_id)
            self.prefetch_next_layers(layer_id)
            # 是否启用该层重计算
            use_recompute = self._should_recompute(layer_id)
            if use_recompute and _ckpt is not None and torch.is_grad_enabled():
                # 在 PyTorch >= 2.0，建议 use_reentrant=False；旧版本没有此参数则退化调用
                try:
                    out = _ckpt(_run_fwd, *args, use_reentrant=self.recompute_use_reentrant, **kwargs)
                except TypeError:
                    # 兼容老版本（无 use_reentrant / 无 kwargs 支持）
                    if kwargs:
                        # 极老版本不支持 kwargs，退化为不重计算（以稳定为先）
                        out = _run_fwd(*args, **kwargs)
                    else:
                        out = _ckpt(_run_fwd, *args)
            else:
                out = _run_fwd(*args, **kwargs)
            # 模拟 on_forward_post_hook 的行为（但不重复进入/退出 saved_tensors hooks）
            if self.release_after_fwd:
                self.offload_layer(layer_id, async_op=True, empty_cache=False)
                self.layer_released_after_fwd[layer_id] = True
            return out

        module._offload_prev_forward = orig_forward
        module.forward = _wrapped_forward

    def _unpatch_forward(self, module: nn.Module):
        """可用于恢复 forward（目前一般不需要调用）。"""
        if getattr(module, "_offload_prev_forward", None) is not None:
            module.forward = module._offload_prev_forward
            module._offload_prev_forward = None

    # --------------------- 训练阶段钩子 --------------------- #
    def synchronize(self):
        """同步所有 offload streams。"""
        if self.enabled:
            if self.h2d_stream is not None:
                self.h2d_stream.synchronize()
            if self.d2h_stream is not None:
                self.d2h_stream.synchronize()

    def on_forward_pre_hook(self, layer_id: int):
        """前向传播前钩子：确保参数在设备上，启动预取。"""
        if not self.enabled:
            return
        self._enter_saved_hooks(layer_id)
        self.ensure_on_device(layer_id)
        self.prefetch_next_layers(layer_id)

    def on_forward_post_hook(self, layer_id: int):
        """前向传播后钩子：可选地 offload 参数。"""
        if not self.enabled:
            return
        self._exit_saved_hooks(layer_id)
        # PP 默认不在 FWD 后 offload；单卡/无 PP 可打开 release_after_fwd
        if self.release_after_fwd:
            self.offload_layer(layer_id, async_op=True, empty_cache=False)
            self.layer_released_after_fwd[layer_id] = True

    def on_backward_pre_hook(self, layer_id: int):
        """反向传播前钩子：JIT 取回参数（如果之前被 offload）。"""
        if not self.enabled:
            return
        # 若 FWD 后释放过，则在 BWD 前 JIT 取回
        if self.layer_released_after_fwd.get(layer_id, False):
            self.prefetch_layer(layer_id, async_op=True)
            evt = self.layer_ready_events.get(layer_id, None)
            if evt is not None:
                torch.cuda.current_stream().wait_event(evt)
            self.layer_released_after_fwd[layer_id] = False

    def on_backward_post_hook(self, layer_id: int):
        """反向传播后钩子：在 PP 中不执行操作，等待统一处理。"""
        if not self.enabled:
            return
        # 注意：不要在这里 offload 梯度！
        # 因为反向传播链可能还没完成，过早 offload 会导致设备不匹配错误。
        # 梯度 offload 应该在整个 backward 完成后，在 on_backward_complete() 中统一处理
        pass
    
    def on_backward_complete(self):
        """整个反向传播完成后的钩子：统一 offload 所有梯度到 CPU。"""
        if not self.enabled or not self.offload_gradients:
            return
        
        # 在整个 backward 完成后，统一 offload 所有层的梯度
        for layer_id in self.layer_params.keys():
            self.offload_layer_gradients(layer_id, async_op=True)
        
        # 等待 D2H 完成
        if self.d2h_stream is not None:
            self.d2h_stream.synchronize()
        
        # 积极释放显存
        if self.aggressive_release:
            torch.cuda.empty_cache()

    def on_optimizer_step_pre(self):
        """优化器步骤前：确保所有参数和梯度在 GPU 上。"""
        if not self.enabled:
            return
        
        # 确保全部参数和梯度在 GPU（处理重算/延迟预取场景）
        for layer_id in self.layer_params.keys():
            self.ensure_on_device(layer_id)
            # 预取梯度回 GPU
            if self.offload_gradients:
                self.prefetch_layer_gradients(layer_id, async_op=True)
        
        if self.h2d_stream is not None:
            self.h2d_stream.synchronize()

        # （可选）优化器状态 JIT 预取（教学用途）
        if self.offload_optimizer_states and self._optimizer is not None:
            stream = self.h2d_stream
            use_async = stream is not None
            for group in self._optimizer.param_groups:
                for p in group['params']:
                    if p.device.type != 'cuda':
                        continue  # 跳过非 CUDA 参数
                    state = self._optimizer.state.get(p, {})
                    for k in ('exp_avg', 'exp_avg_sq'):
                        buf = state.get(k, None)
                        if isinstance(buf, torch.Tensor) and (not buf.is_cuda):
                            gpu_buf = torch.empty_like(buf, device=p.device)
                            self._chunked_copy(gpu_buf, buf, stream, use_async)
                            state[k] = gpu_buf
            if use_async:
                stream.synchronize()

    def on_optimizer_step_post(self):
        """优化器步骤后：将所有参数 offload 到 CPU。性能优化：批量处理，定期清理。"""
        if not self.enabled:
            return
        
        # 统一将所有层参数 offload 到 CPU（PP 友好）
        for layer_id in self.layer_params.keys():
            self.offload_layer(layer_id, async_op=True, empty_cache=False)
            self.layer_released_after_fwd[layer_id] = True

        # （可选）优化器状态 offload（教学用途）
        if self.offload_optimizer_states and self._optimizer is not None:
            stream = self.d2h_stream
            use_async = stream is not None
            for group in self._optimizer.param_groups:
                for p in group['params']:
                    state = self._optimizer.state.get(p, {})
                    for k in ('exp_avg', 'exp_avg_sq'):
                        buf = state.get(k, None)
                        if isinstance(buf, torch.Tensor) and buf.is_cuda:
                            cpu_buf = self._get_cpu_tensor(buf.size(), buf.dtype)
                            self._chunked_copy(cpu_buf, buf, stream, use_async)
                            if use_async:
                                buf.record_stream(stream)
                            state[k] = cpu_buf

        if self.d2h_stream is not None:
            self.d2h_stream.synchronize()

        # 清理已完成的 H2D 源张量
        self._cleanup_completed_h2d_sources()
        # 彻底清理剩余的源张量引用
        self._pending_h2d_src.clear()
        
        # 定期清理张量缓存（避免缓存过大）
        total_cached = sum(len(v) for v in self._cpu_tensor_cache.values())
        if total_cached > self._max_cache_size:
            self._clear_tensor_cache()
        
        # 更积极的显存释放
        if self.aggressive_release:
            torch.cuda.empty_cache()

    # --------------------- 统计 --------------------- #
    def get_stats(self) -> Dict:
        stats = self.stats.copy()
        stats['total_offloaded_mb'] = stats['total_offloaded_bytes'] / 1024**2
        stats['total_prefetched_mb'] = stats['total_prefetched_bytes'] / 1024**2
        stats['total_grad_offloaded_mb'] = stats['total_grad_offloaded_bytes'] / 1024**2
        stats['total_cached_tensors'] = sum(len(v) for v in self._cpu_tensor_cache.values())
        stats['cache_hit_rate'] = (stats['num_cache_hits'] / max(1, stats['num_offload_ops'])) * 100
        return stats

    def print_stats(self):
        s = self.get_stats()
        logger.info("=" * 70)
        logger.info("Tensor Offload Statistics:")
        logger.info(f"  Params offloaded:   {s['total_offloaded_mb']:.2f} MB ({s['num_offload_ops']} ops)")
        logger.info(f"  Params prefetched:  {s['total_prefetched_mb']:.2f} MB ({s['num_prefetch_ops']} ops)")
        logger.info(f"  Grads offloaded:    {s['total_grad_offloaded_mb']:.2f} MB")
        logger.info(f"  Cache hit rate:     {s['cache_hit_rate']:.1f}% ({s['num_cache_hits']} hits)")
        logger.info(f"  Cached tensors:     {s['total_cached_tensors']}")
        logger.info("=" * 70)

# --------- 全局实例与挂钩工具 --------- #
_global_offload_manager: Optional[TensorOffloadManager] = None

def get_offload_manager() -> Optional[TensorOffloadManager]:
    return _global_offload_manager

def initialize_offload_manager(
    enabled: bool = True,
    offload_optimizer_states: bool = False,
    pin_memory: bool = True,
    num_prefetch_layers: int = 1,
    release_after_fwd: bool = False,
    bucket_mb: int = 0,
    pp_mode: bool = True,
    copy_chunk_mb: int = 64,
    include_buffers: bool = False,
    offload_gradients: bool = True,
    aggressive_release: bool = True,
    recompute_enabled: bool = True,
    recompute_policy: str = "auto",
    recompute_min_params_mb: float = 0.0,
    recompute_use_reentrant: Optional[bool] = False,
) -> TensorOffloadManager:
    global _global_offload_manager
    _global_offload_manager = TensorOffloadManager(
        enabled=enabled,
        offload_optimizer_states=offload_optimizer_states,
        pin_memory=pin_memory,
        num_prefetch_layers=num_prefetch_layers,
        release_after_fwd=release_after_fwd,
        bucket_mb=bucket_mb,
        pp_mode=pp_mode,
        copy_chunk_mb=copy_chunk_mb,
        include_buffers=include_buffers,
        offload_gradients=offload_gradients,
        aggressive_release=aggressive_release,
        recompute_enabled=recompute_enabled,
        recompute_policy=recompute_policy,
        recompute_min_params_mb=recompute_min_params_mb,
        recompute_use_reentrant=recompute_use_reentrant,
    )
    return _global_offload_manager

def attach_offload_hooks_to_layer(layer: nn.Module, layer_id: int, mgr: TensorOffloadManager):
    """
    当 mgr.recompute_enabled=True 时，采用"forward 包装 + 重计算"路径；
    否则沿用原先的 pre/post forward hooks。
    """
    # 始终登记层（记录设备/参数/大小等）
    mgr.register_layer(layer_id, layer)

    if mgr.recompute_enabled:
        # 重计算路径：包装 forward，不再注册 forward hooks，避免重入/重复
        mgr._patch_forward_with_recompute(layer, layer_id)
        # 返回一个"兼容句柄"，用于保持接口一致
        class _DummyHandle:
            def remove(self):  # 兼容性：不做事
                pass
        return (_DummyHandle(), _DummyHandle())
    else:
        # 旧路径：使用 hooks（不做重计算）
        def _pre(_module, _inputs):
            mgr.on_forward_pre_hook(layer_id)
            return None

        def _post(_module, _inputs, _outputs):
            mgr.on_forward_post_hook(layer_id)
            return None

        h1 = layer.register_forward_pre_hook(_pre, with_kwargs=False)
        h2 = layer.register_forward_hook(_post, with_kwargs=False)
        return (h1, h2)
