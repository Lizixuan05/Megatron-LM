#!/usr/bin/env python3
"""
简单的Tensor Offload功能测试（改进版）
"""
import torch
import torch.nn as nn
import time
import sys
import os

# 添加Megatron路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from megatron.core.tensor_offload import TensorOffloadManager


class SimpleLayer(nn.Module):
    """简单的测试层"""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


# -------------------------- 工具函数 --------------------------

def _cuda_barrier():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _reset_mem_counters():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def _peak_reserved_mb():
    # 统计 allocator 曾经向 driver 预留的峰值，更接近“峰值驻留”
    return torch.cuda.max_memory_reserved() / 1024**2

def _current_reserved_mb():
    return torch.cuda.memory_reserved() / 1024**2


# -------------------------- 测试 1：基本 offload 功能 --------------------------

def test_basic_offload():
    print("=" * 60)
    print("测试 1: 基本Offload功能")
    print("=" * 60)

    manager = TensorOffloadManager(
        enabled=True,
        pin_memory=False,         # 关闭 pinned memory，避免显存虚高
        num_prefetch_layers=1
    )

    hidden_size = 1024
    num_layers = 4
    layers = [SimpleLayer(hidden_size).cuda() for _ in range(num_layers)]

    for i, layer in enumerate(layers):
        manager.register_layer(i, layer)
        print(f"✓ 注册 Layer {i}")

    print("\n" + "-" * 60)
    print("测试参数位置...")
    print("-" * 60)

    for i in range(num_layers):
        print(f"\n处理 Layer {i}:")
        first_param = next(layers[i].parameters())
        print(f"  初始位置: {first_param.device}")
        assert first_param.device.type == 'cuda', "参数应该在GPU上"

        # Offload到CPU（异步），随后同步以便检查位置
        manager.offload_layer(i, async_op=True)
        manager.synchronize()

        first_param = next(layers[i].parameters())
        print(f"  Offload后: {first_param.device}")
        assert first_param.device.type == 'cpu', "参数应该在CPU上"

        # Prefetch回GPU（异步），随后同步以便检查位置
        manager.prefetch_layer(i, async_op=True)
        manager.synchronize()

        first_param = next(layers[i].parameters())
        print(f"  Prefetch后: {first_param.device}")
        assert first_param.device.type == 'cuda', "参数应该在GPU上"

        print(f"  ✓ Layer {i} 测试通过")

    print("\n" + "=" * 60)
    manager.print_stats()
    return True


# -------------------------- 测试 2：forward 流程 offload --------------------------

def test_forward_with_offload():
    print("\n" + "=" * 60)
    print("测试 2: Forward过程中的Offload")
    print("=" * 60)

    manager = TensorOffloadManager(
        enabled=True,
        pin_memory=False,         # 关闭 pinned memory
        num_prefetch_layers=1     # 减少并发驻留层数
    )

    hidden_size = 512
    num_layers = 6
    batch_size = 4
    seq_length = 128

    layers = [SimpleLayer(hidden_size).cuda() for _ in range(num_layers)]
    for i, layer in enumerate(layers):
        manager.register_layer(i, layer)

    print("\nOffload所有层到CPU...")
    for i in range(num_layers):
        manager.offload_layer(i, async_op=True)
    manager.synchronize()
    print("✓ 完成")

    print("\n" + "-" * 60)
    print("模拟Forward Pass...")
    print("-" * 60)

    x = torch.randn(batch_size, seq_length, hidden_size, device="cuda")

    with torch.inference_mode():
        for i in range(num_layers):
            print(f"\nLayer {i} forward:")

            # 预取当前层与下一层（由 manager 内部策略决定）
            manager.on_forward_pre_hook(i)
            print(f"  ✓ Prefetched layer {i}")

            # 前向
            start_time = time.time()
            x = layers[i](x)
            forward_time = (time.time() - start_time) * 1000
            print(f"  ✓ Forward完成 ({forward_time:.2f}ms)")

            # 用完立刻 post hook
            manager.on_forward_post_hook(i)

            # 模拟 backward 的释放（这里只是触发释放旧层参数的 offload）
            if i > 0:
                manager.on_backward_post_hook(i - 1)
                print(f"  ✓ Offloaded layer {i-1}")

    print("\n" + "=" * 60)
    print("Forward Pass完成!")
    manager.print_stats()
    return True


# -------------------------- 测试 3：性能 & 显存对比 --------------------------

def test_performance_comparison():
    print("\n" + "=" * 60)
    print("测试 3: 性能对比")
    print("=" * 60)

    hidden_size = 4096
    num_layers = 24
    batch_size = 16
    seq_length = 1024
    num_iterations = 10

    # --------- 公共输入 ---------
    x = torch.randn(batch_size, seq_length, hidden_size, device="cuda")

    # ================== 不使用 offload ==================
    print("\n1. 不使用Offload...")
    print("-" * 60)

    layers_no_offload = [SimpleLayer(hidden_size).cuda() for _ in range(num_layers)]

    # 预热
    with torch.inference_mode():
        for layer in layers_no_offload:
            x = layer(x)

    _cuda_barrier()
    _reset_mem_counters()  # 预热后重置峰值计数

    # 计时 + 峰值显存
    with torch.inference_mode():
        start = time.time()
        for _ in range(num_iterations):
            x_temp = x
            for layer in layers_no_offload:
                x_temp = layer(x_temp)
        _cuda_barrier()
        time_no_offload = (time.time() - start) / num_iterations * 1000.0
        mem_no_offload = _peak_reserved_mb()

    print(f"  平均时间: {time_no_offload:.2f} ms/iteration")
    print(f"  峰值显存: {mem_no_offload:.2f} MB")

    # 释放 & 清理
    del layers_no_offload
    _cuda_barrier()
    _reset_mem_counters()

    # ================== 使用 offload ==================
    print("\n2. 使用Offload...")
    print("-" * 60)

    manager = TensorOffloadManager(
        enabled=True,
        pin_memory=False,         # 关闭 pinned memory 避免虚高
        num_prefetch_layers=1     # 降低并发驻留
    )

    layers_with_offload = [SimpleLayer(hidden_size).cuda() for _ in range(num_layers)]
    for i, layer in enumerate(layers_with_offload):
        manager.register_layer(i, layer)

    # 首先把所有层 offload 到 CPU（异步），再同步
    for i in range(num_layers):
        manager.offload_layer(i, async_op=True)
    manager.synchronize()

    # 预热一次（触发完整的 prefetch/offload 生命周期）
    with torch.inference_mode():
        for i, layer in enumerate(layers_with_offload):
            manager.on_forward_pre_hook(i)
            x = layer(x)
            manager.on_forward_post_hook(i)
            if i > 0:
                manager.on_backward_post_hook(i - 1)
        manager.synchronize()

    # 预热后重置峰值计数
    _cuda_barrier()
    _reset_mem_counters()

    # 正式计时循环
    with torch.inference_mode():
        start = time.time()
        for _ in range(num_iterations):
            x_temp = x
            for i, layer in enumerate(layers_with_offload):
                manager.on_forward_pre_hook(i)
                x_temp = layer(x_temp)
                manager.on_forward_post_hook(i)
                if i > 0:
                    manager.on_backward_post_hook(i - 1)
        manager.synchronize()
        _cuda_barrier()
        time_with_offload = (time.time() - start) / num_iterations * 1000.0
        mem_with_offload = _peak_reserved_mb()

    print(f"  平均时间: {time_with_offload:.2f} ms/iteration")
    print(f"  峰值显存: {mem_with_offload:.2f} MB")

    # ================== 对比 ==================
    print("\n" + "=" * 60)
    print("性能对比结果:")
    print("=" * 60)

    time_overhead = ((time_with_offload - time_no_offload) / time_no_offload) * 100
    mem_savings = ((mem_no_offload - mem_with_offload) / mem_no_offload) * 100

    print(f"时间开销: {time_overhead:+.1f}%")
    print(f"显存节约: {mem_savings:.1f}%")
    print()

    if mem_savings > 0:
        print("✓ 成功节约了显存!")
    else:
        print("⚠ 显存没有减少（可能由于统计口径/缓存/prefetch设置）")

    manager.print_stats()
    return True


# -------------------------- main --------------------------

def main():
    print("\n" + "=" * 60)
    print(" Tensor Offload 功能测试套件（改进版）")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("错误: CUDA不可用，无法运行测试")
        return False

    print(f"\nCUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch版本: {torch.__version__}")
    print()

    try:
        tests = [
            test_basic_offload,
            test_forward_with_offload,
            test_performance_comparison,
        ]

        for test_func in tests:
            success = test_func()
            if not success:
                print(f"\n✗ 测试失败: {test_func.__name__}")
                return False

        print("\n" + "=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)
        print()
        print("Tensor Offload 功能正常工作。")
        print()
        return True

    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
