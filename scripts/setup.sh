#!/bin/bash
#SBATCH -J megatron_setup          # 作业名
#SBATCH -p gpu                    # 分区名称（根据集群情况修改）
#SBATCH -N 1                      # 节点数（NNODES）
#SBATCH --ntasks-per-node=1       # 每节点任务数（保持 1 即可）
#SBATCH --gpus-per-node=1         # 每节点 GPU 数量（GPUS_PER_NODE）
#SBATCH -t 48:00:00               # 最长运行时间（根据需要修改）
#SBATCH -o logs/%x_%j.out         # 标准输出日志
#SBATCH -e logs/%x_%j.err         # 错误日志
# 清理环境变量
unset LD_LIBRARY_PATH
# 加载cuda和cudnn
module load cuda/12.4
module load cudnn/9.11.0.98_cuda12
# 初始化 conda 并激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate megatron_env
# 设置cudnn编译路径
export CUDNN_PATH=/data/apps/cudnn/cudnn-linux-x86_64-9.11.0.98_cuda12-archive
export CFLAGS="-I${CUDNN_PATH}/include $CFLAGS"
export CPATH="${CUDNN_PATH}/include:$CPATH"

# 设置基础LD_LIBRARY_PATH
# 需要添加cudnn、cuda、torch的库路径
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib:/data/apps/cuda/12.4/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib
# 设置PYTHONPATH
export PYTHONPATH="/data/home/scyb226/lzx/Megatron-LM:$PYTHONPATH"

echo $LD_LIBRARY_PATH


# Add NCCL header path for transformer_engine compilation
export NCCL_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/include
export CFLAGS="-I${NCCL_INCLUDE_PATH} $CFLAGS"
export CPATH="${NCCL_INCLUDE_PATH}:$CPATH"

# Disable downloading pre-built wheels from GitHub (which may fail due to network issues)
# Force building from source instead by using --no-binary flag
# This prevents pip from trying to download wheels from GitHub
pip install -v --no-build-isolation --no-binary transformer_engine_torch transformer_engine[pytorch] --no-cache-dir

# Install apex with CUDA extensions for Python 3.12
# Apex needs to be compiled for the current Python version
cd /data/home/scyb226/lzx/apex

# Set CUDA paths for compilation
export CUDA_HOME=/data/apps/cuda/12.4
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"  # Support common GPU architectures

# Enable CUDA extensions compilation via environment variables
export APEX_CUDA_EXT=1
export APEX_CPP_EXT=1

# Install apex with CUDA extensions, ensuring it compiles for Python 3.12
pip install -v --no-build-isolation --no-cache-dir .
