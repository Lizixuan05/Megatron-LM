#!/bin/bash
#SBATCH -J test          # 作业名
#SBATCH -p gpu                    # 分区名称（根据集群情况修改）
#SBATCH -N 1                      # 节点数（NNODES）
#SBATCH --ntasks-per-node=1       # 每节点任务数（保持 1 即可）
#SBATCH --gpus-per-node=4         # 每节点 GPU 数量（GPUS_PER_NODE）
#SBATCH -t 1:00:00               # 最长运行时间（根据需要修改）
#SBATCH -o /data/home/scyb226/lzx/Megatron-LM/scripts/logs/%x_%j.out         # 标准输出日志
#SBATCH -e /data/home/scyb226/lzx/Megatron-LM/scripts/logs/%x_%j.err         # 错误日志

unset LD_LIBRARY_PATH
module load cuda/12.4
module load cudnn/9.11.0.98_cuda12
# 初始化 conda 并激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate megatron_env

export CUDNN_PATH=/data/apps/cudnn/cudnn-linux-x86_64-9.11.0.98_cuda12-archive
export CFLAGS="-I${CUDNN_PATH}/include $CFLAGS"
export CPATH="${CUDNN_PATH}/include:$CPATH"

# 将 CUDNN 库路径添加到 LD_LIBRARY_PATH，transformer_engine 需要这些库
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib:/data/apps/cuda/12.4/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib

export PYTHONPATH="/data/home/scyb226/lzx/Megatron-LM:$PYTHONPATH"



echo $LD_LIBRARY_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
#export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO                                                                                                                                        
                                                                          
export NCCL_IB_DISABLE=0                                                                                                                                      
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1                                                                                                        
export NCCL_SOCKET_IFNAME=bond0                                                                                                                               
export NCCL_IB_TIMEOUT=23                                                                                                                                     
export NCCL_IB_RETRY_CNT=13

# 内存优化：减少内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  


TP=1  # 张量模型并行度，根据实际改
PP=4  # pipeline并行度
# 路径参数
DATA_PATH="/data/home/scyb226/lzx/data/megatron/processed_codeparrot_content_document"
TOKENIZER_MODEL="/data/home/scyb226/.cache/models/shakechen/Megatron-LM/hf_models/Llama-2-7b-hf/tokenizer.model"
CHECKPOINT_DIR="/data/home/scyb226/lzx/checkpoints"
SAVE_DIR="/data/home/scyb226/lzx/checkpoints"
# 训练超参数
NUM_LAYERS=32  # Llama-2-7b 通常使用 32 层
HIDDEN_SIZE=4096  # Llama-2-7b 的隐藏层大小
NUM_ATTENTION_HEADS=32  # Llama-2-7b 的注意力头数
SEQ_LENGTH=2048
MAX_POS_EMB=4096
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=16
LR=0.00003
MIN_LR=1e-6
LR_DECAY_STYLE="cosine"
WEIGHT_DECAY=0.01
CLIP_GRAD=1.0
WARMUP=3200
TRAIN_ITERS=320000
SAVE_INTERVAL=10000
LOG_INTERVAL=100
EVAL_INTERVAL=1000
EVAL_ITERS=10
LR_WARMUP_FRACTION=0.01
SPLIT="90,5,5"
# 分布式启动参数
NUM_PROC=4  # 总进程数，需根据机器GPU数调整（必须能被 total_model_size=TP*PP*CP 整除）

# 切换到 Megatron-LM 根目录
# 直接使用绝对路径，在 SLURM 环境下最可靠
MEGATRON_ROOT="/data/home/scyb226/lzx/Megatron-LM"
cd "${MEGATRON_ROOT}" || {
    echo "Error: Cannot cd to ${MEGATRON_ROOT}"
    exit 1
}

echo "Starting Megatron-LM training..."
echo "Current directory: $(pwd)"
echo "Megatron root: ${MEGATRON_ROOT}"
if [ ! -f "${MEGATRON_ROOT}/pretrain_gpt.py" ]; then
    echo "Error: pretrain_gpt.py not found at ${MEGATRON_ROOT}/pretrain_gpt.py"
    exit 1
fi
echo "pretrain_gpt.py found: YES"

torchrun --nproc_per_node=${NUM_PROC} --master_port=29501 "${MEGATRON_ROOT}/pretrain_gpt.py" \
  --data-path ${DATA_PATH} \
  --split ${SPLIT} \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --num-attention-heads ${NUM_ATTENTION_HEADS} \
  --transformer-impl local \
  --tensor-model-parallel-size ${TP} \
  --pipeline-model-parallel-size ${PP} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${MAX_POS_EMB} \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --untie-embeddings-and-output-weights \
  --use-rotary-position-embeddings \
  --no-rope-fusion \
  --normalization RMSNorm \
  --no-persist-layer-norm \
  --no-position-embedding \
  --no-masked-softmax-fusion \
  --attention-softmax-in-fp32 \
  --recompute-granularity selective \
  --init-method-std 0.006 \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style ${LR_DECAY_STYLE} \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${CLIP_GRAD} \
  --lr-warmup-fraction ${LR_WARMUP_FRACTION} \
  --train-iters ${TRAIN_ITERS} \
  --save ${SAVE_DIR} \
  --save-interval ${SAVE_INTERVAL} \
  --log-interval ${LOG_INTERVAL} \
  --eval-interval ${EVAL_INTERVAL} \
  --eval-iters ${EVAL_ITERS} \
  --log-throughput \
  --log-memory-to-tensorboard \
  --bf16 \
  --enable-tensor-offload \
  --tensor-offload-pin-memory \
  --tensor-offload-num-prefetch-layers 3  \
  #--tensor-offload-optimizer-states
  #--tensor-offload-release-after-fwd

