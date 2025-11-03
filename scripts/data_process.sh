#!/bin/bash
#SBATCH -J preprocess_data
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 24:00:00
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

cd /data/home/scyb226/lzx/Megatron-LM/scripts
python prepare_data.py

cd /data/home/scyb226/lzx/Megatron-LM/

python tools/preprocess_data.py \
       --input /data/home/scyb226/lzx/data/megatron/codeparrot_data.json \
       --output-prefix /data/home/scyb226/lzx/data/megatron/processed_codeparrot \
       --tokenizer-model /data/home/scyb226/.cache/models/shakechen/Megatron-LM/hf_models/Llama-2-7b-hf/tokenizer.model \
       --tokenizer-type Llama2Tokenizer \
       --json-keys content \
       --workers 8 \
       --append-eod
