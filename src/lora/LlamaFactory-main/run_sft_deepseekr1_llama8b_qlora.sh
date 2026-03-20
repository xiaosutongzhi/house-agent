#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# 新增：强制 Hugging Face 和 Transformers 走离线模式，彻底切断网络请求
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# ==========================================================

# 你的环境：指定 GPU 5
: "${CUDA_VISIBLE_DEVICES:=5}" 
: "${MODEL_ID:=/mnt/nas/szm/model/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B}"
: "${HF_CACHE_DIR:=/mnt/nas/szm/model}"
: "${OUT_DIR:=/mnt/nas/szm/model/house_agent_deepseekr1_llama8b_lora}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

# If you want to use 2 GPUs on a single machine, set:
#   CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 bash run_sft_deepseekr1_llama8b_qlora.sh
export FORCE_TORCHRUN=${FORCE_TORCHRUN:-0}

llamafactory-cli train \
  --stage sft \
  --do_train True \
  --model_name_or_path "$MODEL_ID" \
  --cache_dir "$HF_CACHE_DIR" \
  --template deepseekr1 \
  --dataset_dir data \
  --dataset house_agent \
  --finetuning_type lora \
  --output_dir "$OUT_DIR" \
  --overwrite_output_dir \
  --preprocessing_num_workers 8 \
  --cutoff_len 384 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 3.0 \
  --lora_target all \
  --lora_rank 16 \
  --lora_dropout 0.05 \
  --quantization_bit 4 \
  --gradient_checkpointing True \
  --enable_thinking False \
  --fp16 True