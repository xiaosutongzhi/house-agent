from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
save_path = "/mnt/nas/szm/model"

# 下载到指定 NAS 目录
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=save_path)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    cache_dir=save_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)