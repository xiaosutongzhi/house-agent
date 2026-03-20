import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ================= 配置区 =================
# 填入你真实的本地绝对路径（就是你之前解决离线报错用的那个哈希文件夹路径）
BASE_MODEL_PATH = "/mnt/nas/szm/model/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B"
# 你刚跑完输出的 lora 文件夹路径
LORA_PATH = "/mnt/nas/szm/model/house_agent_deepseekr1_llama8b_lora"
# ==========================================

print("⏳ 正在加载基础模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
# 以 4-bit 量化加载基础模型，省显存
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config={"load_in_4bit": True} # 如果报错缺 bitsandbytes，可以删掉这行
)

print("⏳ 正在合并 LoRA 权重...")
# 将你的微调权重挂载到基座模型上
model = PeftModel.from_pretrained(base_model, LORA_PATH)
def generate_response(prompt, use_lora=True):
    # 构造对话模板
    messages = [{"role": "user", "content": prompt}]
    
    # 明确要求返回字典 (return_dict=True)
    inputs = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        add_generation_prompt=True,
        return_dict=True
    ).to(model.device)
    
    # 因为 inputs 是字典，使用 **inputs 解包传给 generate
    if not use_lora:
        with model.disable_adapter():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    else:
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
        
    # 获取真正的张量 inputs["input_ids"] 的长度，用来截取生成的新文本
    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # 测试问题（最好用一个不在你训练集里，但场景类似的新问题）
    test_prompt = "你们中介费收这么高，我看网上说这小区物业特别差，你是不是想坑我赶紧接盘？"
    
    print(f"\n【客户提问】: {test_prompt}\n")
    print("-" * 50)
    
    # 1. 原始模型回答
    print("🤖 【原始 DeepSeek-R1-8B 回答】:")
    base_ans = generate_response(test_prompt, use_lora=False)
    print(base_ans)
    
    print("-" * 50)
    
    # 2. 微调模型回答
    print("🤵 【微调后的房产 Agent 回答】:")
    lora_ans = generate_response(test_prompt, use_lora=True)
    print(lora_ans)