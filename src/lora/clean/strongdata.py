import json
import re
import os
import time
from openai import OpenAI

# ================= 配置区 =================
# 替换为你的 DeepSeek API Key
DEEPSEEK_API_KEY = "sk-a61c012a994c4815ad86990665eeab4e" 
DEEPSEEK_BASE_URL = "https://api.deepseek.com" 

# 初始化客户端
client = OpenAI(
    api_key=DEEPSEEK_API_KEY, 
    base_url=DEEPSEEK_BASE_URL
)
# ==========================================

def call_llm_api(prompt, retries=3):
    """
    调用 DeepSeek API，带有简单的重试机制以应对网络抖动
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",  # 指定使用的模型
                messages=[
                    {"role": "system", "content": "你是一个严格遵守格式要求的文本处理引擎。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7, # 0.7 适合生成具备一定多样性的口语化文本
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ API 调用失败 (尝试 {attempt + 1}/{retries}): {e}")
            time.sleep(2) # 失败后等待 2 秒再重试
            
    raise Exception("API 请求多次重试后仍然失败")

def augment_dialogue_data(input_file, output_file):
    augmented_dataset = []
    
    # Prompt 优化：明确要求不输出 markdown 标记，降低解析难度
    prompt_template = """你是一个专业的房地产行业数据增强专家。
我将提供一个【房产经纪人面临的实战情景】以及【对应的标准回答】。
请你从“真实购房客户”的视角，生成 3 种不同语气和表达方式的真实客户话语。

要求：
1. 必须是第一人称（客户本人的口吻），极度口语化，不要像机器生成的。
2. 3 种语气请分别设定为：【直接了当型】、【犹豫试探型】、【挑剔/不耐烦型】。
3. 只输出一个 JSON 数组，格式如：["话语1", "话语2", "话语3"]。
4. 绝对不要输出任何其他解释性文字，不要输出 markdown 代码块标记 (如 ```json)。

【实战情景】：{instruction}
【标准回答参考】：{output}
"""

    if not os.path.exists(input_file):
        print(f"❌ 找不到输入文件: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        try:
            item = json.loads(line.strip())
            orig_instruction = item["instruction"]
            orig_output = item["output"]
            
            prompt = prompt_template.format(
                instruction=orig_instruction,
                output=orig_output
            )
            
            print(f"⏳ 正在增强第 {i+1}/{len(lines)} 条数据...")
            llm_response = call_llm_api(prompt) 
            
            # 使用正则提取方括号包裹的数组内容
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                client_queries = json.loads(json_match.group(0))
                
                for query in client_queries:
                    augmented_dataset.append({
                        "instruction": query,
                        "input": "", 
                        "output": orig_output
                    })
            else:
                print(f"❌ 解析失败: 模型未返回标准 JSON 数组 -> {llm_response}")
                
        except json.JSONDecodeError as e:
             print(f"❌ JSON 解析错误 (第 {i+1} 条): {e}")
        except Exception as e:
            print(f"❌ 处理第 {i+1} 条时发生未知错误: {e}")
            
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in augmented_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"\n🎉 增强完成！原始数据 {len(lines)} 条，成功生成 {len(augmented_dataset)} 条增强数据。")
    print(f"👉 结果已保存至: {output_file}")

if __name__ == "__main__":
    # 确保此处填写的输入文件是你之前清洗过的那一份
    augment_dialogue_data("dialogue_cleaned.jsonl", "dialogue_augmented.jsonl")