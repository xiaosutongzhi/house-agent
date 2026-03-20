import json
import re
import os
import time
from openai import OpenAI

# ================= 配置区 =================
DEEPSEEK_API_KEY = "sk-a61c012a994c4815ad86990665eeab4e" 
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

client = OpenAI(
    api_key=DEEPSEEK_API_KEY, 
    base_url=DEEPSEEK_BASE_URL
)
# ==========================================

def call_llm_api(prompt, retries=3):
    """调用 DeepSeek API，带有重试机制"""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个严格遵守格式要求的文本处理引擎。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ API 调用失败 (尝试 {attempt + 1}/{retries}): {e}")
            time.sleep(2)
            
    raise Exception("API 请求多次重试后仍然失败")

def augment_golden_list(input_file, output_file):
    augmented_dataset = []
    
    # 针对 Golden List 的逆向推导 Prompt
    prompt_template = """你是一个专业的房地产行业数据增强专家。
我将提供一个【房产经纪人的核心服务承诺/价值展示】（这是标准回答）。
请你逆向思考：在真实带看或沟通中，客户说了什么话，才会让经纪人顺理成章地说出这句话？

请生成 3 种不同视角的真实客户原话（第一人称口吻，极度口语化）。
要求：
1. 【痛点抱怨型】：客户对当前看房或交易过程感到疲惫、困惑或遇到麻烦。
2. 【直接求助型】：客户对专业流程不懂，直接向经纪人提问。
3. 【价值质疑型】：客户觉得中介费太贵，或者觉得不需要中介，质疑中介能干什么。
4. 只输出一个 JSON 数组，格式如：["话语1", "话语2", "话语3"]。绝对不要输出其他解释性文字或 markdown 标记。

【经纪人的回答】（你需要根据这个回答来倒推客户的问题）：
{output}
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
            # 注意：这里我们完全丢弃了原有的垃圾 Instruction
            orig_output = item["output"] 
            
            prompt = prompt_template.format(output=orig_output)
            
            print(f"⏳ 正在逆向增强第 {i+1}/{len(lines)} 条服务承诺...")
            llm_response = call_llm_api(prompt) 
            
            # 使用正则提取方括号包裹的数组内容
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                client_queries = json.loads(json_match.group(0))
                
                # 将生成的 3 条真实触发语，分别与原 output 组合
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
            
    print(f"\n🎉 增强完成！原始数据 {len(lines)} 条，成功生成 {len(augmented_dataset)} 条逆向增强数据。")
    print(f"👉 结果已保存至: {output_file}")

if __name__ == "__main__":
    # 确保此处填写的输入文件是你之前清洗过的 golden_list
    augment_golden_list("golden_list_cleaned.jsonl", "golden_list_augmented.jsonl")