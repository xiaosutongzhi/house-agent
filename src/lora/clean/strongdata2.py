import json
import re
import os
import time
from openai import OpenAI

# ================= 配置区 =================
DEEPSEEK_API_KEY = "sk-a61c012a994c4815ad86990665eeab4e"  # 替换为你的真实 Key
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

def augment_house_quotes(input_file, output_file):
    augmented_dataset = []
    
    # 针对专业知识解答的同义裂变 Prompt
    prompt_template = """你是一个专业的房地产行业数据增强专家。
我将提供一个【房产专业知识的主题】以及【经纪人的专业解答】。
请你围绕这个主题，生成 3 种不同语境下的“真实客户提问”（第一人称口吻，口语化，像在实地看房或微信聊天）。

要求：
1. 【咨询请教型】：客户是小白，态度诚恳地向经纪人请教这个选房知识。
2. 【纠结对比型】：客户在考虑利弊，或者在多个选项间犹豫，希望经纪人帮忙分析。
3. 【质疑担忧型】：客户听说了这个房子属性的缺点（如顶楼漏水、西晒、一楼潮湿等），带着防备心提出质疑。
4. 只输出一个 JSON 数组，格式如：["提问1", "提问2", "提问3"]。绝对不要输出其他解释性文字或 markdown 标记。

【专业知识主题】：{instruction}
【经纪人的专业解答】：{output}
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
            
            print(f"⏳ 正在增强第 {i+1}/{len(lines)} 条房产知识...")
            llm_response = call_llm_api(prompt) 
            
            # 使用正则提取方括号包裹的数组内容
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                client_queries = json.loads(json_match.group(0))
                
                # 将生成的 3 条真实发问，分别与原 output 组合
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
            
    print(f"\n🎉 增强完成！原始数据 {len(lines)} 条，成功生成 {len(augmented_dataset)} 条专业知识问答数据。")
    print(f"👉 结果已保存至: {output_file}")

if __name__ == "__main__":
    # 确保此处填写的输入文件是你之前清洗过的 house_quotes
    augment_house_quotes("house_quotes_cleaned.jsonl", "house_quotes_augmented.jsonl")