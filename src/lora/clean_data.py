import json
import time
from openai import OpenAI
import os

# ================= 配置区 =================
# 这里以 DeepSeek 官方 API 为例。如果你用的是其他平台(如阿里云DashScope/智谱等)，请替换 base_url 和 api_key。
BASE_URL = "https://api.deepseek.com/v1"
API_KEY="sk-a61c012a994c4815ad86990665eeab4e"
MODEL_NAME = "deepseek-chat" # 使用对话模型即可，足够聪明且便宜
# 动态获取当前脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 拼接出同级目录下数据文件的绝对路径
INPUT_FILE = os.path.join(SCRIPT_DIR, "house_agent_final_train.jsonl")    
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "house_agent_cleaned.jsonl")
# ==========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def rewrite_agent_response(instruction, old_output):
    """调用大模型，将生硬的指导手册改写为高情商的中介话术"""
    
    system_prompt = """你是一个拥有10年经验的“金牌房产中介”。你的任务是把生硬的【销售培训指导】转化成【自然、高情商、口语化的对客话术】。
严格遵守以下规则：
1. 语气真诚、专业、自然，像真人面对面聊天。
2. 统一使用“姐”、“哥”或“您”来称呼客户，彻底清除原文本中的“李小姐”、“×先生”、“陈先生”等具体称呼。
3. 彻底清除原文本中所有的动作指导或情景分支（例如“走到门口做要开门的样子”、“打假电话”、“如果客户回来该如何说”等内容全部删掉，只保留你说的话）。
4. 你只需要输出改写后的话术纯文本，绝对不要输出任何多余的解释、格式或引言！"""

    user_prompt = f"""
【客户的问题】：{instruction}
【销售指导手册】：{old_output}

请根据手册的意思，直接输出你应该对客户说的话："""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7, # 稍微带点发散性，让话术更自然
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API调用报错: {e}")
        return None

def main():
    print("🚀 开始批量清洗和重写数据集...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        lines = infile.readlines()
        total = len(lines)
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            data = json.loads(line)
            instruction = data.get("instruction", "")
            old_output = data.get("output", "")
            
            print(f"\n正在处理 [{i+1}/{total}]...")
            print(f"原问: {instruction[:30]}...")
            
            # 如果 output 已经是比较正常的对话，可以跳过，但你的数据基本都需要重写
            new_output = rewrite_agent_response(instruction, old_output)
            
            if new_output:
                data["output"] = new_output
                # 写入新文件
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                print("✅ 转换成功")
            else:
                # 如果某一条API报错失败了，保留原数据，避免丢失
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                print("❌ 转换失败，保留原数据")
            
            # 避免触发 API 的并发限制，每条处理完停顿 0.5 秒
            time.sleep(0.5)
            
    print(f"\n🎉 清洗完成！全新且高质量的数据集已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()