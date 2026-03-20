import re
import json
import os

def clean_text(text):
    """深度清理文本中的杂质"""
    # 1. 删除反斜杠（修复 SyntaxError）
    text = re.sub(r'\\', '', text)
    
    # 2. 删除 (2008年8月) 等过期时间标注 
    # 注意：这里改成了通用匹配，兼容全角和半角括号
    text = re.sub(r'[（\(]\d{4}年\d{1,2}月[）\)]', '', text)
    
    # 3. 将所有空白字符（含换行、全角空格）替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_quotes(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"找不到文件: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正则逻辑：匹配中文数字编号（如：一、二、三十一、）开头的问题 
    # 后面跟着“为何”或“为什么”，直到遇到“答：”
    pattern = r'([一二三四五六七八九十百]+、)(.*?)\n?\s*答[：:](.*?)(?=\n[一二三四五六七八九十百]+、|$)'
    
    matches = re.findall(pattern, content, re.DOTALL)
    dataset = []

    for _, question, answer in matches:
        # 清理问题：去除多余空格 
        clean_q = question.strip().replace('\n', '')
        
        # 清理回答：
        # a. 去掉 A：B：C：等选项前缀 
        clean_a = re.sub(r'[A-Z][：:]', '', answer)
        # b. 深度清理，合并为一段流畅的话
        clean_a = clean_text(clean_a)
        
        if clean_q and clean_a:
            dataset.append({
                "instruction": f"请分析一下：{clean_q}",
                "input": "",
                "output": clean_a
            })

    # 写入 JSONL 格式，方便 SFTTrainer 读取
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"处理完成！")
    print(f"从 {input_file} 中提取了 {len(dataset)} 条专业语录数据。")

if __name__ == "__main__":
    # 请确保你的文件名是 house_quotes.txt
    process_quotes("house_quotes.txt", "house_quotes_sft.jsonl")