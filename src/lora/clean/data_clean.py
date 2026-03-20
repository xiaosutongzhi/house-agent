import json
import re
import os

def clean_dialogue(item):
    """清理实战情景对话数据"""
    output = item.get("output", "")
    
    # 1. 移除聊天记录的时间戳和人名，例如 "陈琳 16:57:14 、"
    output = re.sub(r'[\u4e00-\u9fa5]{2,4}\s+\d{1,2}:\d{2}:\d{2}\s*[、，。]*\s*', '', output)
    
    # 2. 清理多余的空白字符
    output = re.sub(r'\s+', ' ', output).strip()
    
    item["output"] = output
    return item

def clean_golden_list(item):
    """清理服务与价值展示数据"""
    output = item.get("output", "")
    
    # 1. 移除机械化的前缀 "我会做到：" 或 "我会做到:"
    output = re.sub(r'^我会做到[：:]\s*', '', output)
    
    item["output"] = output
    return item

def clean_house_quotes(item):
    """清理专业知识解答数据"""
    instruction = item.get("instruction", "")
    output = item.get("output", "")
    
    # 1. 移除问题中单一的前缀 "请分析一下："
    instruction = re.sub(r'^请分析一下[：:]\s*', '', instruction)
    
    # 2. 修复问题和答案粘连的错误 (例如: "为何得房率高的房子好？　省钱...")
    # 寻找全角问号后跟着非空字符的情况
    match = re.search(r'(.*?？)\s*(.+)', instruction)
    if match:
        clean_q = match.group(1).strip()
        leaked_a = match.group(2).strip()
        # 将被挤到 instruction 里的答案拼接回 output
        instruction = clean_q
        output = leaked_a + " " + output
        
    item["instruction"] = instruction
    item["output"] = output
    return item

def process_file(input_path, output_path, clean_func):
    if not os.path.exists(input_path):
        print(f"⚠️ 找不到文件: {input_path}")
        return

    cleaned_data = []
    seen_instructions = set() # 用于去重

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                # 兼容处理可能带有的 复制前缀
                clean_line = re.sub(r'^\\s*', '', line.strip())
                item = json.loads(clean_line)
                
                # 执行专属清理逻辑
                item = clean_func(item)
                
                # 根据 instruction 去重
                inst = item["instruction"]
                if inst not in seen_instructions:
                    seen_instructions.add(inst)
                    cleaned_data.append(item)
                    
            except json.JSONDecodeError as e:
                print(f"解析 JSON 失败: {line[:50]}... 错误: {e}")

    # 写入清理后的文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in cleaned_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"✅ 成功处理 {input_path} -> {output_path} (共 {len(cleaned_data)} 条)")

if __name__ == "__main__":
    # 文件路径配置 (请确保原始文件在同一目录下)
    files_to_process = [
        ("dialogue_sft.jsonl", "dialogue_cleaned.jsonl", clean_dialogue),
        ("golden_list_sft.jsonl", "golden_list_cleaned.jsonl", clean_golden_list),
        ("house_quotes_sft.jsonl", "house_quotes_cleaned.jsonl", clean_house_quotes)
    ]
    
    for in_file, out_file, func in files_to_process:
        process_file(in_file, out_file, func)