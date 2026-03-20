import re
import json
import os

def clean_text_segment(text):
    """深度清理空格、换行和特殊字符"""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[\s　]+', ' ', text)
    return text.strip()

def process_dialogue_data(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 按照数字编号切割段落 (匹配 1、 2、 1． 等)
    segments = re.split(r'\n\s*\d+[．、]', "\n" + content)
    
    dataset = []

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        
        lines = seg.split('\n')
        # 第一行通常是情景描述或客户的第一句话
        scenario = lines[0].strip()
        
        # 2. 提取所有“经纪人：”后的内容（包含话术和动作提示）
        # 使用非贪婪匹配获取经纪人的回复
        body = "\n".join(lines[1:])
        agent_parts = re.findall(r'经纪人：(.*?)(?=客户：|经纪人：|$)', body, re.DOTALL)
        
        # 如果没匹配到（可能是格式不规范），尝试备选匹配
        if not agent_parts:
            agent_parts = re.findall(r'经纪人：(.*)', body)

        # 3. 整合经纪人的所有话术和动作
        full_reply = " ".join([p.strip() for p in agent_parts])
        clean_output = clean_text_segment(full_reply)
        
        if scenario and clean_output:
            dataset.append({
                "instruction": f"作为房产中介经纪人，面对以下实战情景，你该如何处理：{scenario}",
                "input": "",
                "output": clean_output
            })

    # 4. 写入 JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"处理完成！生成了 {len(dataset)} 条实战对白数据。")

if __name__ == "__main__":
    # 请确保你的对白文本保存为 house_dialogue.txt
    process_dialogue_data("house_dialogue.txt", "dialogue_sft.jsonl")