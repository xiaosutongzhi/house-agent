import re
import json
import os

def process_golden_list(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    dataset = []
    
    # 匹配“数字+顿号或点”开头的行
    pattern = r'^\d+[、\.．]\s*(.*)'

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line)
        if match:
            content = match.group(1).strip()
            
            # 策略：根据内容特征，自动生成更有针对性的 Instruction
            if "话术" in content or "说" in content or "： " in content:
                # 针对 101-110 条这种带具体对白的
                parts = content.split('：')
                if len(parts) > 1:
                    scenario = parts[0]
                    speech = parts[1]
                    instruction = f"在房产交易的“{scenario}”环节，作为专业经纪人，你会如何跟客户沟通？"
                    output = speech
                else:
                    instruction = "请提供一段房产经纪人在签约或谈判时的经典沟通话术。"
                    output = content
            else:
                # 针对 1-100 条这种服务标准和承诺
                instruction = f"作为一名专业的房产经纪人，关于“{content[:10]}...”这一项，你如何向客户展示你的服务价值？"
                output = f"我会做到：{content}"

            dataset.append({
                "instruction": instruction,
                "input": "",
                "output": output
            })

    # 写入 JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"处理完成！从列表提取了 {len(dataset)} 条金句数据，已存至 {output_file}")

if __name__ == "__main__":
    process_golden_list("golden_sentences.txt", "golden_list_sft.jsonl")