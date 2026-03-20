import json
import random
import os

def merge_and_shuffle(input_files, output_file):
    combined_data = []
    
    # 1. 逐个读取并验证数据
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 警告: 找不到文件 {file_path}，已跳过。")
            continue
            
        print(f"正在加载: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 确保包含我们需要的三个核心字段
                    if "instruction" in data and "output" in data:
                        # 统一格式规范，如果 input 不存在则补全为空字符串
                        clean_data = {
                            "instruction": data["instruction"],
                            "input": data.get("input", ""),
                            "output": data["output"]
                        }
                        combined_data.append(clean_data)
                except json.JSONDecodeError:
                    print(f"⚠️ 解析错误: {file_path} 第 {line_num} 行格式不对，已跳过。")

    if not combined_data:
        print("❌ 错误: 没有提取到任何有效数据！")
        return

    # 2. 核心步骤：打乱数据顺序 (Shuffle)
    # 这能防止模型在训练时产生局部过拟合（比如连续学了100条金句导致说话风格突变）
    print("正在打乱数据顺序 (Shuffling)...")
    random.seed(42) # 固定随机种子，保证每次打乱的结果一致，方便复现
    random.shuffle(combined_data)

    # 3. 写入最终的训练文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in combined_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("=====================================")
    print(f"✅ 合并成功！总计有效数据: {len(combined_data)} 条")
    print(f"🎉 最终训练集已保存至: {output_file}")
    print("=====================================")

if __name__ == "__main__":
    # 把你前面生成的三个增强版数据集填在这里
    # 如果你的文件名不同，请手动修改一下
    source_files = [
        "house_quotes_augmented.jsonl", 
        "dialogue_augmented.jsonl", 
        "golden_list_augmented.jsonl"
    ]
    
    # 最终喂给 LLaMA-Factory 的文件
    final_output = "house_agent_final_train.jsonl"
    
    merge_and_shuffle(source_files, final_output)