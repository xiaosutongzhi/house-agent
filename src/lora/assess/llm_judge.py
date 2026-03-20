from openai import OpenAI

# 复用你之前清洗数据的 API 设定
API_KEY = "sk-a61c012a994c4815ad86990665eeab4e"
BASE_URL = "https://api.deepseek.com/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def evaluate_models(instruction, ans_a, ans_b):
    judge_prompt = f"""你是一个资深的房产销售总监，现在在考核两个 AI 房产中介。
请根据客户的提问，评估 Model A 和 Model B 的回答。

【客户提问】：{instruction}
【Model A (原始模型)】：{ans_a}
【Model B (微调模型)】：{ans_b}

请从以下三个维度打分（满分 10 分），并给出你的评价理由：
1. 专业度：是否能解决客户关于房产/流程的疑虑？
2. 情绪价值（高情商）：是否安抚了客户的情绪，态度是否像真人中介一样有温度且不生硬？
3. 销售技巧：是否成功引导了下一步动作（如看房、对比数据等）？

输出格式要求：
Model A 总分：X/10
Model B 总分：Y/10
胜出者：[A/B]
详细评价：...
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.3
    )
    print(response.choices[0].message.content)

# 填入你在上一个脚本里得到的真实回答
evaluate_models(
    instruction="你们中介费收这么高，我看网上说这小区物业特别差，你是不是想坑我赶紧接盘？",
    ans_a="把这里替换成原始模型的回答文本", 
    ans_b="把这里替换成你微调后模型的回答文本"
)