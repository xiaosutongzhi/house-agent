import argparse
import os
import re
import time
from datetime import datetime
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import OpenAI as OpenAIType
else:
    OpenAIType = Any

# ==========================================
# 1. 配置说明 (运行时传参或环境变量)
# ==========================================
# 环境变量：DEEPSEEK_API_KEY / OPENAI_API_KEY, BASE_URL
# 建议通过参数传入输入/输出路径，避免与 RAG 主流程耦合

# ==========================================
# 2. 文本清洗与分块模块
# ==========================================
def basic_clean(text: str) -> str:
    """基础正则清洗，去除无用标记和噪音数据"""
    print("🧹 正在进行基础文本清洗...")
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 移除可能存在的 [Image of...] 占位符
    text = re.sub(r"\[Image of.*?\]\s*", "", text)
    # 移除全角空格等多余空白符
    text = re.sub(r"\u3000", " ", text)
    # 移除网页碎片或异常链接行
    text = re.sub(r"^https?://\S+\s*$", "", text, flags=re.MULTILINE)
    # 将多个连续的换行符替换为两个换行符，保持段落清晰
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def remove_commentary_paragraphs(text: str) -> str:
    """规则化去评论：剔除主观解读、媒体口吻、评论性段落"""
    print("🧹 正在进行规则化去评论...")
    # 常见评论/解读/媒体口吻关键词
    commentary_patterns = [
        r"业内指出",
        r"专家(认为|表示)",
        r"研究(院|机构)",
        r"分析师",
        r"评论|解读|观点|认为",
        r"利好|提振|信心|预期|暖冬|回稳|回暖",
        r"据.*?消息",
        r"记者|采访|答记者问",
        r"市场(影响|表现|预期)",
        r"数据显示",
        r"预计|有望",
    ]
    commentary_re = re.compile("|".join(commentary_patterns))

    # 可能包含“硬性政策条款”的信号，出现则尽量保留
    policy_signals = re.compile(
        r"(契税|增值税|个税|土地增值税|税率|免征|减按|征收|预征率|百分点|满\d+年|\d+年|\d+%|\d+\.\d+%|平方米|套住房|首套|二套|家庭唯一)")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    kept: List[str] = []
    removed = 0

    for p in paragraphs:
        if commentary_re.search(p) and not policy_signals.search(p):
            removed += 1
            continue
        kept.append(p)

    print(f"🧽 去评论完成：删除 {removed} 段，保留 {len(kept)} 段")
    return "\n\n".join(kept)

def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    长文本滑动窗口分块。
    为了防止长文档超出 LLM 处理上限或导致注意力丢失，按字符数进行切分，并保留重叠部分以防上下文断裂。
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + max_chars
        # 如果不是最后一块，尽量在换行符处切断，保证句子完整性
        if end < text_length:
            # 往回找最近的换行符
            last_newline = text.rfind('\n', start, end)
            if last_newline != -1 and last_newline > start + (max_chars // 2):
                end = last_newline + 1
                
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        # 步进，减去 overlap 形成滑动窗口
        start = end - overlap
        
    print(f"✂️ 文档已切分为 {len(chunks)} 个数据块 (最大字符数: {max_chars}, 重叠: {overlap})")
    return chunks


def split_by_headings(text: str) -> List[str]:
    """按常见中文标题分段，保留结构，便于后续 RAG 切分"""
    headings = re.compile(r"^(一|二|三|四|五|六|七|八|九|十)[、\.\s]|^\d+[、\.\s]|^#+\s+", re.MULTILINE)
    segments: List[str] = []
    last = 0
    for match in headings.finditer(text):
        start = match.start()
        if start > last:
            segment = text[last:start].strip()
            if segment:
                segments.append(segment)
        last = start
    tail = text[last:].strip()
    if tail:
        segments.append(tail)
    return segments

# ==========================================
# 3. 大模型提纯模块
# ==========================================
def distill_chunk_with_llm(
    client: "OpenAIType",
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    model_name: str,
) -> str:
    """调用 LLM 将单个文本块提纯为结构化 Markdown"""
    
    system_prompt = """你是一个专业、严谨的房地产政策数据处理Agent。
你的任务是从冗长、夹杂新闻套话的政策文本中，提取出纯粹的硬性政策条款，并严格按照Markdown格式输出。
绝对不要编造数据，不要添加任何多余的开头、结尾或问候语。"""
    
    user_prompt = f"""
请阅读以下房地产政策文本片段，提取关键信息并重写为结构化的Markdown格式，用于构建RAG向量知识库。

**核心提取规则：**
1. **必须保留**：具体的税种（契税、增值税、个税等）、税率百分比、面积界定条件（如140平方米及以下）、时间节点（如满2年）、地域差异（如北上广深）。
2. **必须剔除**：新闻修饰语、记者提问、专家主观解读（如“业内指出”、“利好房企”等）、政策背景寒暄。
3. **格式要求**：
   - 使用 `##` 标识大类（例如：## 契税政策）
   - 使用 `###` 标识细分条件（例如：### 首套房）
   - 使用无序列表 `-` 罗列具体的税率和条件，确保句子精炼，方便后续做“句子窗口检索(Sentence-Window)”。

**当前处理的政策文本片段：**
{chunk_text}
    """

    print(f"🚀 正在调用大模型处理第 {chunk_index}/{total_chunks} 块数据...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0, # 温度设为0，确保信息提取的严谨性和稳定性
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ 第 {attempt + 1} 次请求失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(2) # 失败重试前等待
            else:
                print("❌ 达到最大重试次数，跳过此块。")
                return f""

# ==========================================
# 4. 主控管道
# ==========================================
def build_metadata_header(
    source_path: str,
    max_chars: int,
    overlap: int,
    model_name: Optional[str],
    use_llm: bool,
) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    model_value = model_name if use_llm else "none"
    header = (
        "---\n"
        f"source: {os.path.basename(source_path)}\n"
        f"created_utc: {now}\n"
        f"chunk_max_chars: {max_chars}\n"
        f"chunk_overlap: {overlap}\n"
        f"llm_model: {model_value}\n"
        f"llm_enabled: {str(use_llm).lower()}\n"
        "---\n\n"
    )
    return header


def write_output(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ 已写入: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="房地产政策文本清洗与可选 LLM 提纯")
    parser.add_argument("--input", required=True, help="输入原始 txt 路径")
    parser.add_argument("--output", required=True, help="输出 Markdown 路径")
    parser.add_argument("--use-llm", action="store_true", help="启用 LLM 提纯")
    parser.add_argument("--model", default="deepseek-chat", help="LLM 模型名")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "https://api.deepseek.com"), help="LLM Base URL")
    parser.add_argument("--max-chars", type=int, default=2000, help="每块最大字符数")
    parser.add_argument("--overlap", type=int, default=200, help="块之间重叠字符数")
    parser.add_argument("--split-by-heading", action="store_true", help="先按标题分段再进行切块")
    parser.add_argument("--remove-commentary", action="store_true", help="启用规则化去评论")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. 读取原始文件
    if not os.path.exists(args.input):
        print(f"❌ 找不到文件 {args.input}，请确保存放在正确路径。")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 2. 基础清洗
    cleaned_text = basic_clean(raw_text)

    # 2.1 规则化去评论（可选）
    if args.remove_commentary:
        cleaned_text = remove_commentary_paragraphs(cleaned_text)

    # 3. 结构保留（可选）
    if args.split_by_heading:
        segments = split_by_headings(cleaned_text)
        cleaned_text = "\n\n".join(segments)

    # 4. 文本分块
    chunks = chunk_text(cleaned_text, max_chars=args.max_chars, overlap=args.overlap)

    # 5. 输出内容构建
    output_parts: List[str] = [
        build_metadata_header(
            source_path=args.input,
            max_chars=args.max_chars,
            overlap=args.overlap,
            model_name=args.model,
            use_llm=args.use_llm,
        )
    ]

    if args.use_llm:
        try:
            import importlib
            OpenAI = importlib.import_module("openai").OpenAI
        except Exception:
            print("❌ 未安装 openai 包，请先安装后再启用 --use-llm。")
            return

        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ 未设置 API Key，请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。")
            return

        client = OpenAI(api_key=api_key, base_url=args.base_url)
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks, start=1):
            distilled = distill_chunk_with_llm(
                client=client,
                chunk_text=chunk,
                chunk_index=i,
                total_chunks=total_chunks,
                model_name=args.model,
            )
            if distilled:
                output_parts.append(distilled)
                output_parts.append("\n")
    else:
        # 纯清洗模式：直接输出清洗后的文本，便于后续 RAG 切分
        output_parts.append(cleaned_text)

    write_output(args.output, "\n".join(output_parts))


if __name__ == "__main__":
    main()