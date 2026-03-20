import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    import importlib

    load_dotenv = importlib.import_module("dotenv").load_dotenv
    load_dotenv()
except Exception:
    pass

if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_API_BASE", "")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.rag.retrieval import (
    BM25Index,
    build_doc_map,
    bm25_search,
    dense_search,
    get_default_index_path,
    load_index,
    rrf_fuse,
)

# ==========================================
# 0. 初始化真实的向量数据库和模型
# ==========================================
INDEX_PATH = get_default_index_path()
KNOWLEDGE_BASE = load_index(INDEX_PATH)
DOC_MAP = build_doc_map(KNOWLEDGE_BASE)
BM25 = BM25Index(KNOWLEDGE_BASE)

# 初始化 Embedding 模型 (需与建库时一致)
try:
    import importlib

    OpenAIEmbeddings = importlib.import_module("langchain_openai").OpenAIEmbeddings
except Exception:
    raise RuntimeError("langchain_openai 未安装，无法进行向量检索")

embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
embeddings = OpenAIEmbeddings(
    model=embedding_model,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# ==========================================
# 1. 真实的检索逻辑
# ==========================================
def _window_context(item_idx: int, window: int = 2) -> str:
    item = KNOWLEDGE_BASE[item_idx]
    doc_items = DOC_MAP[item.doc_id]
    sent_ids = [x.sent_id for x in doc_items]
    pos = sent_ids.index(item.sent_id)
    start = max(pos - window, 0)
    end = min(pos + window + 1, len(doc_items))
    return "".join([x.text for x in doc_items[start:end]])


def retrieve_documents(question: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """双路检索 + RRF 融合 + 句子窗口上下文。"""
    if not KNOWLEDGE_BASE:
        raise ValueError("知识库为空，请检查 policy_index.jsonl 路径！")

    bm25_hits = bm25_search(KNOWLEDGE_BASE, BM25, question, filters=None, top_k=top_k)
    bm25_rank = [idx for idx, _ in bm25_hits]

    dense_rank: List[int] = []
    if KNOWLEDGE_BASE and KNOWLEDGE_BASE[0].embedding is not None:
        q_vec = embeddings.embed_query(question)
        dense_hits = dense_search(KNOWLEDGE_BASE, q_vec, filters=None, top_k=top_k)
        dense_rank = [idx for idx, _ in dense_hits]

    fused = rrf_fuse([bm25_rank, dense_rank] if dense_rank else [bm25_rank])
    top_indices = fused[:top_k]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        item = KNOWLEDGE_BASE[idx]
        results.append({
            "text": item.text,
            "sent_id": item.sent_id,
            "score": 1.0,
            "context": _window_context(idx, window=2),
        })
    return results

def llm_judge(question: str, context: str) -> int:
    """调用大模型作为裁判，返回 1(相关) / 0(不相关)。"""
    if not context.strip():
        return 0
    try:
        import importlib

        OpenAI = importlib.import_module("openai").OpenAI
    except Exception:
        return 0

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip().strip('"').strip("'")
    if not api_key:
        return 0

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    prompt = (
        "你是检索评估裁判。判断给定检索上下文是否足以回答问题。"
        "只输出 1 或 0。"
    )
    user = f"问题：{question}\n\n检索上下文：{context}\n\n是否相关："
    try:
        models = [
            os.getenv("RERANK_MODEL", "deepseek-r1"),
            os.getenv("RERANK_FALLBACK_MODEL", "deepseek-ai/DeepSeek-R1"),
            "deepseek-chat",
        ]
        for model in models:
            if not model:
                continue
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=4,
                )
                text = (resp.choices[0].message.content or "").strip()
                return 1 if text.startswith("1") else 0
            except Exception as e:
                last_error = e
        print(f"⚠️ LLM 裁判调用失败: {last_error}")
        return 0
    except Exception as e:
        print(f"⚠️ LLM 裁判初始化失败: {e}")
        return 0

# ==========================================
# 2. 评估指标计算逻辑 (已优化：支持文本片段匹配)
# ==========================================

def evaluate_retrieval(eval_data: List[Dict]):
    total_questions = len(eval_data)
    hits = 0
    mrr_sum = 0.0
    relevance_scores = []
    precision_sum = 0.0
    recall_sum = 0.0

    print(f"🚀 开始评估，共 {total_questions} 个测试用例...\n")
    print("-" * 60)

    for idx, item in enumerate(eval_data, 1):
        question = item["question"]
        expected_snippet = item["expected_text_snippet"]
        
        # 1. 执行检索
        retrieved_docs = retrieve_documents(question, top_k=8)
        
        # 2. 计算 Hit Rate & MRR (通过字符串包含来判断是否命中)
        hit_found = False
        relevant_in_topk = 0
        for rank, doc in enumerate(retrieved_docs, start=1):
            # 改进：只要预期的核心片段中有 70% 的字符出现在检索结果中，就认为命中
            # 或者使用更简单的：判断预期的关键数字/名词是否都在结果里
            
            # 这里先用一个简单的“关键词包含”逻辑作为演示：
            # 提取 expected_snippet 中的数字或长词，看是否在 doc["text"] 里
            import re
            numbers = re.findall(r'\d+\.?\d*%?', expected_snippet) # 提取数字和百分号
            
            # 如果预期的片段本身就在结果里，或者关键数字对上了，就算命中
            text_match = expected_snippet in doc["text"] or doc["text"] in expected_snippet
            num_match = all(n in doc["text"] for n in numbers) if numbers else False

            if text_match or num_match:
                relevant_in_topk += 1
                hits += 1
                mrr_sum += 1.0 / rank 
                hit_found = True
                print(f"[{idx}/{total_questions}] ✅ 命中! 排第 {rank} 名 | 问题: {question}")
                break
        
        if not hit_found:
            print(f"[{idx}/{total_questions}] ❌ 未命中 | 问题: {question}")

        # 3. 计算 Precision/Recall (单条相关标准)
        precision = relevant_in_topk / max(len(retrieved_docs), 1)
        recall = 1.0 if hit_found else 0.0
        precision_sum += precision
        recall_sum += recall

        # 4. 计算 Context Relevance
        combined_context = "\n".join([doc["context"] for doc in retrieved_docs])
        rel_score = llm_judge(question, combined_context)
        relevance_scores.append(rel_score)

    print("-" * 60)
    
    # ==========================================
    # 3. 打印最终成绩单
    # ==========================================
    hit_rate = hits / total_questions
    mrr = mrr_sum / total_questions
    avg_relevance = sum(relevance_scores) / total_questions

    avg_precision = precision_sum / total_questions
    avg_recall = recall_sum / total_questions

    print("📊 检索层评估报告 (Retrieval Evaluation Report):")
    print(f"➤ Hit Rate (命中率): {hit_rate * 100:.2f}%  (是否找到了包含关键信息的段落)")
    print(f"➤ MRR (平均倒数排名): {mrr:.4f}  (答案排得有多靠前，满分1.0)")
    print(f"➤ Precision@K: {avg_precision * 100:.2f}%  (Top-K 中相关项占比)")
    print(f"➤ Recall@K: {avg_recall * 100:.2f}%  (是否命中相关项)")
    print(f"➤ Context Relevance: {avg_relevance * 100:.2f}%  (大模型裁判认可率)")


if __name__ == "__main__":
    # 根据 cleaned_policy.md 自动生成的 10 条测试数据集
    real_eval_dataset = [
        {
            "question": "2026年全国首套房的房贷利率大概降到了多少？", 
            "expected_text_snippet": "全国首套房房贷利率普遍降至3.0% 以内"
        },
        {
            "question": "多子女家庭在使用公积金贷款时，额度有什么优惠？", 
            "expected_text_snippet": "多子女家庭最高可上浮20%"
        },
        {
            "question": "在一线城市，个人出售持有满两年的住房，增值税怎么收？", 
            "expected_text_snippet": "对个人销售已购买2年以上（含2年）住房一律免征增值税"
        },
        {
            "question": "面积在140平方米以下的家庭唯一住房，契税税率是多少？", 
            "expected_text_snippet": "面积为140平方米及以下的，减按1%的税率征收契税"
        },
        {
            "question": "购买150平方米的家庭第二套住房，契税按什么标准交？", 
            "expected_text_snippet": "面积为140平方米以上的，减按2%的税率征收契税"
        },
        {
            "question": "三四线城市鼓励农民进城购房，有什么具体补贴？", 
            "expected_text_snippet": "给予每平方米 200-500 元补贴"
        },
        {
            "question": "新市民和青年人申请购房贴息，需要满足什么社保条件？", 
            "expected_text_snippet": "连续缴纳社保 1 年以上即可申请"
        },
        {
            "question": "最新的《住宅项目规范》对新建楼房的层高有什么硬性要求？", 
            "expected_text_snippet": "楼的层高已经从原来的2.8米提高到不低于3米"
        },
        {
            "question": "几层以上的楼房必须加装电梯？", 
            "expected_text_snippet": "4层以上的楼都要加装电梯"
        },
        {
            "question": "针对绿色建筑和适老化社区，财政贴息政策有什么额外优惠？", 
            "expected_text_snippet": "额外享受30 个基点贴息 + 契税减免"
        },
        {
            "question": "国家下调存量房贷利率后，预计平均每年能帮每个家庭省下多少钱？", 
            "expected_text_snippet": "预计全国超5000 万存量房贷家庭年均减负超8000 元"
        },
        {
            "question": "现在购买二套房，最低的首付比例降到了多少？", 
            "expected_text_snippet": "二套房利率同步下调，首付比例最低降至20%"
        },
        {
            "question": "如果我把现在的房子卖了，一年内又买了一套新房，之前交的个人所得税能退吗？", 
            "expected_text_snippet": "出售自有住房并 1 年内重新购房的纳税人，全额退还已缴个税"
        },
        {
            "question": "房贷利息在抵扣个人所得税时，现在每个月的扣除额度是多少？", 
            "expected_text_snippet": "房贷利息个税专项附加扣除额度提高至每月 3000 元"
        },
        {
            "question": "外地人在一线城市（比如北上广深）买房，社保需要连续交满几年？", 
            "expected_text_snippet": "非户籍社保年限缩短至1 年"
        },
        {
            "question": "政府为了去库存，今年计划收购多少平米的存量房转作保障房？", 
            "expected_text_snippet": "计划全年收储超1 亿平方米存量房"
        },
        {
            "question": "对于房地产开发企业，土地增值税的预征率下限统一降低了多少？", 
            "expected_text_snippet": "将各地区土地增值税预征率下限统一降低0.5个百分点"
        },
        {
            "question": "国家针对房地产项目的融资“白名单”，目前审批通过的贷款金额大约有多少？", 
            "expected_text_snippet": "全国“白名单”项目贷款审批金额已经超过7万亿元"
        },
        {
            "question": "住建部提到的“保交房”工作，目前全国已经交付了多少套原本难交付的房子？", 
            "expected_text_snippet": "全国750多万套已售难交付的住房实现交付"
        },
        {
            "question": "买房时地方政府发放的购房消费券，除了买房还能用来买什么东西？", 
            "expected_text_snippet": "可用于家电、家居、汽车消费"
        },
        {
            "question": "个人把房子卖掉，增值税免征的持有年限从几年缩短到了几年？", 
            "expected_text_snippet": "免征年限从5 年缩短至2 年"
        },
        {
            "question": "如果房产证没满2年就卖掉，现在的增值税税率是多少？", 
            "expected_text_snippet": "不满 2 年按 3%简易征收，较此前5% 税率大幅下调"
        },
        {
            "question": "三孩家庭在享受首套房贴息的基础上，还能额外享受多少贴息？", 
            "expected_text_snippet": "在首套贴息基础上，额外叠加20-25 个基点，三孩家庭房贷利率最低可至2.2%"
        },
        {
            "question": "对于老旧小区改造，新政策在隔音方面有什么具体的分贝要求？", 
            "expected_text_snippet": "对楼板的隔音要求，要降低10个分贝"
        },
        {
            "question": "现在的政策规定，如果一个城市的住房去化周期超过多久，就必须停止供应新的宅地？", 
            "expected_text_snippet": "去化超 24 个月城市暂停新增宅地"
        },
        {
            "question": "在一线城市的核心城区，现在的限购政策是怎么规定的？", 
            "expected_text_snippet": "核心城区保留适度限购，外围区域全面放开"
        },
        {
            "question": "针对“卖旧买新”的退税支持政策，目前明确会延续到什么时候？", 
            "expected_text_snippet": "全额退还已缴个税，政策延续至 2027 年底"
        },
        {
            "question": "灵活就业人员现在可以交公积金吗？手续麻烦吗？", 
            "expected_text_snippet": "灵活就业人员全面纳入公积金缴存体系，租房提取、购房提取流程简化，实现 “秒批秒办”"
        },
        {
            "question": "二手房的“带押过户”普及后，有什么好处？", 
            "expected_text_snippet": "无需提前还贷即可完成过户，单套房屋交易成本平均降低3 万 - 8 万元"
        },
        {
            "question": "2026年两会提出的“好房子”建设，在层高上相比以前提高了多少？", 
            "expected_text_snippet": "层高已经从原来的2.8米提高到不低于3米"
        }
    ]
    
    evaluate_retrieval(real_eval_dataset)