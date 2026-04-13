# house-agent 训练/运行记录（SFT → RM → GRPO）

最后更新：2026-04-11

这份文件用于“回放我在仓库里做过什么 + 怎么复现运行”。只写**必要步骤**、**关键产物路径**和**可直接跑的命令**。

---

## 1) 目标与统一输出协议（JSON Schema）

- 目标：让策略模型稳定输出严格 JSON，便于线上解析与打分（SFT/RM/GRPO 全链路同一协议）。
- Schema 文档：`src/lora/data_tools/SCHEMA.md`
- 字段固定为 4 个：`greeting` / `answer` / `evidence` / `next_question`

---

## 2) SFT 数据生成（Schema SFT）

**输入**
- 清洗后的原始对话：`src/lora/house_agent_cleaned.jsonl`

**脚本**
- `src/lora/data_tools/make_house_agent_schema_sft.py`

**输出**
- `src/lora/LlamaFactory-main/data/house_agent_schema_sft.jsonl`

**运行命令**
```bash
# 在仓库根目录执行
source /home/szm/workplace/house_agent/house-agent/.venv-rl/bin/activate
python src/lora/data_tools/make_house_agent_schema_sft.py
```

---

## 3) SFT 训练（LlamaFactory / QLoRA 4bit）

**配置文件**
- `src/lora/LlamaFactory-main/examples/house_agent/qwen25_05b_qlora_sft_house_agent_schema.yaml`

**训练命令**
```bash
source /home/szm/workplace/house_agent/house-agent/.venv-rl/bin/activate
cd src/lora/LlamaFactory-main
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  llamafactory-cli train examples/house_agent/qwen25_05b_qlora_sft_house_agent_schema.yaml \
  | tee train_logs/sft_qwen25_05b_schema.log
```

**产物**
- LoRA SFT 输出目录：`src/lora/LlamaFactory-main/saves/qwen25-0_5b/qlora/sft_house_agent_schema`
- 日志：`src/lora/LlamaFactory-main/train_logs/sft_qwen25_05b_schema.log`

---

## 4) 导出 SFT 合并模型（用于后续 RM 初始化 / RL）

**配置文件**
- `src/lora/LlamaFactory-main/examples/house_agent/qwen25_05b_export_sft_merged.yaml`

**导出命令**
```bash
source /home/szm/workplace/house_agent/house-agent/.venv-rl/bin/activate
cd src/lora/LlamaFactory-main
llamafactory-cli export examples/house_agent/qwen25_05b_export_sft_merged.yaml
```

**产物**
- 合并后的 HF 模型：`src/lora/LlamaFactory-main/saves/qwen25-0_5b/sft_merged_hf`

---

## 5) 偏好标注维度表（Rubric）

- 文档：`src/lora/data_tools/PREFERENCE_RUBRIC.md`
- 设计取向：**格式优先（JSON 可解析 + 字段齐全 + evidence 纪律 + next_question）**。

这个 rubric 支持两种用途：
1) 人工标注/复核 chosen vs rejected
2) LLM-judge 判分/排序

---

## 6) RM 偏好数据生成（chosen/rejected）

**脚本**
- `src/lora/data_tools/make_house_agent_preference.py`

**特点（已实现）**
- 支持 `--deterministic_rejected`：生成多个坏样本候选，然后按启发式 rubric 打分选最差 rejected，减少随机性。

**输出**
- `src/lora/LlamaFactory-main/data/house_agent_pref.json`（sharegpt ranking 格式，LlamaFactory RM 可直接读）

**运行命令（推荐 deterministic）**
```bash
source /home/szm/workplace/house_agent/house-agent/.venv-rl/bin/activate
python src/lora/data_tools/make_house_agent_preference.py \
  --max_samples 5000 \
  --deterministic_rejected
```

---

## 7) RM 训练（版本 A：从 base 初始化）

**配置文件**
- `src/lora/LlamaFactory-main/examples/house_agent/qwen25_05b_qlora_rm_house_agent_pref.yaml`

**训练命令**
```bash
source /home/szm/workplace/house_agent/house-agent/.venv-rl/bin/activate
cd src/lora/LlamaFactory-main
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  llamafactory-cli train examples/house_agent/qwen25_05b_qlora_rm_house_agent_pref.yaml \
  | tee train_logs/rm_qwen25_05b_pref.log
```

**产物**
- RM 输出目录：`src/lora/LlamaFactory-main/saves/qwen25-0_5b/qlora/rm_house_agent_pref`
- 日志：`src/lora/LlamaFactory-main/train_logs/rm_qwen25_05b_pref.log`

---

## 8) RM 训练（版本 B：先 SFT 再 RM，从 SFT merged 初始化）

这是按“先 SFT 冷启动再训练 RM”的路线：RM 直接继承 SFT 的输出风格（尤其是严格 JSON）。

**配置文件（已新增）**
- `src/lora/LlamaFactory-main/examples/house_agent/qwen25_05b_qlora_rm_house_agent_pref_from_sft_merged.yaml`
  - `model_name_or_path: saves/qwen25-0_5b/sft_merged_hf`
  - 为降低显存峰值，已将：`per_device_train_batch_size=4`、`gradient_accumulation_steps=16`

**训练命令（建议单卡空闲 GPU）**
```bash
source /home/szm/workplace/house_agent/house-agent/.venv-rl/bin/activate
cd src/lora/LlamaFactory-main
CUDA_VISIBLE_DEVICES=3 PYTORCH_ALLOC_CONF=expandable_segments:True \
  llamafactory-cli train examples/house_agent/qwen25_05b_qlora_rm_house_agent_pref_from_sft_merged.yaml \
  | tee train_logs/rm_qwen25_05b_pref_from_sft_merged.log
```

**产物**
- RM-from-SFT 输出目录：`src/lora/LlamaFactory-main/saves/qwen25-0_5b/qlora/rm_house_agent_pref_from_sft_merged`
  - 其中包含 `value_head.safetensors`，用于推理打分。
- 日志：`src/lora/LlamaFactory-main/train_logs/rm_qwen25_05b_pref_from_sft_merged.log`

**显存备注（真实踩坑）**
- RM 前向会产生较大的张量（尤其是长序列 + 大词表），多卡自动启动/其他进程占用时容易 OOM。
- 经验优先级：优先用空闲单卡（例如 GPU3）→ 降 `per_device_train_batch_size` → 提高 `gradient_accumulation_steps`。

---

## 9) GRPO 数据（EasyR1 prompt-only）

**脚本**
- `src/lora/data_tools/make_grpo_dataset_easyr1.py`

**输出**
- `src/lora/easyr1/house_agent_grpo.jsonl`

**生成命令**
```bash
source /home/szm/workplace/house_agent/house-agent/.venv-rl/bin/activate
python src/lora/data_tools/make_grpo_dataset_easyr1.py
```

---

## 10) EasyR1：format prompt + reward function（规则 reward）

EasyR1 侧需要两个文件：
- format prompt：`src/lora/easyr1/house_agent_format_prompt.jinja`
- reward function：`src/lora/easyr1/house_agent_reward_function.py`

它们是“可拷贝模板”：把文件拷贝进你的 EasyR1 仓库（或容器）后，在 EasyR1 的 config 里指向对应路径。

**规则 reward 做了什么**
- JSON 可解析
- 必须字段齐全
- `next_question` 非空
- evidence 与 `CONTEXT:` 的一致性（简单 substring 约束；无 context 时倾向 evidence 为空）

---

## 11) EasyR1：与 RM reward 混合（规则 + RM）

我把“RM 混合”做成了**可选开关**：你不设置环境变量，就还是规则 reward；你设置了 RM 路径和权重，就会自动加载 RM 并加到总 reward 里。

### 11.1 你需要的 RM 文件
以 RM-from-SFT 版本为例（推荐先用这个）：
- base（策略风格基座）：`src/lora/LlamaFactory-main/saves/qwen25-0_5b/sft_merged_hf`
- RM adapter：`src/lora/LlamaFactory-main/saves/qwen25-0_5b/qlora/rm_house_agent_pref_from_sft_merged`
- value head：同目录下 `value_head.safetensors`

### 11.2 在 EasyR1 reward worker 里设置环境变量
（示例路径按“在 LlamaFactory-main 目录下运行”写；你如果把模型拷贝到了别处，就改成你的实际路径）

```bash
export HOUSE_AGENT_RM_BASE_MODEL=saves/qwen25-0_5b/sft_merged_hf
export HOUSE_AGENT_RM_ADAPTER_DIR=saves/qwen25-0_5b/qlora/rm_house_agent_pref_from_sft_merged
export HOUSE_AGENT_RULE_WEIGHT=1.0
export HOUSE_AGENT_RM_WEIGHT=0.3
export HOUSE_AGENT_RM_CLIP=5.0
```

### 11.3 混合公式
- 总奖励：`reward = rule_w * rule_reward + rm_w * clip(rm_reward, [-rm_clip, rm_clip])`
- 其中 `rm_reward` 来自 value head 对“(prompt, response)”的打分。

实现位置：`src/lora/easyr1/house_agent_reward_function.py`（已更新为可选加载 RM）。

---

## 12) 关键文件清单（便于以后快速找）

- Schema：`src/lora/data_tools/SCHEMA.md`
- Rubric：`src/lora/data_tools/PREFERENCE_RUBRIC.md`
- SFT 数据脚本：`src/lora/data_tools/make_house_agent_schema_sft.py`
- 偏好数据脚本：`src/lora/data_tools/make_house_agent_preference.py`
- GRPO 数据脚本：`src/lora/data_tools/make_grpo_dataset_easyr1.py`
- EasyR1 prompt 模板：`src/lora/easyr1/house_agent_format_prompt.jinja`
- EasyR1 reward（规则 + 可选 RM 混合）：`src/lora/easyr1/house_agent_reward_function.py`
- LlamaFactory 配置：
  - SFT：`src/lora/LlamaFactory-main/examples/house_agent/qwen25_05b_qlora_sft_house_agent_schema.yaml`
  - RM(base)：`src/lora/LlamaFactory-main/examples/house_agent/qwen25_05b_qlora_rm_house_agent_pref.yaml`
  - RM(from SFT merged)：`src/lora/LlamaFactory-main/examples/house_agent/qwen25_05b_qlora_rm_house_agent_pref_from_sft_merged.yaml`
  - Export merged：`src/lora/LlamaFactory-main/examples/house_agent/qwen25_05b_export_sft_merged.yaml`
- 训练日志：`src/lora/LlamaFactory-main/train_logs/*.log`
