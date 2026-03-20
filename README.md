# 🏠 智能推荐与预定多智能体系统 (AI Multi-Agent System)

这是一个基于多智能体架构（Multi-Agent）构建的智能交互系统。系统利用大语言模型（LLM）进行意图识别与路由，集成了**房源/产品推荐**、**自动化预定**、**政策 RAG 问答**等多个核心业务子图，并支持长短期记忆管理与端侧状态的实时注入。

## 📸 效果演示

<img width="1805" height="1131" alt="image" src="https://github.com/user-attachments/assets/615ca010-2efe-4d48-8245-252ee1dd2132" />

*图 1：【前端隐式获取用户意向】*


<img width="1178" height="417" alt="image" src="https://github.com/user-attachments/assets/62366799-76ec-48cc-b992-fa66d52a005c" />

*图 2：【通过rag模块进行政策长记忆获取】*


<img width="1023" height="126" alt="image" src="https://github.com/user-attachments/assets/997df109-30d2-4638-af7d-ef621e66051c" />

*图 3：【双轨记忆查询（用户核心画像+情景记忆向量存储）+mcp的全局上下文，阳台房和采光好关联】*




## ✨ 核心特性

- 🧠 **多智能体智能路由**：基于 `emotion_router` 动态识别用户意图与情绪，自动分发至推荐、预定、政策解答或闲聊等特定子图（Sub-graphs）。
- 📂 **长短期记忆管理**：
  - 短期记忆：维护当前会话的上下文（Message History）。
  - 长期记忆：通过 `memory_manager` 节点自动抽取用户偏好并持久化，实现跨会话的个性化服务。
- 🔍 **RAG 增强问答**：集成专用语料库与索引，精准解答特定政策与业务规则。
- 🔌 **端侧状态联动 (MCP)**：支持前端（如 Streamlit）与 Redis 联动，实时捕获用户端侧行为状态（`web_mcp_state`），并作为上下文注入 Agent 辅助决策。

## 🏗️ 架构概览


<img width="1448" height="951" alt="image" src="https://github.com/user-attachments/assets/c5969810-8e76-4325-83ad-399c2a537fb4" />

*图 3：【系统主图流转逻辑， Supervisor 路由中枢与四大核心业务子图的调用。】*

系统主要由以下核心模块流转：
1. **入口与记忆**：解析请求并更新/提取长期偏好。
2. **路由分发**：根据意图路由至 `recommend`（推荐）、`reserve`（预定）、`policy_rag`（政策）等业务处理管线。
3. **话术包装**：经过业务节点处理后，统一由 `empathy_generator` 进行最终的情感化话术生成。

## 🚀 快速开始

### 1. 环境准备

确保已安装所需依赖，并在项目根目录配置好大模型相关的环境变量（推荐使用 `.env` 文件）：

```bash
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=your_api_base_url
OPENAI_MODEL=your_model_name

### 2. 启动服务 (基于 Makefile)

本项目提供了便捷的 `make` 命令来启动不同模式：

**启动 Streamlit Web UI (推荐)**
```bash
# 默认端口 8501
make app_and_client 

# 正常启动 CLI
make cli

