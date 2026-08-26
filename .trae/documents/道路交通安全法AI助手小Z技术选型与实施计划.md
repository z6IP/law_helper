# 道路交通安全法 AI 助手「小Z」— 技术选型与实施计划

## 1. Summary（概述）

构建一个基于《中华人民共和国道路交通安全法》的 RAG 智能法律助手「小Z」。

- **后端**：FastAPI 提供 `/api/v1/...` 问答与检索接口，承载文档解析、向量化入库、混合检索（BM25 + 向量 + RRF 融合）、CrossEncoder 重排序、DeepSeek 生成。
- **前端**：Streamlit 单页对话界面，中文 UI，单列居中布局。
- **数据**：项目根目录已存在 `中华人民共和国道路交通安全法_20210429.docx`，按「章/节/条」结构化解析后入库 ChromaDB。
- **LLM**：DeepSeek（OpenAI 兼容接口）。
- **运行环境**：conda 虚拟环境 `law_helper_env`，Python 3.11。

## 2. Current State Analysis（现状分析）

- 项目根目录 `c:\Users\19674\PycharmProjects\law_helper` 目前仅有：
  - `CLAUDE.md`（行为准则，需遵守）
  - `settings.json`（权限约束：允许 Read/write/Bash(np|git|node *)，禁止 `rm -rf`）
  - `.idea/`（PyCharm 工程配置）
  - `中华人民共和国道路交通安全法_20210429.docx`（数据源，已解析验证结构）
- **没有任何业务代码**（无 `app/`、`ui/`、`requirements.txt`、`.env`），属于从零搭建。
- docx 结构已验证：目录 + 八章（第一章 总则 ~ 第八章 附则），部分章下分节；正文为「第X条 + 内容」条目序列。数据干净、可结构化切分。

## 3. 相似开源项目调研

联网检索 GitHub 与国内镜像后，找到以下高度相关的法律问答助手项目：

| 项目 | 地址 | 技术要点 | 与本项目的契合度 |
|---|---|---|---|
| LawCompassRAG | github.com/Zhihui-nzz/LawCompassRAG | BM25 + 向量混合检索、CrossEncoder 重排、BGE-M3、法律域微调 | 高（检索方案几乎一致） |
| LegalFlash-RAG | github.com/F0rJay/Flash-rag | vLLM + FastAPI + LangChain + ChromaDB、Rerank Top50→Top5、流式输出 | 高（架构一致） |
| Juris-RAG | github.com/liuxinglin70-cmyk/Juris-RAG | 多领域独立向量库、可解释多轮问答、引用展示、超范围拒答 | 中高 |
| JurisAgent | github.com/1968320838/JurisAgent | ChromaDB + 智谱 Embedding、合同审查/咨询/案例分析 | 中 |
| ChatLaw（北大） | github.com/PKU-YuanGroup/ChatLaw | 知识图谱 + MoE 多智能体、法律大模型 | 参考（模型为 13B/33B，本地部署重，不作为本次基座） |

**结论**：采「RAG + 混合检索 + 重排序 + 外部 LLM」路线，与 LawCompassRAG / LegalFlash-RAG 同构，但数据聚焦《道路交通安全法》单部法律，规模更小、可本地轻量落地。不采用 ChatLaw 的大模型微调路线（成本高、与「外部 LLM 生成」的既定约束冲突）。

## 4. 技术栈与具体版本

> 标注 ★ 的版本号为本次联网检索（2026-08）确认的当前稳定版；标注 ⚠ 的为执行时用 `pip` 安装最新稳定版并锁定（本次未检索到确切 patch 号，不臆造）。

| 类别 | 库 / 组件 | 版本 | 用途 |
|---|---|---|---|
| 运行时 | Python | 3.11 | 运行环境 |
| Web 框架 | fastapi | ★ 0.135.x | 后端 API |
| ASGI 服务器 | uvicorn | ★ 0.41.x | 启动后端 |
| 前端 | streamlit | ⚠ 1.4x 最新稳定 | 对话 UI |
| 配置 | pydantic-settings | ★ 2.13.x | 读取 .env |
| 数据校验 | pydantic | ★ 2.12.x | schema |
| 环境变量 | python-dotenv | ★ 1.2.x | .env 加载 |
| 向量库 | chromadb | ★ 0.6.2 | 持久化向量存储 |
| Embedding | BAAI/bge-m3（sentence-transformers 加载，ModelScope 下载） | sentence-transformers ★ 5.3.x（或 3.2.1） | 中文语义向量 |
| 稀疏检索 | rank-bm25 | 0.2.2 | BM25 关键词召回 |
| 重排序 | BAAI/bge-reranker-v2-m3（CrossEncoder） | — | Top-K 重排 |
| LLM SDK | openai | ⚠ 最新稳定 | 调 DeepSeek（自定义 base_url） |
| 文档解析 | python-docx | 1.1.2 | 解析法条 docx |
| 文本切分 | langchain-text-splitters | ★ 1.1.x（可选，或手写按条切分） | 切块 |

**Embedding / Reranker 模型下载**：使用 ModelScope 镜像（`MODELSCOPE_CACHE` 或 `modelscope` 下载 + `sentence-transformers` 本地路径加载），避免 HuggingFace.co 直连超时（工程既定约束）。

## 5. Proposed Changes（实施步骤与文件）

### 5.1 项目骨架

```
law_helper/
├── .env                      # LLM 与运行配置（不入库）
├── .env.example              # 配置模板
├── .gitignore                # 排除 .env、chroma/、__pycache__、data/
├── requirements.txt          # 锁定依赖
├── data/                     # 数据源放置目录（docx 拷贝于此，可选）
├── chroma/                   # ChromaDB 持久化目录（运行时生成，不入库）
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI 入口，/api/v1/... 路由注册
│   ├── config.py             # Settings（pydantic-settings），读 .env
│   ├── errors.py             # LawHelperError 异常层级
│   ├── schemas.py            # 请求/响应模型
│   ├── ingestion.py          # docx 解析 + 切条 + 入库
│   ├── embeddings.py         # BGE-M3 封装（ModelScope 加载）
│   ├── retrieval.py          # BM25 + 向量混合检索 + RRF 融合
│   ├── rerank.py             # CrossEncoder Top-5 重排
│   ├── llm.py                # DeepSeek 调用封装
│   └── qa.py                 # 问答编排（检索→重排→生成→引用）
└── ui/
    └── streamlit_app.py      # 前端对话界面
```

### 5.2 各文件 what / why / how

- **`requirements.txt`**：锁定上述依赖版本，供 `pip install -r` 使用。
- **`.env` / `.env.example`**：`OPENAI_API_BASE`（DeepSeek base_url）、`OPENAI_API_KEY`、`LLM_MODEL`（如 `deepseek-chat`）、`CHROMA_DIR`、`DOCX_PATH`、`EMBEDDING_MODEL_DIR`、`RERANK_MODEL_DIR` 等（工程硬约束：LLM 配置必须放 .env）。
- **`app/config.py`**：用 `pydantic-settings.BaseSettings` 读 `.env`，暴露 Settings 单例；前端状态栏读 Settings 拿 LLM 模型名（不读系统环境变量，工程约束）。
- **`app/errors.py`**：定义 `LawHelperError` 基类 + 子类（如 `IngestionError`、`RetrievalError`、`LLMError`），异常统一捕获。
- **`app/ingestion.py`**：
  - 用 `python-docx` 遍历段落，识别「第X章/第X节」标题与「第X条」正文。
  - 每条法条为一个 chunk；metadata 记录 `section_header`（所属章/节标题，如“第一章 总则”“第二章 车辆和驾驶人 / 第二节 机动车驾驶人”）与 `article_no`（条号），以及 `source`（文件名）。
  - 向量 id = `md5(source + section_header)` 保证幂等 upsert（工程硬约束）。
  - 入库 ChromaDB（`chroma/` 持久化）。
- **`app/embeddings.py`**：封装 BGE-M3（`sentence-transformers`），从 ModelScope 本地缓存路径加载，提供 `embed()` / `embed_query()`。
- **`app/retrieval.py`**：BM25（`rank-bm25`）+ ChromaDB 向量检索双路召回；RRF 融合，`BM25_WEIGHT=0.5`、`RRF_LAMBDA=60`（工程硬约束）。
- **`app/rerank.py`**：`bge-reranker-v2-m3` CrossEncoder 对召回结果打分，取 Top-5（工程硬约束）。
- **`app/llm.py`**：`openai` SDK，`base_url` 指向 DeepSeek，`LLM_MODEL` 来自 Settings；封装 `chat()`。
- **`app/qa.py`**：编排 retrieve → rerank → prompt（附带法条原文与出处）→ DeepSeek 生成 → 返回答案 + 引用法条列表。
- **`app/main.py`**：装配各模块，暴露 `/api/v1/chat`（对话）、`/api/v1/ingest`（入库）、`/api/v1/health` 等；统一异常处理（走 `errors.py`）。
- **`ui/streamlit_app.py`**：单列布局、无侧边栏/顶部 logo；初始版面居中对齐；首条消息发送后输入框下移到底部；全部中文文案；上传区置于聊天框上方（用于后续增补法规 docx）。调用后端 `/api/v1/chat`，展示答案与法条引用，状态栏 LLM 型号取 Settings。

## 6. Assumptions & Decisions（假设与决策）

- 数据仅收录《中华人民共和国道路交通安全法》一部（当前 docx），框架支持后续增补。
- LLM 使用 DeepSeek，`deepseek-chat` 为默认模型，可在 `.env` 切换模型名。
- Embedding / Reranker 在**后端进程**加载（前端 Streamlit 不加载，避免模型内存重复占用，工程既定约束）。
- 模型权重经 ModelScope 下载（国内网络稳定）。
- 向量库用 ChromaDB 本地持久化到 `chroma/`。
- 每条法条独立成块（法条较短，不二次切分；极长条目仍在 BGE-M3 8192 token 上限内）。
- 不执行任何 `rm -rf` 类破坏性命令（settings.json 约束）。
- 修改最小化：只新增实现本需求所需的文件，不改动 `CLAUDE.md` / `settings.json`。

## 7. Verification（验证步骤）

1. `conda activate law_helper_env` 后 `python --version` 确认为 3.11。
2. `pip install -r requirements.txt` 成功、无依赖冲突。
3. 配置 `.env`（DeepSeek key、模型下载路径），运行入库脚本：`python -m app.ingestion`，日志显示按条切分并幂等写入 ChromaDB，法条总数与 8 章结构正确。
4. 启动后端 `uvicorn app.main:app --reload`，`http://127.0.0.1:8000/api/v1/health` 返回正常。
5. 用 curl 打 `/api/v1/chat` 问「酒驾会怎么处罚？」，返回答案并附带对应法条出处。
6. 启动前端 `streamlit run ui/streamlit_app.py`，`http://localhost:8501` 正常打开，中文界面、单列居中、首条发送后输入框下移到底部，答案含法条引用。
7. 混合检索结果通过 RRF（权重 0.5 / lambda 60）与 Top-5 重排后可信（抽样验证相关法条命中）。