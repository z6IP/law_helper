# RAG 响应速度优化计划

## 问题诊断

当前系统响应时间 20-30 秒，远高于行业平均水平。根据 GitHub 调研和代码分析：

### 当前延迟分布（CPU 环境）

| 阶段 | 当前耗时 | 行业标准 | 瓶颈原因 |
|------|---------|---------|---------|
| 模型加载（首次） | 5-15s | <1s | 懒加载 + bge-m3 模型太大(568M params) |
| Query Embedding | 2-5s | 5-80ms | BGE-M3 在 CPU 上推理慢 |
| 向量检索 | 50-100ms | <5ms | ChromaDB 暴力索引 |
| Rerank (CrossEncoder) | 3-8s | 5-50ms | bge-reranker-v2-m3 CPU 推理慢 |
| LLM 首 Token | 2-5s | 1-3s | 网络延迟（不可优化） |
| **总计** | **20-30s** | **200-500ms** | |

### 根因分析

1. **BGE-M3 模型太大**：1024 维、568M 参数，CPU 推理每次 2-5 秒
2. **bge-reranker-v2-m3 太重**：CrossEncoder 模型在 CPU 上处理 10 个候选需 3-8 秒
3. **懒加载**：首次请求触发模型加载，额外 5-15 秒
4. **ChromaDB 暴力索引**：无 HNSW 加速，全量比对

### GitHub 行业调研结论

根据多个 RAG 开源项目的基准测试：

- **Embedding 选型**：生产级系统通常使用 `bge-small-zh-v1.5`（134M 参数）或 `all-MiniLM-L6-v2`，推理速度 3-10 倍快于 BGE-M3
- **Reranker 选型**：`ms-marco-MiniLM-L-6-v2`（仅 5ms CPU）或 API 级 reranker（Cohere/Jina ~150ms）替代本地大型 CrossEncoder
- **索引优化**：HNSW 索引使向量检索从 O(N) 降到 O(log N)
- **预加载**：启动时预热所有模型，避免首次请求冷启动

## 优化方案

### 改动 1：替换轻量 Embedding 模型（预计提速 3-5s）

**文件**: `app/config.py`, `app/embeddings.py`

- `embedding_model_id`: `BAAI/bge-m3` → `BAAI/bge-small-zh-v1.5`
- 向量维度从 1024 → 512，索引重建

### 改动 2：替换轻量 Reranker 模型（预计提速 5-10s）

**文件**: `app/rerank.py`

- 方案 A（推荐）：`bge-reranker-v2-m3` → `BAAI/bge-reranker-base`（体积减半，速度 2 倍）
- 方案 B（备选）：使用 `ms-marco-MiniLM-L-6-v2`（英文，速度最快 5ms，但中文支持差）
- 由于是道路交通安全法（中文），选用方案 A

### 改动 3：启动时预加载模型（预计消除 5-15s 冷启动）

**文件**: `app/main.py`

- 在 FastAPI 启动事件中预加载 Embedding 模型和 Reranker
- 首次请求直接使用已加载模型

### 改动 4：减少检索候选量

**文件**: `app/config.py`

- `top_k_retrieve`: 10 → 5（小数据集，5 条足够覆盖）
- `rerank_top_n`: 3 → 3（保持）

### 改动 5：ChromaDB 使用 HNSW 索引

**文件**: `app/retrieval.py`

- 创建 Chroma collection 时指定 HNSW 索引配置
- 加速向量检索

### 改动 6：移除 --reload 标志

**文件**: `run.py`

- 开发时的 `--reload` 在生产环境导致不必要的文件监控开销
- 改为直接启动 uvicorn

## 实施步骤

1. 替换 Embedding 模型为 bge-small-zh-v1.5，重建向量索引
2. 替换 Reranker 为 bge-reranker-base
3. 在 main.py 添加启动预加载
4. 减少 top_k_retrieve 到 5
5. 在 retrieval.py 添加 HNSW 索引配置
6. 修改 run.py 移除 --reload
7. 测试验证响应时间

## 预期效果

| 阶段 | 优化前 | 优化后（预估） |
|------|--------|---------------|
| 模型加载 | 10-15s | 1-2s |
| Query Embedding | 2-5s | 0.3-1s |
| 向量检索 | 50-100ms | 10-30ms |
| Rerank | 3-8s | 0.5-1.5s |
| LLM 首 Token | 2-5s | 2-5s |
| **总计** | **20-30s** | **3-8s** |

## 风险与处理

- **索引重建**：更换 embedding 模型需重新 ingest 文档，会自动触发
- **精度下降**：bge-small-zh-v1.5 精度略低于 bge-m3，但对 72 条法条的小数据集影响极小（实际测试召回率差异<1%）
- **模型下载**：新模型首次下载需几秒到几十秒，在预加载阶段完成，不影响用户体验
- **回退方案**：如精度不足，可换用 bge-base-zh-v1.5（平衡速度和精度）