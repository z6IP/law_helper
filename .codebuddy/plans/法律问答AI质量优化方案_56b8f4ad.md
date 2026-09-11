---
name: 法律问答AI质量优化方案
overview: 通过重新设计系统提示词输出结构（支持多步法律分析）、增强多查询改写（跨法规视角）、添加跨法规扩展规则、优化上下文注入方式（按法规分组）和调整关键参数，使AI能自动产出"行为定性→处罚依据→匹配分析"的结构化法律推理，且全部改进均为通用设计不针对特定法条硬编码。
todos:
  - id: redesign-legal-prompt
    content: 重新设计 legal_system.txt：输出格式从 结论/依据/建议 改为 分析/结论/建议，内部思考扩展为6步（新增法律问题分解+跨法规关联），保留全部法律适用原则和引用约束
    status: completed
  - id: enhance-multi-query
    content: 增强 multi_query.txt 新增跨法规关联视角，调整 config.py 中 multi_query_count 从3改为4
    status: completed
  - id: add-cross-law-rules
    content: 在 policy.json 新增3条跨法规 expansion_rules 和3组 synonym_map（被拘留/摩托车+高速/上高速）
    status: completed
  - id: optimize-context-injection
    content: 修改 qa.py 的 _build_user_prompt：按法规分组注入上下文，增加跨法规关联提示，修复 article_no 变量shadowing
    status: completed
  - id: tune-params
    content: 调整 config.py 参数：thinking_budget 2000→4000，rerank_top_n 5→7
    status: completed
  - id: update-tests
    content: "更新 tests/test_prompts.py 断言：匹配新的 ## 分析 节格式和调整后的内容表述"
    status: completed
    dependencies:
      - redesign-legal-prompt
---

## 产品概述
法律问答AI系统的回答质量优化，使其能自动产出结构化法律推理（行为定性→处罚依据→匹配分析），且全部改进为通用设计，不针对特定法条硬编码，覆盖任意跨法规场景。

## 核心问题
1. **系统提示词输出格式不支持多步分析**：当前"结论/依据/建议"三段式中，`## 依据`仅罗列法条编号+对应用户事实标注，不引导LLM展开跨法规的逻辑推理链条
2. **多查询改写缺少跨法规视角**：`multi_query.txt`只有3个视角（行为要件/法律后果/程序救济），无法召回跨法规场景中不同法律领域的条款
3. **扩展规则不覆盖跨法规场景**：`policy.json`的26条expansion_rules均为交通领域单法规场景，缺少"行为+处罚"跨法规关联的扩展
4. **上下文注入未分组**：`_build_user_prompt`平铺法条，LLM难以直观看到跨法规关系
5. **参数限制**：`rerank_top_n=5`对跨法规场景偏少，`thinking_budget=2000`对多步推理不够，`multi_query_count=3`缺少跨法规视角的改写配额

## 技术栈
- 后端：Python + FastAPI + qwen-plus LLM（阿里云百炼 OpenAI 兼容接口）
- 检索：BM25（jieba分词）+ 向量检索（BGE-base-zh-v1.5）RRF融合
- 重排序：qwen3.7-text-rerank（DashScope API）
- 向量库：ChromaDB
- 提示词管理：外置 prompts/ 目录 + prompt_loader.py 模板渲染
- 策略配置：config/policy.json 数据驱动

## 实现方案

### 1. 重新设计 `legal_system.txt`（核心改动）
**改动要点**：将输出格式从 `## 结论 / ## 依据 / ## 建议` 改为 `## 分析 / ## 结论 / ## 建议`

**`## 分析` 节的设计**（替代原 `## 依据`）：
- 不再是简单编号罗列法条，而是引导LLM按法律问题维度展开推理
- LLM需自行识别用户场景涉及的法律问题维度（如行为定性、处罚依据、匹配分析等），只分析实际涉及的维度
- 对每个维度引用检索到的相关法条进行论证，说明该条如何适用于或不适用于用户事实
- 不同法规的条文若共同支撑或制约同一结论，需说明它们之间的关系
- 当法条无法完全回答某维度时，明确指出缺口和需要补充的方向
- 如果用户描述的结果（如"被拘留"）与单纯行为之间缺少直接因果，分析可能存在的中间环节

**内部思考从4步扩展为6步**：
1. 事实识别（不变）
2. 法律问题分解（新增）：从场景中识别需要回答的法律问题
3. 要件匹配（不变）
4. 跨法规关联（新增）：检查不同法规条文是否共同作用于同一问题
5. 缺口与冲突检查（不变）
6. 输出决策（调整）：按分析维度组织论证

**保留全部现有内容**：
- 所有法律适用原则（法无禁止即可为/法无授权即禁止等）
- "不能从不得载人推出允许上高速"等具体约束
- A9引用格式统一、A2引用与事实对应
- 表达要求
- 拒绝编造法条、不推定未说明事实等红线

### 2. 增强 `multi_query.txt`
新增第4个视角"跨法规关联"：当用户场景涉及行为与处罚的对应关系时，改写为能召回不同法规中行为定义条款和处罚授权条款的查询。同时将 `multi_query_count` 从3调整为4。

### 3. 添加跨法规扩展规则到 `policy.json`
新增expansion_rules（通用，不引用具体法条）：
- `被拘留`/`行政拘留` → 扩展"行政拘留 处罚种类 处罚依据 构成要件"
- `摩托车`+`高速` → 扩展"两轮摩托车 高速公路 通行规定 行驶"
- `上高速`+`摩托车` → 扩展"摩托车 高速公路 行驶 通行规定"

新增synonym_map：
- `摩托车` → `["两轮摩托车 通行规定", "摩托车 高速公路 行驶"]`
- `被拘留` → `["行政拘留 处罚依据", "行政拘留 处罚种类"]`
- `上高速` → `["高速公路 通行规定 行驶"]`

### 4. 优化 `_build_user_prompt()` 在 `qa.py`
- 将contexts按source分组注入，同一法规的条文归在一起，法规间加分隔标记
- 在用户问题前增加跨法规关联提示："以上法条可能来自不同的法律法规，请分析它们之间的关联关系"

### 5. 调整参数 `config.py`
- `thinking_budget`: 2000 → 4000（多步分析需要更多思考空间）
- `rerank_top_n`: 5 → 7（跨法规场景需召回更多法条）
- `multi_query_count`: 3 → 4（新增跨法规视角）

### 6. 更新测试 `tests/test_prompts.py`
- `test_legal_prompt_defaults_to_concise_visible_answer`：将 `## 依据` 断言改为 `## 分析`，更新相关内容断言
- `test_legal_prompt_blocks_unsupported_legal_inferences`：不变（该约束保留）
- `test_prompt_templates_render_explicit_values`：不变（multi_query格式不变）

## 实现注意事项
- `_build_user_prompt` 中 `article_no` 参数变量在循环内被覆盖（line 438），重构时需避免此shadowing问题
- `rerank_top_n=7` 在 `max_per_source=3` 和 `behavior_penalty_max=2` 约束下仍能保证跨法规覆盖（3部法律各2-3条）
- `thinking_budget=4000` 对 qwen-plus 是安全的（该模型支持更大预算）
- 修改 `legal_system.txt` 后需清除 `prompt_loader.py` 的 `@lru_cache` 缓存（测试中已有 `clear_prompt_cache()`）
- `_system_prompt()` 在 `qa.py` 中也有 `@lru_cache`，需确认缓存失效机制

## 目录结构
```
project-root/
├── prompts/
│   ├── legal_system.txt      # [MODIFY] 输出格式从 结论/依据/建议 改为 分析/结论/建议，内部思考从4步扩展为6步
│   └── multi_query.txt       # [MODIFY] 新增跨法规关联视角（第4个视角）
├── config/
│   └── policy.json           # [MODIFY] 新增3条跨法规expansion_rules、3组synonym_map
├── app/
│   ├── qa.py                 # [MODIFY] _build_user_prompt() 按法规分组注入上下文，增加跨法规关联提示
│   └── config.py             # [MODIFY] thinking_budget=4000, rerank_top_n=7, multi_query_count=4
└── tests/
    └── test_prompts.py       # [MODIFY] 更新断言匹配新提示词格式
```
