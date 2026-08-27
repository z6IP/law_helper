# 展示大模型思考过程 — 实施计划

## Summary

在现有 RAG 流式问答链路上，把阿里云百炼推理模型流式输出中的 `reasoning_content` 字段（思考过程）透传到前端，在 AI 回答气泡上方增加一个可折叠的「查看思考过程」展开器（与现有「查看引用法条」展开器风格一致），默认收起。保留当前 `LLM_MODEL=deepseek-v4-flash-0731` 先实测，若该模型不吐 `reasoning_content`，则前端不显示该展开器（优雅降级）。

## Current State Analysis

### 现有流式协议（NDJSON，每行一个 JSON 事件）

`app/main.py` 的 `/api/v1/chat/stream` 端点逐行 yield NDJSON，目前只有两种事件类型：
- `{"type": "references", "references": [...]}` — 检索到的引用法条（最先发出，可能为空数组）
- `{"type": "delta", "content": "..."}` — LLM 正文字本增量

### LLM 调用层 `app/llm.py`

`BailianClient.chat_stream` 用 OpenAI 兼容 SDK 流式调用，当前**只读 `delta.content`**：
```python
for chunk in resp:
    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
        yield chunk.choices[0].delta.content
```
未读取 `delta.reasoning_content` 字段（百炼推理模型 qwq/deepseek-r1/qwen3-reasoner 会通过该字段返回思考过程）。

### 编排层 `app/qa.py`

`answer_stream` 生成器：
1. 先 `yield {"type": "references", ...}`
2. 调用 `get_llm().chat_stream(SYSTEM_PROMPT, user_prompt)`，把每个 delta 直接 `yield {"type": "delta", "content": delta}`

### 前端 `ui/streamlit_app.py`

`_stream_chat(question, refs_holder)` 用 `urllib.request` 读 NDJSON 流，**yield 字符串**（仅正文 delta），引用法条写入 `refs_holder["references"]`。

渲染：
```python
with st.chat_message("assistant"):
    answer = st.write_stream(_stream_chat(question, refs_holder))
    references = refs_holder.get("references", [])
    _render_references(references)
```

`st.write_stream` 接受字符串生成器，自动累积渲染（打字机效果）。引用法条通过 `_render_references` 在正文下方放 `st.expander("查看引用法条")`。

### 关键约束

- 推理模型流式输出顺序：**先全部 `reasoning_content`，再 `content`**（百炼约定）。
- `st.write_stream` 一次只能写一个 placeholder，无法交错写两个区域 → 需要放弃 `st.write_stream`，改用手动累积 + `placeholder.markdown` 原地更新。
- 当前 `LLM_MODEL=deepseek-v4-flash-0731` 是否吐 `reasoning_content` 未知，前端必须优雅降级（无 reasoning 时不显示展开器）。

## Proposed Changes

### 1. `app/llm.py` — `chat_stream` 区分 reasoning / content

**What**: `chat_stream` 改为 yield 元组 `(kind, text)`，kind ∈ `{"reasoning", "content"}`，分别从 `delta.reasoning_content` 和 `delta.content` 读取。

**Why**: 让编排层能区分两种文本，分别走不同 NDJSON 事件。

**How**:
```python
def chat_stream(self, system_prompt, user_prompt):
    self._ensure_loaded()
    settings = get_settings()
    try:
        resp = self._client.chat.completions.create(
            model=settings.llm_model,
            messages=[...],
            temperature=0.2,
            stream=True,
        )
        for chunk in resp:
            if not (chunk.choices and chunk.choices[0].delta):
                continue
            delta = chunk.choices[0].delta
            # 思考过程（推理模型才有，普通模型该字段为 None）
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                yield ("reasoning", rc)
            # 正文
            if delta.content:
                yield ("content", delta.content)
    except Exception as exc:
        raise LLMError(f"大模型调用失败：{exc}") from exc
```

**注意**: 用 `getattr(delta, "reasoning_content", None)` 兜底——OpenAI SDK 的 `Delta` 没有 `reasoning_content` 字段时不会报错（Pydantic 模型允许额外字段或返回 None）。

### 2. `app/qa.py` — `answer_stream` 新增 `reasoning` 事件

**What**: 把 `(kind, text)` 元组映射为 NDJSON 事件：
- `("reasoning", x)` → `{"type": "reasoning", "content": x}`
- `("content", x)` → `{"type": "delta", "content": x}`（保持现有事件名，前端兼容旧逻辑）

**Why**: 在现有协议上扩展一个新事件类型，不破坏 `delta`/`references`。

**How**: 修改 `answer_stream` 中两处调用 `chat_stream` 的循环（无意义输入分支 + 正常 RAG 分支）：

```python
for kind, text in get_llm().chat_stream(SYSTEM_PROMPT, user_prompt):
    if kind == "reasoning":
        yield {"type": "reasoning", "content": text}
    else:
        yield {"type": "delta", "content": text}
```

两处分支（trivial 拒答分支、有法条的正常分支）都同样改。无相关法条时 LLM 简短拒答，也可能带思考过程，前端同样展示。

### 3. `ui/streamlit_app.py` — 前端处理 reasoning 事件 + 折叠展开器

**What**:
- `_stream_chat` 改为 yield 元组 `(kind, payload)`：`("references", [...])` / `("reasoning", "chunk")` / `("delta", "chunk")`。
- 渲染逻辑放弃 `st.write_stream`，改用手动累积 + `placeholder.markdown` 原地更新。创建「查看思考过程」展开器（默认收起），位于 AI 回答正文上方；无 reasoning 时不显示。

**Why**:
- `st.write_stream` 无法把一个流拆到两个 placeholder（reasoning 区 + 正文区）。
- 推理模型先吐 reasoning 再吐 content，手动累积可正确处理该顺序，且无 reasoning 时优雅降级。

**How** — 替换 `_stream_chat` 与 AI 消息渲染块：

```python
def _stream_chat(question: str, refs_holder: dict):
    """调用后端流式接口，yield (kind, payload) 元组。
    payload: references 事件 -> list[dict]；reasoning/delta 事件 -> str 文本块。
    """
    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND_URL}/api/v1/chat/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                data = json.loads(line)
                t = data.get("type")
                if t == "references":
                    yield ("references", data.get("references", []))
                elif t == "reasoning":
                    yield ("reasoning", data.get("content", ""))
                elif t == "delta":
                    yield ("delta", data.get("content", ""))
    except Exception as exc:
        yield ("delta", f"后端调用失败：{exc}")
```

AI 消息渲染（替换 `with chat_placeholder: with st.chat_message("assistant"): ...` 块）：

```python
with chat_placeholder:
    with st.chat_message("assistant"):
        # 思考过程展开器容器（无 reasoning 时保持为空，不显示）
        reasoning_container = st.empty()
        # 正文区
        answer_placeholder = st.empty()

        reasoning_buf: list[str] = []
        answer_buf: list[str] = []
        reasoning_inner = None  # 懒创建：首次收到 reasoning 时才建 expander
        references: list[dict] = []

        for kind, data in _stream_chat(question, refs_holder):
            if kind == "references":
                references = data
            elif kind == "reasoning":
                if reasoning_inner is None:
                    expander = reasoning_container.expander(
                        "查看思考过程", expanded=False
                    )
                    reasoning_inner = expander.empty()
                reasoning_buf.append(data)
                reasoning_inner.markdown("".join(reasoning_buf))
            elif kind == "delta":
                answer_buf.append(data)
                answer_placeholder.markdown("".join(answer_buf))

        answer_text = "".join(answer_buf)
        _render_references(references)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "references": references,
            "reasoning": "".join(reasoning_buf) or None,  # 无思考时存 None
        }
    )
    st.rerun()
```

历史消息回填渲染（`for msg in st.session_state.messages` 循环）也要支持 reasoning：
```python
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("reasoning"):
            with st.expander("查看思考过程", expanded=False):
                st.markdown(msg["reasoning"])
        st.write(msg["content"])
        if msg["role"] == "assistant":
            _render_references(msg.get("references", []))
```

CSS 对齐：现有 CSS 已让 `stExpander` 左对齐到输入框左缘，新增的「查看思考过程」展开器会自动继承同一规则，无需额外 CSS。

## Assumptions & Decisions

1. **保留 `deepseek-v4-flash-0731` 先实测**：若该模型不吐 `reasoning_content`，前端 `reasoning_container` 保持为空（`st.empty()`），不显示展开器，正文正常流式渲染 — 优雅降级，不报错。
2. **非流式接口 `/api/v1/chat` 暂不支持 reasoning**：前端只用流式接口，非流式 `chat()` 保持返回 `(text, references)` 不变。后续如需可同步扩展。
3. **放弃 `st.write_stream`，改用手动累积 + `placeholder.markdown`**：因 `st.write_stream` 无法把单个流拆到两个 placeholder（reasoning 区 + 正文区）。手动累积的视觉效果与 `write_stream` 基本一致（原地更新文本），可接受。
4. **reasoning 事件协议字段**：`{"type": "reasoning", "content": "..."}`，与 `delta` 平行，不修改 `delta` 语义，保持后向兼容。
5. **思考过程只读 `delta.reasoning_content`**：不读 `delta.reasoning`（部分模型用短名）—— 百炼 OpenAI 兼容接口统一用 `reasoning_content`。若实测发现该模型用别的字段名，再加 `getattr` 兜底。
6. **历史消息存储 reasoning**：存入 `session_state.messages[i]["reasoning"]`，刷新页面后历史 AI 消息的思考过程仍可展开查看。无思考时存 `None`，回填时跳过渲染展开器。
7. **拒答分支也展示思考**：trivial 输入走拒答分支时，LLM 仍可能输出思考过程，前端同样展示（保持一致性）。

## Verification Steps

1. **后端流式协议验证** — 启动后端，运行 `_debug_backend_stream.py`，提一个真实法律问题（如「酒驾怎么罚」），确认输出行中除 `references` / `delta` 外出现 `{"type": "reasoning", "content": "..."}` 事件。若没有 reasoning 事件，说明当前模型不吐思考，需考虑换模型（不在本次计划内）。
2. **前端渲染验证** — 启动前端，提「酒驾怎么罚」：
   - 若模型吐 reasoning：AI 回答气泡上方出现「查看思考过程」展开器（默认收起），点开能看到完整思考文本；正文在下方流式渲染；底部「查看引用法条」展开器正常。
   - 若模型不吐 reasoning：不显示「查看思考过程」展开器，正文正常流式渲染，引用法条正常。
3. **历史消息回填验证** — 发送一条消息后刷新页面（或触发 rerun），确认 AI 历史消息的思考展开器仍可展开查看。
4. **拒答分支验证** — 输入「你好」（trivial），确认拒答消息也走流式，无 reasoning 时无展开器；有 reasoning 时正常展示。
5. **trivial query 与无相关法条分支** — 输入「1」（trivial）和「天气怎么样」（无相关法条），确认前端不崩、不显示空展开器、不显示引用法条展开器。
