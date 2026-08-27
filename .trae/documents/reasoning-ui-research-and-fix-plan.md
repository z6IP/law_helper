# 思考过程（Reasoning）UI 研究与修复计划

## 1. Summary

当前 `law_helper` Streamlit 前端的思考过程展示存在以下体验问题：
- 默认展开/收起状态不符合预期；
- 从“思考中”切换到“思考完成”时，用户手动展开的思考框会被自动折叠或闪烁；
- 使用 `st.status` 无法持久化用户展开状态，且完成状态图标与文字语义不一致。

本计划先调研 DeepSeek、Qwen、GitHub Copilot 等主流产品的思考过程交互流程，然后基于调研结论重新设计并修复当前实现，最终交付一个稳定、符合用户预期的思考过程 UI。

## 2. Current State Analysis

### 2.1 前端代码位置
- `ui/streamlit_app.py:347-411`：流式输出时创建 `st.status("思考中", state="running", expanded=False)` 作为思考容器，实时写入 `reasoning` 内容，首个 `delta` 到达后 `update(label="思考完成", state="complete")`。
- `ui/streamlit_app.py:321-324`：历史消息用 `st.status("思考完成", state="complete", expanded=False)` 展示已保存的 reasoning。
- `app/qa.py:156-217`：后端通过 `/api/v1/chat/stream` 推送 `progress` / `reasoning` / `delta` / `references` 事件。

### 2.2 已知问题
1. **`st.status` 无 `key` 参数**：Streamlit 无法跨 rerun 保持用户展开/收起状态，每次事件触发 rerun 后都会回到 `expanded=False`。
2. **`state="complete"` 与展开状态冲突**：在部分 Streamlit 版本中，`complete` 状态会强制折叠内容；保持 `running` 又导致“思考完成”后图标仍在转圈。
3. **默认状态混乱**：之前为满足“我要看思考过程”临时改成 `expanded=True`，后又被要求“默认应该是关闭思考过程”，说明当前组件无法同时满足“默认收起”和“展开后不被刷新”。
4. **末尾 `st.rerun()` 已移除**：去掉了 rerun 以避免闪烁，但历史消息渲染与流式渲染仍使用不同组件类型，状态无法继承。

## 3. 行业最佳实践调研

### 3.1 DeepSeek / DeepSeek R1
- **数据流**：API 通过 `reasoning_content`（非流式）或 `delta.reasoning_content`（流式）独立返回思维链，与 `content` 分离。
- **UI 形态**：
  - 使用可折叠的“Reasoning Process”块，默认**收起**；
  - 块标题显示状态（如 `Thinking…` / `Reasoning Process`），左侧有状态图标；
  - 流式阶段在折叠面板内实时追加 reasoning tokens；
  - 完成后面板标题变为“Reasoning Process”或“Thought briefly”，保持用户最后展开的折叠状态。
- **关键设计原则**：
  - 思考内容**不默认占用主屏幕**，避免长 CoT 淹没最终答案；
  - 用户主动展开后，状态切换不再折叠面板；
  - 最终答案与 reasoning 在视觉层级上严格区分。

来源：[DeepSeek 思考模式 API 文档](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode)、[DeepSeek Thinking UI 示例](https://github.com/lewismessthecode/Deepseek-Thinkinhg-UI)、[DeepCopilot Issue #52](https://github.com/deep-copilot/DeepCopilot/issues/52)

### 3.2 Qwen / 通义千问
- **数据流**：OpenAI 兼容接口通过 `extra_body={"enable_thinking": True}` 开启；流式中先输出 `reasoning_content`，再输出 `content`。
- **UI 形态**：
  - 网页版 Playground 在输入框底部提供“Deep Thinking”开关；
  - 思考过程以折叠块形式呈现，默认**收起**；
  - 思考阶段显示一行紧凑标题（如 `∴ Thinking… 3s`），不展开时不占用空间；
  - 支持 `Ctrl+O` / `Alt+T` 全局展开/收起所有思考块；
  - 长思考任务中保持块高度恒定（1 行标题），避免页面上下跳动。
- **关键设计原则**：
  - **默认隐藏流式预览**，防止高度抖动；
  - 思考块与后续工具调用/最终答案按时间顺序**内联排列**，而不是全部堆在顶部；
  - 用户展开后保持滚动位置，不做上下文切换。

来源：[QwenCloud Thinking 文档](https://docs.qwencloud.com/developer-guides/text-generation/thinking)、[qwen-code PR #8077](https://github.com/QwenLM/qwen-code/pull/8077)

### 3.3 GitHub Copilot / CopilotKit
- **数据流**：Copilot 本身不暴露真实 CoT tokens，仅显示“Thought for Ns”计时标签；CopilotKit 通过 `typing` + `informative` activity 或 `REASONING_MESSAGE_START/CONTENT` 事件传输 reasoning。
- **UI 形态**：
  - 以 inline chip / pill 形式嵌入对话流（如 `[Thought for 2s] → tool calls → final answer`）；
  - 不默认展开，点击后展开查看详细 reasoning；
  - 无真实内容时仅显示计时，避免空白块。
- **关键设计原则**：
  - 思考指示器**内联**于动作之前，而不是顶部聚合；
  - 默认保持对话流紧凑；
  - reasoning 作为可展开的辅助信息，不抢占主答案视觉焦点。

来源：[Microsoft Copilot Studio Activity Trace](https://learn.microsoft.com/et-ee/microsoft-copilot-studio/agents-experience/preview-overview)、[Showing Agent Reasoning in Custom UIs](http://microsoft.github.io/mcscatblog/posts/show-reasoning-agents-sdk/)、[CopilotKit Issue #3420](https://github.com/CopilotKit/CopilotKit/issues/3420)

### 3.4 共性的设计原则总结
1. **默认收起**：长 CoT 默认不可见，避免淹没最终答案。  
2. **状态标签可变，折叠状态不变**：标题可以从 `Thinking…` 变为 `Thought briefly` / `思考完成`，但用户手动展开后不应被强制折叠。  
3. **流式内容写入折叠容器内部**：即使容器收起，内部文本仍实时追加，展开即可看到完整历史。  
4. **紧凑标题，防止抖动**：未展开时只显示一行状态标题，高度恒定。  
5. **最终答案与 reasoning 视觉分离**：reasoning 使用辅助色/边框/图标与正文区分。

## 4. Proposed Changes

### 4.1 技术选型：改用 `st.expander` + 前端 JS 动态标题

`st.status` 无法传入 `key`，导致无法持久化展开状态；而 `st.expander` 支持 `key`，可以跨 rerun 保持用户展开/收起状态。我们将：
- 用 `st.expander` 作为思考过程容器；
- 通过 `st.iframe` 注入 JS，在流式过程中动态修改 expander 的 `<summary>` 标题文字和图标；
- 这样既能保持 `key` 带来的状态持久化，又能让标题在“思考中”和“思考完成”之间切换。

### 4.2 具体修改

#### A. `ui/streamlit_app.py` 流式渲染块（当前 `L347-411` 附近）
1. 将 `think_ph.status(...)` 替换为：
   ```python
   think_key = f"think_{len(st.session_state.messages)}"
   think_ph = st.empty()
   think_expander = think_ph.expander("思考中", expanded=False, key=think_key)
   reasoning_inner = think_expander.empty()
   ```
2. 在 expander 创建后，通过 `st.iframe` 注入一段 JS，为该 expander 的 `<summary>` 添加 `thinking-in-progress` class，并插入旋转 spinner SVG/字符。
3. `progress` / `reasoning` 事件仍只更新 `reasoning_inner.markdown(...)` 和标题文字（通过 JS）。
4. 首个 `delta` 到达时，通过 JS 将标题从“思考中”改为“思考完成”，移除 spinner，保留 expander 折叠状态。
5. 非推理模型时，`think_ph.empty()` 移除整个 expander。

#### B. `ui/streamlit_app.py` 历史消息块（当前 `L317-324` 附近）
1. 历史消息同样使用 `st.expander`，标题固定为“思考完成”，使用与流式期不同的 key（如 `f"hist_think_{msg_idx}"`）。
2. 这样历史消息默认收起，且每条历史消息的展开状态独立保持。

#### C. 新增/复用 JS 注入函数
- 在 `_position_chat_input` 类似的 `st.iframe` 注入模式基础上，新增 `_update_expander_label(key, label, is_thinking)` 函数。
- JS 通过 `data-testid` 或自定义 `data-think-key` 属性定位到目标 expander 的 `<summary>`，修改其 innerHTML。

#### D. CSS 调整
- 移除当前未使用的 `.thinking-spinner` / `.thinking-done` / `.done-icon` 样式（可选，保持代码整洁）。
- 为 JS 注入的 spinner 添加 CSS 旋转动画。
- 为“思考完成”状态添加绿色勾选图标样式。

### 4.3 事件流与状态机

| 阶段 | 前端状态 | expander 标题 | expanded | 内部内容 |
|---|---|---|---|---|
| 发送问题后 | 等待后端 | “思考中” + spinner | 默认 False | 空 |
| progress 事件 | 检索/重排中 | “思考中” / 进度文案 + spinner | 用户控制 | 空或已追加的 reasoning |
| reasoning 事件 | LLM 思考中 | “思考中” + spinner | 用户控制 | 实时追加 |
| 首个 delta | 思考完成 | “思考完成” + 勾选 | 用户控制 | 完整 reasoning |
| 后续 delta | 输出正文中 | “思考完成” + 勾选 | 用户控制 | 不变 |
| 非推理模型 | 无 thinking | 移除 expander | - | - |

## 5. Assumptions & Decisions

1. **默认收起**：遵循 DeepSeek/Qwen/Copilot 的共性设计，reasoning 默认不展开，避免长 CoT 占用主屏幕。  
2. **`st.expander` 优于 `st.status`**：虽然 `st.status` 自带状态图标，但它不支持 `key`，无法满足“保持用户展开状态”的硬性要求。  
3. **JS 注入修改标题**：Streamlit 原生不支持动态修改 expander 标题，必须通过 JS 操作 DOM。该模式与现有 `_position_chat_input` 的 `st.iframe` 注入一致。  
4. **历史消息与当前消息 key 不同**：流式期用基于消息索引的 key，历史记录用基于 `msg_idx` 的 key。两者状态不共享，但各自稳定，避免跨会话状态串扰。  
5. **reasoning 内容仍实时追加到折叠容器内部**：即使 expander 收起，内部 `reasoning_inner` 持续更新，用户展开即可看到完整流式历史。  
6. **不引入新的依赖**：继续基于 Streamlit + 原生 JS/CSS 实现，不引入额外前端框架。

## 6. Verification Steps

1. 启动后端和 Streamlit 前端。  
2. 发送一个法律问题，观察：
   - 思考过程默认收起，只显示一行“思考中”标题带旋转图标；
   - 主动展开思考框后，可以看到实时流式的 reasoning 内容；
   - 思考完成（首个正文 delta 到达）后，标题变为“思考完成”带勾选图标；
   - 标题变化过程中，思考框**不自动折叠、不闪烁、不重复开合**。  
3. 继续等待正文流式输出，确认正文正常追加，思考框保持用户最后操作的状态。  
4. 发送第二个问题，确认：
   - 上一个问题的思考过程在历史记录中显示为“思考完成”，默认收起；
   - 展开历史消息的思考过程，内容完整，且展开状态稳定。  
5. 测试非推理模型（如果配置允许），确认无思考内容时不显示思考框。  
6. 测试快速刷新页面或切换对话，确认无 `DuplicateWidgetID` 或 JS 错误。

## 7. 参考来源

- DeepSeek 思考模式 API 文档：https://api-docs.deepseek.com/zh-cn/guides/thinking_mode  
- DeepSeek Thinking UI (Streamlit 示例)：https://github.com/lewismessthecode/Deepseek-Thinkinhg-UI  
- DeepCopilot 思考块设计讨论：https://github.com/deep-copilot/DeepCopilot/issues/52  
- QwenCloud Thinking 文档：https://docs.qwencloud.com/developer-guides/text-generation/thinking  
- qwen-code PR #8077（思考块高度稳定与内联展开）：https://github.com/QwenLM/qwen-code/pull/8077  
- Microsoft Copilot Studio Activity Trace：https://learn.microsoft.com/et-ee/microsoft-copilot-studio/agents-experience/preview-overview  
- Showing Agent Reasoning in Custom UIs：http://microsoft.github.io/mcscatblog/posts/show-reasoning-agents-sdk/  
- CopilotKit reasoning 消息实现讨论：https://github.com/CopilotKit/CopilotKit/issues/3420
