# 界面问题修复实施计划

## 问题描述
1. **左下角多余横杠**：发送消息后，界面左下角出现一条不该有的短横线（疑似 `st.iframe` 注入脚本后残留的可视痕迹，或 `stBottomBlockContainer` 默认分隔线）。
2. **用户消息未右对齐**：尽管现有代码使用 `[aria-label]` 选择器和注入 JS 设置 `marginLeft:auto`，但用户消息仍然靠左显示，说明当前选择器/方法在实际 DOM 中未生效。

## 仓库调研结论

### 现有代码结构（ui/streamlit_app.py）
- 第 18-58 行：通过 `st.markdown` 注入全局 CSS。
  - 第 38 行隐藏了头像 `[data-testid*="stChatMessageAvatar"]`，但该元素依然存在于 DOM 中，仍可被 `:has()` 选择器命中。
  - 第 40-54 行使用 `[data-testid="stChatMessageContent"][aria-label="Chat message from user"]` 尝试用户消息右对齐——**该 `aria-label` 在当前 Streamlit 版本中可能不存在或不一致，导致选择器失效**。
- 第 205-261 行 `_inject_alignment_js()`：通过 `st.iframe` + `window.parent.document` 注入 JS，用 `MutationObserver` + `setInterval` 定期设置 `marginLeft`。这层 JS 依赖 aria-label 属性，存在同样的选择器失效风险。
- 第 148 行（`_position_chat_input` 中的 `st.iframe`）以及第 261 行（`_inject_alignment_js` 中的 `st.iframe`）两处都使用了 1x1 iframe 注入脚本，**iframe 默认带有边框/外边距，可能是左下角那条横杠的来源**。
- 另一个可能：`[data-testid="stBottomBlockContainer"]` 或 `[data-testid="stBottom"]` 在新版 Streamlit 中有默认的 `border-top` 分隔线。

### 网上搜索结论（版本独立方案）
根据 Streamlit 官方论坛和 GitHub issue 最新验证可用方案：

**用户消息右对齐（推荐使用 `:has()` + `data-testid`，不依赖不稳定的 emotion-cache class）：**
```css
/* 含用户头像的聊天消息外层容器 → 整体右推 */
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    display: flex;
    justify-content: flex-end; /* 整行靠右 */
}
/* 用户消息内容容器文本右对齐（对内容内部的所有元素生效） */
.stChatMessage:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] * {
    text-align: right;
}
/* AI 消息保持靠左（显式） */
.stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) {
    display: flex;
    justify-content: flex-start;
}
```
> 来源：GitHub streamlit/streamlit#13441 @zhaowei0315 2026-01-27 验证通过版本 1.53.1

**去除分隔线 / 底部容器多余边框：**
```css
/* 底部输入块容器：去除默认背景 + 顶部分隔线 */
[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
    border-top: none !important;
    box-shadow: none !important;
}
/* 注入用 iframe：完全隐藏（不占空间） */
iframe[title="streamlit_app.components.html.iframe"] {
    display: none !important;
    border: none !important;
    outline: none !important;
    width: 0 !important;
    height: 0 !important;
}
/* 兜底：所有 Streamlit 生成的 iframe 隐藏边框 */
iframe {
    border: 0 !important;
    outline: 0 !important;
}
```

## 文件和模块
- `ui/streamlit_app.py`：
  1. 更新第 18-58 行的全局 CSS 注入块，按上述方案替换右对齐选择器、添加底部边框去除和 iframe 隐藏规则。
  2. 移除 `_inject_alignment_js()` 函数（第 205-261 行）和其调用（第 268 行）——新的 `:has()` CSS 方案已覆盖其职责，且移除该 iframe 注入有助于消除左下角横杠。
  3. `_position_chat_input()` 中的 iframe 保留（它负责输入框定位），但通过 CSS 让 iframe 本身完全不可见即可。

## 实施步骤（依赖顺序）
1. **步骤 1：重写右对齐 CSS（使用 `:has()` + `data-testid` 版本独立方案）**
   - 在 `st.markdown(style...)` 块中，删除当前基于 `[aria-label]` 的选择器（第 45-54 行），替换为：
     - `.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) { display:flex; justify-content:flex-end; }`
     - 其内部 `[data-testid="stChatMessageContent"] * { text-align:right; }`
     - 对应 assistant 的显式 `justify-content:flex-start`
     - 保留 `[data-testid="stChatMessageContent"]` 的 `max-width:80%` 和 `width:fit-content`
2. **步骤 2：去除底部横杠**
   - 在同一个 `<style>` 块中追加：
     - `[data-testid="stBottomBlockContainer"]` → `background:transparent; border-top:none; box-shadow:none;`
     - `iframe` → `border:0; outline:0;`
     - 隐藏所有 Streamlit 注入的脚本 iframe（通过属性选择器或全局 `iframe { display:none \!important }` 兜底，但注意不要影响可能未来需要的真实 iframe）
3. **步骤 3：移除冗余的 `_inject_alignment_js()`**
   - 删除第 205-261 行函数定义、删除第 268 行调用。
   - 原因：`:has()` CSS 方案在 DOM 层直接生效，不依赖 JS 轮询，性能更稳定，且少了一个 iframe 注入源（降低横杠出现概率）。

## 依赖和注意事项
- `:has()` 选择器：现代浏览器（Chrome 105+、Edge 105+、Safari 15.4+、Firefox 121+）均支持。国内主流环境兼容。
- Streamlit 内部 `data-testid` 约定：`stChatMessageAvatarUser` 和 `stChatMessageAvatarAssistant` 从 1.40 版本起稳定存在（当前项目大概率 >= 1.40）。
- 虽然第 38 行 `display:none` 隐藏了头像元素，但 `:has()` 是结构选择器，只要 DOM 中存在该元素即能匹配，不受 `display` 影响。
- 本方案不触碰后端代码（`app/`）、不改变流式输出逻辑、不增加任何依赖。

## 验证
1. 启动 `streamlit run ui/streamlit_app.py`，初始界面检查：
   - 「我是小Z」标题和副标题显示正常，输入框在其下方。
2. 输入任意问题并发送，检查：
   - 用户消息气泡整体靠右对齐（不再贴左）。
   - AI 回答依然靠左对齐。
   - 左下角无短横线/分隔线（对比修复前的截图）。
   - 输入框平滑回落至底部。
   - 引用法条展开器工作正常。
3. 刷新页面后再次发消息重复验证，确认 `:has()` 方案无需 JS 轮询即可生效。

## 风险与兜底
- **风险 A：特定 Streamlit 版本下 `data-testid="stChatMessageAvatarUser"` 名称变化** → 回退方案：同时保留 `aria-label` 选择器作为兜底并叠写 `!important`，或用浏览器 F12 检查实际 DOM 的 data-testid 再微调。
- **风险 B：iframe 全局隐藏误伤** → 不使用 `iframe { display:none }` 粗暴写法，改为仅针对脚本容器隐藏（如 `width:0;height:0;border:0`），不影响正常 iframe 内容。
- **风险 C：移除 `_inject_alignment_js()` 后在流式输出瞬间样式闪跳** → `:has()` 是同步 CSS 选择器，流式内容追加时浏览器即时重新计算布局，一般无闪烁；如仍有问题，可恢复该函数作为次要兜底（但先隐藏其 iframe 的边框）。
