# 聊天气泡对齐修复计划

## 问题分析

### 当前状态
- 用户消息和 AI 消息都显示在左侧/中间，没有实现"用户右对齐、AI 左对齐"的效果

### 根因
当前 CSS 使用 `:has([data-testid="stChatMessageAvatarUser"])` 选择器来区分用户消息，但存在以下问题：
1. Streamlit 1.61.0 的 `data-testid` 属性命名可能已变化
2. `:has()` 伪类选择器在 Streamlit 嵌套上下文可能表现不稳定
3. 头像被 `display:none` 隐藏后，`:has()` 仍然应该能匹配，但如果 Streamlit 内部改变了 DOM 结构（如使用 `visibility:hidden` 或完全移除元素），选择器会静默失败

## 实施计划

### 修改文件
- `ui/streamlit_app.py`: 修改 CSS 样式和添加 JavaScript 注入

### 具体步骤

#### 步骤 1：修改 CSS 样式
移除依赖 `:has()` 的选择器，改用基于 class 的样式：

```python
# 用户消息靠右
.msg-user {
    flex-direction: row-reverse !important;
}
.msg-user [data-testid="stChatMessageContent"] {
    text-align: right;
    margin-left: auto;
}

# AI 消息靠左（默认行为，无需特殊处理）
```

#### 步骤 2：添加 JavaScript 注入
在页面加载后，通过 JS 扫描所有 `stChatMessage` 元素，根据其内容（用户/助手）添加对应的 class：

```javascript
// 遍历所有 chat message 元素
document.querySelectorAll('[data-testid="stChatMessage"]').forEach(el => {
    // 检查是否包含用户头像相关元素
    if (el.querySelector('[data-testid*="AvatarUser"]') || 
        el.textContent.includes('用户消息')) {
        el.classList.add('msg-user');
    } else {
        el.classList.add('msg-assistant');
    }
});
```

#### 步骤 3：增强版 JS 方案（更可靠）
由于 Streamlit 可能使用不同的 DOM 结构，改用基于位置或文本内容的检测方法：

```javascript
// 使用 MutationObserver 监听 DOM 变化
// 根据消息在父容器中的位置或其他特征来判断角色
```

### 验证
1. 重启 Streamlit 应用
2. 发送用户消息，确认气泡右对齐
3. AI 回复后，确认气泡左对齐
4. 检查是否影响其他功能（如引用法条展开器）

### 风险与处理
- **风险**：Streamlit 更新可能改变 DOM 结构
- **处理**：使用 MutationObserver 持续监听，确保每次 DOM 变化后都正确添加 class
- **风险**：JS 注入时机可能不对
- **处理**：使用 `window.addEventListener('load')` 和 MutationObserver 双重保障
