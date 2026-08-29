"""小Z 前端（Streamlit）。

单列居中布局、无侧边栏/顶部 logo；统一使用聊天输入框（无独立发送按钮），
初始版面显示标题与副标题，发送首条消息后标题与副标题消失，对话历史从顶部排列。
全部中文文案。
"""
from __future__ import annotations

import json

import httpx
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="小Z · 道路交通安全法助手", layout="centered")

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { display: none; }
        header[data-testid="stHeader"] { background: transparent; }
        /* 聚焦时不显示红色边框（聊天输入框） */
        [data-testid="stChatInput"] > div {
            border-color: rgba(128, 128, 128, 0.4) !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] textarea:focus,
        [data-testid="stChatInput"] textarea:focus-visible {
            outline: none !important;
            box-shadow: none !important;
        }
        /* 聊天输入框：上移/回落平滑过渡 */
        [data-testid="stBottom"] {
            transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        }
        /* 隐藏头像（用户 + AI） */
        [data-testid*="stChatMessageAvatar"] { display: none !important; }
        /* 清除所有聊天消息外层的左右 padding（默认 16px），
           让消息内容的边缘可以与输入框左右边缘严格对齐；
           同时清除外层背景色和圆角，把背景/圆角改挂在【用户消息的内层内容容器】上，
           这样【用户消息】的短消息背景框就不会撑满整行，只会贴合文字大小；
           【AI 消息】仍保留 Streamlit 默认 block 布局，确保流式输出文字始终可见 */
        .stChatMessage {
            padding-left: 0 !important;
            padding-right: 0 !important;
            background: transparent !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }

        /* ========== 用户消息 ==========
           - 外层：右对齐到输入框右缘
           - 内层：气泡给一点内边距，短内容按最小宽度居中
           - 文字：气泡内居中
         */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            display: flex !important;
            justify-content: flex-end !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] {
            display: inline-flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: auto !important;
            flex: 0 1 auto !important;
            padding: 8px 14px !important;
            border-radius: 4px !important;
            background-color: rgba(128, 128, 128, 0.15) !important;
            color: inherit !important;
            margin-left: auto !important;
            margin-right: 0 !important;
            box-sizing: border-box !important;
            max-width: 100% !important;
            min-width: 60px !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] * {
            text-align: center !important;
        }
        /* 用户消息内的块级元素占满 flex 容器宽度，使短内容能在 min-width 内居中 */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] > *:first-child,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] p,
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] div {
            width: 100% !important;
            max-width: 100% !important;
            text-align: center !important;
            margin: 0 !important;
        }

        /* ========== AI 消息 ==========
           - 完全不覆盖 width / display / flex / padding / margin，
             让 Streamlit 流式输出的容器按默认 block 100% 宽度渲染，避免文字丢失；
           - 只保留：外层左对齐（贴输入框左缘）、文字左对齐、并显式指定颜色为页面前景色。
         */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            display: flex !important;
            justify-content: flex-start !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
            [data-testid="stChatMessageContent"] {
            /* 仅控制对齐方式与可见性，不改盒模型 */
            text-align: left !important;
            color: inherit !important;
            opacity: 1 !important;
            visibility: visible !important;
            margin-left: 0 !important;
            margin-right: auto !important;
            max-width: 100% !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
            [data-testid="stChatMessageContent"] * {
            text-align: left !important;
            color: inherit !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        /* 让"查看引用法条"展开器（stExpander）对齐到 AI 消息的左侧（输入框左缘） */
        [data-testid="stExpander"] {
            max-width: 100% !important;
            margin-left: 0 !important;
            margin-right: auto !important;
        }
        /* ======== 自定义 spinner / 完成状态 ======== */
        .thinking-spinner {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: #888;
            font-size: 0.95em;
            margin-top: 4px;
        }
        .thinking-spinner .spinner-ring {
            width: 14px;
            height: 14px;
            border: 2px solid rgba(128, 128, 128, 0.3);
            border-top-color: #888;
            border-radius: 50%;
            animation: thinking-spin 0.8s linear infinite;
            flex-shrink: 0;
            display: inline-block;
        }
        .thinking-spinner.thinking-done {
            color: #2a9d8f;
        }
        .thinking-spinner .done-icon {
            width: 14px;
            height: 14px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            flex-shrink: 0;
        }
        @keyframes thinking-spin {
            to { transform: rotate(360deg); }
        }
        /* ---- 去除底部横杠 ---- */
        /* 底部输入块容器：去除默认背景、顶部分隔线和阴影 */
        [data-testid="stBottomBlockContainer"] {
            background: transparent !important;
            border-top: none !important;
            box-shadow: none !important;
            padding-top: 0 !important;
        }
        [data-testid="stBottom"] {
            border-top: none !important;
            box-shadow: none !important;
        }
        /* 所有 iframe 去除边框、轮廓（注入脚本用的 1x1 iframe 不留痕迹） */
        iframe {
            border: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        /* st.iframe 注入产生的空容器隐藏 */
        [data-testid="stIframe"] iframe {
            width: 0 !important;
            height: 0 !important;
            opacity: 0 !important;
            visibility: hidden !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _stream_chat(question: str, refs_holder: dict):
    """调用后端流式接口，yield (kind, payload) 元组。

    使用 httpx 逐行流式读取（iter_lines），确保 reasoning 内容真正实时推送，
    而不是等待整段思考完成后才一次性显示。

    payload:
      - references 事件 -> list[dict]（引用法条列表）
      - reasoning 事件 -> str（思考过程文本块）
      - delta 事件 -> str（正文文本块）
    """
    url = f"{BACKEND_URL}/api/v1/chat/stream"
    payload = {"question": question}
    try:
        with httpx.Client(timeout=180) as client:
            with client.stream("POST", url, json=payload) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    t = data.get("type")
                    if t == "references":
                        yield ("references", data.get("references", []))
                    elif t == "progress":
                        yield ("progress", data.get("content", ""))
                    elif t == "reasoning":
                        yield ("reasoning", data.get("content", ""))
                    elif t == "delta":
                        yield ("delta", data.get("content", ""))
    except httpx.ConnectError:
        yield ("delta", "⚠️ 后端服务尚未就绪，请稍后刷新页面重试。")
    except Exception as exc:  # noqa: BLE001
        # 连接类错误给出友好提示，其他错误保留原始信息
        msg = str(exc)
        if "10061" in msg or "ConnectionRefused" in msg:
            yield ("delta", "⚠️ 后端服务尚未就绪，请稍后刷新页面重试。")
        else:
            yield ("delta", f"后端调用失败：{exc}")


if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_references(references: list[dict]) -> None:
    if not references:
        return
    # 显式 expanded=False + 唯一 key，避免多消息间展开状态串扰
    with st.expander("查看引用法条", expanded=False, key=f"refs_{id(references)}"):
        for ref in references:
            source = ref.get("source", "")
            article_no = ref.get("article_no", "")
            header = f"《{source}》{article_no}" if source else article_no
            if ref.get("section_header"):
                header += f"（{ref['section_header']}）"
            st.markdown(f"**{header}**")
            st.write(ref.get("text", ""))


def _position_chat_input(is_welcome: bool) -> None:
    """初始时把底部输入框上移到副标题下方，发送首条消息后平滑回落底部。

    通过 st.iframe 注入一段 JS（同源）定位父页面的 [data-testid="stBottom"]，
    配合上面的 CSS transition 实现平滑过渡。
    """
    mode = "welcome" if is_welcome else "chat"
    js = """
    <script>
    (function () {
        window.frameElement.parentElement.style.display = "none";
        var mode = "__MODE__";
        var doc = window.parent.document;
        var gap = 24;
        function position() {
            var input = doc.querySelector('[data-testid="stBottom"]');
            if (!input) { return; }
            if (mode === "welcome") {
                var sub = doc.querySelector(".welcome-sub");
                if (!sub) { return; }
                var inputTop = input.getBoundingClientRect().top;
                var desiredTop = sub.getBoundingClientRect().bottom + gap;
                var dy = inputTop - desiredTop;
                // 首次定位不带动画：先禁用过渡，直接落在副标题下方
                input.style.transition = "none";
                input.style.transform = dy > 0
                    ? "translateY(-" + dy + "px)"
                    : "translateY(0px)";
                void input.offsetHeight;
                input.style.transition = "";
            } else {
                input.style.transform = "translateY(0px)";
            }
        }
        function tryPosition() {
            var input = doc.querySelector('[data-testid="stBottom"]');
            var ready = !!input;
            if (mode === "welcome") {
                ready = ready && !!doc.querySelector(".welcome-sub");
            }
            if (ready) { position(); }
            else { setTimeout(tryPosition, 50); }
        }
        tryPosition();
        window.parent.addEventListener("resize", position);
    })();
    </script>
    """.replace("__MODE__", mode)
    st.iframe(js, width=1, height=1)


# ---- 标题区域占位符（提交问题后立即清空，让对话内容占据顶部）----
title_placeholder = st.empty()

# ---- 对话历史占位符（用于控制渲染顺序）----
chat_placeholder = st.container()

# ---- 初始状态：渲染标题 + 副标题 ----
# 只有在没有消息历史时才显示标题（已发送过消息的 rerun 不渲染标题）
if not st.session_state.messages:
    with title_placeholder:
        st.markdown(
            '<style>'
            '.welcome-title{text-align:center;font-size:2.5rem;font-weight:700;margin-top:14vh;}'
            '.welcome-sub{text-align:center;color:#888;margin-bottom:1.5rem;}'
            '</style>'
            '<div class="welcome-title">我是小Z</div>'
            '<div class="welcome-sub">你的道路交通安全法智能助手，请随时提问</div>',
            unsafe_allow_html=True,
        )

# ---- 渲染对话历史 ----
with chat_placeholder:
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("reasoning"):
                # 历史思考过程：用 st.status 展示，标题为"思考完成"，默认收起
                with st.status("思考完成", state="complete", expanded=False):
                    st.markdown(msg["reasoning"])
            # assistant 引用法条渲染
            st.write(msg["content"])
            if msg["role"] == "assistant":
                _render_references(msg.get("references", []))

# ---- 底部输入框（流式输出）----
if question := st.chat_input("请输入您的法律问题"):
    # 立即清空标题占位符 —— 对话内容覆盖原标题位置
    title_placeholder.empty()

    # 立即将输入框平滑回落到底部
    _position_chat_input(False)

    # 追加并渲染用户问题（靠右）
    st.session_state.messages.append({"role": "user", "content": question})
    with chat_placeholder:
        with st.chat_message("user"):
            st.write(question)

    # 流式渲染 AI 回答（靠左）
    refs_holder: dict = {}
    with chat_placeholder:
        with st.chat_message("assistant"):
            # ===== 思考过程：st.status 容器（标题即状态指示器）=====
            # 标题在思考中为"思考中"，首个正文 delta 到达后变为"思考完成"；
            # 默认收起；update 时不传 expanded，尽量保持用户展开/收起状态。
            think_ph = st.empty()
            status_container = think_ph.status("思考中", state="running", expanded=False)
            reasoning_inner = status_container.empty()

            reasoning_buf: list[str] = []
            answer_buf: list[str] = []
            answer_placeholder = st.empty()  # 流式正文占位符（替换式更新，避免重复渲染）
            references: list[dict] = []
            thinking_done = False
            has_llm_reasoning = False
            spinner_label = "思考中"  # 当前状态标签文案

            for kind, data in _stream_chat(question, refs_holder):
                if kind == "references":
                    references = data
                elif kind == "progress":
                    # 进度文案显示在状态标签上，不写入思考过程
                    spinner_label = data
                    status_container.update(label=data, state="running")
                elif kind == "reasoning":
                    # 真正的 LLM 思考内容（推理模型才有）
                    reasoning_buf.append(data)
                    reasoning_inner.markdown("".join(reasoning_buf))
                    if not has_llm_reasoning:
                        has_llm_reasoning = True
                        # 进入 LLM 思考阶段，状态标签恢复为"思考中"
                        if spinner_label != "思考中":
                            spinner_label = "思考中"
                            status_container.update(label="思考中", state="running")
                elif kind == "delta":
                    # 首个 delta 到达：思考结束，状态标签变为"思考完成"
                    if not thinking_done:
                        thinking_done = True
                        status_container.update(label="思考完成", state="complete")
                    answer_buf.append(data)
                    answer_placeholder.markdown("".join(answer_buf))

            answer_text = "".join(answer_buf)

            # 只有思考没有正文的情况：同样切换到"思考完成"
            if not thinking_done:
                status_container.update(label="思考完成", state="complete")
                thinking_done = True

            # 非推理模型（无思考内容）：移除思考过程组件
            if not has_llm_reasoning:
                think_ph.empty()

            _render_references(references)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "references": references,
            # 仅当有真正的 LLM 思考内容时才保存 reasoning（不含预热文本）
            "reasoning": "".join(reasoning_buf) if has_llm_reasoning else None,
        }
    )
    # 不再 st.rerun()：流式渲染的结果已包含最终状态（思考完成 + 展开），
    # 避免 rerun 导致 st.status 重建而闪烁。


# 初始状态上移输入框、发送后回落到底部（同一元素平滑过渡）
_position_chat_input(not st.session_state.messages)