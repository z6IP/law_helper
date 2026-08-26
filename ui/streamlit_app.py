"""小Z 前端（Streamlit）。

单列居中布局、无侧边栏/顶部 logo；统一使用聊天输入框（无独立发送按钮），
初始版面显示标题与副标题，发送首条消息后标题与副标题消失，对话历史从顶部排列。
全部中文文案。
"""
from __future__ import annotations

import json
import urllib.request

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
           - 内层：气泡贴合文字（inline-block）、给一点内边距（不贴字）
           - 文字：气泡内左对齐
         */
        .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
            display: flex !important;
            justify-content: flex-end !important;
        }
        .stChatMessage:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] {
            display: inline-block !important;
            width: auto !important;
            flex: 0 1 auto !important;
            padding: 5px 14px !important;
            border-radius: 4px !important;
            background-color: #1e1e2e !important;
            margin-left: auto !important;
            margin-right: 0 !important;
            box-sizing: border-box !important;
            max-width: 100% !important;
        }
        .stChatMessage:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] * {
            text-align: left !important;
        }
        /* 用户消息内的块级元素不要撑破 inline-block 容器的收缩宽度 */
        .stChatMessage:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] > *:first-child,
        .stChatMessage:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] p,
        .stChatMessage:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] div {
            width: auto !important;
            max-width: 100% !important;
        }

        /* ========== AI 消息 ==========
           - 完全不覆盖 width / display / flex / padding / margin，
             让 Streamlit 流式输出的容器按默认 block 100% 宽度渲染，避免文字丢失；
           - 只保留：外层左对齐（贴输入框左缘）、文字左对齐、并显式指定颜色为页面前景色。
         */
        .stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) {
            display: flex !important;
            justify-content: flex-start !important;
        }
        .stChatMessage:has([data-testid="stChatMessageAvatarAssistant"])
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
        .stChatMessage:has([data-testid="stChatMessageAvatarAssistant"])
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
    """调用后端流式接口，逐段产出回答文本；引用法条写入 refs_holder["references"]。"""
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
                if data.get("type") == "references":
                    refs_holder["references"] = data.get("references", [])
                elif data.get("type") == "delta":
                    yield data.get("content", "")
    except Exception as exc:  # noqa: BLE001
        yield f"后端调用失败：{exc}"


if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_references(references: list[dict]) -> None:
    if not references:
        return
    with st.expander("查看引用法条"):
        for ref in references:
            header = ref.get("article_no", "")
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
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
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
            answer = st.write_stream(_stream_chat(question, refs_holder))
            references = refs_holder.get("references", [])
            _render_references(references)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "references": references}
    )
    st.rerun()


# 初始状态上移输入框、发送后回落到底部（同一元素平滑过渡）
_position_chat_input(not st.session_state.messages)