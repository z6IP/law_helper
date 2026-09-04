"""后台问答任务：把「生成过程」从 HTTP 连接解耦为独立后台线程。

用户发送问题后，生成在独立 daemon 线程中执行，直到给出完整答案：
- 切换历史会话 / 新建对话 / 切换到其他浏览器页面都不会中断生成；
- 客户端主动断开（刷新 / 关闭标签页）时，后台线程继续运行，并把结果写回会话存储。

会话落库分两阶段：
1. 提交任务时立即写入「标题 + 用户消息」（保证用户问题不丢失，刷新后会话仍在）；
2. 生成完成后写回完整「助手消息」（正文 + reasoning + 引用法条）。

约束：同一会话同一时刻只允许一个生成任务（单用户本地工具，进程内内存注册表即可）。
"""
from __future__ import annotations

import threading

from app import answer_cache, session_store
from app.errors import LawHelperError
from app.qa import answer_stream
from app.tracing import finish_trace, start_trace


class JobConflictError(LawHelperError):
    status_code = 409
    message = "该会话正在生成回答"


class Job:
    """单次后台问答任务，按 session_id 唯一。"""

    def __init__(
        self,
        session_id: str,
        question: str,
        history: list[dict],
        title: str,
        document_text: str | None = None,
        file_names: list[str] | None = None,
        attachments: list[dict] | None = None,
    ):
        self.session_id = session_id
        self.question = question
        self.history = history
        self.title = title
        self.document_text = document_text
        self.file_names = file_names or []
        self.attachments = attachments or []

        self.status = "running"  # running / done / error
        self.error: str | None = None
        self.events: list[dict] = []
        self.answer_text = ""
        self.references: list[dict] = []
        self.reasoning_text = ""

        self._cond = threading.Condition()
        self._thread = threading.Thread(
            target=self._run, name=f"job-{session_id[:8]}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def append_event(self, ev: dict) -> None:
        with self._cond:
            self.events.append(ev)
            self._cond.notify_all()

    def read_from(self, cursor: int) -> tuple[list[dict], str]:
        """无阻塞快照：返回 [cursor:] 事件与当前状态。"""
        with self._cond:
            return list(self.events[cursor:]), self.status

    def wait_for(self, cursor: int) -> None:
        """阻塞直到出现新事件或任务结束。"""
        with self._cond:
            while len(self.events) <= cursor and self.status == "running":
                self._cond.wait()

    def _set_status(self, status: str) -> None:
        with self._cond:
            self.status = status
            self._cond.notify_all()

    def _run(self) -> None:
        cached = answer_cache.get(self.question) if not self.history and not self.document_text else None
        start_trace(
            kind="chat_stream",
            question=self.question,
            session_id=self.session_id,
            cache_hit=cached is not None,
        )
        try:
            if cached is not None:
                self.references = cached["references"]
                self.answer_text = cached["answer"]
                self.append_event({"type": "references", "references": self.references})
                self.append_event({"type": "delta", "content": self.answer_text})
            else:
                for payload in answer_stream(self.question, self.history, self.document_text):
                    self.append_event(payload)
                    t = payload.get("type")
                    if t == "references":
                        self.references = payload.get("references") or []
                    elif t == "reasoning":
                        self.reasoning_text += payload.get("content") or ""
                    elif t == "delta":
                        self.answer_text += payload.get("content") or ""

            # 答案缓存仅对「无历史的单轮问题且带引用」启用（与同步 /chat 一致）
            if not self.history and self.references and not self.document_text:
                answer_cache.put(self.question, self.answer_text, self.references)

            self.answer_text = self.answer_text or "（未生成有效回答）"
            _persist(self, with_answer=True)
            self.append_event({"type": "done"})
            self._set_status("done")
        except Exception as exc:  # noqa: BLE001 - 兜底异常，避免线程静默退出
            message = getattr(exc, "message", None) or str(exc)
            self.error = message
            self.append_event({"type": "error", "content": message})
            # 生成失败也落库错误提示，避免刷新后助手消息丢失
            if not self.answer_text:
                self.answer_text = message
            _persist(self, with_answer=True)
            self._set_status("error")
        finally:
            finish_trace(status="error" if self.error else "ok")


def _persist(job: Job, with_answer: bool) -> None:
    """按会话落库；with_answer=False 时仅写入用户消息（生成前），True 时写入完整助手回复。"""
    user_msg: dict = {"role": "user", "content": job.question}
    if job.file_names:
        user_msg["fileNames"] = job.file_names
    if job.attachments:
        user_msg["attachments"] = job.attachments
    messages: list[dict] = list(job.history) + [user_msg]
    if with_answer:
        messages.append(
            {
                "role": "assistant",
                "content": job.answer_text,
                "references": job.references,
                "reasoning": job.reasoning_text or None,
            }
        )
    session_store.save(job.session_id, job.title, messages)


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def submit_chat_job(
    session_id: str,
    question: str,
    history: list[dict],
    title: str,
    document_text: str | None = None,
    file_names: list[str] | None = None,
    attachments: list[dict] | None = None,
) -> Job:
    """创建并启动一个后台问答任务；同会话已有任务在跑时抛出 JobConflictError。"""
    with _JOBS_LOCK:
        existing = _JOBS.get(session_id)
        if existing is not None and existing.status == "running":
            raise JobConflictError()
        job = Job(session_id, question, history, title, document_text, file_names, attachments)
        _JOBS[session_id] = job

    _persist(job, with_answer=False)  # 立即落盘标题 + 用户消息
    job.start()
    return job


def get_job(session_id: str) -> Job | None:
    with _JOBS_LOCK:
        return _JOBS.get(session_id)


def list_running_session_ids() -> list[str]:
    with _JOBS_LOCK:
        return [sid for sid, job in _JOBS.items() if job.status == "running"]