"""微批合并 merge_stream_events 的单元测试。

验证：delta/reasoning 事件按阈值合并后，拼接文本与逐事件完全一致、
非 delta 事件即时透传、事件顺序保持、不同 kind 分缓冲。
"""
from __future__ import annotations

from app.jobs import _DELTA_MERGE_CHARS, merge_stream_events


def collect(stream):
    return list(merge_stream_events(iter(stream)))


class TestMergeStreamEvents:
    def test_merges_small_deltas_until_stream_end(self):
        chunks = ["你", "好", "，", "请", "问"]
        events = [{"type": "delta", "content": c} for c in chunks]
        out = collect(events)
        assert len(out) == 1
        assert out[0]["type"] == "delta"
        assert out[0]["content"] == "".join(chunks)

    def test_non_delta_events_flush_and_pass_through(self):
        events = [
            {"type": "progress", "content": "正在检索"},
            {"type": "delta", "content": "答"},
            {"type": "references", "references": []},
            {"type": "delta", "content": "案"},
            {"type": "done"},
        ]
        out = collect(events)
        assert [e["type"] for e in out] == [
            "progress",
            "delta",
            "references",
            "delta",
            "done",
        ]
        delta_texts = [e["content"] for e in out if e["type"] == "delta"]
        assert "".join(delta_texts) == "答案"

    def test_delta_and_reasoning_kept_separate(self):
        events = [
            {"type": "reasoning", "content": "思考"},
            {"type": "delta", "content": "正文"},
            {"type": "reasoning", "content": "继续"},
        ]
        out = collect(events)
        assert [e["type"] for e in out] == ["reasoning", "delta", "reasoning"]
        assert out[0]["content"] == "思考"
        assert out[1]["content"] == "正文"
        assert out[2]["content"] == "继续"

    def test_concatenation_equals_stream_order(self):
        events = [
            {"type": "delta", "content": "a"},
            {"type": "delta", "content": "b"},
            {"type": "progress", "content": "x"},
            {"type": "delta", "content": "c"},
        ]
        out = collect(events)
        deltas = [e["content"] for e in out if e["type"] == "delta"]
        assert "".join(deltas) == "abc"

    def test_char_threshold_flushes_mid_stream(self):
        big = "x" * (_DELTA_MERGE_CHARS + 1)
        events = [
            {"type": "delta", "content": big},
            {"type": "delta", "content": "tail"},
        ]
        out = collect(events)
        assert len(out) >= 2
        delta_texts = [e["content"] for e in out if e["type"] == "delta"]
        assert "".join(delta_texts) == big + "tail"

    def test_empty_stream(self):
        assert collect([]) == []

    def test_non_content_payloads_preserved(self):
        # progress/references 等非 delta 事件原样透传，不被合并或改写
        events = [
            {"type": "references", "references": [{"source": "法A"}]},
            {"type": "progress", "content": "正在思考"},
            {"type": "done"},
        ]
        assert collect(events) == events
