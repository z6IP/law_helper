from types import SimpleNamespace

import app.llm as llm_module


class FakeCompletions:
    def __init__(self, stream=False):
        self.stream = stream
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            delta = SimpleNamespace(reasoning_content="分析", content="答案")
            chunk = SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(delta=delta)],
            )
            return iter([chunk])
        message = SimpleNamespace(content="答案")
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=message)],
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def test_sync_call_keeps_thinking_disabled(monkeypatch):
    settings = SimpleNamespace(
        llm_model="test-model",
        answer_temperature=0.35,
        thinking_enabled=True,
        thinking_budget=1234,
    )
    client = FakeClient()
    model = llm_module.BailianClient()
    model._client = client
    monkeypatch.setattr(llm_module, "get_settings", lambda: settings)

    assert model.chat("system", "user") == "答案"

    call = client.chat.completions.calls[0]
    assert call["temperature"] == 0.35
    assert call["extra_body"] == {"enable_thinking": False}


def test_stream_call_uses_configured_thinking_strategy(monkeypatch):
    settings = SimpleNamespace(
        llm_model="test-model",
        answer_temperature=0.15,
        thinking_enabled=False,
        thinking_budget=987,
    )
    client = FakeClient()
    model = llm_module.BailianClient()
    model._client = client
    monkeypatch.setattr(llm_module, "get_settings", lambda: settings)

    assert list(model.chat_stream("system", "user")) == [
        ("reasoning", "分析"),
        ("content", "答案"),
    ]

    call = client.chat.completions.calls[0]
    assert call["temperature"] == 0.15
    assert call["extra_body"] == {
        "enable_thinking": False,
        "thinking_budget": 987,
    }
