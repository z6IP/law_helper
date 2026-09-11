from types import SimpleNamespace

import pytest

import app.prompt_loader as prompt_loader


def test_prompt_templates_render_explicit_values(monkeypatch):
    monkeypatch.setattr(
        prompt_loader,
        "get_settings",
        lambda: SimpleNamespace(prompt_dir="prompts"),
    )
    prompt_loader.clear_prompt_cache()

    prompt = prompt_loader.render_prompt("multi_query", query_count=4)

    assert "4" in prompt
    assert "只做语言层改写" in prompt


def test_prompt_template_reports_missing_values(monkeypatch):
    monkeypatch.setattr(
        prompt_loader,
        "get_settings",
        lambda: SimpleNamespace(prompt_dir="prompts"),
    )
    prompt_loader.clear_prompt_cache()

    with pytest.raises(ValueError, match="缺少变量"):
        prompt_loader.render_prompt("off_topic")


def test_all_prompt_files_are_present():
    expected = {
        "legal_system",
        "off_topic",
        "context_resolve",
        "meta_answer",
        "multi_query",
        "title_system",
        "title_user",
        "ocr_default",
        "domain_traffic",
    }
    assert expected <= {
        path.stem for path in prompt_loader._prompt_directory().glob("*.txt")
    }


def test_legal_prompt_blocks_unsupported_legal_inferences(monkeypatch):
    monkeypatch.setattr(
        prompt_loader,
        "get_settings",
        lambda: SimpleNamespace(prompt_dir="prompts"),
    )
    prompt_loader.clear_prompt_cache()

    prompt = prompt_loader.render_prompt("legal_system", law_list="《测试法规》")

    assert "不能从“仅规定了某一限制”推出“许可了其他行为”" in prompt


@pytest.mark.parametrize(
    "template_name",
    ["refusal_no_law", "refusal_insufficient", "refusal_out_of_scope"],
)
def test_refusal_prompts_render_question(template_name, monkeypatch):
    """A5：新增拒答模板必须可渲染且包含用户问题。"""
    monkeypatch.setattr(
        prompt_loader,
        "get_settings",
        lambda: SimpleNamespace(prompt_dir="prompts"),
    )
    prompt_loader.clear_prompt_cache()

    prompt = prompt_loader.render_prompt(template_name, question="测试问题？")
    assert "测试问题？" in prompt


def test_legal_prompt_defaults_to_concise_visible_answer(monkeypatch):
    monkeypatch.setattr(
        prompt_loader,
        "get_settings",
        lambda: SimpleNamespace(prompt_dir="prompts"),
    )
    prompt_loader.clear_prompt_cache()

    prompt = prompt_loader.render_prompt("legal_system", law_list="《测试法规》")

    assert "结论只写 2-3 句话" in prompt
    assert "不能为了简短而省略解释" in prompt
    assert "建议是回答中最重要的部分" in prompt
    assert "必须按以下三个分类输出" in prompt
    assert "## 分析" in prompt
    assert "## 结论" in prompt
    assert "## 建议" in prompt
    assert "分类标题必须独占一行" in prompt
    assert "不要使用“……”拼接不连续的原文" in prompt
    assert "跨法规关联" in prompt


@pytest.mark.parametrize(
    "case_word",
    ["不得载人", "允许上高速", "无证驾驶", "冲卡", "扰乱秩序", "立法法", "禁令标志", "摩托车", "扣车", "行政拘留"],
)
def test_legal_prompt_has_no_case_hardcoding(monkeypatch, case_word):
    """去硬编码守护：摩托车上高速被拘留案例的特有词不得出现在通用提示词中。"""
    monkeypatch.setattr(
        prompt_loader,
        "get_settings",
        lambda: SimpleNamespace(prompt_dir="prompts"),
    )
    prompt_loader.clear_prompt_cache()

    prompt = prompt_loader.render_prompt("legal_system", law_list="《测试法规》")

    assert case_word not in prompt


def test_domain_traffic_prompt_renders(monkeypatch):
    """交通领域提示词可独立渲染，且无占位符变量。"""
    monkeypatch.setattr(
        prompt_loader,
        "get_settings",
        lambda: SimpleNamespace(prompt_dir="prompts"),
    )
    prompt_loader.clear_prompt_cache()

    prompt = prompt_loader.render_prompt("domain_traffic")

    assert "禁令标志" in prompt
    assert "上位法授权" in prompt


def test_match_domains_hits_traffic(monkeypatch):
    import app.qa as qa

    monkeypatch.setattr(
        qa,
        "get_policy",
        lambda: {
            "domains": {
                "traffic": {
                    "prompt": "domain_traffic",
                    "law_keywords": ["道路交通安全法", "机动车驾驶证"],
                }
            }
        },
    )

    contexts = [{"metadata": {"source": "中华人民共和国道路交通安全法"}}]

    assert qa._match_domains(contexts) == ("traffic",)


def test_match_domains_no_hit(monkeypatch):
    import app.qa as qa

    monkeypatch.setattr(
        qa,
        "get_policy",
        lambda: {
            "domains": {
                "traffic": {
                    "prompt": "domain_traffic",
                    "law_keywords": ["道路交通安全法"],
                }
            }
        },
    )

    contexts = [{"metadata": {"source": "中华人民共和国民法典"}}]

    assert qa._match_domains(contexts) == ()


def test_compose_system_prompt_injects_domain(monkeypatch):
    import app.qa as qa

    monkeypatch.setattr(
        qa,
        "get_policy",
        lambda: {
            "domains": {
                "traffic": {
                    "prompt": "domain_traffic",
                    "law_keywords": ["道路交通安全法"],
                }
            }
        },
    )
    monkeypatch.setattr(qa, "_system_prompt", lambda: "基础提示词")
    monkeypatch.setattr(qa, "_domain_prompt", lambda d: f"[{d}领域段]")
    events = []
    monkeypatch.setattr(qa, "event", lambda name, **kw: events.append((name, kw)))

    result = qa._compose_system_prompt(
        [{"metadata": {"source": "中华人民共和国道路交通安全法"}}]
    )

    assert result == "基础提示词\n\n[traffic领域段]"
    assert events == [("prompt.domain_injected", {"domains": ["traffic"]})]


def test_compose_system_prompt_no_domain_returns_base(monkeypatch):
    import app.qa as qa

    monkeypatch.setattr(
        qa,
        "get_policy",
        lambda: {
            "domains": {
                "traffic": {
                    "prompt": "domain_traffic",
                    "law_keywords": ["道路交通安全法"],
                }
            }
        },
    )
    monkeypatch.setattr(qa, "_system_prompt", lambda: "基础提示词")

    result = qa._compose_system_prompt(
        [{"metadata": {"source": "中华人民共和国民法典"}}]
    )

    assert result == "基础提示词"


def test_compose_system_prompt_degrades_on_missing_domain(monkeypatch):
    import app.qa as qa

    monkeypatch.setattr(
        qa,
        "get_policy",
        lambda: {
            "domains": {
                "traffic": {
                    "prompt": "domain_traffic",
                    "law_keywords": ["道路交通安全法"],
                }
            }
        },
    )
    monkeypatch.setattr(qa, "_system_prompt", lambda: "基础提示词")

    def _raise(*_args, **_kwargs):
        raise FileNotFoundError("提示词模板不存在")

    monkeypatch.setattr(qa, "_domain_prompt", _raise)
    events = []
    monkeypatch.setattr(qa, "event", lambda name, **kw: events.append((name, kw)))

    result = qa._compose_system_prompt(
        [{"metadata": {"source": "中华人民共和国道路交通安全法"}}]
    )

    assert result == "基础提示词"
    assert events == [("prompt.domain_missing", {"domains": ["traffic"]})]
