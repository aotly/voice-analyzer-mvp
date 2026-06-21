from ai_reports import _raise_safe_api_error, build_report_prompt, generate_ai_report
from voice_core import GIRL


class FakeResponse:
    def __init__(self):
        self.status_code = 503
        self.reason = "Service Unavailable"


def test_openai_without_key_falls_back_to_rule_report(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    profile = {
        "labels": ["明亮听感"],
        "pitch": {"mean_hz": 220, "range_hz": 70},
        "scores": {"lightness": 80, "front_resonance": 60, "stability": 50},
        "formant_spacing_hz": 1100,
        "voice_identity": {"primary_label": "中性少女", "secondary_tags": []},
    }

    report, source = generate_ai_report(profile, "更自然女声", "OpenAI", api_key="")

    assert source == "本地规则"
    assert "声音鉴定标签" in report


def test_api_errors_do_not_expose_query_keys():
    try:
        _raise_safe_api_error(FakeResponse(), "Gemini")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected safe API error")

    assert "503" in message
    assert "key=" not in message
    assert "googleapis.com" not in message


def test_report_prompt_requires_f3_f4_and_voice_label():
    profile = {
        "labels": ["明亮听感"],
        "pitch": {"mean_hz": 270, "range_hz": 135},
        "formants": {"f1": 840, "f2": 2196, "f3": 3318, "f4": 4018},
        "concept_scores": {GIRL: 80},
        "voice_identity": {"primary_label": "中性少女"},
    }

    prompt = build_report_prompt(profile, "更自然女声")

    assert "F3" in prompt
    assert "F4" in prompt
    assert "声音鉴定标签" in prompt
    assert "不超过 7 个字" in prompt
    assert "文学描述" in prompt
    assert "不是用户真实性别" in prompt
