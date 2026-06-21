from __future__ import annotations

import json
import os
from typing import Literal

import requests

from voice_core import generate_rule_report


Provider = Literal["本地规则", "OpenAI", "Gemini"]


def build_report_prompt(profile: dict, user_goal: str) -> str:
    return f"""
你是一个声音鉴定与声音训练产品里的声学报告助手。你必须基于结构化数据写报告，不能只写泛泛鼓励。

安全边界：
1. 不做医学诊断。
2. 不判断用户真实生理性别或身份。
3. 可以分析“听感倾向”“声学结构”“训练方向”“模仿可塑性”。
4. “雄激素化听感痕迹”只能解释为声音厚度、低共振、粗糙度等声学倾向，不能写成身体判定。
5. “少女系、少年系、轻御系、正太系、青年系”等词只能表示声线风格相似度，不是用户真实性别、年龄、人格或身份判断。

必须输出这些段落：
## 声线画像
先给一个通用声学画像，例如：明亮轻质型、气声轻质型、低频厚重型、清亮稳定型。不要直接用少女、少年、御姐、正太作为首页结论。

## 核心维度解释
必须按这些通用维度解释：轻薄感、声音女性感、声音年轻感、明亮感、声音硬朗感、气息感、共鸣前置感、颗粒感/质感、声音温度、稳定自然度、雄激素化痕迹、声音可塑性。

## 声音鉴定标签
这部分只作为“风格接近方向”，不是核心评分，也不是身份判断。给一个简短但有辨识度的风格标签，不超过 7 个字。可以参考：清亮年轻、轻声旁白、中性少年、冷感清亮、气泡青年。不要把所有信息都塞进标签，复杂气质放到副标签和文学描述里。

## 副标签
给 5-8 个短副标签，例如：轻薄、微气声、高前置、颗粒边、暖声、生活女声、MTF高潜力、轻声学男性化特征。

## 一段文学描述
用 80-160 字描述这个声音的气质，要具体、有画面、有用心感，但不要油腻，不要过度夸张。可以写“像什么”，但不要攻击用户。

## 关键声学解释
必须逐项解释 F0、F1、F2、F3、F4、formant spacing、HNR。不能漏 F3 和 F4。

## 概念评分解释
解释少女感、少年感、幼感、御姐/成熟感、轻薄感、厚度感、清亮感、气泡/粗糙感、共鸣前置、声道短感、MTF训练友好度、模仿可塑性里最关键的 5-8 项。

## 训练优先级
给 3 条具体建议，每条说明为什么。

## 下一次复测建议
告诉用户下一次应该录什么、怎么对比。

用户目标：{user_goal or "未填写，按通用声线优化目标处理"}

声音鉴定标签候选：{profile.get("voice_identity", {}).get("primary_label")}
文学描述草稿：{profile.get("voice_identity", {}).get("literary_description")}

声学数据 JSON：
{json.dumps(profile, ensure_ascii=False, indent=2)}
""".strip()


def build_chat_prompt(profile: dict, report: str, history: list[dict], user_message: str) -> str:
    compact_history = history[-8:]
    return f"""
你是声音分析产品里的持续对话教练。用户已经拿到一份声音报告，现在继续追问。

回答要求：
1. 回答要基于这份声学数据和报告，不要凭空捏造。
2. 如果用户问训练方法，给具体练习。
3. 如果用户问标签是否准确，要解释依据和不确定性。
4. 不做医学诊断，不判断真实生理性别或身份。

声学数据：
{json.dumps(profile, ensure_ascii=False, indent=2)}

已有报告：
{report}

最近对话：
{json.dumps(compact_history, ensure_ascii=False, indent=2)}

用户新问题：{user_message}
""".strip()


def generate_ai_report(
    profile: dict,
    user_goal: str,
    provider: Provider,
    api_key: str = "",
    model: str = "",
) -> tuple[str, str]:
    if provider == "OpenAI":
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        selected_model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        if key:
            return _openai_text(build_report_prompt(profile, user_goal), key, selected_model), f"OpenAI / {selected_model}"

    if provider == "Gemini":
        key = api_key or os.getenv("GEMINI_API_KEY", "")
        selected_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if key:
            return _gemini_text(build_report_prompt(profile, user_goal), key, selected_model), f"Gemini / {selected_model}"

    return generate_rule_report(profile, user_goal=user_goal), "本地规则"


def generate_chat_reply(
    profile: dict,
    report: str,
    history: list[dict],
    user_message: str,
    provider: Provider,
    api_key: str = "",
    model: str = "",
) -> tuple[str, str]:
    if provider == "OpenAI":
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        selected_model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        if key:
            return _openai_text(build_chat_prompt(profile, report, history, user_message), key, selected_model), f"OpenAI / {selected_model}"

    if provider == "Gemini":
        key = api_key or os.getenv("GEMINI_API_KEY", "")
        selected_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if key:
            return _gemini_text(build_chat_prompt(profile, report, history, user_message), key, selected_model), f"Gemini / {selected_model}"

    label = profile.get("voice_identity", {}).get("primary_label", "当前声线")
    return (
        f"从本地规则看，你现在更接近“{label}”。如果要继续细化，建议先围绕 F0 稳定度、F2-F4 前置程度和 HNR 做复测。你可以问我：怎么降低气泡感、怎么更少女、怎么更自然、怎么做下一次录音对比。",
        "本地规则",
    )


def _openai_text(prompt: str, api_key: str, model: str) -> str:
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "input": prompt},
        timeout=45,
    )
    _raise_safe_api_error(response, "OpenAI")
    data = response.json()
    if data.get("output_text"):
        return data["output_text"]

    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("text"):
                chunks.append(content["text"])
    return "\n".join(chunks).strip() or "OpenAI 返回了空内容。"


def _gemini_text(prompt: str, api_key: str, model: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    response = requests.post(
        url,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=45,
    )
    _raise_safe_api_error(response, "Gemini")
    data = response.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "\n".join(part.get("text", "") for part in parts).strip() or "Gemini 返回了空内容。"


def _raise_safe_api_error(response: requests.Response, provider: str) -> None:
    if response.status_code < 400:
        return

    reason = response.reason or "请求失败"
    if response.status_code == 503:
        detail = "服务暂时不可用，可能是模型繁忙、区域访问受限，或当前模型暂不可用。"
    elif response.status_code in {401, 403}:
        detail = "Key 无效、权限不足，或当前账号没有这个模型的访问权限。"
    elif response.status_code == 404:
        detail = "模型名或接口地址不可用，请换一个模型名。"
    elif response.status_code == 429:
        detail = "请求过多或额度不足，请稍后再试。"
    else:
        detail = reason

    raise RuntimeError(f"{provider} API 调用失败：HTTP {response.status_code}。{detail}")
