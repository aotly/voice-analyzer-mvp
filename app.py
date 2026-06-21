from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st

from ai_reports import generate_ai_report, generate_chat_reply
from voice_core import VOICE_DIMENSIONS, analyze_uploaded_bytes


LOCAL = "本地规则"


def fmt_value(value: object, suffix: str = "") -> str:
    if value is None:
        return "待上传"
    return f"{value}{suffix}"


def build_placeholder_profile() -> dict:
    explanations = {
        name: {
            "circle_terms": "待上传",
            "evaluates": "待上传",
            "score": None,
            "level": "待上传",
        }
        for name in VOICE_DIMENSIONS
    }
    return {
        "dimension_scores": {name: 0 for name in VOICE_DIMENSIONS},
        "dimension_explanations": explanations,
        "pitch": {"mean_hz": None, "min_hz": None, "max_hz": None, "range_hz": None, "std_hz": None},
        "formants": {"f1": None, "f2": None, "f3": None, "f4": None},
        "formant_spacing_hz": None,
        "hnr_db": None,
        "layers": {
            "physical": {
                "F0_mean": None,
                "F0_min": None,
                "F0_max": None,
                "F1": None,
                "F2": None,
                "F3": None,
                "F4": None,
                "Formant dispersion": None,
                "CPP": None,
                "HNR": None,
                "Jitter": None,
                "Shimmer": None,
            }
        },
        "pitch_series": [],
        "recording_quality": None,
    }


def dimension_level(score: int, low: str, mid: str, high: str) -> str:
    if score < 40:
        return low
    if score < 68:
        return mid
    return high


def score_status(score: int | None, has_audio: bool) -> str:
    if not has_audio or score is None:
        return "上传音频后生成"
    if score < 35:
        return "偏低"
    if score < 68:
        return "中等"
    if score < 86:
        return "偏高"
    return "很高"


def build_overview(profile: dict, has_audio: bool) -> dict[str, object]:
    if not has_audio:
        return {
            "portrait": "等待上传",
            "sentence": "上传一段 wav/flac 音频后，声镜会生成声线画像、12 维声学特征图、专业参数和 AI 持续对话。",
            "chips": ["等待音频", "12维评分", "F0-F4", "AI问答"],
            "age_label": "待分析",
            "age_band": "--",
        }

    scores = profile["dimension_scores"]
    bright = scores.get("明亮感", 0)
    thin = scores.get("轻薄感", 0)
    breath = scores.get("气息感", 0)
    front = scores.get("共鸣前置感", 0)
    stable = scores.get("稳定自然度", 0)
    age = scores.get("声音年轻感", 0)
    rough = scores.get("颗粒感/质感", 0)
    flexible = scores.get("声音可塑性", 0)

    if bright >= 65 and thin >= 60:
        portrait = "明亮轻质型"
    elif breath >= 65 and thin >= 55:
        portrait = "气声轻质型"
    elif rough >= 65:
        portrait = "颗粒质感型"
    elif stable >= 70:
        portrait = "稳定自然型"
    else:
        portrait = "均衡表达型"

    age_band = "18 - 25" if age >= 72 else "22 - 32" if age >= 55 else "28 - 40" if age >= 38 else "35+"
    age_label = "年轻化偏高" if age >= 72 else "偏年轻" if age >= 55 else "成熟均衡" if age >= 38 else "成熟化明显"
    chips = [
        ("明亮感较强", bright),
        ("轻质感明显", thin),
        ("共鸣偏前", front),
        ("气声边缘较显", breath),
        ("清亮自然", stable),
        ("可塑性较强", flexible),
    ]
    top_chips = [name for name, _ in sorted(chips, key=lambda item: item[1], reverse=True)[:5]]
    sentence = f"这是一条偏{dimension_level(bright, '暗厚', '均衡', '明亮')}、偏{dimension_level(thin, '厚实', '均衡', '轻质')}的声音，共鸣{dimension_level(front, '偏后', '居中', '较前')}，气声边缘{dimension_level(breath, '较少', '适中', '明显')}，整体更接近清亮自然的表达方向。"
    return {"portrait": portrait, "sentence": sentence, "chips": top_chips, "age_label": age_label, "age_band": age_band}


def professional_items(profile: dict) -> list[tuple[str, str]]:
    physical = profile["layers"]["physical"]
    return [
        ("F0", fmt_value(physical.get("F0_mean"), " Hz")),
        ("F0 范围", f"{fmt_value(physical.get('F0_min'), ' Hz')} - {fmt_value(physical.get('F0_max'), ' Hz')}"),
        ("F1", fmt_value(physical.get("F1"), " Hz")),
        ("F2", fmt_value(physical.get("F2"), " Hz")),
        ("F3", fmt_value(physical.get("F3"), " Hz")),
        ("F4", fmt_value(physical.get("F4"), " Hz")),
        ("Formant Spacing", fmt_value(physical.get("Formant dispersion"), " Hz")),
        ("CPP", fmt_value(physical.get("CPP"), " dB")),
        ("HNR", fmt_value(physical.get("HNR"), " dB")),
        ("Jitter", fmt_value(physical.get("Jitter"), " %")),
        ("Shimmer", fmt_value(physical.get("Shimmer"), " %")),
    ]


def microphone_input(label: str):
    audio_input = getattr(st, "audio_input", None)
    if audio_input:
        return audio_input(label, sample_rate=44100, key="recorded_audio")

    experimental_audio_input = getattr(st, "experimental_audio_input", None)
    if experimental_audio_input:
        return experimental_audio_input(label, key="recorded_audio")

    st.info("当前 Streamlit 版本暂不支持网页录音，请先用上传音频。")
    return None


def quality_cards(profile: dict, has_audio: bool) -> str:
    quality = profile.get("recording_quality") if has_audio else None
    if not quality:
        recording_score = "--"
        device_score = "--"
        recording_label = "待录制"
        device_label = "待评估"
        issues = ["录音或上传后生成录音质量与设备质量评估。"]
        tips = ["建议 10-30 秒，安静环境，手机或麦克风离嘴 15-25 cm。"]
        metrics = {}
    else:
        recording_score = quality["recording_score"]
        device_score = quality["device_score"]
        recording_label = quality["recording_label"]
        device_label = quality["device_label"]
        issues = quality["issues"]
        tips = quality["tips"]
        metrics = quality["metrics"]

    issue_items = "".join(f"<li>{item}</li>" for item in issues[:3])
    tip_items = "".join(f"<li>{item}</li>" for item in tips[:3])
    metrics_text = " · ".join(
        item
        for item in [
            f"{metrics.get('duration_sec')} 秒" if metrics.get("duration_sec") else "",
            f"{metrics.get('sample_rate_hz')} Hz" if metrics.get("sample_rate_hz") else "",
            f"音量 {metrics.get('rms_dbfs')} dBFS" if metrics.get("rms_dbfs") is not None else "",
            f"削波 {metrics.get('clipping_percent')}%" if metrics.get("clipping_percent") is not None else "",
        ]
        if item
    )
    metrics_html = f"<div class='quality-metrics'>{metrics_text}</div>" if metrics_text else ""

    return f"""
    <div class="quality-grid">
      <div class="quality-card">
        <div class="quality-label">录音质量</div>
        <div class="quality-score">{recording_score}<span>{recording_label}</span></div>
        <ul>{issue_items}</ul>
      </div>
      <div class="quality-card">
        <div class="quality-label">设备质量</div>
        <div class="quality-score">{device_score}<span>{device_label}</span></div>
        <ul>{tip_items}</ul>
      </div>
    </div>
    {metrics_html}
    """


def secret_or_env(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, default) or "")


def resolve_ai_config() -> tuple[str, str, str]:
    preferred = secret_or_env("VOICESCOPE_PROVIDER")
    openai_key = secret_or_env("OPENAI_API_KEY")
    gemini_key = secret_or_env("GEMINI_API_KEY")

    if preferred == "OpenAI" and openai_key:
        return "OpenAI", openai_key, secret_or_env("OPENAI_MODEL", "gpt-5-mini")
    if preferred == "Gemini" and gemini_key:
        return "Gemini", gemini_key, secret_or_env("GEMINI_MODEL", "gemini-2.5-flash")
    if openai_key:
        return "OpenAI", openai_key, secret_or_env("OPENAI_MODEL", "gpt-5-mini")
    if gemini_key:
        return "Gemini", gemini_key, secret_or_env("GEMINI_MODEL", "gemini-2.5-flash")
    return LOCAL, "", ""


st.set_page_config(page_title="声镜 VoiceScope", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 24% -10%, rgba(34, 160, 151, .12), transparent 28%),
            linear-gradient(180deg, #fbfaf7 0%, #f3f0ea 100%);
        color: #202322;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #152123 0%, #0d1415 100%);
        border-right: 1px solid rgba(255,255,255,.05);
        min-width: 260px;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stCaption {
        color: #f7f1e8 !important;
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: #fffdf8 !important;
        color: #1f2827 !important;
        border-color: #e5ddd4 !important;
    }
    section[data-testid="stSidebar"] input::placeholder,
    section[data-testid="stSidebar"] textarea::placeholder {
        color: #7f8784 !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] button {
        color: #f7f1e8 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button * {
        color: #1f2827 !important;
    }
    section[data-testid="stSidebar"] button:disabled,
    section[data-testid="stSidebar"] button[disabled] {
        background: #f4f3f2 !important;
        color: #8d8175 !important;
        border-color: #e1d8cf !important;
    }
    .block-container { padding: 1.35rem 1.45rem 2.4rem; max-width: 1480px; }
    .capture-card {
        background: rgba(255, 254, 250, .96);
        border: 1px solid #e7ddd1;
        border-radius: 12px;
        padding: 16px 18px 12px;
        margin-bottom: 14px;
        box-shadow: 0 12px 26px rgba(35, 30, 24, .05);
    }
    .capture-title {
        font-size: 17px;
        font-weight: 850;
        margin-bottom: 2px;
        color: #202322;
    }
    .capture-copy {
        color: #626a67;
        font-size: 13px;
        margin-bottom: 8px;
        line-height: 1.55;
    }
    .vs-card {
        background: rgba(255, 254, 250, .94);
        border: 1px solid #e7ddd1;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 16px 34px rgba(35, 30, 24, .06);
    }
    .overview {
        display: grid;
        grid-template-columns: 92px 1fr 270px;
        gap: 24px;
        align-items: center;
        margin-bottom: 14px;
    }
    .overview > div { min-width: 0; }
    .sound-icon {
        width: 78px; height: 78px; border-radius: 50%;
        background: #f2eee7; border: 1px solid #e6ddd2;
        display: flex; align-items: center; justify-content: center;
        color: #0f8d8a; font-size: 34px; font-weight: 800;
    }
    .headline {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 8px;
        word-break: keep-all;
        overflow-wrap: normal;
        line-height: 1.25;
    }
    .headline span { color: #0f8d8a; }
    .overview-copy { color: #3f4544; font-size: 15px; line-height: 1.75; }
    .chip {
        display: inline-block; padding: 7px 14px; border-radius: 999px;
        margin: 14px 8px 0 0; font-weight: 650; font-size: 14px;
        background: #e8f4f1; color: #0f7773;
    }
    .chip:nth-child(2n) { background: #f6eadc; color: #a95f2d; }
    .chip:nth-child(3n) { background: #fbe9e2; color: #b95748; }
    .age-card {
        border: 1px solid #e7ddd2; background: #faf7f1; border-radius: 10px;
        padding: 17px 18px; min-height: 124px;
    }
    .age-title { color: #68706d; font-size: 14px; font-weight: 700; }
    .age-main { color: #202322; font-size: 24px; font-weight: 800; margin-top: 8px; }
    .age-sub { color: #646c69; margin-top: 6px; font-size: 14px; }
    .section-title { font-size: 17px; font-weight: 800; margin: 0 0 14px 0; }
    .score-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
    }
    .score-pill {
        border: 1px solid #e8ded3;
        background: linear-gradient(180deg, #fffefd 0%, #faf6ef 100%);
        border-radius: 10px;
        padding: 9px 12px;
    }
    .score-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
    }
    .score-name { color: #28302e; font-size: 13px; font-weight: 760; }
    .score-value { color: #0e8f89; font-size: 19px; font-weight: 850; }
    .score-status { color: #7b7167; font-size: 12px; margin-top: 4px; }
    .score-bar {
        height: 5px;
        border-radius: 999px;
        background: #eee7dc;
        overflow: hidden;
        margin-top: 7px;
    }
    .score-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #8bc7c0, #0e908a);
    }
    .quality-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }
    .quality-card {
        border: 1px solid #e8ded3;
        background: linear-gradient(180deg, #fffefd 0%, #faf6ef 100%);
        border-radius: 10px;
        padding: 13px 14px;
    }
    .quality-label { color: #68706d; font-size: 13px; font-weight: 800; }
    .quality-score {
        margin-top: 4px;
        color: #0e8f89;
        font-size: 28px;
        font-weight: 900;
        line-height: 1.15;
    }
    .quality-score span {
        color: #202322;
        font-size: 14px;
        font-weight: 800;
        margin-left: 8px;
    }
    .quality-card ul {
        margin: 9px 0 0 18px;
        padding: 0;
        color: #59615f;
        font-size: 13px;
        line-height: 1.55;
    }
    .quality-metrics {
        color: #776f65;
        font-size: 12px;
        margin-top: 8px;
    }
    .param-grid {
        display: grid; grid-template-columns: repeat(11, minmax(92px, 1fr));
        border: 1px solid #e7ded3; border-radius: 10px; overflow: hidden; background: #fffdf8;
    }
    .param-cell { padding: 13px 10px; text-align: center; border-right: 1px solid #ebe3d9; }
    .param-cell:last-child { border-right: none; }
    .param-name { font-size: 12px; color: #39413f; font-weight: 800; }
    .param-value { margin-top: 8px; font-size: 13px; color: #28302e; }
    .query-chip {
        border: 1px solid #dcd3c9; background: #fffdf8; border-radius: 7px;
        padding: 12px 14px; color: #45504d; font-size: 14px;
    }
    .tiny-note { color: #776f65; font-size: 13px; }
    .spacer { height: 14px; }
    @media (max-width: 980px) {
        .overview { grid-template-columns: 82px 1fr; align-items: start; }
        .age-card { grid-column: 1 / -1; }
        .headline { font-size: 26px; }
        .score-grid { grid-template-columns: 1fr; }
        .quality-grid { grid-template-columns: 1fr; }
        .param-grid { grid-template-columns: repeat(4, minmax(112px, 1fr)); }
    }
    @media (max-width: 560px) {
        .block-container { padding: 2.15rem .58rem 1.6rem; }
        section[data-testid="stSidebar"] {
            min-width: min(88vw, 360px) !important;
            width: min(88vw, 360px) !important;
        }
        .capture-card, .vs-card { border-radius: 10px; padding: 14px; }
        .overview { grid-template-columns: 1fr; gap: 14px; }
        .sound-icon { width: 64px; height: 64px; font-size: 28px; }
        .headline { font-size: 22px; overflow-wrap: anywhere; }
        .overview-copy { font-size: 14px; line-height: 1.65; }
        .chip { margin: 9px 5px 0 0; padding: 6px 11px; font-size: 13px; }
        .age-main { font-size: 22px; }
        .quality-score { font-size: 24px; }
        .param-grid { grid-template-columns: repeat(2, minmax(112px, 1fr)); }
        [data-testid="stHorizontalBlock"] { gap: .65rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="capture-card">
      <div class="capture-title">录音或上传音频</div>
      <div class="capture-copy">手机端优先用“录音”，电脑端也可以上传 wav/flac/aiff/aif。完成后会自动分析声线、设备质量和录音质量。</div>
      <div class="tiny-note">云端版本：2026-06-21-2</div>
    </div>
    """,
    unsafe_allow_html=True,
)
record_tab, upload_tab = st.tabs(["直接录音", "上传文件"])
with record_tab:
    recorded = microphone_input("点击录音，建议 10-30 秒")
with upload_tab:
    uploaded = st.file_uploader("上传音频", type=["wav", "flac", "aiff", "aif"], key="uploaded_audio")

with st.expander("分析设置", expanded=False):
    user_goal = st.text_area("目标声线", placeholder="例如：更自然女声、更清亮、更轻薄、更适合配音...", height=88)
    analysis_context = st.selectbox(
        "分析语境",
        ["日常对话（安静环境）", "女声/女性用户", "男声/男性用户", "中性声线", "MTF声音训练", "FTM声音训练", "配音/角色声线"],
        index=0,
    )
    provider, api_key, model = resolve_ai_config()
    st.caption("AI 报告由后台配置；未配置云端 Key 时自动使用本地规则。建议 10-30 秒，安静环境，正常说话或朗读同一段文本。")

audio_input = recorded or uploaded
audio_name = getattr(audio_input, "name", "recorded-audio.wav") if audio_input is not None else ""
audio_suffix = ".wav"
if uploaded is not None and "." in uploaded.name:
    audio_suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()

has_audio = audio_input is not None
profile = build_placeholder_profile()
analysis_error = None

if has_audio:
    try:
        profile = analyze_uploaded_bytes(audio_input.getvalue(), suffix=audio_suffix)
    except Exception as exc:
        analysis_error = exc
        has_audio = False

if analysis_error:
    st.error("音频分析失败。当前先显示占位工作台；建议用 wav/flac、10-30 秒、噪声较小的语音重试。")
    st.caption(str(analysis_error))

overview = build_overview(profile, has_audio)
chips_html = "".join(f"<span class='chip'>{chip}</span>" for chip in overview["chips"])

st.markdown(
    f"""
    <div class="vs-card overview">
      <div class="sound-icon">▥</div>
      <div>
        <div class="headline">声线画像：<span>{overview["portrait"]}</span></div>
        <div class="overview-copy">{overview["sentence"]}</div>
        <div>{chips_html}</div>
      </div>
      <div class="age-card">
        <div class="age-title">声学年龄听感</div>
        <div class="age-main">{overview["age_label"]}</div>
        <div class="age-sub">听感区间 {overview["age_band"]}</div>
        <div class="tiny-note">非真实年龄判断</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("<div class='section-title'>录音设备与录音质量</div>", unsafe_allow_html=True)
    st.markdown(quality_cards(profile, has_audio), unsafe_allow_html=True)

st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
left, right = st.columns([0.95, 1.65])

with left:
    with st.container(border=True):
        st.markdown("<div class='section-title'>多维声学特征雷达图（12 维度）</div>", unsafe_allow_html=True)
        radar_values = [profile["dimension_scores"].get(name, 0) for name in VOICE_DIMENSIONS]
        radar_fig = go.Figure(
            data=[
                go.Scatterpolar(
                    r=radar_values + [radar_values[0]],
                    theta=VOICE_DIMENSIONS + [VOICE_DIMENSIONS[0]],
                    fill="toself",
                    line_color="#128c8a",
                    fillcolor="rgba(18, 140, 138, 0.16)",
                )
            ]
        )
        radar_fig.update_layout(
            height=470,
            margin=dict(l=36, r=36, t=18, b=18),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(255,255,255,0)",
                radialaxis=dict(range=[0, 100], tickfont=dict(size=10, color="#818984"), gridcolor="#d9ded8"),
                angularaxis=dict(tickfont=dict(size=11, color="#687370"), gridcolor="#e6e1d8"),
            ),
            showlegend=False,
        )
        st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

with right:
    with st.container(border=True):
        st.markdown("<div class='section-title'>多维声学特征评分（12 维度）</div>", unsafe_allow_html=True)
        score_cards = []
        for dim in VOICE_DIMENSIONS:
            raw_score = profile["dimension_scores"].get(dim)
            score = int(raw_score) if has_audio and raw_score is not None else None
            value = str(score) if score is not None else "--"
            width = score if score is not None else 0
            score_cards.append(
                f'<div class="score-pill">'
                f'<div class="score-top">'
                f'<div class="score-name">{dim}</div>'
                f'<div class="score-value">{value}</div>'
                f'</div>'
                f'<div class="score-status">{score_status(score, has_audio)}</div>'
                f'<div class="score-bar"><div class="score-fill" style="width:{width}%"></div></div>'
                f'</div>'
            )
        st.markdown(f"<div class='score-grid'>{''.join(score_cards)}</div>", unsafe_allow_html=True)

st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
with st.expander("专业声学参数（点击展开）", expanded=True):
    cells = "".join(
        f"<div class='param-cell'><div class='param-name'>{name}</div><div class='param-value'>{value}</div></div>"
        for name, value in professional_items(profile)
    )
    st.markdown(f"<div class='param-grid'>{cells}</div>", unsafe_allow_html=True)

st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>AI 持续对话 <span class='tiny-note'>向 AI 提问，深入理解你的声音，并获取个性化建议。</span></div>", unsafe_allow_html=True)
quick_cols = st.columns(4)
quick_questions = [
    "我的声音适合做哪些类型的内容？",
    "如何让声音更通透自然？",
    "如何让中高频更明亮但不刺耳？",
    "怎样提高声音的稳定度？",
]
for col, text in zip(quick_cols, quick_questions):
    col.markdown(f"<div class='query-chip'>{text}</div>", unsafe_allow_html=True)

report_goal = f"分析语境：{analysis_context}\n目标声线：{user_goal}"
if has_audio:
    report_key = f"report::{audio_name}::{provider}::{model}::{report_goal}"
    if st.session_state.get("report_key") != report_key:
        try:
            report, report_source = generate_ai_report(profile, report_goal, provider, api_key, model)
        except Exception as exc:
            st.warning("AI 报告调用失败，已退回本地规则报告。")
            st.caption(str(exc).split("?key=", 1)[0])
            report, report_source = generate_ai_report(profile, report_goal, LOCAL)
        st.session_state.report_key = report_key
        st.session_state.report = report
        st.session_state.report_source = report_source
        st.session_state.voice_chat = []

    with st.expander("查看 AI 完整报告", expanded=False):
        st.caption(f"报告来源：{st.session_state.report_source}")
        st.markdown(st.session_state.report)

    for message in st.session_state.get("voice_chat", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("输入你的问题，例如：如何让我的声音更清亮自然？")
    if question:
        st.session_state.voice_chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            reply, chat_source = generate_chat_reply(
                profile,
                st.session_state.report,
                st.session_state.voice_chat,
                f"{report_goal}\n\n用户追问：{question}",
                provider,
                api_key,
                model,
            )
        except Exception as exc:
            reply, chat_source = generate_chat_reply(profile, st.session_state.report, st.session_state.voice_chat, question, LOCAL)
            st.caption(str(exc).split("?key=", 1)[0])
        st.session_state.voice_chat.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.caption(f"来源：{chat_source}")
            st.markdown(reply)
else:
        st.text_input("输入你的问题，例如：如何让我的声音更清亮自然？", disabled=True, placeholder="上传音频后即可开始对话")

st.markdown("<p class='tiny-note'>提示：本产品只提供声学倾向和训练反馈，不用于医学诊断、生理性别判断或身份判定。</p>", unsafe_allow_html=True)
