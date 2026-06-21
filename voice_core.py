from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


GIRL = "\u5c11\u5973\u7cfb\u76f8\u4f3c\u5ea6"
BOY = "\u5c11\u5e74\u7cfb\u76f8\u4f3c\u5ea6"
YOUNG = "\u5e7c\u6001\u97f3\u8272"
MATURE = "\u8f7b\u5fa1\u7cfb\u76f8\u4f3c\u5ea6"
THIN = "\u8f7b\u8584\u611f"
THICK = "\u539a\u5ea6\u611f"
CLEAR = "\u6e05\u4eae\u611f"
BREATH = "\u6c14\u58f0\u611f"
ROUGH = "\u6c14\u6ce1/\u7c97\u7cd9\u611f"
FRONT = "\u5171\u9e23\u524d\u7f6e"
SHORT_TRACT = "\u58f0\u9053\u77ed\u611f"
ACTIVE = "\u97f3\u9ad8\u6d3b\u8dc3\u5ea6"
STABLE = "\u97f3\u9ad8\u7a33\u5b9a\u5ea6"
FEMININE = "\u5973\u6027\u611f"
MASCULINE = "\u7537\u6027\u611f"
NEUTRAL = "\u4e2d\u6027\u611f"
MTF = "MTF\u8bad\u7ec3\u53cb\u597d\u5ea6"
MASC_TRACE = "\u58f0\u5b66\u7537\u6027\u5316\u7279\u5f81"
IMITATION = "\u6a21\u4eff\u53ef\u5851\u6027"

ATTRIBUTE_CATALOG = {
    "\u7269\u7406\u5c42": [
        "F0", "F0\u8303\u56f4", "F0\u6ce2\u52a8", "F1", "F2", "F3", "F4", "F1-F2 spacing",
        "F2-F3 spacing", "F3-F4 spacing", "Formant dispersion", "HNR", "Jitter", "Shimmer",
        "CPP", "Spectral Tilt", "LTAS", "\u9891\u8c31\u91cd\u5fc3", "\u9ad8\u9891\u80fd\u91cf", "\u4f4e\u9891\u80fd\u91cf",
    ],
    "\u611f\u77e5\u5c42": [
        "\u5973\u6027\u542c\u611f", "\u7537\u6027\u542c\u611f", "\u4e2d\u6027\u542c\u611f", "\u97f3\u8272\u5e74\u8f7b\u5ea6",
        "\u5e7c\u6001\u97f3\u8272", "\u9752\u5e74\u97f3\u8272", "\u6210\u719f\u97f3\u8272", "\u8f7b\u8584\u611f", "\u539a\u5ea6\u611f",
        "\u6e05\u4eae\u611f", "\u901a\u900f\u611f", "\u9510\u5229\u611f", "\u67d4\u548c\u611f", "\u6c14\u58f0\u611f",
        "\u5171\u9e23\u4f4d\u7f6e", "\u5f00\u53e3\u5ea6", "\u9897\u7c92\u611f", "\u751c\u5ea6", "\u51b7\u611f",
        "\u538b\u8feb\u611f", "\u9f3b\u8154\u611f", "\u53e3\u8154\u660e\u4eae\u5ea6", "\u80f8\u8154\u539a\u5ea6",
        "\u54ac\u5b57\u67d4\u5ea6", "\u8bed\u6d41\u8f7b\u5feb\u5ea6", "\u6c14\u6ce1\u611f", "\u6c99\u611f",
    ],
    "\u7ed3\u6784\u5c42": [
        "\u58f0\u5e26\u8d28\u91cf", "\u58f0\u9053\u957f\u5ea6", "\u5589\u4f4d\u4e60\u60ef", "\u5171\u9e23\u524d\u7f6e",
        "\u58f0\u9053\u77ed\u611f", "\u95ed\u5408\u6548\u7387", "\u6f0f\u6c14\u7a0b\u5ea6", "\u58f0\u5b66\u7537\u6027\u5316\u7279\u5f81",
        "\u58f0\u5e26\u7537\u6027\u5316", "\u58f0\u9053\u7537\u6027\u5316", "\u5171\u9e23\u7537\u6027\u5316",
        "MTF\u8bad\u7ec3\u53cb\u597d\u5ea6", "\u6a21\u4eff\u53ef\u5851\u6027", "\u76ee\u6807\u8ddd\u79bb",
        "\u8bad\u7ec3\u74f6\u9888", "\u7a33\u5b9a\u6027", "\u75b2\u52b3\u98ce\u9669", "\u53ef\u6301\u7eed\u53d1\u58f0",
    ],
    "\u98ce\u683c\u5c42": [
        "\u4e3b\u6807\u7b7e", "\u526f\u6807\u7b7e", "\u98ce\u683c\u5f52\u7c7b", "\u58f0\u97f3DNA", "\u6210\u957f\u65b9\u5411",
        "\u5c11\u5973\u7cfb\u76f8\u4f3c\u5ea6", "\u5c11\u5e74\u7cfb\u76f8\u4f3c\u5ea6", "\u8f7b\u5fa1\u7cfb\u76f8\u4f3c\u5ea6",
        "\u52a8\u6f2b\u7cfb", "\u5e7f\u64ad\u5267\u7cfb", "\u751f\u6d3b\u7cfb", "\u4e3b\u64ad\u7cfb", "CV\u7cfb",
        "\u5076\u50cf\u7cfb", "\u751c\u59b9", "\u5143\u6c14", "\u6e05\u51b7", "\u75c5\u5f31", "\u77e5\u6027",
        "\u5fa1\u7cfb", "\u6b63\u592a", "\u4e2d\u6027\u5c11\u5e74", "\u9633\u5149", "\u6175\u61d2", "\u51b7\u5fa1",
        "\u8f7b\u5fa1", "\u70df\u55d3\u5fa1", "MTF\u5f0f\u5c11\u5973", "\u81ea\u7136\u751f\u6d3b\u5973\u58f0",
    ],
}

VOICE_DIMENSIONS = [
    "\u8f7b\u8584\u611f",
    "\u58f0\u97f3\u5973\u6027\u611f",
    "\u58f0\u97f3\u5e74\u8f7b\u611f",
    "\u660e\u4eae\u611f",
    "\u58f0\u97f3\u786c\u6717\u611f",
    "\u6c14\u606f\u611f",
    "\u5171\u9e23\u524d\u7f6e\u611f",
    "\u9897\u7c92\u611f/\u8d28\u611f",
    "\u58f0\u97f3\u6e29\u5ea6",
    "\u7a33\u5b9a\u81ea\u7136\u5ea6",
    "\u96c4\u6fc0\u7d20\u5316\u75d5\u8ff9",
    "\u58f0\u97f3\u53ef\u5851\u6027",
]

DIMENSION_META = {
    "\u8f7b\u8584\u611f": {
        "circle_terms": "\u8f7b\u3001\u8584\u3001\u5ae9\u3001\u5e72\u51c0\u3001\u4e0d\u538b\u55d3",
        "evaluates": "\u58f0\u5e26\u91cd\u91cf\u3001\u4f4e\u9891\u5e95\u76d8\u3001\u539a\u5ea6",
        "score_meaning": "\u5206\u6570\u8d8a\u9ad8\uff0c\u8d8a\u8f7b\u3001\u8d8a\u8584\u3001\u4f4e\u9891\u5e95\u76d8\u8d8a\u5c11\u3002",
    },
    "\u58f0\u97f3\u5973\u6027\u611f": {
        "circle_terms": "\u5973\u58f0\u611f\u3001\u7537\u58f0\u611f\u3001\u4e2d\u6027\u3001\u6027\u522b\u6a21\u7cca",
        "evaluates": "\u7efc\u5408 F0\u3001\u5171\u632f\u3001\u91cd\u91cf\u3001\u8bed\u6c14\u4e60\u60ef",
        "score_meaning": "\u5206\u6570\u8d8a\u9ad8\uff0c\u8d8a\u504f\u9ad8\u9891\u5973\u6027\u5316\u542c\u611f\uff1b\u4e0d\u4ee3\u8868\u771f\u5b9e\u6027\u522b\u3002",
    },
    "\u58f0\u97f3\u5e74\u8f7b\u611f": {
        "circle_terms": "\u5c11\u5e74\u611f\u3001\u5c11\u5973\u611f\u3001\u5e7c\u6001\u3001\u6210\u719f\u3001\u8001\u6210",
        "evaluates": "\u58f0\u97f3\u91cd\u91cf\u3001\u660e\u4eae\u5ea6\u3001\u8bed\u6c14\u5f39\u6027",
        "score_meaning": "\u5206\u6570\u8d8a\u9ad8\uff0c\u8d8a\u5e74\u8f7b\u5316\u3001\u8f7b\u5feb\uff1b\u4e0d\u4ee3\u8868\u771f\u5b9e\u5e74\u9f84\u3002",
    },
    "\u660e\u4eae\u611f": {
        "circle_terms": "\u4eae\u3001\u6e05\u3001\u900f\u3001\u8106\u3001\u524d\u7f6e",
        "evaluates": "\u9ad8\u9891\u5b58\u5728\u611f\u3001\u53e3\u8154\u524d\u7f6e\u3001\u6e05\u6670\u5ea6",
        "score_meaning": "\u5206\u6570\u8d8a\u9ad8\uff0c\u8d8a\u4eae\u3001\u8d8a\u6e05\u6670\u3001\u9ad8\u9891\u8d8a\u6d3b\u8dc3\u3002",
    },
    "\u58f0\u97f3\u786c\u6717\u611f": {
        "circle_terms": "\u8f6f\u3001\u786c\u3001\u9510\u3001\u7cef\u3001\u51b7\u3001\u653b\u51fb\u6027",
        "evaluates": "\u58f0\u97f3\u8fb9\u7f18\u3001\u95ed\u5408\u5f3a\u5ea6\u3001\u53d1\u58f0\u65b9\u5f0f",
        "score_meaning": "\u5206\u6570\u8d8a\u9ad8\uff0c\u8d8a\u786c\u3001\u8d8a\u9510\u3001\u8fb9\u7f18\u8d8a\u660e\u663e\u3002",
    },
    "\u6c14\u606f\u611f": {
        "circle_terms": "\u6c14\u58f0\u3001\u8d34\u8033\u3001\u865a\u3001\u98d8\u3001\u5b9e",
        "evaluates": "\u6f0f\u6c14\u6bd4\u4f8b\u3001\u95ed\u5408\u677e\u7d27\u3001\u4eb2\u5bc6\u611f",
        "score_meaning": "\u5206\u6570\u8d8a\u9ad8\uff0c\u6c14\u58f0\u8fb9\u7f18\u548c\u6f0f\u6c14\u611f\u8d8a\u660e\u663e\u3002",
    },
    "\u5171\u9e23\u524d\u7f6e\u611f": {
        "circle_terms": "\u524d\u3001\u540e\u3001\u9f3b\u3001\u80f8\u3001\u53e3\u8154\u3001\u5934\u8154",
        "evaluates": "\u58f0\u97f3\u5728\u8eab\u4f53/\u8154\u4f53\u91cc\u7684\u7a7a\u95f4\u4f4d\u7f6e",
        "score_meaning": "\u5206\u6570\u8d8a\u9ad8\uff0c\u8d8a\u504f\u524d\u7f6e\u3001\u53e3\u8154/\u5934\u8154\u4eae\u611f\u8d8a\u660e\u663e\u3002",
    },
    "\u9897\u7c92\u611f/\u8d28\u611f": {
        "circle_terms": "\u6c99\u3001\u54d1\u3001\u7cd9\u3001\u78e8\u7802\u3001\u4e1d\u6ed1",
        "evaluates": "\u566a\u58f0\u3001\u58f0\u5e26\u6469\u64e6\u3001\u97f3\u8272\u5e72\u51c0\u7a0b\u5ea6",
        "score_meaning": "\u5206\u6570\u8d8a\u9ad8\uff0c\u9897\u7c92\u3001\u6c99\u611f\u3001\u7c97\u7cd9\u611f\u8d8a\u660e\u663e\u3002",
    },
    "\u58f0\u97f3\u6e29\u5ea6": {
        "circle_terms": "\u51b7\u3001\u6e05\u51b7\u3001\u4e2d\u6027\u3001\u6e29\u67d4\u3001\u751c\u6696",
        "evaluates": "\u97f3\u8272\u7684\u51b7\u6696\u3001\u4eb2\u548c\u5ea6\u548c\u60c5\u7eea\u8272\u5f69",
        "score_meaning": "\u5206\u6570\u8d8a\u9ad8\u8d8a\u504f\u6e29\u6696\u4eb2\u548c\uff0c\u8d8a\u4f4e\u8d8a\u504f\u51b7\u611f\u6216\u758f\u79bb\u3002",
    },
    "\u7a33\u5b9a\u81ea\u7136\u5ea6": {
        "circle_terms": "\u7a33\u3001\u7aef\u7740\u3001\u5939\u3001\u88c5\u3001\u7528\u529b\u3001\u4e0d\u81ea\u7136",
        "evaluates": "\u957f\u6837\u672c\u7ef4\u6301\u80fd\u529b\u548c\u53d1\u58f0\u8d1f\u62c5",
        "score_meaning": "\u5206\u6570\u8d8a\u9ad8\uff0c\u8d8a\u7a33\u5b9a\u3001\u8d8a\u81ea\u7136\u3001\u8d8a\u4e0d\u50cf\u786c\u5939\u6216\u786c\u6491\u3002",
    },
    "\u96c4\u6fc0\u7d20\u5316\u75d5\u8ff9": {
        "circle_terms": "\u539a\u3001\u4f4e\u3001\u58f0\u9053\u957f\u3001\u80f8\u611f\u3001\u7c97\u7cd9\u3001\u538b\u55d3",
        "evaluates": "\u58f0\u5e26\u91cd\u91cf\u3001\u4f4e\u9891\u539a\u5ea6\u3001\u5171\u632f\u504f\u540e\u3001\u7c97\u7cd9\u5ea6",
        "score_meaning": "\u5206\u6570\u8d8a\u9ad8\uff0c\u8d8a\u6709\u4f4e\u539a\u3001\u504f\u540e\u3001\u7c97\u7cd9\u7b49\u58f0\u5b66\u75d5\u8ff9\uff1b\u8fd9\u4e0d\u662f\u8eab\u4f53\u6216\u8eab\u4efd\u5224\u65ad\u3002",
    },
    "\u58f0\u97f3\u53ef\u5851\u6027": {
        "circle_terms": "\u53ef\u7ec3\u3001\u4e0a\u9650\u9ad8\u3001\u53ef\u6a21\u4eff\u3001\u98ce\u683c\u53ef\u6269\u5c55",
        "evaluates": "\u97f3\u9ad8\u5f39\u6027\u3001\u7a33\u5b9a\u5ea6\u3001\u5171\u9e23\u53ef\u8c03\u7a0b\u5ea6\u3001\u566a\u58f0\u63a7\u5236",
        "score_meaning": "\u5206\u6570\u8d8a\u9ad8\uff0c\u8d8a\u9002\u5408\u505a\u98ce\u683c\u8bad\u7ec3\u3001\u76ee\u6807\u58f0\u7ebf\u6a21\u4eff\u548c\u957f\u671f\u590d\u6d4b\u3002",
    },
}


def _clean_numbers(values: Iterable[float | int | None]) -> list[float]:
    clean: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            clean.append(number)
    return clean


def summarize_pitch(values: Iterable[float | int | None]) -> dict[str, int | None]:
    voiced = _clean_numbers(values)
    if not voiced:
        return {"mean_hz": None, "min_hz": None, "max_hz": None, "range_hz": None, "std_hz": None}
    mean_hz = int(round(float(np.mean(voiced))))
    min_hz = int(round(min(voiced)))
    max_hz = int(round(max(voiced)))
    return {
        "mean_hz": mean_hz,
        "min_hz": min_hz,
        "max_hz": max_hz,
        "range_hz": max_hz - min_hz,
        "std_hz": int(round(float(np.std(voiced)))),
    }


def estimate_formant_spacing(formants: dict[str, float | int | None]) -> int | None:
    ordered = _clean_numbers([formants.get("f1"), formants.get("f2"), formants.get("f3"), formants.get("f4")])
    if len(ordered) < 2:
        return None
    gaps = [ordered[index + 1] - ordered[index] for index in range(len(ordered) - 1)]
    positive_gaps = [gap for gap in gaps if gap > 0]
    return int(round(float(np.mean(positive_gaps)))) if positive_gaps else None


def classify_pitch_feel(mean_hz: int | float | None) -> str:
    if mean_hz is None:
        return "\u97f3\u9ad8\u4fe1\u606f\u4e0d\u8db3"
    if mean_hz < 145:
        return "\u4f4e\u6c89/\u539a\u91cd\u503e\u5411"
    if mean_hz < 200:
        return "\u4e2d\u6027\u504f\u660e\u4eae\u503e\u5411"
    return "\u660e\u4eae/\u5973\u6027\u5316\u542c\u611f\u503e\u5411"


def classify_spacing_feel(spacing_hz: int | float | None) -> str:
    if spacing_hz is None:
        return "\u5171\u9e23\u4fe1\u606f\u4e0d\u8db3"
    if spacing_hz < 950:
        return "\u58f0\u9053\u8f83\u957f/\u5171\u9e23\u504f\u539a\u503e\u5411"
    if spacing_hz < 1150:
        return "\u4e2d\u6027\u5171\u9e23\u503e\u5411"
    return "\u58f0\u9053\u8f83\u77ed/\u5171\u9e23\u504f\u524d\u503e\u5411"


def _score(value: float, low: float, high: float) -> int:
    if high == low:
        return 0
    normalized = (value - low) / (high - low)
    return int(round(max(0, min(1, normalized)) * 100))


def _inverse_score(value: float, low: float, high: float) -> int:
    return 100 - _score(value, low, high)


def _mix(*weighted_values: tuple[int, float]) -> int:
    total_weight = sum(abs(weight) for _, weight in weighted_values)
    if total_weight == 0:
        return 0
    return int(round(sum(value * weight for value, weight in weighted_values) / total_weight))


def _clamp(value: int) -> int:
    return max(0, min(100, value))


def grade(value: int, labels: tuple[str, str, str, str, str]) -> str:
    if value < 20:
        return labels[0]
    if value < 40:
        return labels[1]
    if value < 60:
        return labels[2]
    if value < 80:
        return labels[3]
    return labels[4]


def build_concept_scores(
    pitch_summary: dict[str, int | None],
    formants: dict[str, int | None],
    spacing: int | None,
    hnr_db: int | None,
) -> dict[str, int]:
    mean = float(pitch_summary.get("mean_hz") or 0)
    pitch_range = float(pitch_summary.get("range_hz") or 0)
    pitch_std = float(pitch_summary.get("std_hz") or 0)
    f1 = float(formants.get("f1") or 0)
    f2 = float(formants.get("f2") or 0)
    f3 = float(formants.get("f3") or 0)
    f4 = float(formants.get("f4") or 0)
    spacing_value = float(spacing or 0)
    hnr = float(hnr_db or 0)

    pitch_bright = _score(mean, 150, 260)
    pitch_youth = _score(mean, 165, 285)
    range_energy = _score(pitch_range, 45, 170)
    front_resonance = _score(spacing_value, 880, 1230)
    short_tract = _score(spacing_value, 900, 1280)
    formant_brightness = _mix(
        (_score(f2, 1300, 2300), 0.35),
        (_score(f3, 2400, 3600), 0.35),
        (_score(f4, 3300, 4600), 0.3),
    )
    hnr_clean = _score(hnr, 8, 24)
    roughness = _inverse_score(hnr, 7, 20)
    instability = _score(pitch_std, 18, 65)

    thinness = _mix((pitch_bright, 0.35), (front_resonance, 0.25), (formant_brightness, 0.25), (_inverse_score(f1, 450, 900), 0.15))
    thickness = 100 - thinness
    youthful = _mix((pitch_youth, 0.45), (front_resonance, 0.25), (range_energy, 0.15), (roughness, -0.15))
    boyish = _mix((_score(mean, 150, 230), 0.35), (_score(f1, 500, 850), 0.25), (range_energy, 0.2), (roughness, 0.2))
    girlish = _mix((pitch_bright, 0.4), (front_resonance, 0.25), (formant_brightness, 0.3), (hnr_clean, 0.05))
    mature = _mix((_score(mean, 175, 245), 0.25), (_inverse_score(pitch_range, 60, 170), 0.15), (thickness, 0.25), (hnr_clean, 0.2), (_inverse_score(front_resonance, 40, 90), 0.15))
    feminine = _mix((girlish, 0.55), (front_resonance, 0.25), (thinness, 0.2))
    masculine = _mix((thickness, 0.4), (_inverse_score(spacing_value, 850, 1200), 0.35), (roughness, 0.25))
    neutral = 100 - abs(feminine - masculine)
    mtf_friendly = _mix((feminine, 0.45), (front_resonance, 0.25), (hnr_clean, 0.15), (_inverse_score(roughness, 30, 85), 0.15))

    return {
        GIRL: _clamp(girlish),
        BOY: _clamp(boyish),
        YOUNG: _clamp(youthful),
        MATURE: _clamp(mature),
        THIN: _clamp(thinness),
        THICK: _clamp(thickness),
        CLEAR: _clamp(formant_brightness),
        BREATH: _clamp(_mix((roughness, 0.55), (_inverse_score(hnr, 8, 22), 0.45))),
        ROUGH: _clamp(roughness),
        FRONT: _clamp(front_resonance),
        SHORT_TRACT: _clamp(short_tract),
        ACTIVE: _clamp(range_energy),
        STABLE: _clamp(100 - instability),
        FEMININE: _clamp(feminine),
        MASCULINE: _clamp(masculine),
        NEUTRAL: _clamp(neutral),
        MTF: _clamp(mtf_friendly),
        MASC_TRACE: _clamp(masculine),
        IMITATION: _clamp(_mix((range_energy, 0.35), (hnr_clean, 0.25), (front_resonance, 0.2), (_inverse_score(roughness, 35, 90), 0.2))),
    }


def make_voice_identity(concept_scores: dict[str, int]) -> dict:
    girl = concept_scores.get(GIRL, concept_scores.get("\u5c11\u5973\u611f", 0))
    boy = concept_scores.get(BOY, concept_scores.get("\u5c11\u5e74\u611f", 0))
    mature = concept_scores.get(MATURE, concept_scores.get("\u5fa1\u59d0/\u6210\u719f\u611f", 0))
    young = concept_scores.get(YOUNG, concept_scores.get("\u5e7c\u611f", 0))
    rough = concept_scores.get(ROUGH, concept_scores.get("\u6c14\u6ce1/\u7c97\u7cd9\u611f", 0))
    front = concept_scores.get(FRONT, concept_scores.get("\u5171\u9e23\u524d\u7f6e", 0))
    thin = concept_scores.get(THIN, 0)
    clear = concept_scores.get(CLEAR, 0)
    mtf = concept_scores.get(MTF, 0)
    masc = concept_scores.get(MASC_TRACE, 0)

    if girl >= 72 and boy >= 58:
        label = "\u4e2d\u6027\u5c11\u5973"
    elif girl >= 78 and mature >= 46:
        label = "\u5c11\u5fa1\u5c11\u5973"
    elif mature >= 62 and girl >= 55:
        label = "MTF\u5fa1\u59d0"
    elif girl >= 70:
        label = "\u6e05\u4eae\u5c11\u5973"
    elif boy >= 72 and young >= 62:
        label = "\u5e7c\u6001\u6b63\u592a"
    elif boy >= 65:
        label = "\u6c14\u6ce1\u5c11\u5e74" if rough >= 55 else "\u5e72\u51c0\u5c11\u5e74"
    elif rough >= 70 and front < 50:
        label = "\u6c14\u6ce1\u9752\u5e74"
    else:
        label = "\u4e2d\u6027\u6e05\u4eae"

    secondary = [
        grade(thin, ("\u539a\u91cd", "\u504f\u539a", "\u5747\u8861", "\u8f7b\u8584", "\u6781\u8f7b\u8584")),
        grade(clear, ("\u6d51\u6d4a", "\u666e\u901a", "\u6e05\u4eae", "\u901a\u900f", "\u6c34\u6676\u611f")),
        grade(rough, ("\u4e1d\u6ed1", "\u8f7b\u9897\u7c92", "\u9897\u7c92", "\u7802\u611f", "\u6c14\u6ce1\u6c99\u611f")),
        grade(front, ("\u540e\u7f6e\u5171\u9e23", "\u504f\u540e\u5171\u9e23", "\u4e2d\u6027\u5171\u9e23", "\u504f\u524d\u5171\u9e23", "\u9ad8\u524d\u7f6e\u5171\u9e23")),
        grade(young, ("\u6210\u719f", "\u9752\u5e74", "\u5e74\u8f7b", "\u504f\u5e7c", "\u5e7c\u6001")),
    ]
    if mtf >= 70:
        secondary.append("MTF\u9ad8\u6f5c\u529b")
    if masc >= 55:
        secondary.append("\u8f7b\u58f0\u5b66\u7537\u6027\u5316\u7279\u5f81")
    if front >= 60 and not any("\u524d\u7f6e" in tag for tag in secondary):
        secondary.append("\u9ad8\u524d\u7f6e\u5171\u9e23")
    if girl >= 72 and boy >= 58:
        secondary.append("\u5c11\u5973\u5c11\u5e74\u878d\u5408")
    if rough >= 70:
        secondary.append("\u62d6\u978b\u5f0f\u6c14\u6ce1\u8fb9")

    voice_dna = "V-" + "".join(
        [
            "N" if concept_scores.get(NEUTRAL, 0) >= 55 else ("F" if concept_scores.get(FEMININE, 0) > concept_scores.get(MASCULINE, 0) else "M"),
            "S" if thin >= 55 else "D",
            "P" if clear >= 55 else "M",
            "A" if rough >= 55 else "C",
        ]
    )
    style_family = "\u751f\u6d3b\u7cfb"
    if girl >= 70 and clear >= 60:
        style_family = "\u52a8\u6f2b\u7cfb / \u5076\u50cf\u7cfb"
    elif mature >= 60:
        style_family = "\u5e7f\u64ad\u5267\u7cfb / CV\u7cfb"
    elif boy >= 65:
        style_family = "\u5c11\u5e74\u7cfb / \u751f\u6d3b\u7cfb"

    if mtf >= 70:
        growth = "MTF\u81ea\u7136\u751f\u6d3b\u5973\u58f0 -> \u8f7b\u5fa1\u5973\u58f0"
    elif girl >= boy:
        growth = "\u81ea\u7136\u5c11\u5973\u97f3 -> \u5c11\u5fa1\u8fb9\u754c\u58f0"
    else:
        growth = "\u6e05\u723d\u5c11\u5e74\u97f3 -> \u4e2d\u6027\u9752\u5e74\u58f0"

    resonance = "\u5171\u9e23\u9760\u524d\uff0c\u4eae\u5ea6\u6bd4\u8f83\u660e\u663e" if front >= 55 else "\u5171\u9e23\u4f4d\u7f6e\u8f83\u5c45\u4e2d\uff0c\u539a\u5ea6\u8fd8\u7559\u7740"
    texture = "\u5e26\u4e00\u70b9\u7802\u783e\u548c\u6c14\u6ce1\u8fb9\u7f18" if rough >= 55 else "\u8fb9\u7f18\u5e72\u51c0\uff0c\u7ebf\u6761\u6e05\u695a"
    age = "\u504f\u5e7c\u3001\u504f\u8f7b" if young >= 65 else "\u5e74\u9f84\u611f\u4e0d\u91cd"
    literary = f"\u8fd9\u662f\u4e00\u6761{label}\uff1a{resonance}\uff0c{texture}\uff0c\u6574\u4f53{age}\u3002\u5b83\u50cf\u4e00\u675f\u88ab\u64e6\u4eae\u7684\u7ec6\u5149\uff0c\u5916\u5c42\u8f7b\uff0c\u91cc\u9762\u4ecd\u6709\u4e00\u70b9\u771f\u5b9e\u7684\u9897\u7c92\u611f\u3002"

    return {
        "primary_label": label,
        "secondary_tags": secondary[:7],
        "style_family": style_family,
        "voice_dna": voice_dna,
        "growth_direction": growth,
        "literary_description": literary,
        "style_disclaimer": "\u8fd9\u4e9b\u662f\u58f0\u7ebf\u98ce\u683c\u76f8\u4f3c\u5ea6\uff0c\u4e0d\u662f\u7528\u6237\u771f\u5b9e\u5e74\u9f84\u3001\u6027\u522b\u6216\u4eba\u683c\u5224\u65ad\u3002",
    }


def build_layers(
    pitch_summary: dict[str, int | None],
    formants: dict[str, int | None],
    spacing: int | None,
    hnr_db: int | None,
    concept_scores: dict[str, int],
    identity: dict,
) -> tuple[dict, dict[str, str]]:
    physical = {
        "F0_mean": pitch_summary.get("mean_hz"),
        "F0_min": pitch_summary.get("min_hz"),
        "F0_max": pitch_summary.get("max_hz"),
        "F0_range": pitch_summary.get("range_hz"),
        "F0_std": pitch_summary.get("std_hz"),
        "F1": formants.get("f1"),
        "F2": formants.get("f2"),
        "F3": formants.get("f3"),
        "F4": formants.get("f4"),
        "Formant dispersion": spacing,
        "HNR": hnr_db,
        "Jitter": None,
        "Shimmer": None,
        "CPP": None,
        "Spectral Tilt": None,
        "LTAS": None,
    }
    perceptual = {name: concept_scores[name] for name in [FEMININE, MASCULINE, NEUTRAL, GIRL, BOY, YOUNG, MATURE, THIN, CLEAR, BREATH, ROUGH, FRONT] if name in concept_scores}
    structural = {name: concept_scores[name] for name in [SHORT_TRACT, MTF, MASC_TRACE, IMITATION, STABLE, ACTIVE] if name in concept_scores}
    style = {
        "\u4e3b\u6807\u7b7e": identity["primary_label"],
        "\u526f\u6807\u7b7e": "\u3001".join(identity["secondary_tags"]),
        "\u98ce\u683c\u5f52\u7c7b": identity["style_family"],
        "\u58f0\u97f3DNA": identity["voice_dna"],
        "\u6210\u957f\u65b9\u5411": identity["growth_direction"],
    }
    layers = {"physical": physical, "perceptual": perceptual, "structural": structural, "style": style}
    summaries = {
        "\u7269\u7406\u5c42": "\u7ed9\u4e13\u4e1a\u7528\u6237\u770b\uff1aF0\u3001F1-F4\u3001spacing\u3001HNR\u7b49\u539f\u59cb\u4f20\u611f\u5668\u6570\u636e\u3002",
        "\u611f\u77e5\u5c42": "\u7ed9\u666e\u901a\u7528\u6237\u770b\uff1a\u8f7b\u8584\u611f\u3001\u6e05\u4eae\u611f\u3001\u6c14\u58f0\u611f\u3001\u5171\u9e23\u4f4d\u7f6e\u3001\u97f3\u8272\u5e74\u8f7b\u5ea6\u3002",
        "\u7ed3\u6784\u5c42": "\u63a8\u6d4b\u58f0\u9053\u3001\u5171\u9e23\u548c\u8bad\u7ec3\u53ef\u5851\u6027\uff0c\u53ea\u8868\u793a\u58f0\u5b66\u503e\u5411\u3002",
        "\u98ce\u683c\u5c42": "\u6700\u9002\u5408\u5206\u4eab\uff1a\u4e3b\u6807\u7b7e\u3001\u526f\u6807\u7b7e\u3001\u58f0\u97f3DNA\u3001\u6210\u957f\u65b9\u5411\u3002",
    }
    return layers, summaries


def build_dimension_scores(concept_scores: dict[str, int]) -> dict[str, int]:
    feminine = concept_scores.get(FEMININE, 0)
    masculine = concept_scores.get(MASCULINE, 0)
    neutral = concept_scores.get(NEUTRAL, 0)
    thin = concept_scores.get(THIN, 0)
    thick = concept_scores.get(THICK, 0)
    clear = concept_scores.get(CLEAR, 0)
    rough = concept_scores.get(ROUGH, 0)
    breath = concept_scores.get(BREATH, 0)
    front = concept_scores.get(FRONT, 0)
    stable = concept_scores.get(STABLE, 0)
    active = concept_scores.get(ACTIVE, 0)
    young = concept_scores.get(YOUNG, 0)
    mature = concept_scores.get(MATURE, 0)

    return {
        "\u8f7b\u8584\u611f": thin,
        "\u58f0\u97f3\u5973\u6027\u611f": _clamp(_mix((feminine, 0.5), (_inverse_score(masculine, 35, 85), 0.2), (neutral, 0.3))),
        "\u58f0\u97f3\u5e74\u8f7b\u611f": _clamp(_mix((young, 0.55), (_inverse_score(mature, 25, 80), 0.25), (active, 0.2))),
        "\u660e\u4eae\u611f": clear,
        "\u58f0\u97f3\u786c\u6717\u611f": _clamp(_mix((rough, 0.35), (thick, 0.25), (_inverse_score(breath, 25, 80), 0.15), (front, 0.25))),
        "\u6c14\u606f\u611f": breath,
        "\u5171\u9e23\u524d\u7f6e\u611f": front,
        "\u9897\u7c92\u611f/\u8d28\u611f": rough,
        "\u58f0\u97f3\u6e29\u5ea6": _clamp(_mix((clear, 0.35), (breath, 0.25), (thin, 0.2), (_inverse_score(rough, 35, 90), 0.2))),
        "\u7a33\u5b9a\u81ea\u7136\u5ea6": _clamp(_mix((stable, 0.45), (_inverse_score(breath, 35, 90), 0.2), (_inverse_score(rough, 35, 90), 0.2), (neutral, 0.15))),
        "\u96c4\u6fc0\u7d20\u5316\u75d5\u8ff9": concept_scores.get(MASC_TRACE, 0),
        "\u58f0\u97f3\u53ef\u5851\u6027": concept_scores.get(IMITATION, 0),
    }


def build_dimension_explanations(dimension_scores: dict[str, int]) -> dict[str, dict[str, str]]:
    explanations: dict[str, dict[str, str]] = {}
    for name in VOICE_DIMENSIONS:
        meta = DIMENSION_META[name]
        score = dimension_scores[name]
        explanations[name] = {
            **meta,
            "score": score,
            "level": grade(score, ("\u5f88\u4f4e", "\u504f\u4f4e", "\u4e2d\u7b49", "\u504f\u9ad8", "\u5f88\u9ad8")),
        }
    return explanations


def build_voice_profile(
    pitch_summary: dict[str, int | None],
    formants: dict[str, float | int | None],
    hnr_db: float | int | None = None,
) -> dict:
    rounded_formants = {key: _round_or_none(value) for key, value in formants.items()}
    mean_hz = pitch_summary.get("mean_hz")
    spacing = estimate_formant_spacing(rounded_formants)
    pitch_label = classify_pitch_feel(mean_hz)
    spacing_label = classify_spacing_feel(spacing)
    hnr_rounded = _round_or_none(hnr_db)
    concept_scores = build_concept_scores(pitch_summary, rounded_formants, spacing, hnr_rounded)
    dimension_scores = build_dimension_scores(concept_scores)
    dimension_explanations = build_dimension_explanations(dimension_scores)
    identity = make_voice_identity(concept_scores)
    layers, summaries = build_layers(pitch_summary, rounded_formants, spacing, hnr_rounded, concept_scores, identity)

    labels = [identity["primary_label"], *identity["secondary_tags"][:3], pitch_label, spacing_label]
    if hnr_rounded is not None and concept_scores.get(ROUGH, 0) >= 55:
        labels.append("\u6c14\u58f0\u6216\u7c97\u7cd9\u611f\u504f\u660e\u663e")

    return {
        "labels": labels,
        "voice_identity": identity,
        "scores": {
            "lightness": concept_scores[THIN],
            "front_resonance": concept_scores[FRONT],
            "stability": concept_scores[STABLE],
            "feminine_listening_feel": concept_scores[FEMININE],
        },
        "concept_scores": concept_scores,
        "dimension_scores": dimension_scores,
        "dimension_explanations": dimension_explanations,
        "layers": layers,
        "layer_summaries": summaries,
        "attribute_catalog": ATTRIBUTE_CATALOG,
        "pitch": pitch_summary,
        "formants": rounded_formants,
        "formant_spacing_hz": spacing,
        "hnr_db": hnr_rounded,
    }


def _round_or_none(value: float | int | None) -> int | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return int(round(float(value)))


def analyze_audio_file(file_path: str | Path) -> dict:
    import parselmouth

    sound = parselmouth.Sound(str(file_path))
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_summary = summarize_pitch(pitch_values)

    formant = sound.to_formant_burg()
    duration = sound.get_total_duration()
    sample_times = np.linspace(duration * 0.2, duration * 0.8, num=9) if duration > 0 else []
    formants: dict[str, int | None] = {}
    for formant_index in range(1, 5):
        values = []
        for sample_time in sample_times:
            value = formant.get_value_at_time(formant_index, float(sample_time))
            if value and math.isfinite(value):
                values.append(value)
        formants[f"f{formant_index}"] = _round_or_none(float(np.median(values))) if values else None

    hnr_db = None
    try:
        harmonicity = sound.to_harmonicity_cc()
        hnr_values = _clean_numbers(harmonicity.values.flatten())
        hnr_db = float(np.median(hnr_values)) if hnr_values else None
    except Exception:
        hnr_db = None

    profile = build_voice_profile(pitch_summary=pitch_summary, formants=formants, hnr_db=hnr_db)
    profile["duration_sec"] = round(duration, 2)
    profile["pitch_series"] = [
        {"time": round(float(time), 3), "hz": _round_or_none(value)}
        for time, value in zip(pitch.xs(), pitch_values)
        if value and value > 0
    ]
    return profile


def analyze_uploaded_bytes(data: bytes, suffix: str = ".wav") -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        temp_path = Path(tmp.name)
    try:
        return analyze_audio_file(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def generate_rule_report(profile: dict, user_goal: str = "") -> str:
    identity = profile.get("voice_identity", {})
    pitch = profile.get("pitch", {})
    formants = profile.get("formants", {})
    spacing = profile.get("formant_spacing_hz")
    concept_scores = profile.get("concept_scores", {})
    top_scores = sorted(concept_scores.items(), key=lambda item: item[1], reverse=True)[:8]
    top_text = "\u3001".join(f"{name}{score}\u5206" for name, score in top_scores)
    goal_sentence = f"\u76ee\u6807\u58f0\u7ebf\uff1a{user_goal}\u3002" if user_goal.strip() else "\u76ee\u6807\u58f0\u7ebf\uff1a\u672a\u586b\u5199\uff0c\u5148\u6309\u901a\u7528\u58f0\u97f3\u4f18\u5316\u7ed9\u5efa\u8bae\u3002"

    return "\n\n".join(
        [
            f"## \u58f0\u97f3\u9274\u5b9a\u6807\u7b7e\n{identity.get('primary_label', '')}",
            f"## \u58f0\u97f3DNA\n{identity.get('voice_dna', '')}\n\n\u526f\u6807\u7b7e\uff1a{'\u3001'.join(identity.get('secondary_tags', []))}",
            f"## \u6587\u5b66\u63cf\u8ff0\n{identity.get('literary_description', '')}",
            f"## \u5173\u952e\u58f0\u5b66\u6570\u636e\n\u5e73\u5747 F0\uff1a{pitch.get('mean_hz')} Hz\uff1bF0 \u8303\u56f4\uff1a{pitch.get('min_hz')}-{pitch.get('max_hz')} Hz\uff1bF1-F4\uff1a{formants.get('f1')} / {formants.get('f2')} / {formants.get('f3')} / {formants.get('f4')} Hz\uff1bformant spacing\uff1a{spacing} Hz\uff1bHNR\uff1a{profile.get('hnr_db')} dB\u3002",
            f"## \u6982\u5ff5\u8bc4\u5206\n{top_text}",
            f"## \u6210\u957f\u65b9\u5411\n{identity.get('growth_direction', '')}\n\n{goal_sentence}\u4e0b\u4e00\u6b65\u5efa\u8bae\u4f18\u5148\u770b HNR\u3001F0 \u7a33\u5b9a\u5ea6\u548c F2-F4 \u524d\u7f6e\u7a0b\u5ea6\uff0c\u4e0d\u8981\u53ea\u9760\u62ac\u9ad8 F0\u3002",
            "## \u514d\u8d23\u58f0\u660e\n\u8fd9\u662f\u58f0\u5b66\u503e\u5411\u548c\u542c\u611f\u8bad\u7ec3\u53cd\u9988\uff0c\u4e0d\u662f\u533b\u5b66\u8bca\u65ad\uff0c\u4e5f\u4e0d\u7528\u4e8e\u5224\u65ad\u751f\u7406\u6027\u522b\u6216\u8eab\u4efd\u3002",
        ]
    )
