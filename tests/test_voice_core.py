from voice_core import (
    ATTRIBUTE_CATALOG,
    FRONT,
    GIRL,
    IMITATION,
    MATURE,
    MTF,
    ROUGH,
    THIN,
    YOUNG,
    BOY,
    build_voice_profile,
    classify_pitch_feel,
    estimate_formant_spacing,
    make_voice_identity,
    assess_recording_quality,
    summarize_pitch,
    VOICE_DIMENSIONS,
)


def test_summarize_pitch_ignores_unvoiced_values():
    summary = summarize_pitch([0, 180, 190, None, 200])

    assert summary["mean_hz"] == 190
    assert summary["min_hz"] == 180
    assert summary["max_hz"] == 200
    assert summary["range_hz"] == 20


def test_estimate_formant_spacing_uses_adjacent_formants():
    spacing = estimate_formant_spacing({"f1": 550, "f2": 1650, "f3": 2750, "f4": 3850})

    assert spacing == 1100


def test_classify_pitch_feel_maps_common_ranges():
    assert classify_pitch_feel(115)
    assert classify_pitch_feel(175)
    assert classify_pitch_feel(235)


def test_build_voice_profile_combines_metrics_into_labels():
    profile = build_voice_profile(
        pitch_summary={"mean_hz": 225, "min_hz": 190, "max_hz": 260, "range_hz": 70, "std_hz": 24},
        formants={"f1": 600, "f2": 1800, "f3": 3000, "f4": 4200},
        hnr_db=22,
    )

    assert profile["labels"]
    assert profile["scores"]["lightness"] >= 60


def test_voice_profile_adds_concept_scores_and_identity():
    profile = build_voice_profile(
        pitch_summary={"mean_hz": 270, "min_hz": 210, "max_hz": 345, "range_hz": 135, "std_hz": 40},
        formants={"f1": 840, "f2": 2196, "f3": 3318, "f4": 4018},
        hnr_db=10,
    )

    assert profile["concept_scores"][GIRL] >= 70
    assert profile["concept_scores"][ROUGH] >= 50
    assert "primary_label" in profile["voice_identity"]
    assert profile["voice_identity"]["literary_description"]
    assert profile["voice_identity"]["voice_dna"]
    assert len(profile["voice_identity"]["primary_label"]) <= 7
    assert len(profile["voice_identity"]["secondary_tags"]) >= 5
    assert profile["voice_identity"]["style_family"]
    assert profile["layers"]["physical"]["F3"] == 3318
    assert "感知层" in profile["layer_summaries"]
    assert profile["attribute_catalog"]["感知层"]
    assert set(VOICE_DIMENSIONS).issubset(profile["dimension_scores"])
    assert set(VOICE_DIMENSIONS).issubset(profile["dimension_explanations"])


def test_make_voice_identity_names_mixed_young_bright_voice():
    identity = make_voice_identity(
        {
            GIRL: 82,
            BOY: 66,
            MATURE: 25,
            YOUNG: 70,
            ROUGH: 60,
            FRONT: 58,
        }
    )

    assert "少女" in identity["primary_label"]
    assert len(identity["primary_label"]) <= 7
    assert "文学描述" not in identity["literary_description"]
    assert identity["voice_dna"].startswith("V-")
    assert len(identity["secondary_tags"]) >= 5


def test_voice_identity_can_name_mtf_training_style():
    identity = make_voice_identity(
        {
            "女性感": 74,
            GIRL: 68,
            MTF: 78,
            "声学男性化特征": 42,
            FRONT: 70,
            THIN: 66,
            "清亮感": 72,
            "气声感": 38,
        }
    )

    assert "MTF" in identity["growth_direction"] or "自然生活女声" in identity["growth_direction"]
    assert any("前置" in tag for tag in identity["secondary_tags"])


def test_identity_declares_style_not_user_demographics():
    profile = build_voice_profile(
        pitch_summary={"mean_hz": 210, "min_hz": 170, "max_hz": 255, "range_hz": 85, "std_hz": 28},
        formants={"f1": 640, "f2": 1900, "f3": 3100, "f4": 4050},
        hnr_db=18,
    )

    assert "style_disclaimer" in profile["voice_identity"]
    assert "不是" in profile["voice_identity"]["style_disclaimer"]


def test_attribute_catalog_covers_many_voice_dimensions():
    total = sum(len(values) for values in ATTRIBUTE_CATALOG.values())

    assert total >= 60
    assert "轻薄感" in ATTRIBUTE_CATALOG["感知层"]
    assert "声道长度" in ATTRIBUTE_CATALOG["结构层"]
    assert GIRL in ATTRIBUTE_CATALOG["风格层"]
    assert "风格归类" in ATTRIBUTE_CATALOG["风格层"]


def test_voice_dimensions_have_circle_terms_and_explanations():
    expected = [
        "轻薄感",
        "声音女性感",
        "声音年轻感",
        "明亮感",
        "声音硬朗感",
        "气息感",
        "共鸣前置感",
        "颗粒感/质感",
        "声音温度",
        "稳定自然度",
        "雄激素化痕迹",
        "声音可塑性",
    ]

    assert VOICE_DIMENSIONS == expected

    profile = build_voice_profile(
        pitch_summary={"mean_hz": 210, "min_hz": 170, "max_hz": 255, "range_hz": 85, "std_hz": 28},
        formants={"f1": 640, "f2": 1900, "f3": 3100, "f4": 4050},
        hnr_db=18,
    )

    for name in expected:
        detail = profile["dimension_explanations"][name]
        assert 0 <= profile["dimension_scores"][name] <= 100
        assert detail["circle_terms"]
        assert detail["evaluates"]
        assert detail["score_meaning"]


def test_assess_recording_quality_rewards_clear_speech_sample():
    samples = [0.0, 0.08, -0.09, 0.12, -0.11, 0.06] * 8000

    quality = assess_recording_quality(
        samples=samples,
        sample_rate_hz=44100,
        duration_sec=12.0,
        hnr_db=19,
        voiced_frames=120,
        total_frames=150,
    )

    assert quality["recording_score"] >= 75
    assert quality["device_score"] >= 75
    assert quality["recording_label"] in {"良好", "优秀"}
    assert quality["device_label"] in {"良好", "优秀"}
    assert quality["metrics"]["duration_sec"] == 12.0


def test_assess_recording_quality_flags_clipping_and_short_duration():
    samples = [0.0, 0.99, -1.0, 0.0] * 3000

    quality = assess_recording_quality(
        samples=samples,
        sample_rate_hz=8000,
        duration_sec=2.0,
        hnr_db=6,
        voiced_frames=5,
        total_frames=60,
    )

    assert quality["recording_score"] < 55
    assert quality["device_score"] < 60
    assert any("爆音" in item or "削波" in item for item in quality["issues"])
    assert any("8 kHz" in item for item in quality["tips"])
