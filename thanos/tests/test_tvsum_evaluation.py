import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_tvsum.py"
spec = importlib.util.spec_from_file_location("evaluate_tvsum", SCRIPT)
tvsum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tvsum)


def test_catalog_and_annotations_are_complete():
    catalog = tvsum.load_catalog(tvsum.DEFAULT_TVSUM / "data" / "ydata-tvsum50-info.tsv")
    annotations = tvsum.load_annotation_lines(tvsum.DEFAULT_TVSUM / "data" / "ydata-tvsum50-anno.tsv")
    assert len(catalog) == len(annotations) == 50
    assert all(len(annotations[row["video_id"]]) == 20 for row in catalog)


def test_gold_knapsack_uses_15_percent_and_last_short_shot():
    # 430 frames -> 64-frame budget. A 60-frame shot + final 10-frame
    # shot cannot both fit, so a very high-scoring tail wins by shot mean.
    ratings = [1] * 420 + [5] * 10
    mask = tvsum.gold_mask(ratings)
    assert sum(mask) == 10
    assert all(mask[420:])


def test_interval_projection_and_f1():
    predicted = tvsum.intervals_mask([{"start": 2.0, "end": 4.0}], 100, 10.0)
    gold = tvsum.intervals_mask([{"start": 3.0, "end": 5.0}], 100, 10.0)
    assert sum(predicted) == sum(gold) == 20
    assert tvsum.f1(predicted, gold) == pytest.approx(0.5)
    assert tvsum.recall(predicted, gold) == pytest.approx(0.5)
    assert tvsum.precision(predicted, gold) == pytest.approx(0.5)


def test_invalid_intervals_are_rejected():
    with pytest.raises(ValueError):
        tvsum.intervals_mask([{"start": -1, "end": 2}], 100, 10.0)
    with pytest.raises(ValueError):
        tvsum.intervals_mask([{"start": 4, "end": 2}], 100, 10.0)


def test_score_only_knapsack_picks_higher_value_shot():
    shots = [
        {"start_seconds": 0.0, "end_seconds": 6.0, "importance_score": 0.1},
        {"start_seconds": 6.0, "end_seconds": 12.0, "importance_score": 0.9},
    ]
    assert tvsum.select_scored_shots(shots, 400, 40.0) == [{"start": 6.0, "end": 12.0}]


def test_uniform_duration_matched_baseline():
    assert sum(tvsum.uniform_mask(600)) == 90
    assert sum(tvsum.uniform_mask(600, 60)) == 60
    assert sum(tvsum.uniform_mask(1200, 120)) == 120


def test_wrong_video1_is_not_used_as_tvsum(monkeypatch, tmp_path):
    (tmp_path / "video1.mp4").write_bytes(b"not actual video")
    monkeypatch.setattr(tvsum, "video_duration", lambda _: 1363.5)
    row = {"video_id": "AwmHb44_ouw", "alias": "video1", "length": "5:54"}
    path, duration, error = tvsum.find_video(tmp_path, row)
    assert path is duration is None
    assert "beklenen" in error
