from __future__ import annotations

from data.validator import DataValidator


def test_data_validator_detects_gap(sample_candles):
    df = sample_candles.drop(index=[10, 11]).reset_index(drop=True)
    validator = DataValidator(interval_ms=300_000)
    clean, report = validator.validate(df)
    assert len(clean) == len(df)
    assert report.missing_count == 2
    assert report.missing_ratio > 0
