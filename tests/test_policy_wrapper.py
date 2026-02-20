from __future__ import annotations

import pytest

from agents.policy_wrapper import create_policy


def test_create_baseline_policy_and_act():
    policy = create_policy("buy_and_hold")
    policy.reset()
    assert 0.0 <= policy.act(None, {"close": 100.0, "logret_1": 0.0}) <= 1.0


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        create_policy("unknown")
