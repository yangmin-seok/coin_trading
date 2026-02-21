from __future__ import annotations

import pytest

from src.coin_trading.config.schema import TrainConfig
from src.coin_trading.pipelines.train_flow.train import _resolve_training_device


class _CudaStub:
    def __init__(self, available: bool, count: int):
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class _TorchStub:
    def __init__(self, available: bool, count: int):
        self.cuda = _CudaStub(available, count)
        self.__version__ = "stub-0.0"


def test_train_config_device_alias_gpu_maps_to_cuda():
    cfg = TrainConfig(device="gpu")
    assert cfg.device == "cuda"


def test_train_config_device_invalid_value_raises():
    with pytest.raises(ValueError, match="train.device must be one of"):
        TrainConfig(device="tpu")


def test_resolve_training_device_auto_cpu(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _TorchStub(False, 0))
    assert _resolve_training_device("auto") == "cpu"


def test_resolve_training_device_auto_cuda(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _TorchStub(True, 1))
    assert _resolve_training_device("auto") == "cuda:0"


def test_resolve_training_device_alias_gpu(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _TorchStub(True, 2))
    assert _resolve_training_device("gpu") == "cuda:0"


def test_resolve_training_device_invalid_value_raises(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _TorchStub(True, 2))
    with pytest.raises(ValueError, match="unsupported train.device"):
        _resolve_training_device("cuda:bad")


def test_resolve_training_device_cuda_unavailable_raises(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _TorchStub(False, 0))
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _resolve_training_device("cuda:0")
