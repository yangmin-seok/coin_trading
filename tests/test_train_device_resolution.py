from __future__ import annotations

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


def test_resolve_training_device_auto_cpu(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", _TorchStub(False, 0))
    assert _resolve_training_device("auto") == "cpu"


def test_resolve_training_device_auto_cuda(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", _TorchStub(True, 1))
    assert _resolve_training_device("auto") == "cuda:0"


def test_resolve_training_device_invalid_cuda_fallback(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", _TorchStub(False, 0))
    assert _resolve_training_device("cuda:0") == "cpu"
