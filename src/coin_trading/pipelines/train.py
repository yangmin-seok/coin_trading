"""Training pipeline entry module.

공개 API:
- ``run``: 학습 파이프라인 실행 진입점.
- run_id 규칙은 ``pipelines.run_manager.make_run_id()`` (``<YYYYMMDD_HHMMSSZ>_<git_sha7>``)를 따른다.

비공개/미지원 API:
- ``ensure_training_candles`` 및 ``summarize_dataset_for_training`` 같은 과거 헬퍼는
  더 이상 ``pipelines.train`` 에서 제공하지 않는다.
- 데이터 준비/요약 기능은 ``src.coin_trading.pipelines.train_flow.data`` 에서만 사용한다.
"""

from __future__ import annotations

from src.coin_trading.pipelines.train_flow.orchestrator import run

__all__ = ["run"]


if __name__ == "__main__":
    print(run())
