# Coin Trading 구현 상태 (통합본)

중복 문서를 통합해, 이 파일 하나에서 폴더/함수/남은 작업을 확인할 수 있게 정리했습니다.

## 1) 구현 체크 요약

- config, features, env, run artifact 기록: 구현
- train 파이프라인: 구현(합성 데이터 기반 에피소드 실행 + summary 기록)
- trade 실거래 루프: 부분 구현(런타임 조립/리컨실만 구현)
- 주문/리스크/유저WS/텔레그램: 미구현(placeholder)

## 2) 폴더/함수별 상태

## config/
- `loader.load_config`: YAML + `COIN_TRADING__` env override + schema validate
- `schema.*`: reward/execution/features/split/app 검증 모델

상태: 구현됨

## data/
- `io.build_partition_path`, `write_candles_parquet`, `read_candles_parquet`
- `validator.DataValidator`
- `downloader.normalize_time_unit` (+ downloader 골격)

상태: 부분 구현

## features/
- `common.update_features` (지표 계산 핵심)
- `offline.compute_offline`
- `online.OnlineFeatureEngine`
- `parity_test.replay_and_compare`

상태: 구현됨

## env/
- `execution_model.ExecutionModel`
- `reward.compute_reward`
- `trading_env.TradingEnv`
- `recorder.StepRecorder`

상태: 구현됨

## agents/
- `baselines`: BuyAndHold/MACrossover/VolTarget
- `policy_wrapper.create_policy`: 정책 팩토리 + 액션 clamp
- `sb3_ppo.PPOPolicy`, `sb3_sac.SACPolicy`: 선택적 SB3 로더 shim

상태: train 기준 구현됨(SB3는 optional dependency)

## pipelines/
- `run_manager.*`: run id/meta/manifest/hash
- `train.run`: run artifact 생성 + 합성 데이터 에피소드 실행 + `train_summary.json` 저장
- `trade.build_runtime/reconcile_once`: 런타임 조립/리컨실

상태: train 구현됨, trade 부분 구현

## execution/
- `marketdata`: close-candle 처리/gapfill/state 저장
- `reconcile`: 내부/거래소 잔고 비교
- `orders`, `risk`: placeholder

상태: 부분 구현

## integrations/
- `binance_rest`, `binance_ws_market`: 구현
- `binance_ws_user`, `telegram`: placeholder

상태: 부분 구현

## monitoring/
- `metrics.RuntimeCounters`, `MetricsLogger`
- `alerts.AlertEngine`
- `drift`: placeholder

상태: 부분 구현

## 3) 우선순위 TODO

1. `execution/orders.py`, `execution/risk.py` 구현
2. `integrations/binance_ws_user.py` 구현
3. `pipelines/trade.py` full event loop + graceful shutdown
4. `integrations/telegram.py` 연동
5. drift/alert 규칙 고도화
