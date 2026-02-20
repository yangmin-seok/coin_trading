# Coin Trading 코드맵 + 구현 상태 점검

이 문서는 현재 코드 기준으로 **"구현됨/부분 구현/미구현"** 상태를 빠르게 확인하기 위한 요약 문서입니다.

기준: `main` 코드 트리(`/workspace/coin_trading`).

---

## 0) 요청사항 구현 여부 체크

- [x] 새 MD 문서에 폴더/함수별 기능 요약 작성
- [x] 폴더/함수별로 남은 구현 항목(TODO) 정리
- [x] README 사용 가이드(처음~끝) 업데이트

---

## 1) 폴더별/함수별 기능 요약

## config/

- `schema.py`
  - `RewardConfig`, `ExecutionConfig`, `FeatureWindowsConfig`, `FeaturesConfig`, `SplitConfig`, `AppConfig`
  - 역할: 앱 전역 설정을 Pydantic 모델로 검증.
- `loader.py`
  - `_set_nested(...)`: 중첩 키 경로(`a.b.c`)에 값 설정.
  - `load_config(...)`: YAML 로드 + `APP__...` 환경변수 override 적용 + `AppConfig` 검증.

상태: **구현됨**

## data/

- `io.py`
  - `build_partition_path(...)`: 파티션 경로 생성.
  - `write_candles_parquet(...)`: 일자별 파케이 저장.
  - `read_candles_parquet(...)`: 파케이 로드.
- `validator.py`
  - `ValidationReport`: 검증 결과 컨테이너.
  - `DataValidator`: 정렬/중복/결측 gap/이상치 검증.
- `downloader.py`
  - `HistoricalDownloader`: 심볼/인터벌 메타만 가진 최소 골격.
  - `normalize_time_unit(...)`: time 단위를 ms/us 기준으로 정규화.

상태: **부분 구현** (다운로드 실동작은 골격 위주)

## features/

- `common.py`
  - `FeatureState`: 온라인 피처 계산 상태 저장.
  - `_mean`, `_std`, `_rsi`, `_ema`: 지표 계산 유틸.
  - `update_features(...)`: 캔들 1개씩 받아 피처 갱신.
- `offline.py`
  - `compute_offline(...)`: 히스토리 기반 배치 피처 생성.
- `online.py`
  - `OnlineFeatureEngine`: 실시간 스트림 피처 계산 엔진.
- `parity_test.py`
  - `replay_and_compare(...)`: offline vs online 값 일치성 확인.

상태: **핵심 구현됨**

## env/

- `execution_model.py`
  - `ExecutionResult`, `ExecutionModel`: 수수료/슬리피지 포함 체결 모델.
- `reward.py`
  - `compute_reward(...)`: 수익/턴오버/DD 페널티 결합 보상.
- `recorder.py`
  - `StepRecorder`: 스텝 로그 기록.
- `trading_env.py`
  - `TradingEnv`: 관측/액션/보상/포지션 갱신 포함 환경.

상태: **구현됨**

## execution/

- `marketdata.py`
  - `interval_to_ms(...)`: 인터벌 문자열→밀리초.
  - `CandleClosedEvent`: 종료 캔들 이벤트 타입.
  - `MemoryStateStore`: 마지막 처리 timestamp 저장.
  - `GapFiller`: 누락 구간 REST 보정.
  - `MarketDataWS`: WS 수신 + close 캔들만 큐에 전달.
- `reconcile.py`
  - `ReconcileResult`, `Reconciler`: 내부/거래소 자산 비교.
- `state.py`
  - `PortfolioState`: 내부 포트폴리오 상태 모델.
- `orders.py`
  - `OrderManager`: 시장가 주문 생성/취소, 목표 포지션→주문 의도 변환, execution report 반영.
- `risk.py`
  - `RiskManager`: target clamp, drawdown/leverage 기반 승인 판단.

상태: **핵심 구현됨**

## integrations/

- `binance_ws_market.py`
  - `BinanceMarketWSConfig`, `extract_kline_payload(...)`: 바이낸스 kline payload 정규화.
- `binance_rest.py`
  - `BinanceRESTClient`: REST 요청(klines/account/open orders) + 재시도.
- `binance_ws_user.py`
  - `extract_user_payload(...)`, `BinanceUserWS`: user 이벤트 파싱 및 큐 전달.
- `telegram.py`
  - `TelegramSender.send_text(...)`: Telegram Bot API `sendMessage` 호출.

상태: **부분 구현** (재연결/listenKey/재시도 정책 고도화 필요)

## monitoring/

- `metrics.py`
  - `RuntimeCounters`, `MetricsLogger`: 카운터 집계 + JSONL 배출.
- `alerts.py`
  - `Alert`, `AlertEngine`: 리컨실/드로우다운 알림 룰.
- `drift.py`
  - 드리프트 모듈 뼈대.

상태: **부분 구현**

## pipelines/

- `run_manager.py`
  - `make_run_id`, `git_sha`, `_git_dirty`, `write_meta`, `write_data_manifest`, `write_feature_manifest`, `write_train_manifest`, `implementation_hash`.
- `train.py`
  - `_train_data_glob(...)`: 학습용 파케이 경로 패턴 생성.
  - `load_training_candles(...)`: 학습 데이터셋 로드/정렬/중복제거.
  - `summarize_dataset_for_training(...)`: split 별 row 수와 feature NaN 비율 산출.
  - `run()`: 실행 폴더 생성 + meta/data/feature/train manifest + dataset summary 기록.
- `trade.py`
  - `TradeRuntime`, `build_runtime`, `reconcile_once`, `process_market_event`, `run(max_events, timeout_s)`.
  - queue 이벤트를 피처/정책/리스크/주문의도/메트릭으로 연결하는 기본 루프 포함.
- `test.py`
  - `run()`: 스캐폴드 문자열 반환.

상태: **부분 구현** (기본 실행 루프 구현, 실주문 송신/복구 자동화는 추가 필요)

## agents/

- `baselines.py`
  - `BaselinePolicy`, `BuyAndHold`, `MACrossover`, `VolTarget`: 베이스라인 정책.
- `policy_wrapper.py`
  - `PolicyWrapper`: 공통 `predict(...)` 인터페이스로 정책 추상화.
- `sb3_ppo.py`, `sb3_sac.py`
  - `PPOPolicyAdapter`, `SACPolicyAdapter`: SB3 `model.predict(...)` 결과를 공통 act 규격으로 변환.

상태: **부분 구현** (실제 학습/로드 파이프라인 연동은 미구현)

---

## 2) 반드시 구현해야 할 TODO (우선순위)

1. `pipelines/trade.py` 실주문 전송/복구 자동화
   - 주문 요청 체결 추적, 실패 시 재시도/보상 트랜잭션, graceful shutdown.
2. `integrations/binance_ws_user.py` 고도화
   - listenKey 발급/keepalive/만료 대응, 연결 복구 상태머신 추가.
3. `integrations/telegram.py` 운영 품질 강화
   - 네트워크 예외 처리, 백오프, 레이트리밋 대응.
4. `agents/*` 고도화
   - SB3 모델 저장/로드/평가 루틴과 파이프라인 연동.
5. 운영 안정성
   - 장애 복구(runbook), 메트릭 대시보드, 실운영 알람 라우팅 강화.

---

## 3) 빠른 결론

- **지금 상태는 “기본 뼈대 + 일부 핵심 구현 완료” 단계**.
- 학습용 메타데이터 기록, 피처 계산/파리티, 시장데이터 close-candle 처리, 리컨실/모니터링 기초는 준비됨.
- 실제 운영 트레이딩(주문/리스크/유저스트림/알림)은 아직 미완이며, 해당 구간이 핵심 개발 잔여분.


## 4) 이번 구현 계획표/진행 결과

| 단계 | 작업 | 상태 |
|---|---|---|
| 1 | 구조 점검 및 범위 확정 | 완료 |
| 2 | orders/risk/user ws/telegram/policy wrapper 구현 | 완료 |
| 3 | trade runtime에 이벤트 처리 루프(`process_market_event`) 연결 | 완료 |
| 4 | README + 코드맵 문서 업데이트 | 완료 |
| 5 | 테스트 실행 및 검증 | 완료 |
