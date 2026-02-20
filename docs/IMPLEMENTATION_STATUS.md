# Coin Trading Implementation Status (handoff-safe)

이 문서는 **다음 세션에서도 끊기지 않도록** 현재 구현 상태와 남은 작업을 폴더별로 기록한다.
기준 커밋은 문서 업데이트 시점의 `HEAD`를 따른다.

## 1) 폴더별 현황

### config/
- 구현됨
  - `schema.py`: 앱/보상/실행/피처/split 타입 검증
  - `loader.py`: yaml + env override 로딩
- 수정 필요
  - env override 타입 캐스팅 강화(현재 문자열 주입 중심)
  - secret 분리 로더(예: `.env`/vault) 명시화

### data/
- 구현됨
  - `io.py`: parquet 파티션 경로/읽기/쓰기
  - `validator.py`: 중복/결측(gap)/이상치 검사
  - `downloader.py`: 계약/시간단위 정규화(ms/us)
- 구현 필요
  - bulk zip + checksum 실제 다운로드
  - REST 증분 병합/재시도 정책

### features/
- 구현됨
  - online-first feature 계산 경로
  - offline replay + parity 테스트 유틸
- 수정 필요
  - feature window/파라미터를 config와 완전 연동
  - 구현 해시/manifest에 parity 결과(steps, max diff) 반영

### env/
- 구현됨
  - t 관측, t+1(next_open) 체결 시맨틱
  - reward 파라미터 주입
  - step recorder
- 수정 필요
  - obs 스키마 검증(누락 feature 강제 에러 옵션)
  - 거래소 필터(minQty/stepSize/minNotional) 제약 반영

### execution/
- 구현됨
  - `marketdata.py`: closed-candle only 처리, gapfill, last_ts 저장
  - `state.py`: 포트폴리오 상태
- 이번 턴 추가
  - `reconcile.py`: 스냅샷 기반 불일치 감지 로직
- 구현 필요
  - order/risk 실제 정책 및 이벤트 반영

### integrations/
- 구현됨
  - `binance_ws_market.py`: raw/combined payload normalize
- 이번 턴 추가
  - `binance_rest.py`: klines/account/open orders 최소 REST 어댑터
- 구현 필요
  - `binance_ws_user.py`: user data stream(WebSocket API) 구독/이벤트 라우팅

### monitoring/
- 구현 필요
  - metrics jsonl/steps parquet 출력
  - alerts 정책(DD/연속손실/reconcile mismatch/ws 재연결 과다)

### pipelines/
- 구현됨
  - train run artifact(meta/data/feature manifest) 저장
- 이번 턴 추가
  - trade runtime 조립 함수(시장데이터 + gapfill + reconcile)
- 구현 필요
  - 실제 event loop(run_forever), graceful shutdown, 재연결 카운트 메트릭

### tests/
- 구현됨
  - data/features/env/run-manager/market-ws 단위테스트
- 주의
  - 현재 환경에서 numpy 미설치로 `pytest -q` 전체 실행 불가
- 구현 필요
  - reconcile 단위테스트
  - trade runtime 조립 smoke test

---

## 2) 지금 바로 이어서 할 일 (우선순위)
1. `integrations/binance_rest.py` 실제 호출 구현 (timeout/retry/backoff 포함).
2. `execution/reconcile.py` 주기 대조 로직(잔고/포지션 mismatch 임계치).
3. `pipelines/trade.py`에서 runtime 컴포넌트 조립 + 상태 체크 루프 연결.
4. `monitoring/metrics.py`에 ws reconnect/gapfill/reconcile mismatch 카운터 기록.

## 3) 수정 시 주의할 불변 규칙
- 저장/학습 데이터 기준은 UTC, bar 식별자는 open_time(ms, UTC)
- 실시간 kline은 `k.x == true`만 전략 입력
- env는 관측 t, 체결은 t+1(next_open)
- offline/online feature parity 유지
- run artifact(meta/data/feature manifest) 누락 금지
