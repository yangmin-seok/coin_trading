# coin_trading

RL 기반 코인 트레이딩 프레임워크입니다.  
현재는 **학습 파이프라인(train probe) + 실시간 트레이드 런타임(runtime) + 거래소 연동 + 알림 모듈**의 기본 흐름이 연결되어 있습니다.

---

## 0) 프로젝트 구조 총람 (전체 파악)

아래는 저장소 루트 기준 구조입니다.

```text
<repo-root>
├─ README.md
├─ pyproject.toml
├─ docs/
│  ├─ README.md
│  └─ CODEBASE_FUNCTION_MAP.md
├─ data/                     # 데이터 I/O/검증/다운로더 유틸(루트 패키지)
├─ env/                      # TradingEnv, reward/execution model, recorder
├─ src/coin_trading/
│  ├─ agents/                # baseline/SB3 정책 어댑터
│  ├─ config/                # 설정 스키마 + 로더
│  ├─ execution/             # marketdata/order/risk/reconcile/state
│  ├─ features/              # offline/online 피처 계산
│  ├─ integrations/          # Binance REST/WS + Telegram
│  ├─ monitoring/            # metrics/alerts/drift
│  └─ pipelines/             # train/trade/test/run_manager 진입점
└─ tests/                    # 유닛/통합 테스트
```

핵심 포인트:

- 실행 진입점은 `src.coin_trading.pipelines.*` 모듈 경로입니다.
- 실행은 저장소 루트에서 `python -m src.coin_trading.pipelines.<name>` 형태로 수행합니다.
- `PYTHONPATH`를 수동으로 설정하지 않아도 됩니다.

---

## 1) 경로 변경 반영: 실행 기준 위치

최근 구조 기준 실행 기준은 **저장소 루트**입니다.

```bash
cd .
```

아래처럼 모듈 경로를 명시해 실행하세요.

```bash
python -m src.coin_trading.pipelines.train
```

> 참고: 로컬 환경에서 패키지명이 `src.coin_traiding`로 설정되어 있다면, 아래 `src.coin_trading` 부분만 해당 이름으로 치환해 실행하세요.

---

## 2) 빠른 시작

### 2-1. 요구사항

- Python 3.11+
- (권장) 가상환경

### 2-2. 설치

```bash
conda create -n coin_trading python=3.11 -y
conda activate coin_trading
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

네트워크 제한 환경에서는 의존성 설치가 실패할 수 있습니다.

### 2-3. 최소 실행 (명시적 모듈 경로)

```bash
python -m src.coin_trading.pipelines.train
```

> 참고: 로컬 환경에서 패키지명이 `src.coin_traiding`로 설정되어 있다면, 아래 `src.coin_trading` 부분만 해당 이름으로 치환해 실행하세요.

성공하면 콘솔 마지막 줄에 `run_id`가 출력됩니다.

요점은 "학습 성능" 자체보다, **파이프라인이 end-to-end로 정상 동작하는지 확인하는 것**입니다.

## 3) Train은 "목표"가 아니라 "런타임 점검 게이트"

요점은 "학습 성능" 자체보다, **파이프라인이 end-to-end로 정상 동작하는지 확인하는 것**입니다.

- 캔들 데이터 로딩/부트스트랩
- 오프라인 피처 계산
- TradingEnv + baseline 정책 1회 probe
- run 산출물(manifest/summary/artifact) 생성

즉, Train은 "실거래 전 점검 단계" 역할에 가깝습니다.

### 3-1. 실행 순서 상세

`src.coin_trading.pipelines.train.run()` 기준:

1. 설정 로드 (`load_config`)
2. `run_id` 생성, `runs/<run_id>/` 생성
3. `config.yaml`, `meta.json` 기록
4. 학습용 캔들 로드 (`data/processed/...`)
5. 없으면 bootstrap 캔들 생성 + parquet 저장 시도
6. 데이터셋/split/피처 NaN 요약 생성
7. `TradingEnv`에서 baseline(`VolTarget`) probe 롤아웃
8. `data_manifest.json`, `feature_manifest.json`, `train_manifest.json`, `dataset_summary.json` 기록

### 3-2. Train 결과에서 확인할 핵심 파일

`runs/<run_id>/`:

- `train_manifest.json`: 현재 상태(ready/blocked), probe 요약
- `data_manifest.json`: bootstrap 여부, 커버리지
- `feature_manifest.json`: 피처 정의/구현 해시
- `dataset_summary.json`: split row/NaN 비율
- `train_probe_summary.json`: steps/reward/equity
- `train_probe/trace.csv`, `train_probe/reward_equity.svg`: probe 추적 아티팩트

---

## 4) 실시간으로 "어떻게 주고받는지" (Runtime 이벤트 흐름)

아래는 `src.coin_trading.pipelines.trade` 기준의 런타임 흐름입니다.

### 4-1. 런타임 구성(build_runtime)

- `BinanceRESTClient`: REST 조회
- `MarketDataWS`: 마켓 websocket 이벤트 수신
- `GapFiller`: 누락 candle을 REST로 메움
- `OnlineFeatureEngine`: 실시간 피처 업데이트
- `VolTarget`: 목표 포지션 계산
- `RiskManager`: 목표 승인/거절
- `OrderManager`: 주문 의도 변환
- `MetricsLogger`: JSONL 메트릭 기록
- `AlertEngine`: 리컨실/드로우다운 알림 판단

### 4-2. 시장 데이터 수신 → 내부 큐 전달

1. WS 메시지 수신
2. `kline.x == true`(종가 확정 캔들)만 처리
3. timestamp gap 발생 시 `GapFiller.fill(...)` 호출
4. 보정 이벤트 + 신규 이벤트를 `asyncio.Queue`에 넣음
5. 마지막 처리 시각(`last_ts`) 저장

### 4-3. 큐 소비 시 의사결정 루프

`process_market_event(runtime, event)`:

1. 포트폴리오 mark-to-market
2. 실시간 피처 업데이트
3. 정책이 목표 비중 산출 (`policy.act`)
4. 리스크 승인 (`risk.approve_target`)
5. 승인 시 주문 의도(`target_to_intent`) 생성
6. 메트릭 카운터/구조화 로그 emit

### 4-4. 최소 점검 커맨드

```bash
python -c "from src.coin_trading.pipelines.trade import run; print(run())"
```

이 명령은 runtime 조립이 되는지 빠르게 확인합니다.

---

## 5) User Stream (실시간 체결/잔고 이벤트) 처리

`src.coin_trading.integrations.binance_ws_user.BinanceUserWS`는 user stream payload를 이벤트 타입별로 분류하여 큐로 전달합니다.

- 처리 이벤트 타입:
  - `outboundAccountPosition`
  - `executionReport`
  - `balanceUpdate`
  - 기타는 `other`
- 입력이 combined stream 형식이면 `data` 필드를 자동 추출
- 최종적으로 `UserStreamEvent(event_type, payload)`를 큐에 저장

이 구조를 사용하면 주문 체결/잔고 변경을 runtime loop에서 비동기 소비하도록 확장할 수 있습니다.

---

## 6) 텔레그램 메시지는 어떻게 보내는가

`src.coin_trading.integrations.telegram.TelegramSender`가 Telegram Bot API `sendMessage`를 직접 호출합니다.

### 6-1. 동작 방식

- URL: `https://api.telegram.org/bot<token>/sendMessage`
- JSON payload:
  - `chat_id`
  - `text`
  - `disable_notification`
- `urllib.request.urlopen`으로 POST 전송
- 응답 JSON(dict) 반환

### 6-2. 사용 예시

```python
from src.coin_trading.integrations.telegram import TelegramSender

sender = TelegramSender(bot_token="<BOT_TOKEN>", chat_id="<CHAT_ID>")
resp = sender.send_text("[coin_trading] runtime started")
print(resp)
```

### 6-3. 운영 시 권장

현재 구현은 기본 전송만 포함합니다. 운영에서는 아래를 추가 권장합니다.

- 네트워크 예외 처리
- 재시도/백오프
- 레이트리밋 대응
- 알림 코드별 포맷 통일

---

## 7) 구역별 점검 체크리스트

### A. Train 구역

```bash
RUN_ID=$(python -m src.coin_trading.pipelines.train | tail -n 1)
cat runs/$RUN_ID/train_manifest.json
cat runs/$RUN_ID/data_manifest.json
```

체크 포인트:
- `train_manifest.status == "ready"`
- `data_manifest.bootstrap_generated` 값 확인

### B. Runtime 구역

```bash
python -c "from src.coin_trading.pipelines.trade import run; print(run())"
```

체크 포인트:
- runtime ready 문자열 출력
- symbol/interval 정상 표시

### C. User Stream 구역

- user event가 `executionReport` 등으로 분류되는지 확인
- 큐 consumer에서 이벤트 누락 없이 읽는지 확인

### D. Telegram 구역

- 토큰/챗ID로 `send_text` 정상 응답 확인
- 실패 시 재시도 정책(추후 구현)을 붙일 지점 정의

---

## 8) 테스트

```bash
pytest -q
```

환경 제약으로 전체 테스트가 어려우면 최소한:

```bash
python -m compileall .
```
