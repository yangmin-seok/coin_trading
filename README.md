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

## 2-4. RL action/position 정의

- `action`은 **목표 익스포저 비율(target exposure ratio)** 입니다.
- 범위는 `[-1.0, 1.0]`이며, `-1.0`은 최대 숏, `0.0`은 중립, `1.0`은 최대 롱을 의미합니다.
- 환경 관측치의 `position_ratio`는 signed ratio를 유지하며 학습 안정성을 위해 `[-1.0, 1.0]`로 clip합니다.


## 3) Train은 **실제 모델 학습/선택 단계**

이제 Train은 단순 점검이 아니라, **워크포워드 학습 → 검증/테스트 평가 → 모델 선택**까지 수행하는 핵심 단계입니다.

- 캔들 데이터 로딩/부트스트랩
- 오프라인 피처 계산 + 스케일링
- RL 학습(PPO/SAC)
- 주기적 검증(Val) 평가 + 얼리 스톱
- Test 평가 및 리포트/시각화 생성

### 3-1. 실행 순서 상세

`src.coin_trading.pipelines.train.run()` 기준:

1. 설정 로드 (`load_config`)
2. `run_id` 생성, `runs/<run_id>/` 생성
3. `config.yaml`, `meta.json` 기록
4. 학습용 캔들 로드 (`data/processed/...`)
5. 없으면 bootstrap 캔들 생성 + parquet 저장 시도
6. 워크포워드 split 생성
7. split별 학습/평가 반복 (reward 타입 × 반복 실험 포함)
8. best 실험 선택 + 요약 리포트 저장

### 3-2. Train 결과에서 확인할 핵심 파일

`runs/<run_id>/`:

- `reports/model_train_summary.json`: 워크포워드 전체 결과, 선택된 best 실험, 지표 요약
- `walkforward_XX/.../reports/learning_curve.csv|json`: 학습 이력 원본 수치
- `walkforward_XX/.../plots/learning_curve.svg`: 검증 지표 변화(러닝커브)
- `walkforward_XX/.../reports/val_trace/*`: 검증 구간 트레이스/시각화
- `walkforward_XX/.../reports/test_trace/*`: 테스트 구간 트레이스/시각화
- `walkforward_XX/.../artifacts/metrics.json`: 최종 val/test metrics + 아티팩트 경로

### 3-3. Train 이미지 해석 가이드 (중요)

아래는 Train에서 자주 보는 이미지와 **해석 기준**입니다.

1. `plots/learning_curve.svg`  
   - 무엇을 보나: 학습 진행 중 검증(Val) 성능 변화.
     - Left 축: `val_return_pct`, `val_pnl_pct`
     - Right 축: `val_sharpe`, `val_turnover`, `val_cost_pnl_ratio`
   - 좋다고 볼 수 있는 패턴:
     - `val_return_pct` 우상향 + `val_sharpe`가 0 이상에서 개선
     - `val_turnover`가 과도하게 치솟지 않음
     - `val_cost_pnl_ratio`가 하향 또는 낮은 수준 유지
   - 경계 패턴:
     - 수익률은 오르는데 `val_cost_pnl_ratio`도 급상승 → 거래비용으로 실익 악화 가능성
     - `val_sharpe` 변동성이 매우 큼 → 정책이 불안정하거나 과최적화 가능성

2. `reports/val_trace/reward_equity.svg`, `reports/test_trace/reward_equity.svg`  
   - 무엇을 보나: 전략 수익률(좌축) vs 기준선(cash hold, buy&hold).
   - 좋다고 볼 수 있는 패턴:
     - 전략 곡선이 buy&hold 대비 장기적으로 위에 위치
     - 급락 후 회복이 빠르고, 최종 수익률이 기준선 대비 우위
   - 해석 팁:
     - Val에서 좋고 Test에서도 유사하면 일반화 가능성↑
     - Val만 좋고 Test에서 꺾이면 과적합 신호로 해석

3. `reports/val_trace/drawdown_turnover.svg`, `reports/test_trace/drawdown_turnover.svg`  
   - 무엇을 보나: Drawdown(손실 구간 깊이)과 유효 포지션/노출 변화.
   - 좋다고 볼 수 있는 패턴:
     - 드로우다운 피크(최대낙폭)가 제한적
     - 포지션 전환이 과도하게 잦지 않음(비용 관점 유리)
   - 경계 패턴:
     - 급격한 포지션 전환 + 깊은 드로우다운 동반 → 리스크/비용 동시 악화

4. `reports/val_trace/reward_components_timeseries.png`, `reports/test_trace/reward_components_timeseries.png`  
   - 무엇을 보나: 보상 구성요소(`reward_pnl`, `reward_cost`, `reward_penalty`, `reward`)의 시계열.
   - 좋다고 볼 수 있는 패턴:
     - `reward_pnl` 기여가 우세하고,
     - `reward_cost`, `reward_penalty`가 장기적으로 reward를 압도하지 않음
   - 경계 패턴:
     - `reward_cost`/`reward_penalty` 진폭이 커서 총 `reward`가 지속적으로 눌림
     - 특정 구간에서 penalty가 급등 → 해당 리스크 제약(예: DD, inactivity) 재튜닝 필요

5. (실행 옵션에 따라) 공통 리스크 플롯
   - `drawdown_curve.png`: train/valid/test의 drawdown 비교.
     - 해석: Test 구간의 drawdown이 Train 대비 비정상적으로 크면 일반화 실패 가능성.
   - `monthly_returns_heatmap.png`: 월별 수익 기여의 계절성/편향 확인.
     - 해석: 몇 개 월에만 성과가 집중되면 레짐 의존성(취약성) 가능성.

실무적으로는 **단일 이미지 하나**보다 아래 3개를 함께 보시면 됩니다.
- 수익성: `reward_equity.svg`
- 안정성: `drawdown_turnover.svg`
- 비용/패널티 구조: `reward_components_timeseries.png`

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

워크포워드 최소 커버리지/리포트 검증:

```bash
RUN_ID=$(python -m src.coin_trading.pipelines.train | tail -n 1)
export RUN_ID
python - <<'PY'
import json, pathlib, os
run_id = os.environ["RUN_ID"]
summary_path = pathlib.Path("runs") / run_id / "reports" / "model_train_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
print("walkforward_runs:", summary["walkforward_runs"])
print("walkforward_shortfall:", summary["walkforward_shortfall"])
assert summary["walkforward_runs"] >= 3
assert summary["walkforward_shortfall"] is None
PY
```

`model_train_summary.json`의 `walkforward_coverage_check`에는
`data_end`와 `next_fold_required_test_end` 비교 결과가 기록되며,
데이터가 부족한 경우 `walkforward_coverage_adjustment`에 분할 축소 적용 내역이 남습니다.

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
