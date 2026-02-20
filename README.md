# coin_trading

RL 기반 코인 트레이딩 프레임워크입니다.  
현재는 **학습 데이터/피처/환경/런 메타 기록/시장데이터 처리**와 함께, 주문/리스크/user stream/알림/정책 래퍼의 기본 구현이 포함되어 있습니다.

---

## 1) 목표: "train 한 번"으로 바로 런 산출물 만들기

이 프로젝트는 `python -m pipelines.train` **1회 실행만으로** 아래를 보장하도록 구성되어 있습니다.

- `runs/<run_id>/` 산출물 생성
- 데이터가 없으면 `data/processed/...`에 **부트스트랩(합성) 캔들 데이터 자동 생성**
- `train_manifest.json` 상태를 `ready`로 기록

즉, 처음 클론한 환경에서도 최소한 "학습 파이프라인이 끝까지 도는지"를 바로 검증할 수 있습니다.

---

## 2) 빠른 시작

### 2-1. 요구사항

- Python 3.11+
- (권장) 가상환경

### 2-2. 설치

```bash
python -m venv .venv

# bash / zsh
source .venv/bin/activate

python -m pip install --upgrade pip

# bash
python -m pip install -e .[dev]

# zsh (글롭 방지: 대괄호를 quote 처리)
python -m pip install -e '.[dev]'
# 또는
python -m pip install -e ".[dev]"
```

네트워크 제한 환경에서는 의존성 설치가 실패할 수 있습니다.

### 2-3. 진짜 최소 실행(핵심)

```bash
python -m pipelines.train
```

성공하면 콘솔에 `run_id`가 출력됩니다.

예시:

```text
demo-btcusdt-5m-3-20260101-010203
```

---

## 3) train 1회 실행 시 생성되는 파일

`runs/<run_id>/` 아래:

- `config.yaml`: 실행 시점 설정 스냅샷
- `meta.json`: 실행 메타 정보(시간/git 등)
- `data_manifest.json`: 데이터/커버리지/부트스트랩 여부
- `feature_manifest.json`: 피처 컬럼/윈도우/구현 해시
- `train_manifest.json`: 학습 준비 상태 + probe epochs/model 요약
- `dataset_summary.json`: split별 row 및 피처 NaN 요약
- `train_probe_summary.json`: reward/equity probe 요약

`data_manifest.json`에서 아래 키를 확인하세요.

- `bootstrap_generated: true` → 실제 데이터가 없어 자동 합성 데이터 생성됨
- `bootstrap_persisted: true` → 합성 데이터를 `data/processed` parquet로 저장함
- `bootstrap_persisted: false` → parquet 엔진(pyarrow/fastparquet) 미설치로 메모리에서만 사용함
- `bootstrap_generated: false` → 기존 `data/processed` 데이터 사용됨

---


## 3-1) 자주 묻는 질문 (train)

- **Q. train epoch은 어느정도야?**  
  현재 `pipelines.train`은 딥러닝 학습 루프가 아니라, **1 epoch 성격의 baseline probe 롤아웃**을 수행합니다.
  `train_manifest.json`의 `epochs`가 `1`로 기록됩니다.

- **Q. 모델은 어떤 걸 사용해?**  
  현재 probe 모델은 `VolTarget-baseline`입니다. (`agents.baselines.VolTarget`)
  실제 SB3 PPO/SAC 학습 파이프라인은 아직 별도 고도화가 필요합니다.

- **Q. train이 잘 되는지 reward를 볼 수 있어?**  
  가능합니다. train 실행 시 `runs/<run_id>/train_probe/` 아래에:
  - `trace.csv` (step, signal=buy/hold/sell, reward, equity 등)
  - `reward_equity.svg` (reward/equity 추이)
  가 자동 생성됩니다.

## 4) 설정 방법

기본 설정 파일은 `config/default.yaml` 입니다.

주요 키:

- `mode`: 실행 모드 (`demo`/`live`/`backtest`)
- `exchange`, `market`, `symbol`, `interval`
- `reward.*`: 보상 함수 파라미터
- `execution.*`: 수수료/슬리피지/행동 변화 제한
- `features.version`, `features.windows.*`
- `split.train/val/test`: 데이터 분할 기준 날짜

환경변수 오버라이드는 `COIN_TRADING__` prefix를 사용합니다.

예시:

```bash
export COIN_TRADING__SYMBOL=ETHUSDT
export COIN_TRADING__INTERVAL=1m
python -m pipelines.train
```

`config.loader.load_config()`가 YAML + 환경변수를 합쳐 검증합니다.

---

## 5) 실행 가이드 (처음부터 끝까지)

### Step A. train 한 번 실행

```bash
python -m pipelines.train
```

### Step B. 산출물 상태 확인

```bash
RUN_ID=$(python -m pipelines.train | tail -n 1)
cat runs/$RUN_ID/train_manifest.json
cat runs/$RUN_ID/data_manifest.json
```

확인 포인트:

- `train_manifest.status == "ready"`
- `data_manifest.bootstrap_generated` 값

### Step C. 트레이드 런타임 빌드/점검

```bash
python -c "from pipelines.trade import run; print(run())"
```

기본 모드에서는 런타임 조립 상태를 반환하고,  
`run(max_events=...)`로 큐 이벤트 소비/피처 업데이트/리스크 판단/주문 의도 생성까지 점검할 수 있습니다.

### Step D. 리컨실 로직 단독 점검 예시

`pipelines.trade.reconcile_once(...)`로 내부 잔고 vs 거래소 잔고 비교/알림을 확인할 수 있습니다.

---

## 6) 테스트

```bash
pytest -q
```

환경 제약으로 전체 테스트가 어려우면 최소한:

```bash
python -m compileall .
```

---


### 6-1. 백테스트 의사결정 추적 시각화(테스트 아티팩트)

아래 테스트는 백테스트 스텝에서 모델의 buy/hold/sell 신호(`filled_qty` 기반)와 `reward`, `equity` 추이를 아티팩트로 저장합니다.

```bash
pytest -q tests/test_backtest_trace_visualization.py
```

생성물(테스트 tmp 디렉토리):
- `trace.csv`: step별 signal(reward/equity 포함)
- `reward_equity.svg`: reward/equity 라인 차트

## 7) 현재 구현 범위 / 비구현 범위

구현됨(핵심):

- 설정 로딩/검증
- 피처 계산(offline/online) + parity 확인 유틸
- 트레이딩 환경/보상/체결 모델
- Binance REST + market WS payload 처리
- 리컨실/메트릭/알림 엔진 기본
- run artifact 기록
- **train 시 데이터 미존재 시 자동 bootstrap 데이터 생성**

비구현 또는 부분구현:

- `pipelines/trade.py`의 실거래 주문 전송/복구 자동화 고도화
- `integrations/binance_ws_user.py`의 listenKey 발급/갱신 자동화
- `integrations/telegram.py`의 재시도/레이트리밋 대응
- SB3 PPO/SAC 학습 파이프라인 실연결

상세 체크리스트는 `docs/CODEBASE_FUNCTION_MAP.md` 참고.

---

## 8) 권장 다음 작업

1. 실주문 lifecycle(전송/체결추적/실패복구) 고도화
2. user stream listenKey lifecycle 자동화
3. alert → telegram 운영 품질(재시도/한도) 강화
4. SB3 학습/로드/평가 파이프라인 연결
