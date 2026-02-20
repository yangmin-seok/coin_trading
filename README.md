# coin_trading

RL 기반 코인 트레이딩 프레임워크 스캐폴드입니다.  
현재는 **데이터/피처/환경/런 메타 기록/시장데이터 처리의 핵심 골격**에 더해, 주문/리스크/user stream/알림/정책 래퍼의 기본 구현이 포함되어 있습니다.

## 1) 빠른 시작

## 1-1. 요구사항

- Python 3.11+
- (권장) 가상환경

## 1-2. 설치

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

네트워크 제한 환경에서는 의존성 설치가 실패할 수 있습니다.

## 1-3. 기본 테스트

```bash
pytest -q
```

또는 문법 컴파일 확인만 할 경우:

```bash
python -m compileall .
```

---

## 2) 설정 방법

기본 설정 파일은 `config/default.yaml` 입니다.

- 주요 키
  - `mode`: 실행 모드 (`demo` 등)
  - `exchange`, `market`, `symbol`, `interval`
  - `reward.*`: 보상 함수 파라미터
  - `execution.*`: 수수료/슬리피지/행동 변화 제한
  - `features.version`, `features.windows.*`
  - `split.train/val/test`

환경변수 오버라이드는 `APP__` prefix를 사용합니다.

예시:

```bash
export APP__SYMBOL=ETHUSDT
export APP__INTERVAL=1m
```

`config.loader.load_config()`가 YAML + 환경변수를 합쳐 검증합니다.

---

## 3) 프로젝트 구조

- `config/`: 설정 스키마 + 로더
- `data/`: 데이터 저장/검증/다운로더 뼈대
- `features/`: offline/online 피처 + parity 테스트
- `env/`: 트레이딩 환경/보상/체결 모델
- `execution/`: 시장데이터, 상태, 리컨실 + 기본 orders/risk
- `integrations/`: Binance REST/WS Market + WS User/Telegram 기본 송수신
- `monitoring/`: 메트릭/알림
- `pipelines/`: train/trade/test 파이프라인 조립
- `tests/`: 단위/통합 테스트
- `docs/CODEBASE_FUNCTION_MAP.md`: 폴더/함수별 구현 상태 + TODO 문서

---

## 4) 실행 가이드 (처음부터 끝까지)

## Step A. 설정 확인

1. `config/default.yaml`을 프로젝트 목적에 맞게 수정합니다.
2. 필요한 값만 `APP__...` 환경변수로 덮어씁니다.

## Step B. 학습 실행 산출물(run artifact) 생성

```bash
python -m pipelines.train
```

실행 후 `runs/<run_id>/` 아래에 다음이 생성됩니다.

- `config.yaml`
- `meta.json`
- `data_manifest.json`
- `feature_manifest.json`

## Step C. 트레이드 런타임 빌드/점검

```bash
python -c "from pipelines.trade import run; print(run())"
```

기본 모드에서는 런타임 조립 상태를 반환하고,
`run(max_events=...)`로 큐 이벤트 소비/피처 업데이트/리스크 판단/주문 의도 생성까지 점검할 수 있습니다.

## Step D. 리컨실 로직 단독 점검 예시

`pipelines.trade.reconcile_once(...)`를 통해 내부 잔고 vs 거래소 잔고 비교/알림을 테스트할 수 있습니다.

## Step E. 테스트 수행

```bash
pytest -q
```

환경 제약으로 전체 테스트가 어려우면 최소한 아래를 권장합니다.

```bash
python -m compileall .
```

---

## 5) 현재 구현 범위 / 비구현 범위

구현됨(핵심):

- 설정 로딩/검증
- 피처 계산(offline/online) + parity 확인 유틸
- 트레이딩 환경/보상/체결 모델
- Binance REST + market WS payload 처리
- 리컨실/메트릭/알림 엔진 기본
- run artifact 기록

비구현 또는 부분구현:

- `pipelines/trade.py`의 실거래 주문 전송/복구 자동화 고도화
- `integrations/binance_ws_user.py`의 listenKey 발급/갱신 자동화
- `integrations/telegram.py`의 재시도/레이트리밋 대응
- SB3 PPO/SAC 학습 파이프라인 실연결

상세 체크리스트는 `docs/CODEBASE_FUNCTION_MAP.md` 참고.

---

## 6) 권장 다음 작업

1. user stream + orders/risk를 먼저 완성해 실거래 루프를 닫기
2. trade loop graceful shutdown/retry 정책 강화
3. monitoring(alert → telegram) 운영 연계
4. SB3 에이전트 실연결 및 백테스트 자동화
