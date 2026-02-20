# coin_trading

RL 기반 코인 트레이딩 프레임워크입니다.
현재는 **train(기본 백테스트 학습 루프)까지 동작**하도록 정리되어 있고,
실거래 주문/유저스트림 일부는 후속 TODO 입니다.

## 1) 설치

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## 2) 설정

기본 설정 파일: `config/default.yaml`

환경변수 오버라이드 prefix는 `COIN_TRADING__` 입니다.

```bash
export COIN_TRADING__SYMBOL=ETHUSDT
export COIN_TRADING__INTERVAL=1m
```

## 3) 처음부터 끝까지 실행

### Step A. 학습(train)

```bash
python -m pipelines.train
```

생성 결과(`runs/<run_id>/`):

- `config.yaml`
- `meta.json`
- `data_manifest.json`
- `feature_manifest.json`
- `train_summary.json`

`train`은 현재 합성 캔들 데이터 기반으로 feature 계산 + `TradingEnv` 에피소드(기본 `buy_and_hold`)를 실행해 summary를 남깁니다.

### Step B. 트레이드 런타임 조립 확인

```bash
python -c "from pipelines.trade import run; print(run())"
```

### Step C. 테스트

```bash
pytest -q
```

환경상 전체 테스트가 어려우면:

```bash
python -m compileall .
```

## 4) 현재 상태 문서

구현 현황/TODO 단일 문서는 아래를 참고하세요.

- `docs/IMPLEMENTATION_STATUS.md`

## 5) 남은 주요 TODO

- `execution/orders.py`, `execution/risk.py` 실구현
- `integrations/binance_ws_user.py` 유저 스트림 구현
- `integrations/telegram.py` 알림 전송 연동
- `pipelines/trade.py` full event loop 구현
