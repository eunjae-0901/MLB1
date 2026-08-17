# 새 방법론 1: 동적 위험 예측 데이터

Notion의 `(발표)MLB 투수 단기 부상 예측 연구 - (5) 동적 위험 예측으로 연구 재설계 (~26.08.18)`에 맞춘 데이터 파이프라인이다.

## 데이터 정의

- 예측 단위: 투수 × 현재 snapshot
- snapshot: 실제 등판일 가운데 직전 선택 시점과 최소 5일 간격인 시점
- 입력 정보: snapshot 당일까지 알려진 정보만 사용
- 주 타깃: 향후 100일 이내 strict shoulder/elbow IL
- 보조 타깃: 향후 30일 및 60일 이내 strict shoulder/elbow IL
- 회귀 타깃: 다음 strict shoulder/elbow IL까지 남은 일수
- 시간 분할: Train 2016–2021, Validation 2022–2023, Final Test 2024–2025
- 역할 구분: starter / bullpen
- 이벤트 가중치: 하나의 부상 사건에 연결된 양성 snapshot 가중치의 합이 1이 되도록 설정

## 실행

```powershell
python "새 방법론1 파일/01_collect_and_match_targets.py"
```

## 산출물

- `data/dynamic_snapshot_targets.parquet`: snapshot, 타깃, 이벤트 가중치
- `data/pitcher_game_features.parquet`: 투수-경기 단위 원천 feature
- `results/target_balance.csv`: split/role/horizon별 타깃 비율
- `results/event_coverage.csv`: 부상 이벤트별 연결 snapshot 수
- `results/generalization_cohorts.csv`: 기존 선수/완전 신규 선수별 100일 타깃 구성
- `results/data_audit.json`: 중복, 기간, 관찰 마감일 검사

대용량 `data/` 산출물은 Git에서 제외되며 스크립트로 재생성한다. 결과 요약 CSV와 audit JSON은 추적한다.

## 우측 검열 처리

부상 원자료의 마지막 관찰일보다 예측 horizon이 뒤로 넘어가는 snapshot은 음성으로 확정할 수 없다. 따라서 `observed_30d`, `observed_60d`, `observed_100d`에 추적 완료 여부를 저장하고, 추적이 끝나지 않은 타깃은 결측 처리한다. 모델 학습과 평가는 해당 horizon의 결측 행을 제외해야 한다.

다음 단계에서는 이 index를 기준으로 최근 100일을 20개의 5일 bin으로 만들고, Train에서만 전처리·불균형 대응·threshold 선택을 수행한다.

## 논문 평가 전략

주 분석은 동일 선수의 과거 시즌 부상 및 workload 이력을 유지한다. 단, 선수 ID 자체는 모델 feature로 사용하지 않는다. 전체 Test 성능을 주 결과로 보고하고, Train 기간에 한 번도 등장하지 않은 `new_player` Test 성능을 추가 일반화 검증으로 함께 보고한다. `seen_in_train`과 `evaluation_cohort` 열로 두 코호트를 구분한다.

## 모델 실험

1. `01_XGBoost_CPU.ipynb`: 사건 가중치와 class balance를 적용한 CPU 주 기준선
2. `02_LSTM_GPU.ipynb`: 2025 논문 LSTM을 동적 snapshot 분류+회귀로 확장
3. `03_ViT_GPU.ipynb`: 2025 논문 ViT를 동적 snapshot 분류+회귀로 확장
4. `04_ResNet_GPU.ipynb`: LSTM의 신규 선수 일반화·calibration 개선용 1D Residual TCN

공통 입력은 `02_build_100d_sequences.py`가 생성하는 20×33 시계열이다. XGBoost는 실행 완료 상태이며 LSTM과 ViT는 CUDA GPU가 필요하다. GPU 실행 후 `05_combine_all_results.py`를 실행하면 `results/all_models_summary.csv`와 비교 PNG가 생성된다.
