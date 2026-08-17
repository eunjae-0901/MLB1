# 데이터 설명서

## [데이터 구조]

```text
당장 필요한 파일/
data/
  - bullpen_dataset.csv : 불펜 투수 rolling-window 데이터 (135,743행 x 65컬럼)
  - starter_dataset.csv : 선발 투수 rolling-window 데이터 (41,758행 x 65컬럼)
  - data_description.md : 이 설명서

01_XGBoost.ipynb : XGBoost + Bayesian Optimization 모델 (불펜/선발, 이진분류/3종분류)
02_DNN.ipynb     : DNN(MLP) + Bayesian Optimization 모델 (불펜/선발, 이진분류/3종분류)
```

두 CSV는 각각 `03_build_rolling_dataset.ipynb`(→ 필요없는거 폴더)로 Statcast 투구 기록에서
만든 최종 학습용 데이터입니다. 원본 원시 데이터(Statcast, 부상자명단, 선수 정보 등)와
이 데이터를 만드는 파이프라인 스크립트는 전부 `필요없는거/mlb_injury_project/` 안에
백업되어 있습니다.

## [행/컬럼 의미]

각 행은 **한 투수의 특정 시점 rolling-window(최근 등판 기록을 모은 것) 하나**를 의미합니다.
불펜은 최근 14일, 선발은 최근 3경기 기준으로 만들었습니다.

### 기본 식별자

| 컬럼 | 설명 |
| --- | --- |
| `player_id` | 선수 고유 ID |
| `window_end_date` | 이 window가 끝나는 기준일(가장 최근 등판일) |
| `p_throws` | 투구팔(R/L) |
| `age` | 기준일 시점 나이 |
| `height_inches`, `weight_lb` | 신장/체중 |
| `birth_country` | 출신 국가 |

### 등판 관련 변수

| 컬럼 | 설명 |
| --- | --- |
| `days_since_prev_game` | 직전 등판으로부터 경과일 |
| `n_pitches_window`, `n_batters_faced_window` | window 내 총 투구수/상대타자수 |
| `n_appearances_window` | window 내 등판 횟수 |
| `innings_pitched_window` | window 내 이닝 수 |
| `complete_games_window` | window 내 완투 수 |

### 구종별 지표 (`w_` 접두사)

구종 그룹 7개(FB=속구, SI=싱커, CT=커터, SL=슬라이더, CB=커브, CH=체인지업, SP=스플리터)별로
사용비율(`pct`)/구속(`v`)/수평무브먼트(`x`)/수직무브먼트(`z`)/익스텐션(`ext`)/회전수(`spin`)를
window 평균으로 담았습니다. `w_v_all` 등 `_all` 접미사는 전체 구종 평균입니다.

### 라벨 및 분리 기준

| 컬럼 | 설명 |
| --- | --- |
| `il_start_date` | (부상 발생 시) 부상자명단 등재일. 부상이 없으면 결측 |
| `injury_class_strict` | 부상 부위 상세 분류 |
| `days_to_injury` | window 종료일로부터 부상까지 남은 일수 |
| `label` | **0**=안 다침, **1**=어깨 부상, **2**=팔꿈치 부상, **3**=그 외 부상(허리/무릎 등 무관한 부상) |
| `split` | `train` / `val` / `test` — 시즌 경계 기준으로 미리 나눠둔 값. 같은 부상 사례가 여러 split에 걸쳐 나오지 않도록 buffer를 두고 분리함 |

**이진분류**로 쓸 때는 `label==3`(그 외) 행을 제외하고 `label>0`을 1로 합쳐서 사용합니다
(0=안 다침 vs 1=어깨 또는 팔꿈치). **3종분류**는 마찬가지로 `label==3` 행만 제외하고
0/1/2 그대로 사용합니다. 두 노트북(`01_XGBoost.ipynb`, `02_DNN.ipynb`) 모두 이 처리를
내부에서 자동으로 해줍니다.
