"""
여러 스크립트가 공유하는 지표(feature) 정의.
03_build_rolling_dataset.py, 05_build_sequence_arrays.py, eda/build_sequence_flat_for_review.py가
동일한 METRICS/PITCH_GROUPS 정의를 써야 각 산출물의 컬럼 이름/의미가 서로 어긋나지 않는다.
"""

PITCH_GROUPS = ["FB", "SI", "CT", "SL", "CB", "CH", "SP"]

# (짧은 이름, SUM에 쓸 표현식, 결측 여부를 판단할 원본 컬럼)
# release_speed/pfx_x/pfx_z/release_extension/release_spin_rate는 트래킹 장비 오류로
# 경기 전체가 통째로 NULL이 되는 경우가 있는데(예: 2019-09-26 한 경기 전체), 이걸
# n_pitches(전체 투구수)로 나누면 평균이 크게 낮게 잡히는 버그가 있었다. 그래서 각 지표마다
# "값이 실제로 있는 투구 수"를 따로 세서 그걸로 나눈다.
METRICS = [
    ("v", "release_speed", "release_speed"),
    ("x", "pfx_x * 12", "pfx_x"),
    ("z", "pfx_z * 12", "pfx_z"),
    ("ext", "release_extension", "release_extension"),
    ("spin", "release_spin_rate", "release_spin_rate"),
]
