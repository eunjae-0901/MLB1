"""
01_xgboost_bayesian.py, 02_dnn_bayesian.py가 공유하는 입력변수 정리 유틸.

지금 rolling-window 데이터는 구종 그룹(FB/SI/CT/SL/CB/CH/SP)별로 구속/무브먼트/
익스텐션/회전수/사용비율을 전부 따로 담고 있어서 숫자 컬럼이 56개나 된다. 이 중에는
서로 거의 같은 정보를 담은 컬럼(예: 전체 평균 구속 w_v_all과 직구 평균 구속 w_vFB는
투수 대부분이 직구를 제일 많이 던지니 상관관계가 매우 높음)이 섞여있어서, 이런 중복을
줄이면 모델이 더 안정적으로 학습될 수 있다.

여기서는 두 가지 방법을 제공한다.
  1) select_uncorrelated_features: 상관관계가 너무 높은(기본 0.9 초과) 컬럼 쌍 중
     하나를 제거해서 중복을 줄인다. XGBoost/DNN 둘 다 적용 가능하고, 어떤 원래
     변수가 남았는지 그대로 보여서 해석하기 쉽다.
  2) apply_pca: 살아남은 숫자 컬럼들을 주성분(PCA)으로 압축한다. 해석은 어려워지지만
     차원을 더 줄이고 싶을 때(특히 표본 수 대비 변수가 너무 많은 DNN 쪽에) 쓴다.

두 방법 다 train 데이터 기준으로만 계산해서(fit) val/test에는 그 기준을 그대로
적용(transform)한다 - val/test 정보가 전처리 단계에서 새어들어가는 걸 막기 위함이다.
"""
import numpy as np
import pandas as pd


def select_uncorrelated_features(train_df: pd.DataFrame, candidate_cols: list[str],
                                  threshold: float = 0.9) -> list[str]:
    """train_df 기준 상관행렬을 계산해서, candidate_cols를 순서대로 훑으며
    이미 채택된 컬럼과 상관계수 절댓값이 threshold를 넘으면 그 컬럼을 버린다.
    남은(채택된) 컬럼 이름 리스트를 반환한다."""
    corr = train_df[candidate_cols].corr().abs()

    kept: list[str] = []
    dropped: list[tuple[str, str, float]] = []
    for col in candidate_cols:
        is_redundant = False
        for k in kept:
            c = corr.loc[col, k]
            if pd.notna(c) and c > threshold:
                dropped.append((col, k, c))
                is_redundant = True
                break
        if not is_redundant:
            kept.append(col)

    print(f"[feature_selection] 후보 {len(candidate_cols)}개 -> 상관관계(>|{threshold}|) 제거 후 {len(kept)}개")
    for col, kept_as, c in dropped:
        print(f"  제거: {col}  (기존 {kept_as}와 상관계수 {c:.3f})")
    return kept


def apply_pca(train_arr: np.ndarray, *other_arrs: np.ndarray, n_components: float = 0.95):
    """train_arr(이미 표준화된 상태 가정)로 PCA를 학습하고, train/그 외 배열들을
    같은 기준으로 변환한다. n_components가 1 미만의 실수면 "설명 분산 비율"
    기준으로 필요한 주성분 개수를 자동으로 정한다."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(train_arr)
    print(f"[feature_selection] PCA: 입력 {train_arr.shape[1]}차원 -> {train_pca.shape[1]}개 주성분"
          f" (설명 분산 비율 합계 {pca.explained_variance_ratio_.sum():.3f})")

    out = [train_pca]
    for arr in other_arrs:
        out.append(pca.transform(arr))
    return pca, out
