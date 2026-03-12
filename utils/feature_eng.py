#%%



from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# Dataset T -> T+1
def make_t_to_t1_dataset(
    df: pd.DataFrame,
    year_t: int,
    year_t1: int,
    id_col: str = "RA",
    year_col: str = "Ano",
    target_col: str = "Target_Risco",
    out_target_col: str = "Target_Risco_t1",
    suffix_t: str = "_t",
) -> pd.DataFrame:
    """
    Pareia registros por aluno (id_col) entre year_t e year_t1:
      - features: colunas do ano T com sufixo _t
      - target  : target do ano T+1 como out_target_col

    Saída contém id_col (sem sufixo) e out_target_col + features _t.
    """
    df_t = df[df[year_col] == year_t].copy()
    df_t1 = df[df[year_col] == year_t1].copy()

    if df_t.empty or df_t1.empty:
        raise ValueError(f"Ano(s) sem dados: T={year_t} ({len(df_t)}), T+1={year_t1} ({len(df_t1)})")

    # 1 linha por aluno/ano
    df_t = df_t.sort_values(id_col).drop_duplicates(subset=[id_col], keep="first")
    df_t1 = df_t1.sort_values(id_col).drop_duplicates(subset=[id_col], keep="first")

    # Features do ano T: tudo exceto target
    feat_cols = [c for c in df_t.columns if c != target_col]
    df_t = df_t[feat_cols].copy()

    # Target do ano T+1
    df_t1 = df_t1[[id_col, target_col]].copy().rename(columns={target_col: out_target_col})

    paired = df_t.merge(df_t1, on=id_col, how="inner")

    # sufixo nas features
    rename_map = {c: f"{c}{suffix_t}" for c in paired.columns if c not in [id_col, out_target_col]}
    return paired.rename(columns=rename_map)


def drop_all_nan_and_constants(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """
    Remove colunas:
      - 100% NaN no treino
      - constantes no treino (nunique <= 1)
    """
    removed_all_nan = [c for c in X_train.columns if X_train[c].isna().all()]
    X_train2 = X_train.drop(columns=removed_all_nan, errors="ignore")
    X_test2 = X_test.drop(columns=[c for c in removed_all_nan if c in X_test.columns], errors="ignore")

    removed_const = [c for c in X_train2.columns if X_train2[c].nunique(dropna=False) <= 1]
    X_train3 = X_train2.drop(columns=removed_const, errors="ignore")
    X_test3 = X_test2.drop(columns=[c for c in removed_const if c in X_test2.columns], errors="ignore")

    info = {"removed_all_nan": removed_all_nan, "removed_constant": removed_const}
    return X_train3, X_test3, info


# Pipeline builders
def _make_onehot():
    # compat sklearn: sparse_output (novo) vs sparse (antigo)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [
        c for c in X.columns
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string")
    ]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_onehot()),
            ]), categorical_cols),
        ],
        remainder="drop",
    )


@dataclass(frozen=True)
class RFParams:
    n_estimators: int = 400
    random_state: int = 42
    class_weight: str | dict[str, Any] | None = "balanced"
    n_jobs: int = -1


def build_rf_pipeline(X_train: pd.DataFrame, params: RFParams = RFParams()) -> Pipeline:
    pre = build_preprocessor(X_train)
    model = RandomForestClassifier(
        n_estimators=params.n_estimators,
        random_state=params.random_state,
        class_weight=params.class_weight,
        n_jobs=params.n_jobs,
    )
    return Pipeline([("preprocess", pre), ("model", model)])


# Model schema alignment
def align_to_model_schema(
    X: pd.DataFrame,
    model: Pipeline,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Alinha X ao schema real de features usado pelo pipeline exportado:
      - remove colunas extras
      - adiciona colunas faltantes com NaN
      - reordena conforme feature_names_in_ do preprocessor

    Retorna (X_alinhado, info).
    """
    pre = model.named_steps.get("preprocess")
    if pre is None or not hasattr(pre, "feature_names_in_"):
        raise ValueError("Pipeline não possui preprocess.feature_names_in_. Modelo não parece ser o esperado.")

    expected = list(pre.feature_names_in_)
    extras = [c for c in X.columns if c not in expected]
    missing = [c for c in expected if c not in X.columns]

    X2 = X.copy()
    for c in missing:
        X2[c] = np.nan

    X2 = X2[expected].copy()
    info = {"extras_removed": extras, "missing_filled": missing}
    return X2, info