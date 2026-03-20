#%%
"""02_machine_learning.py

Objetivo:
- Construir dataset longitudinal T→T+1 (ano T como features; ano T+1 como target)
- Treinar RandomForest + pipeline de pré-processamento
- Exportar:
  * modelo joblib
  * predições CSV
  * metadata JSON

Observação:
- Funções compartilhadas ficam em src/feature_eng.py
- Paths padronizados em src/paths.py
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from utils.paths import ABT_PATH, ML_OUT_DIR
from utils.feature_eng import (
    make_t_to_t1_dataset,
    drop_all_nan_and_constants,
    RFParams,
    build_rf_pipeline,
)

TRAIN_PAIR = (2022, 2023)
TEST_PAIR = (2023, 2024)

THRESHOLD_FINAL = 0.45

RF_PARAMS = RFParams(
    n_estimators=400,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

# proxies do ano T (ajuste se necessário)
PROXY_DROP = ["IAN_t"]

def main() -> None:
    ML_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ABT_PATH.exists():
        raise FileNotFoundError(f"ABT não encontrada: {ABT_PATH}")

    df = pd.read_parquet(ABT_PATH)

    train_pair = make_t_to_t1_dataset(df, *TRAIN_PAIR)
    test_pair = make_t_to_t1_dataset(df, *TEST_PAIR)

    y_train = train_pair["Target_Risco_t1"].astype(int)
    y_test = test_pair["Target_Risco_t1"].astype(int)

    drop_cols = ["RA", "Ano_t", "Target_Risco_t1"]
    X_train = train_pair.drop(columns=[c for c in drop_cols if c in train_pair.columns], errors="ignore")
    X_test = test_pair.drop(columns=[c for c in drop_cols if c in test_pair.columns], errors="ignore")

    # Proxy drop (apenas se existir)
    X_train = X_train.drop(columns=[c for c in PROXY_DROP if c in X_train.columns], errors="ignore")
    X_test = X_test.drop(columns=[c for c in PROXY_DROP if c in X_test.columns], errors="ignore")

    X_train, X_test, dropped_info = drop_all_nan_and_constants(X_train, X_test)

    clf = build_rf_pipeline(X_train, params=RF_PARAMS)
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_thr = (y_proba >= THRESHOLD_FINAL).astype(int)

    print("Threshold final:", THRESHOLD_FINAL)
    print(classification_report(y_test, y_pred_thr, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_thr))
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")

    # -----------------
    # Export
    # -----------------
    model_path = ML_OUT_DIR / "modelo_rf_t_to_t1.joblib"
    joblib.dump(clf, model_path)

    pred_path = ML_OUT_DIR / "predicoes_2024_usando_2023_thr045.csv"
    out_pred = pd.DataFrame(
        {
            "RA": test_pair["RA"].astype(str).values,
            "Target_Risco_2024_real": y_test.values,
            "Prob_Risco_2024": y_proba,
            "Flag_Acionar": y_pred_thr,
            "Pred_2024_thr045": y_pred_thr,
        }
    )
    out_pred.to_csv(pred_path, index=False, encoding="utf-8")

    meta_path = ML_OUT_DIR / "metadata_t_to_t1_thr045.json"

    feat_used = getattr(clf.named_steps["preprocess"], "feature_names_in_", None)
    if feat_used is not None:
        feat_used = list(map(str, list(feat_used)))

    meta = {
        "train_pair": TRAIN_PAIR,
        "test_pair": TEST_PAIR,
        "threshold": THRESHOLD_FINAL,
        "rf_params": {
            "n_estimators": RF_PARAMS.n_estimators,
            "random_state": RF_PARAMS.random_state,
            "class_weight": RF_PARAMS.class_weight,
            "n_jobs": RF_PARAMS.n_jobs,
        },
        "proxy_drop": PROXY_DROP,
        "dropped_info": dropped_info,
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "auc_test": float(auc),
        "features_used": feat_used,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[OK] Export concluído.")
    print("Modelo:", model_path)
    print("Predições:", pred_path)
    print("Metadata:", meta_path)
    print(f"Acionados: {int(y_pred_thr.sum())}/{len(y_pred_thr)} ({(y_pred_thr.mean()*100):.1f}%)")


if __name__ == "__main__":
    main()