#%%

import re
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path.cwd()
ABT_PATH = '../data/02_processed/abt.parquet'
MODEL_PATH = '../outputs/ml_t_to_t1/modelo_rf_t_to_t1.joblib'

TRAIN_PAIR = (2022, 2023)
TEST_PAIR = (2023, 2024)

DROP_COLS_MODEL = {"RA", "Ano_t", "Target_Risco_t1"}
SUSPECT_REGEX = re.compile(r"(defas|target|risco|label|nivel|ideal|atual)", re.IGNORECASE)

RANDOM_STATE = 42
THRESHOLD_DIAG = 0.45

ID_COL = "RA"
ANO_COL = "Ano"
TARGET_ABT = "Target_Risco"

def corr_with_target_numeric(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    num = X.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        return pd.Series(dtype=float)
    tmp = num.copy()
    tmp["_y_"] = y.values
    return tmp.corr(numeric_only=True)["_y_"].abs().sort_values(ascending=False)

def permutation_auc_drop(clf: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, top_n: int = 12):
    base = clf.predict_proba(X_test)[:, 1]
    base_auc = roc_auc_score(y_test, base)

    rng = np.random.default_rng(RANDOM_STATE)
    drops = []
    for col in X_test.columns:
        Xp = X_test.copy()
        vals = Xp[col].values.copy()
        idx = np.arange(len(vals))
        rng.shuffle(idx)
        Xp[col] = vals[idx]
        proba = clf.predict_proba(Xp)[:, 1]
        auc = roc_auc_score(y_test, proba)
        drops.append((col, base_auc - auc))

    imp = pd.DataFrame(drops, columns=["feature", "auc_drop"]).sort_values("auc_drop", ascending=False)
    return base_auc, imp.head(top_n)

def print_section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def make_t_to_t1_dataset(df: pd.DataFrame, year_t: int, year_t1: int) -> pd.DataFrame:
    ID_COL = "RA"
    ANO_COL = "Ano"
    TARGET_ABT = "Target_Risco"

    df_t = df[df[ANO_COL] == year_t].copy()
    df_t1 = df[df[ANO_COL] == year_t1].copy()

    df_t = df_t.sort_values(ID_COL).drop_duplicates(subset=[ID_COL], keep="first")
    df_t1 = df_t1.sort_values(ID_COL).drop_duplicates(subset=[ID_COL], keep="first")

    feat_cols = [c for c in df_t.columns if c != TARGET_ABT]
    df_t = df_t[feat_cols].copy()

    df_t1 = df_t1[[ID_COL, TARGET_ABT]].rename(columns={TARGET_ABT: "Target_Risco_t1"})

    paired = df_t.merge(df_t1, on=ID_COL, how="inner")
    rename_map = {c: f"{c}_t" for c in paired.columns if c not in [ID_COL, "Target_Risco_t1"]}
    return paired.rename(columns=rename_map)

def hash_df(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df, index=False)

def drop_all_nan_and_constants(X_train: pd.DataFrame, X_test: pd.DataFrame):
    removed_all_nan = [c for c in X_train.columns if X_train[c].isna().all()]
    X_train2 = X_train.drop(columns=removed_all_nan, errors="ignore")
    X_test2 = X_test.drop(columns=[c for c in removed_all_nan if c in X_test.columns], errors="ignore")

    removed_const = [c for c in X_train2.columns if X_train2[c].nunique(dropna=False) <= 1]
    X_train3 = X_train2.drop(columns=removed_const, errors="ignore")
    X_test3 = X_test2.drop(columns=[c for c in removed_const if c in X_test2.columns], errors="ignore")

    return X_train3, X_test3, {"removed_all_nan": removed_all_nan, "removed_constant": removed_const}

def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    categorical_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline([("preprocess", preprocessor), ("model", model)])

print_section("CONFIG")
print("ABT_PATH:", ABT_PATH)
print("MODEL_PATH:", MODEL_PATH)
print("TRAIN_PAIR:", TRAIN_PAIR, "| TEST_PAIR:", TEST_PAIR)

df = pd.read_parquet(ABT_PATH)

print_section("0) Sanidade do ABT")
print("Shape:", df.shape)
print("Cols:", list(df.columns))
print("\nDistribuição por ano:")
print(df["Ano"].value_counts().sort_index())
print("\nDistribuição do target (ABT):")
print(df["Target_Risco"].value_counts(dropna=False))

print_section("1) Construção dos pares T→T+1")
train_pair = make_t_to_t1_dataset(df, *TRAIN_PAIR)
test_pair = make_t_to_t1_dataset(df, *TEST_PAIR)
print("Train_pair shape:", train_pair.shape)
print("Test_pair  shape:", test_pair.shape)

print_section("2) Checagens de schema (anti-leakage)")
feature_cols_full = [c for c in train_pair.columns if c != "Target_Risco_t1"]

bad_t1_in_X = [c for c in feature_cols_full if c.endswith("_t1") or "t1" in str(c).lower()]
print("Colunas com 't1' em features:", bad_t1_in_X if bad_t1_in_X else "Nenhuma")

non_t_features = [
    c for c in train_pair.columns
    if (c not in [ID_COL, "Target_Risco_t1"]) and (not str(c).endswith("_t"))
]
print("Features sem sufixo _t:", non_t_features if non_t_features else "Nenhuma")

suspects = [c for c in feature_cols_full if SUSPECT_REGEX.search(str(c))]
print("Colunas suspeitas por nome (features):", suspects if suspects else "Nenhuma")

print_section("3) Split (pares) e montagem X/y")

y_train = train_pair["Target_Risco_t1"].astype(int)
y_test = test_pair["Target_Risco_t1"].astype(int)

X_train = train_pair.drop(columns=[c for c in DROP_COLS_MODEL if c in train_pair.columns], errors="ignore")
X_test = test_pair.drop(columns=[c for c in DROP_COLS_MODEL if c in test_pair.columns], errors="ignore")

X_train, X_test, dropped = drop_all_nan_and_constants(X_train, X_test)
print("X_train:", X_train.shape, "| X_test:", X_test.shape)
print("Removidos 100% NaN:", dropped["removed_all_nan"])

print_section("3b) Modelo exportado e schema real de features")
model = joblib.load(MODEL_PATH)

expected = list(model.named_steps["preprocess"].feature_names_in_)
extras_train = [c for c in X_train.columns if c not in expected]
extras_test = [c for c in X_test.columns if c not in expected]
missing_train = [c for c in expected if c not in X_train.columns]
missing_test = [c for c in expected if c not in X_test.columns]

for c in missing_train:
    X_train[c] = np.nan
for c in missing_test:
    X_test[c] = np.nan

X_train_model = X_train[expected].copy()
X_test_model = X_test[expected].copy()

print("Features esperadas pelo modelo:", len(expected))
print("Extras removidas (treino):", extras_train if extras_train else "Nenhuma")
print("Extras removidas (teste) :", extras_test if extras_test else "Nenhuma")
print("Faltantes preenchidas com NaN (treino):", missing_train if missing_train else "Nenhuma")
print("Faltantes preenchidas com NaN (teste) :", missing_test if missing_test else "Nenhuma")
print("X_train_model:", X_train_model.shape, "| X_test_model:", X_test_model.shape)

print_section("4) Overlap de linhas idênticas (hash de features)")
common_cols = sorted(set(X_train_model.columns).intersection(set(X_test_model.columns)))
train_hash = hash_df(X_train_model[common_cols])
test_hash = hash_df(X_test_model[common_cols])
overlap = len(set(train_hash).intersection(set(test_hash)))
print(f"Overlap de hashes (features) train vs test: {overlap} / {len(test_hash)}")

print_section("5) Correlação (numéricas) com y no treino (apenas features do modelo exportado)")
corr = corr_with_target_numeric(X_train_model, y_train)
if len(corr) == 0:
    print("Sem numéricas para correlação.")
else:
    print(corr.head(15).to_string())
    corr_no_y = corr.drop(labels=["_y_"], errors="ignore")
    if (corr_no_y >= 0.98).any():
        print("\n⚠️ ALERTA: correlação >= 0.98 em feature numérica (proxy forte).")
        print(corr_no_y[corr_no_y >= 0.98].head(10).to_string())

print_section("6) Avaliação do MODELO EXPORTADO (AUC train vs AUC test)")
proba_train = model.predict_proba(X_train_model)[:, 1]
proba_test = model.predict_proba(X_test_model)[:, 1]

auc_train = roc_auc_score(y_train, proba_train)
auc_test = roc_auc_score(y_test, proba_test)
print(f"ROC-AUC train: {auc_train:.4f}")
print(f"ROC-AUC test : {auc_test:.4f}")

y_pred_thr = (proba_test >= THRESHOLD_DIAG).astype(int)
print(f"\nThreshold diagnóstico ({THRESHOLD_DIAG}) – relatório rápido:")
print(classification_report(y_test, y_pred_thr, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_thr))
print(f"\nResumo operacional: acionados={int(y_pred_thr.sum())}/{len(y_pred_thr)} ({(y_pred_thr.mean()*100):.1f}%)")


print_section("7) Permutation importance (queda de AUC no test)")
base_auc, imp = permutation_auc_drop(model, X_test_model, y_test, top_n=12)
print(f"Base ROC-AUC (test): {base_auc:.4f}")
print(imp.to_string(index=False))

print_section("8) Label permutation test (anti-leakage)")
N_PERM = 30
PERM_P95_OK = 0.62
PERM_P95_WARN = 0.65

rng = np.random.default_rng(RANDOM_STATE)
aucs = []

for i in range(N_PERM):
    y_perm = pd.Series(rng.permutation(y_train.values), index=y_train.index)

    clf_perm = build_pipeline(X_train_model)
    clf_perm.fit(X_train_model, y_perm)

    proba_perm = clf_perm.predict_proba(X_test_model)[:, 1]
    aucs.append(roc_auc_score(y_test, proba_perm))

aucs = np.array(aucs, dtype=float)
mean_auc = float(aucs.mean())
p95_auc = float(np.quantile(aucs, 0.95))
max_auc = float(aucs.max())

print(f"Permutation AUC mean: {mean_auc:.4f}")
print(f"Permutation AUC p95 : {p95_auc:.4f}")
print(f"Permutation AUC max : {max_auc:.4f}")

if p95_auc <= PERM_P95_OK:
    print(f"OK: p95 <= {PERM_P95_OK:.2f}. Evidência contra leakage estrutural.")
elif p95_auc <= PERM_P95_WARN:
    print(f"⚠️ ATENÇÃO: p95 entre {PERM_P95_OK:.2f} e {PERM_P95_WARN:.2f}. "
          f"Pode ser variação estatística; sugerindo aumentar N_PERM e revisar schema.")
else:
    print(f"🚨 ALERTA: p95 > {PERM_P95_WARN:.2f}. Suspeita forte de leakage/bug de pipeline/schema.")

print("\n[OK] Checks T→T+1 concluídos.")