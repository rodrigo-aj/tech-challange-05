#%%

from pathlib import Path

# Raiz do repositório: assume execução a partir de qualquer subpasta do projeto

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "01_raw"
PROCESSED_DIR = DATA_DIR / "02_processed"

HIST_DIR = DATA_DIR / "03_hist"
HIST_PARQUET_DIR = PROCESSED_DIR / "hist_parquet"

ABT_PATH = PROCESSED_DIR / "abt.parquet"

OUTPUTS_DIR = REPO_ROOT / "outputs"
EDA_OUT_DIR = OUTPUTS_DIR / "eda"
ML_OUT_DIR = OUTPUTS_DIR / "ml_t_to_t1"
MODEL_DIR = ML_OUT_DIR / "modelo_rf_t_to_t1.joblib"

REPORTS_DIR = REPO_ROOT / "reports"