#%%
"""01_data_ingestion.py

Objetivo:
- Ler o XLSX oficial (data/01_raw/base-raw.xlsx)
- Padronizar colunas por ano
- Gerar ABT consolidada (data/02_processed/abt.parquet)

Observação:
- Funções de padronização ficam em src/data_cleaning.py
"""

from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.paths import RAW_DIR, PROCESSED_DIR, ABT_PATH
from utils.data_cleaning import AbtConfig, build_abt_from_xlsx

INPUT_XLSX = RAW_DIR / "base-raw.xlsx"

SHEETS = {
    "PEDE2022": 2022,
    "PEDE2023": 2023,
    "PEDE2024": 2024,
}

CFG = AbtConfig(metrics=("IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV"))

def main() -> None:
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_XLSX}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    abt = build_abt_from_xlsx(
        xlsx_path=INPUT_XLSX,
        sheets=SHEETS,
        cfg=CFG,
        target_rule="defasagem_lt_0",
    )

    abt.to_parquet(ABT_PATH, index=False)
    print("[OK] ABT gerada:")
    print("- path :", ABT_PATH)
    print("- shape:", abt.shape)
    print("- anos :", sorted(abt["Ano"].dropna().unique().tolist()))


if __name__ == "__main__":
    main()