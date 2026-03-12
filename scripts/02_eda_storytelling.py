#%%
"""02_eda_storytelling.py

Objetivo:
- Carregar ABT e gerar gráficos/insights (EDA)
- Salvar figuras em outputs/eda

Observação:
- Este arquivo mantém a lógica de EDA no próprio notebook/script,
  mas usa src/paths.py para caminhos padronizados.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.paths import ABT_PATH, EDA_OUT_DIR

EDA_OUT_DIR.mkdir(parents=True, exist_ok=True)

PEDRA_ORDER = ["QUARTZO", "AGATA", "AMETISTA", "TOPAZIO"]

def _done(title: str):
    print(f"{title} - Concluído")

def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(EDA_OUT_DIR / name, dpi=150)
    plt.close()

def bin_series(s: pd.Series, q: int = 4, label_prefix: str = "") -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
    try:
        b = pd.qcut(s2, q=q, duplicates="drop")
        return b.astype(str).map(lambda x: f"{label_prefix}{x}")
    except Exception:
        return pd.cut(s2, bins=q).astype(str).map(lambda x: f"{label_prefix}{x}")
    
def bin_fixed_ian(s: pd.Series) -> pd.Series:
    """
    Faixas fixas (ajuste se quiser alinhar ao dicionário oficial):
    Aqui é só uma discretização coerente para visualização.
    """
    s2 = pd.to_numeric(s, errors="coerce")
    bins = [-np.inf, 4, 6, 8, np.inf]
    labels = ["IAN <= 4", "IAN 4–6", "IAN 6–8", "IAN > 8"]
    return pd.cut(s2, bins=bins, labels=labels)
    
def safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=False)

def plot_charts(df: pd.DataFrame, num_questao: int) -> None:
    match int(num_questao):
        case 1:
            # ==========================
            # Q1 — IAN
            # ==========================
            if {"IAN", "Ano"}.issubset(df.columns):
                title = "[Q1] Perfil de defasagem (IAN) por ano — % por faixa"
                df_tmp = safe_copy(df)
                df_tmp["IAN_faixa"] = bin_fixed_ian(df_tmp["IAN"])

                tab = (
                    df_tmp.groupby(["Ano", "IAN_faixa"], dropna=False)
                    .size()
                    .reset_index(name="n")
                )
                tab["pct"] = tab["n"] / tab.groupby("Ano")["n"].transform("sum")

                plt.figure(figsize=(10, 4))
                pivot = tab.pivot(index="Ano", columns="IAN_faixa", values="pct").fillna(0)
                pivot.plot(kind="bar", stacked=True, ax=plt.gca())
                plt.title(title)
                plt.ylabel("Proporção de alunos")
                plt.xlabel("Ano")
                plt.legend(title="Faixa IAN", bbox_to_anchor=(1.02, 1), loc="upper left")
                savefig("Q1_ian_faixas_por_ano.png")
                plt.show()
                _done(title)

                title = "[Q1b] Distribuição do IAN por ano (violino)"
                plt.figure(figsize=(10, 4))
                sns.violinplot(data=df, x="Ano", y="IAN", inner="quartile", cut=0)
                plt.title(title)
                savefig("Q1b_ian_violin_por_ano.png")
                plt.show()
                _done(title)

        case 2:
            # ==========================
            # Q2 — IDA
            # ==========================
            if {"IDA", "Ano"}.issubset(df.columns):
                title = "[Q2] IDA — tendência (média) por ano"
                trend = df.groupby("Ano", dropna=False)["IDA"].mean().reset_index()

                plt.figure(figsize=(8, 4))
                sns.lineplot(data=trend, x="Ano", y="IDA", marker="o")
                plt.title(title)
                plt.ylabel("IDA médio")
                savefig("Q2_ida_media_por_ano.png")
                plt.show()
                _done(title)

                title = "[Q2b] IDA — distribuição por ano (boxplot)"
                plt.figure(figsize=(10, 4))
                sns.boxplot(data=df, x="Ano", y="IDA")
                plt.title(title)
                savefig("Q2b_ida_box_por_ano.png")
                plt.show()
                _done(title)

                fase_col = None
                for cand in ["Fase", "fase", "Periodo", "Período", "Bimestre", "Trimestre"]:
                    if cand in df.columns:
                        fase_col = cand
                        break

                if fase_col:
                    title = f"[Q2c] IDA — evolução por {fase_col} e Ano"
                    tmp = df.groupby(["Ano", fase_col], dropna=False)["IDA"].mean().reset_index()
                    plt.figure(figsize=(10, 4))
                    sns.lineplot(data=tmp, x=fase_col, y="IDA", hue="Ano", marker="o")
                    plt.title(title)
                    plt.ylabel("IDA médio")
                    savefig("Q2c_ida_por_fase_e_ano.png")
                    plt.show()
                    _done(title)

        case 3:
            # ==========================
            # Q3 — IEG vs IDA/IPV (barras por faixas)
            # ==========================
            if {"IEG", "IPV", "Ano"}.issubset(df.columns):
                title = "[Q3-alt] IPV médio por faixa de IEG (por Ano) — sem NaN"
                df_tmp = df.copy()
                df_tmp = df_tmp[df_tmp["IEG"].notna() & df_tmp["IPV"].notna()].copy()
                df_tmp["IEG_faixa"] = bin_series(df_tmp["IEG"], q=4, label_prefix="IEG ")

                grp = df_tmp.groupby(["Ano", "IEG_faixa"], dropna=False)["IPV"].mean().reset_index()

                plt.figure(figsize=(12, 4))
                sns.barplot(data=grp, x="IEG_faixa", y="IPV", hue="Ano")
                plt.xticks(rotation=30, ha="right")
                plt.title(title)
                plt.ylabel("IPV médio")
                savefig("Q3_alt_ipv_medio_por_faixa_ieg_por_ano_sem_nan.png")
                plt.show()
                _done(title)

            if {"IEG", "IDA", "Ano"}.issubset(df.columns):
                title = "[Q3-alt] IDA médio por faixa de IEG (por Ano) — sem NaN"
                df_tmp = df.copy()
                df_tmp = df_tmp[df_tmp["IEG"].notna() & df_tmp["IDA"].notna()].copy()
                df_tmp["IEG_faixa"] = bin_series(df_tmp["IEG"], q=4, label_prefix="IEG ")

                grp = df_tmp.groupby(["Ano", "IEG_faixa"], dropna=False)["IDA"].mean().reset_index()

                plt.figure(figsize=(12, 4))
                sns.barplot(data=grp, x="IEG_faixa", y="IDA", hue="Ano")
                plt.xticks(rotation=30, ha="right")
                plt.title(title)
                plt.ylabel("IDA médio")
                savefig("Q3_alt_ida_medio_por_faixa_ieg_por_ano_sem_nan.png")
                plt.show()
                _done(title)

        case 4:
            # ==========================
            # Q4 — IAA coerência (barras por faixas + contagens)
            # ==========================
            if {"IAA", "IDA", "Ano"}.issubset(df.columns):
                title = "[Q4-alt] IDA médio por faixa de IAA (por Ano)"
                df_tmp = df.copy()
                df_tmp = df_tmp[df_tmp["IAA"].notna() & df_tmp["IDA"].notna()].copy()
                df_tmp["IAA_faixa"] = bin_series(df_tmp["IAA"], q=4, label_prefix="IAA ")

                counts = df_tmp.groupby(["Ano", "IAA_faixa"], dropna=False).size().reset_index(name="n")
                print("\n[Q4] Contagem de alunos por Ano x Faixa IAA (IDA):")
                print(counts.pivot(index="IAA_faixa", columns="Ano", values="n").fillna(0).astype(int))

                grp = df_tmp.groupby(["Ano", "IAA_faixa"], dropna=False)["IDA"].mean().reset_index()

                plt.figure(figsize=(12, 4))
                sns.barplot(data=grp, x="IAA_faixa", y="IDA", hue="Ano")
                plt.xticks(rotation=30, ha="right")
                plt.title(title)
                plt.ylabel("IDA médio")
                savefig("Q4_alt_ida_medio_por_faixa_iaa_por_ano.png")
                plt.show()
                _done(title)

            if {"IAA", "IEG", "Ano"}.issubset(df.columns):
                title = "[Q4-alt] IEG médio por faixa de IAA (por Ano)"
                df_tmp = df.copy()
                df_tmp = df_tmp[df_tmp["IAA"].notna() & df_tmp["IEG"].notna()].copy()
                df_tmp["IAA_faixa"] = bin_series(df_tmp["IAA"], q=4, label_prefix="IAA ")

                counts = df_tmp.groupby(["Ano", "IAA_faixa"], dropna=False).size().reset_index(name="n")
                print("\n[Q4] Contagem de alunos por Ano x Faixa IAA (IEG):")
                print(counts.pivot(index="IAA_faixa", columns="Ano", values="n").fillna(0).astype(int))

                grp = df_tmp.groupby(["Ano", "IAA_faixa"], dropna=False)["IEG"].mean().reset_index()

                plt.figure(figsize=(12, 4))
                sns.barplot(data=grp, x="IAA_faixa", y="IEG", hue="Ano")
                plt.xticks(rotation=30, ha="right")
                plt.title(title)
                plt.ylabel("IEG médio")
                savefig("Q4_alt_ieg_medio_por_faixa_iaa_por_ano.png")
                plt.show()
                _done(title)

        case _:
            raise ValueError("num_questao deve ser um inteiro de 1 a 10.")

def main() -> None:
    if not ABT_PATH.exists():
        raise FileNotFoundError(f"ABT não encontrada: {ABT_PATH}")

    df = pd.read_parquet(ABT_PATH)

    plot_charts(df, 1)

if __name__ == "__main__":
    main()