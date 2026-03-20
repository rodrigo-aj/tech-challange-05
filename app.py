import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import joblib

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

THRESHOLD_FINAL = 0.45
from utils.paths import ABT_PATH, MODEL_DIR

# Proxy drop (igual ao treino)
PROXY_DROP = ["IAN_t"]

# Colunas que não entram no modelo (iguais ao treino)
DROP_COLS = ["RA", "Ano_t", "Target_Risco_t1"]

def extract_idaluno_from_ra(ra: pd.Series) -> pd.Series:
    """RA-123 -> 123"""
    s = ra.astype(str).str.extract(r"RA-(\d+)")[0]
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def make_features_for_year(df_abt: pd.DataFrame, year_t: int) -> pd.DataFrame:
    """
    Monta um DF com o mesmo padrão de features de treino:
    - pega Ano == year_t
    - remove duplicatas por RA
    - remove Target_Risco (do ABT)
    - renomeia colunas (exceto RA) com sufixo _t
    """
    if "Ano" not in df_abt.columns or "RA" not in df_abt.columns:
        raise ValueError("ABT precisa conter colunas 'RA' e 'Ano'.")

    df_t = df_abt[df_abt["Ano"] == year_t].copy()
    df_t = df_t.sort_values("RA").drop_duplicates(subset=["RA"], keep="first")

    # remove target do ABT (se existir)
    if "Target_Risco" in df_t.columns:
        df_t = df_t.drop(columns=["Target_Risco"], errors="ignore")

    # renomeia tudo (menos RA) para *_t
    rename_map = {c: f"{c}_t" for c in df_t.columns if c != "RA"}
    df_t = df_t.rename(columns=rename_map)

    return df_t


def align_features_to_model(X: pd.DataFrame, clf) -> pd.DataFrame:
    """
    Alinha colunas do X às colunas esperadas pelo preprocessor do pipeline.
    - remove extras
    - adiciona faltantes com NaN
    - reordena
    """
    preprocess = clf.named_steps.get("preprocess", None)
    if preprocess is None:
        raise ValueError("Pipeline sem etapa 'preprocess'. Verifique o modelo exportado.")

    expected = getattr(preprocess, "feature_names_in_", None)
    if expected is None:
        # fallback: usar colunas atuais
        return X.copy()

    expected = list(map(str, list(expected)))

    X2 = X.copy()
    # adiciona faltantes
    for c in expected:
        if c not in X2.columns:
            X2[c] = np.nan
    # remove extras e reordena
    X2 = X2[expected].copy()
    return X2


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {path}")
    return joblib.load(path)


@st.cache_data
def load_abt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ABT não encontrada em: {path}")
    df = pd.read_parquet(path)
    # normaliza
    df["RA"] = df["RA"].astype(str).str.strip()
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")
    return df


def predict_for_year(df_abt: pd.DataFrame, clf, year_t: int) -> pd.DataFrame:
    """
    Retorna dataframe com RA + Prob + Flags para todos do ano_t.
    """
    feats = make_features_for_year(df_abt, year_t)

    # drop cols que não entram no modelo
    X = feats.drop(columns=[c for c in DROP_COLS if c in feats.columns], errors="ignore")

    # proxy drop (se existir)
    X = X.drop(columns=[c for c in PROXY_DROP if c in X.columns], errors="ignore")

    # alinha ao modelo
    X_model = align_features_to_model(X, clf)

    proba = clf.predict_proba(X_model)[:, 1]
    flag = (proba >= THRESHOLD_FINAL).astype(int)

    out = pd.DataFrame(
        {
            "RA": feats["RA"].values,
            "Ano_T": year_t,
            "Ano_T1_previsto": year_t + 1,
            "Prob_Risco_T1": proba,
            "Flag_Acionar": flag,
        }
    )
    out = out.sort_values("Prob_Risco_T1", ascending=False).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Passos Mágicos - Predição de Risco", layout="wide")

st.title("Predição de risco de defasagem (T→T+1)")
st.caption(
    "Modelo treinado para estimar a probabilidade de um aluno entrar em risco no próximo ano "
    "a partir dos indicadores do ano atual (Ano T)."
)

# Sidebar: paths / sanity
with st.sidebar:

    st.divider()
    st.subheader("Notas")
    st.write(
        "A coluna Prob_Risco_T1 é um score contínuo (0–1). "
        "Flag_Acionar é a decisão operacional baseada no threshold."
    )

# Load data/model
try:
    clf = load_model(MODEL_DIR)
    abt = load_abt(ABT_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

years = sorted([int(y) for y in abt["Ano"].dropna().unique().tolist()])
if not years:
    st.error("Nenhum ano encontrado na ABT.")
    st.stop()

tabs = st.tabs(["Aluno", "Priorizar"])

# -----------------------------------------------------------------------------
# Tab 1: Aluno individual
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Avaliação individual")

    colA, colB, colC = st.columns([1, 2, 1])

    with colA:
        year_t = st.selectbox("Ano T (features)", years, index=len(years) - 2 if len(years) >= 2 else 0)

    # lista de RAs do ano
    df_year = abt[abt["Ano"] == year_t].copy()
    ra_list = sorted(df_year["RA"].astype(str).unique().tolist())

    with colB:
        ra_filter = st.text_input("Filtro (opcional) para RA", value="")
        if ra_filter.strip():
            ra_filtered = [r for r in ra_list if ra_filter.strip().lower() in r.lower()]
        else:
            ra_filtered = ra_list

        if not ra_filtered:
            st.warning("Nenhum RA encontrado com esse filtro.")
            st.stop()

        ra_sel = st.selectbox("Selecionar RA", ra_filtered)

    with colC:
        st.write("")
        st.write("")
        run_btn = st.button("Calcular probabilidade", use_container_width=True)

    if run_btn:
        # monta features do ano selecionado
        feats_year = make_features_for_year(abt, year_t)
        row = feats_year[feats_year["RA"] == ra_sel].copy()
        if row.empty:
            st.error("RA não encontrado no ano selecionado.")
            st.stop()

        # prepara X
        X = row.drop(columns=[c for c in DROP_COLS if c in row.columns], errors="ignore")
        X = X.drop(columns=[c for c in PROXY_DROP if c in X.columns], errors="ignore")
        X_model = align_features_to_model(X, clf)

        proba = float(clf.predict_proba(X_model)[:, 1][0])
        flag = int(proba >= THRESHOLD_FINAL)

        st.markdown("### Resultado")
        st.write(f"Ano T: {year_t}  |  Ano T+1 previsto: {year_t + 1}")
        st.write(f"Probabilidade de risco (T+1): {proba:.4f}")
        decision_text = (
            "Recomendação: Priorizar acompanhamento/intervenção para risco no próximo ano"
            if flag == 1
            else "Recomendação: Manter acompanhamento padrão (sem prioridade adicional)"
        )
        st.write(decision_text)
        st.caption(f"Regra: prioridade quando probabilidade ≥ {THRESHOLD_FINAL:.2f}.")

        st.divider()
        st.markdown("### Indicadores do ano T (entrada do modelo)")
        # mostra os indicadores do ABT do ano T (linha original)
        original = abt[(abt["Ano"] == year_t) & (abt["RA"] == ra_sel)].copy()
        if not original.empty:
            st.dataframe(original.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Linha original não encontrada na ABT (inesperado).")

# -----------------------------------------------------------------------------
# Tab 2: Priorizar (ranking)
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Priorizar alunos (ranking)")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        year_t_rank = st.selectbox("Ano T para ranking", years, index=len(years) - 2 if len(years) >= 2 else 0, key="year_rank")

    with col2:
        top_n = st.number_input("Top N", min_value=10, max_value=5000, value=50, step=10)

    with col3:
        st.write("")
        st.write("")
        gen_btn = st.button("Gerar ranking", use_container_width=True)

    if gen_btn:
        ranked = predict_for_year(abt, clf, year_t_rank)

        ranked["Recomendacao"] = np.where(
            ranked["Flag_Acionar"] == 1,
            "Priorizar acompanhamento/intervenção (risco no próximo ano)",
            "Acompanhamento padrão",
        )

        acionados = int(ranked["Flag_Acionar"].sum())
        total = int(len(ranked))
        pct = float(acionados / total * 100) if total else 0.0

        st.markdown("### Resumo")
        st.write(f"Ano T: {year_t_rank}  |  Ano T+1 previsto: {year_t_rank + 1}")
        st.write(f"Acionados (threshold={THRESHOLD_FINAL:.2f}): {acionados}/{total} ({pct:.1f}%)")

        st.divider()
        st.markdown("### Top alunos por probabilidade")
        show = ranked.head(int(top_n)).copy()
        cols = ["RA", "Ano_T", "Ano_T1_previsto", "Prob_Risco_T1", "Recomendacao", "Flag_Acionar"]
        st.dataframe(show[cols], use_container_width=True)