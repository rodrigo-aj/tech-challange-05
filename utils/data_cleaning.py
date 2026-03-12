#%%



import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


# Text normalization
def strip_accents(text: str) -> str:
    """Remove acentuação mantendo caracteres base."""
    if text is None:
        return text
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(ch)
    )


def normalize_spaces(s: str) -> str:
    """Trim + colapsa múltiplos espaços."""
    return re.sub(r"\s+", " ", str(s)).strip()


def normalize_pedra(val) -> str:
    """Normaliza categoria 'Pedra' (caixa alta, sem acento, espaços normalizados)."""
    if pd.isna(val):
        return val
    s = strip_accents(str(val))
    s = normalize_spaces(s).upper()
    return s


# Column helpers
def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Retorna o primeiro nome de coluna existente em df dentre candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve colunas duplicadas (mesmo nome repetido).
    Estratégia: renomear duplicadas com sufixo __dupN.
    Isso evita df['COL'] retornar DataFrame (causa do TypeError no to_numeric).
    """
    cols = list(df.columns)
    seen: dict[str, int] = {}
    new_cols = []
    for c in cols:
        key = str(c)
        if key not in seen:
            seen[key] = 0
            new_cols.append(key)
        else:
            seen[key] += 1
            new_cols.append(f"{key}__dup{seen[key]}")
    out = df.copy()
    out.columns = new_cols
    return out


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Converte colunas para numérico (coerce)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ABT standardization
@dataclass(frozen=True)
class AbtConfig:
    metrics: tuple[str, ...] = ("IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV")


def standardize_year_df(df: pd.DataFrame, year: int, cfg: AbtConfig = AbtConfig()) -> pd.DataFrame:
    """
    Padroniza um DF anual para colunas canônicas:
      - RA
      - Ano
      - Defasagem
      - INDE
      - Pedra
      - métricas comuns (cfg.metrics) se existirem
    """
    df = dedupe_columns(df.copy())

    # RA
    ra_col = pick_first_existing(df, ["RA", "Ra", "ra"])
    if ra_col and ra_col != "RA":
        df.rename(columns={ra_col: "RA"}, inplace=True)

    # Defasagem: pode vir como Defas, DEFASAGEM etc.
    def_col = pick_first_existing(df, ["Defasagem", "Defas", "DEFASAGEM", "DEFAS"])
    if def_col and def_col != "Defasagem":
        df.rename(columns={def_col: "Defasagem"}, inplace=True)

    # INDE: pode vir como "INDE 22" / "INDE 2023" / etc.
    inde_candidates = ["INDE", f"INDE {str(year)[-2:]}", f"INDE {year}"]
    inde_candidates += [c for c in df.columns if str(c).strip().upper().startswith("INDE")]
    inde_col = pick_first_existing(df, inde_candidates)
    if inde_col and inde_col != "INDE":
        df.rename(columns={inde_col: "INDE"}, inplace=True)

    # Pedra: pode vir como "Pedra 22" / "Pedra 2023" / etc.
    pedra_candidates = ["Pedra", f"Pedra {str(year)[-2:]}", f"Pedra {year}"]
    pedra_candidates += [c for c in df.columns if str(c).strip().lower().startswith("pedra")]
    pedra_col = pick_first_existing(df, pedra_candidates)
    if pedra_col and pedra_col != "Pedra":
        df.rename(columns={pedra_col: "Pedra"}, inplace=True)

    if "Pedra" in df.columns:
        df["Pedra"] = df["Pedra"].apply(normalize_pedra)

    # Ano
    df["Ano"] = int(year)

    # Seleção de colunas úteis
    keep = ["RA", "Ano", "Pedra", "Defasagem", "INDE"]
    keep += [m for m in cfg.metrics if m in df.columns]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    # Tipagem
    if "RA" in out.columns:
        out["RA"] = out["RA"].astype(str).str.strip()

    num_cols = ["Defasagem", "INDE"] + [m for m in cfg.metrics if m in out.columns]
    out = coerce_numeric(out, num_cols)

    return out


def build_abt_from_xlsx(
    xlsx_path,
    sheets: dict[str, int],
    cfg: AbtConfig = AbtConfig(),
    target_rule: str = "defasagem_lt_0",
) -> pd.DataFrame:
    """
    Lê um XLSX com múltiplas abas (um ano por aba), padroniza e concatena em ABT.

    target_rule:
      - "defasagem_lt_0": Target_Risco = (Defasagem < 0)
    """
    xls = pd.ExcelFile(xlsx_path)
    frames = []

    for sheet, year in sheets.items():
        if sheet not in xls.sheet_names:
            raise ValueError(f"Aba '{sheet}' não encontrada. Abas disponíveis: {xls.sheet_names}")

        raw = pd.read_excel(xlsx_path, sheet_name=sheet)
        std = standardize_year_df(raw, year, cfg=cfg)

        if "Defasagem" not in std.columns:
            raise ValueError(f"Coluna Defasagem ausente após padronização no ano {year} (aba {sheet}).")

        if target_rule == "defasagem_lt_0":
            std["Target_Risco"] = (std["Defasagem"] < 0).astype("int8")
        else:
            raise ValueError(f"target_rule desconhecida: {target_rule}")

        frames.append(std)

    abt = pd.concat(frames, ignore_index=True)
    return abt