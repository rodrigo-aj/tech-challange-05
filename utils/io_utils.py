#%%

from pathlib import Path
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_encoding(sample: bytes) -> str:
    try:
        import chardet  # type: ignore
        enc = (chardet.detect(sample) or {}).get("encoding")
        return enc or "utf-8"
    except Exception:
        return "utf-8"


def guess_sep(first_line: str) -> str:
    return ";" if first_line.count(";") > first_line.count(",") else ","


def read_csv_auto(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    sample = raw[:20000]
    enc = detect_encoding(sample)

    text = sample.decode(enc, errors="ignore")
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    sep = guess_sep(first_line)

    for encoding in [enc, "utf-8-sig", "latin1", "cp1252", "utf-8"]:
        try:
            return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
        except Exception:
            continue

    alt_sep = "," if sep == ";" else ";"
    return pd.read_csv(path, sep=alt_sep, encoding="latin1", low_memory=False)


def to_parquet(df: pd.DataFrame, out_path: Path, engine: str = "fastparquet") -> None:
    ensure_dir(out_path.parent)
    df.to_parquet(out_path, engine=engine, compression="snappy", index=False)