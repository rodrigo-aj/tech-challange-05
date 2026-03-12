#%%

from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path("../data").resolve()
INPUT_DIR = BASE_DIR / "03_hist"
OUTPUT_DIR = BASE_DIR / "02_processed" / "03_hist_parquet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _detect_encoding(sample: bytes) -> str:
    try:
        import chardet  # type: ignore
        enc = (chardet.detect(sample) or {}).get("encoding")
        return enc or "utf-8"
    except Exception:
        return "utf-8"

def _guess_sep(first_line: str) -> str:
    return ";" if first_line.count(";") > first_line.count(",") else ","

def read_csv_safe(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    sample = raw[:20000]
    enc = _detect_encoding(sample)

    text = sample.decode(enc, errors="ignore")
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    sep = _guess_sep(first_line)

    for encoding in [enc, "utf-8-sig", "latin1", "cp1252", "utf-8"]:
        try:
            return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
        except Exception:
            continue

    alt_sep = "," if sep == ";" else ";"
    return pd.read_csv(path, sep=alt_sep, encoding="latin1", low_memory=False)

def to_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="snappy", index=False)

def to_snake(text: str) -> str:
    text = str(text).strip().replace(" ", "_")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.lower().strip("_")

def strip_tb_prefix(snake_name: str) -> str:
    return snake_name[3:] if snake_name.startswith("tb_") else snake_name

def is_under_folder(rel_parts: tuple[str, ...], folder_name: str) -> bool:
    return any(p.lower() == folder_name.lower() for p in rel_parts)

def find_entity(csv_path: Path) -> str:
    rel = csv_path.relative_to(INPUT_DIR)
    for part in rel.parts[:-1]:
        if part.startswith("Tb"):
            return part
    return csv_path.stem

def is_under_merge(rel_parts: tuple[str, ...]) -> bool:
    return any(p.lower() == "merge" for p in rel_parts)

def first_subfolder_under_entity(rel_parts: tuple[str, ...], entity: str) -> str | None:
    parts = list(rel_parts)
    try:
        i = parts.index(entity)
    except ValueError:
        return None
    if len(parts) >= i + 3:
        return parts[i + 1]
    return None

def base_file_prefix(csv_path: Path, entity: str) -> str:
    ent_snake = to_snake(entity)
    stem_snake = to_snake(csv_path.stem)

    suffix = stem_snake
    if suffix.startswith(ent_snake):
        suffix = suffix[len(ent_snake):].strip("_")

    if not suffix:
        return ent_snake
    return f"{ent_snake}_{suffix}"

def ensure_unique(path: Path) -> Path:
    if not path.exists():
        return path
    base = path.stem
    suffix = path.suffix
    k = 2
    while True:
        cand = path.with_name(f"{base}_v{k}{suffix}")
        if not cand.exists():
            return cand
        k += 1

def get_output_path(csv_path: Path) -> Path:
    rel = csv_path.relative_to(INPUT_DIR)
    rel_parts = rel.parts

    # Outras tabelas
    if is_under_folder(rel_parts, "Outras tabelas"):
        stem_snake = to_snake(csv_path.stem)
        stem_snake = strip_tb_prefix(stem_snake)
        name = f"tb_diversos_{stem_snake}.parquet"
        return ensure_unique(OUTPUT_DIR / name)

    entity = find_entity(csv_path)
    ent_snake = to_snake(entity)

    # Merge
    if is_under_merge(rel_parts):
        if csv_path.name.lower() == "merged_data.csv":
            name = f"{ent_snake}_merge.parquet"
            return ensure_unique(OUTPUT_DIR / name)

        file_snake = to_snake(csv_path.stem)
        name = f"{ent_snake}_merge_{file_snake}.parquet"
        return ensure_unique(OUTPUT_DIR / name)

    # Fora do merge
    sub = first_subfolder_under_entity(rel_parts, entity)
    sub_snake = to_snake(sub) if sub else "base"

    file_prefix = base_file_prefix(csv_path, entity)
    name = f"{file_prefix}_{sub_snake}.parquet"
    return ensure_unique(OUTPUT_DIR / name)

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR não encontrado: {INPUT_DIR}")

    csv_list = list(INPUT_DIR.rglob("*.csv"))
    print(f"Iniciando leitura em: {INPUT_DIR}")
    print(f"Salvando parquets em: {OUTPUT_DIR}")
    print(f"CSV encontrados: {len(csv_list)}\n")

    if not csv_list:
        raise RuntimeError(f"Nenhum CSV encontrado em: {INPUT_DIR}")

    manifest = []
    for csv_file in sorted(csv_list):
        try:
            target_path = get_output_path(csv_file)
            df = read_csv_safe(csv_file)
            to_parquet(df, target_path)

            manifest.append({
                "csv_origem": str(csv_file.relative_to(INPUT_DIR)),
                "parquet_destino": str(target_path.name),
                "status": "OK",
                "linhas": int(df.shape[0]),
                "colunas": int(df.shape[1]),
            })
            print(f"Convertido: {target_path.name}")
        except Exception as e:
            manifest.append({
                "csv_origem": str(csv_file.relative_to(INPUT_DIR)),
                "status": f"ERRO: {e}",
            })
            print(f"✖ Erro em {csv_file.name}: {e}")

    manifest_path = OUTPUT_DIR / "manifest_hist_parquet.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"\nProcesso finalizado! Manifesto salvo em: {manifest_path}")

if __name__ == "__main__":
    main()