import re
import hashlib
from pathlib import Path
import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'datasets'
OLY_DIR = DATA_DIR / 'olympics'
COUNTRIES_CSV = DATA_DIR / 'countries' / 'countries of the world.csv'
OUT_DIR = REPO_ROOT / 'outputs'
OUT_DIR.mkdir(exist_ok=True)


def country_id_from_name(name: str) -> str:
    if pd.isna(name):
        name = ''
    key = re.sub(r'\s+', ' ', str(name)).strip().lower()
    return hashlib.md5(key.encode('utf-8')).hexdigest()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: re.sub(r'\s+', '_', str(c).strip().lower()))
    return df


def read_olympics_files() -> pd.DataFrame:
    frames = []
    if not OLY_DIR.exists():
        print(f'No olympics directory found at {OLY_DIR}', file=sys.stderr)
        return pd.DataFrame()
    for f in sorted(OLY_DIR.glob('*.csv')):
        m = re.search(r'(\d{4})', f.name)
        year = int(m.group(1)) if m else None
        try:
            # prefer the default engine; avoid engine='python' together with low_memory
            df = pd.read_csv(f, low_memory=False, encoding='utf-8')
        except Exception:
            df = pd.read_csv(f, low_memory=False, encoding='latin1')
        df = normalize_columns(df)
        df['source_file'] = f.name
        df['year'] = year
        for c in ('nation', 'country', 'team'):
            if c in df.columns:
                df = df.rename(columns={c: 'country'})
                break
        if 'country' not in df.columns:
            df['country'] = None
        for medal in ('gold', 'silver', 'bronze', 'total', 'rank'):
            if medal in df.columns:
                df[medal] = pd.to_numeric(df[medal].astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce')
        df['country_id'] = df['country'].astype(str).apply(country_id_from_name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_olympics_table() -> pd.DataFrame:
    return read_olympics_files()


def build_countries_table() -> pd.DataFrame:
    if not COUNTRIES_CSV.exists():
        print(f'Countries CSV not found at {COUNTRIES_CSV}', file=sys.stderr)
        return pd.DataFrame()
    try:
        df = pd.read_csv(COUNTRIES_CSV, low_memory=False, encoding='utf-8')
    except Exception:
        df = pd.read_csv(COUNTRIES_CSV, low_memory=False, encoding='latin1')
    df = normalize_columns(df)
    for c in ('country', 'name', 'country_name'):
        if c in df.columns:
            df = df.rename(columns={c: 'country'})
            break
    if 'country' not in df.columns:
        df['country'] = None
    df['country_id'] = df['country'].astype(str).apply(country_id_from_name)
    for col in df.columns:
        if re.search(r'(population|area|density|gdp|percapita|index|%|rate)', col):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce')
    return df


def upsert_parquet(df: pd.DataFrame, out_path: Path, key_cols):
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, df], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=key_cols, keep='last')
    else:
        combined = df.copy()
    combined.to_parquet(out_path, index=False)
    return combined


def run_pipeline():
    print('Reading countries CSV...')
    countries = build_countries_table()
    countries_path = OUT_DIR / 'countries.parquet'
    print(f'Upserting countries -> {countries_path}')
    countries_written = upsert_parquet(countries, countries_path, key_cols=['country_id'])
    print(f'wrote {len(countries_written)} country rows')

    print('Reading olympics CSVs...')
    olympics = build_olympics_table()
    olympics_path = OUT_DIR / 'olympics.parquet'
    print(f'Upserting olympics -> {olympics_path}')
    olympics_written = upsert_parquet(olympics, olympics_path, key_cols=['country_id', 'year'])
    print(f'wrote {len(olympics_written)} olympics rows')

    print('Creating denormalized artifact...')
    merged = olympics_written.merge(countries_written, on='country_id', how='left', suffixes=('_olymp', '_country'))
    merged_path = OUT_DIR / 'olympics_denormalized.parquet'
    merged.to_parquet(merged_path, index=False)
    print(f'wrote denormalized -> {merged_path} ({len(merged)} rows)')
    return {'countries': countries_written, 'olympics': olympics_written, 'merged': merged}


if __name__ == '__main__':
    try:
        artifacts = run_pipeline()
        print('Pipeline completed successfully.')
    except Exception as e:
        print('Pipeline failed with exception:', e, file=sys.stderr)
        raise
