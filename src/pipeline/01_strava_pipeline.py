"""
PIPELINE 01 - STRAVA (PT-BR)
=============================
Extrai e processa o arquivo .zip exportado do Strava (export em português).
Gera um CSV limpo e enriquecido com features para análise.

Como usar:
    python 01_strava_pipeline.py --zip caminho/para/export.zip
"""

import os
import zipfile
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ── Configurações ─────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw/strava")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Mapeamento de colunas PT-BR → nomes padronizados
COL_MAP = {
    "Data da atividade":          "activity_date",
    "Tipo de atividade":          "activity_type",
    "Nome da atividade":          "activity_name",
    "Distância":                  "distance_raw",
    "Tempo de movimentação":      "moving_time_s",
    "Tempo decorrido":            "elapsed_time_s",
    "Ganho de elevação":          "elevation_gain",
    "Perda de elevação":          "elevation_loss",
    "Velocidade média":           "avg_speed",
    "Velocidade máx.":            "max_speed",
    "Frequência cardíaca média":  "avg_heart_rate",
    "Frequência cardíaca máxima": "max_heart_rate",
    "Frequência cardíaca máxima.1": "max_heart_rate",
    "Cadência média":             "avg_cadence",
    "Calorias":                   "calories",
    "Esforço relativo":           "suffer_score",
    "Esforço relativo.1":         "suffer_score",
    "Carga de treinamento":       "training_load",
    "Total de passos":            "total_steps",
    "Hora de início":             "start_time",
    "Temperatura média":          "avg_temperature",
    "Umidade":                    "humidity",
}

# ── 1. Extração do ZIP ────────────────────────────────────────────────────────
def extract_zip(zip_path: str) -> None:
    print(f"[1/5] Extraindo {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)
    print(f"      ✓ Extraído em {RAW_DIR}")

# ── 2. Leitura do activities.csv ──────────────────────────────────────────────
def load_activities() -> pd.DataFrame:
    print("[2/5] Lendo activities.csv ...")
    candidates = list(RAW_DIR.rglob("activities.csv"))
    if not candidates:
        raise FileNotFoundError("activities.csv não encontrado. Verifique o ZIP do Strava.")
    df = pd.read_csv(candidates[0])
    print(f"      ✓ {len(df)} atividades | {df.shape[1]} colunas")
    return df

# ── 3. Limpeza e padronização ─────────────────────────────────────────────────
def clean_activities(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Limpando e padronizando dados ...")

    # Renomeia apenas as colunas que existem
    rename = {k: v for k, v in COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Parseia data (formato pt-BR: "15 de fev. de 2020, 21:59:14")
    df["activity_date"] = pd.to_datetime(df["activity_date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["activity_date"])
    df = df.sort_values("activity_date").reset_index(drop=True)

    # Distância: vírgula decimal → float, já vem em km no export PT-BR
    if "distance_raw" in df.columns:
        df["distance_km"] = (
            df["distance_raw"].astype(str)
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
        # Strava PT-BR exporta em km; se valores > 500 provavelmente está em metros
        if df["distance_km"].median() > 200:
            df["distance_km"] /= 1000

    # Converte colunas numéricas
    num_cols = ["moving_time_s","elapsed_time_s","elevation_gain","elevation_loss",
                "avg_speed","max_speed","avg_heart_rate","max_heart_rate",
                "avg_cadence","calories","suffer_score","training_load",
                "total_steps","avg_temperature","humidity"]
    for c in num_cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(",", ".", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )

    print(f"      ✓ Dataset limpo: {len(df)} linhas, {df.shape[1]} colunas")
    return df

# ── 4. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/5] Criando features ...")

    df["year"]       = df["activity_date"].dt.year
    df["month"]      = df["activity_date"].dt.month
    df["month_name"] = df["activity_date"].dt.strftime("%b")
    df["weekday"]    = df["activity_date"].dt.day_name()
    df["week_num"]   = df["activity_date"].dt.isocalendar().week.astype(int)
    df["hour"]       = df["activity_date"].dt.hour

    # Pace (min/km) — usa tempo de movimentação
    if "moving_time_s" in df.columns and "distance_km" in df.columns:
        df["moving_time_min"] = df["moving_time_s"] / 60
        mask = df["distance_km"] > 0.1
        df["pace_min_km"] = np.nan
        df.loc[mask, "pace_min_km"] = df.loc[mask, "moving_time_min"] / df.loc[mask, "distance_km"]
        df["pace_min_km"] = df["pace_min_km"].clip(lower=2, upper=30)

    # Tipo normalizado (PT-BR → padrão)
    if "activity_type" in df.columns:
        df["activity_type_clean"] = df["activity_type"].str.strip()

    # Carga semanal
    if "distance_km" in df.columns:
        df["weekly_distance_km"] = (
            df.groupby(["year", "week_num"])["distance_km"]
            .transform("sum")
        )

    # Flag de corrida
    if "activity_type_clean" in df.columns:
        df["is_run"] = df["activity_type_clean"].str.contains("Corrida|Run", na=False, case=False)

    print(f"      ✓ Features criadas. Total de colunas: {df.shape[1]}")
    return df

# ── 5. Exportação ─────────────────────────────────────────────────────────────
def save_processed(df: pd.DataFrame) -> Path:
    print("[5/5] Salvando dataset processado ...")
    out_path = PROCESSED_DIR / "strava_processed.csv"
    df.to_csv(out_path, index=False)

    runs = df[df.get("is_run", pd.Series(False, index=df.index))]
    print(f"\n{'='*50}")
    print(f"  PIPELINE STRAVA CONCLUÍDO!")
    print(f"  Total atividades : {len(df)}")
    print(f"  Corridas         : {len(runs)}")
    print(f"  Período          : {df['activity_date'].min().date()} → {df['activity_date'].max().date()}")
    print(f"  Tipos            : {df['activity_type_clean'].value_counts().to_dict() if 'activity_type_clean' in df.columns else 'N/A'}")
    print(f"{'='*50}\n")
    return out_path

# ── Main ──────────────────────────────────────────────────────────────────────
def run(zip_path: str):
    extract_zip(zip_path)
    df = load_activities()
    df = clean_activities(df)
    df = engineer_features(df)
    save_processed(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Strava")
    parser.add_argument("--zip", required=True, help="Caminho para o export.zip do Strava")
    args = parser.parse_args()
    run(args.zip)
