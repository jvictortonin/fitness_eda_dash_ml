"""
PIPELINE 03 - MERGE STRAVA + APPLE HEALTH
==========================================
Combina os datasets processados do Strava e Apple Health
em um dataset unificado por data para análises cruzadas.

Como usar:
    python 03_merge_datasets.py
    (Requer que os pipelines 01 e 02 já tenham sido executados)
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR  = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def load_strava() -> pd.DataFrame:
    path = PROCESSED_DIR / "strava_processed.csv"
    if not path.exists():
        raise FileNotFoundError("strava_processed.csv não encontrado. Execute 01_strava_pipeline.py primeiro.")
    df = pd.read_csv(path, parse_dates=["activity_date"])
    df["date"] = df["activity_date"].dt.date
    df["date"] = pd.to_datetime(df["date"])
    print(f"[Strava] {len(df)} atividades carregadas")
    return df

def load_apple_health() -> pd.DataFrame:
    path = PROCESSED_DIR / "apple_health_daily.csv"
    if not path.exists():
        print("[Apple Health] apple_health_daily.csv não encontrado. Continuando só com Strava.")
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"[Apple Health] {len(df)} dias carregados | colunas: {list(df.columns)}")
    return df

def aggregate_strava_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega atividades Strava por dia (pode haver mais de uma por dia)."""
    agg = df.groupby("date").agg(
        num_activities        = ("activity_date", "count"),
        total_distance_km     = ("distance_km",   "sum"),
        total_moving_time_min = ("moving_time_min","sum"),
        avg_pace_min_km       = ("pace_min_km",    "mean"),
        avg_heart_rate        = ("average_heart_rate", "mean") if "average_heart_rate" in df.columns else ("distance_km", "first"),
        total_elevation       = ("elevation_gain", "sum") if "elevation_gain" in df.columns else ("distance_km", "first"),
        weekly_distance_km    = ("weekly_distance_km", "first"),
    ).reset_index()
    return agg

def engineer_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features adicionais no dataset combinado."""
    df = df.sort_values("date").reset_index(drop=True)

    # Dias sem treino
    df["rest_day"] = df["num_activities"] == 0 if "num_activities" in df.columns else False

    # Rolling 7 dias de distância
    if "total_distance_km" in df.columns:
        df["rolling_7d_distance"] = df["total_distance_km"].rolling(7, min_periods=1).sum()
        df["rolling_28d_distance"] = df["total_distance_km"].rolling(28, min_periods=1).sum()

    # Monotonia de treino (desvio padrão / média dos últimos 7 dias)
    if "total_distance_km" in df.columns:
        roll_std  = df["total_distance_km"].rolling(7, min_periods=2).std()
        roll_mean = df["total_distance_km"].rolling(7, min_periods=2).mean()
        df["training_monotony"] = roll_std / (roll_mean + 1e-5)

    return df

def run():
    print("=" * 50)
    print("  MERGE STRAVA + APPLE HEALTH")
    print("=" * 50)

    strava = load_strava()
    apple  = load_apple_health()

    strava_daily = aggregate_strava_daily(strava)

    if apple is not None:
        combined = pd.merge(strava_daily, apple, on="date", how="outer")
    else:
        combined = strava_daily

    combined = engineer_combined_features(combined)
    combined = combined.sort_values("date").reset_index(drop=True)

    out_path = FEATURES_DIR / "combined_dataset.csv"
    combined.to_csv(out_path, index=False)

    print(f"\n✓ Dataset combinado salvo em {out_path}")
    print(f"  Período: {combined['date'].min()} → {combined['date'].max()}")
    print(f"  Shape: {combined.shape}")
    print(f"  Colunas: {list(combined.columns)}")

if __name__ == "__main__":
    run()
