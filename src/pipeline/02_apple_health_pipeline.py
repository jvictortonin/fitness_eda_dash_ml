"""
PIPELINE 02 - APPLE HEALTH
===========================
Extrai e processa o arquivo export.zip do Apple Health (app Saúde do iPhone).
Gera CSVs separados por categoria de dado.

Como usar:
    python 02_apple_health_pipeline.py --zip caminho/para/export.zip
"""

import os
import zipfile
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Configurações ─────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw/apple_health")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Tipos de dados que queremos extrair do XML
RECORD_TYPES = {
    "HKQuantityTypeIdentifierStepCount":              "steps",
    "HKQuantityTypeIdentifierHeartRate":              "heart_rate",
    "HKQuantityTypeIdentifierRestingHeartRate":       "resting_heart_rate",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv",
    "HKQuantityTypeIdentifierActiveEnergyBurned":     "active_calories",
    "HKQuantityTypeIdentifierBasalEnergyBurned":      "basal_calories",
    "HKQuantityTypeIdentifierDistanceWalkingRunning": "distance_walk_run",
    "HKQuantityTypeIdentifierVO2Max":                 "vo2max",
    "HKQuantityTypeIdentifierBodyMass":               "weight",
    "HKQuantityTypeIdentifierSleepAnalysis":          "sleep",
    "HKCategoryTypeIdentifierSleepAnalysis":          "sleep",
}

# ── 1. Extração do ZIP ────────────────────────────────────────────────────────
def extract_zip(zip_path: str) -> Path:
    print(f"[1/5] Extraindo {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)
    print(f"      ✓ Extraído em {RAW_DIR}")

    # Localiza o export.xml
    xmls = list(RAW_DIR.rglob("export.xml"))
    if not xmls:
        raise FileNotFoundError("export.xml não encontrado no ZIP do Apple Health.")
    return xmls[0]

# ── 2. Parse do XML ───────────────────────────────────────────────────────────
def parse_health_xml(xml_path: Path) -> pd.DataFrame:
    print(f"[2/5] Parseando XML (pode demorar alguns minutos para arquivos grandes)...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    total = 0
    for record in root.iter("Record"):
        total += 1
        rtype = record.get("type", "")
        if rtype in RECORD_TYPES:
            records.append({
                "type":       rtype,
                "category":   RECORD_TYPES[rtype],
                "value":      record.get("value"),
                "unit":       record.get("unit"),
                "start_date": record.get("startDate"),
                "end_date":   record.get("endDate"),
                "source":     record.get("sourceName"),
            })

    df = pd.DataFrame(records)
    print(f"      ✓ {total} registros totais | {len(df)} registros relevantes extraídos")
    return df

# ── 3. Limpeza ────────────────────────────────────────────────────────────────
def clean_health(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Limpando dados ...")

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"]   = pd.to_datetime(df["end_date"],   errors="coerce")
    df["value"]      = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["start_date", "value"])
    df = df.sort_values("start_date").reset_index(drop=True)

    # Campos de data
    df["date"]    = df["start_date"].dt.date
    df["year"]    = df["start_date"].dt.year
    df["month"]   = df["start_date"].dt.month
    df["weekday"] = df["start_date"].dt.day_name()

    print(f"      ✓ Dataset limpo: {len(df)} registros")
    return df

# ── 4. Agrega por dia ─────────────────────────────────────────────────────────
def aggregate_daily(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    print("[4/5] Agregando dados por dia ...")
    daily_datasets = {}

    agg_rules = {
        "steps":            ("value", "sum"),
        "active_calories":  ("value", "sum"),
        "basal_calories":   ("value", "sum"),
        "distance_walk_run":("value", "sum"),
        "heart_rate":       ("value", "mean"),
        "resting_heart_rate":("value","mean"),
        "hrv":              ("value", "mean"),
        "vo2max":           ("value", "mean"),
        "weight":           ("value", "mean"),
    }

    for category, (col, agg) in agg_rules.items():
        subset = df[df["category"] == category].copy()
        if subset.empty:
            continue
        if agg == "sum":
            daily = subset.groupby("date")[col].sum().reset_index()
        else:
            daily = subset.groupby("date")[col].mean().reset_index()
        daily.columns = ["date", category]
        daily["date"] = pd.to_datetime(daily["date"])
        daily_datasets[category] = daily
        print(f"      ✓ {category}: {len(daily)} dias")

    return daily_datasets

# ── 5. Salva CSVs ─────────────────────────────────────────────────────────────
def save_processed(df_raw: pd.DataFrame, daily: dict) -> None:
    print("[5/5] Salvando datasets ...")

    # Raw completo
    df_raw.to_csv(PROCESSED_DIR / "apple_health_raw.csv", index=False)

    # Um CSV consolidado com todos os dados diários
    if daily:
        from functools import reduce
        dfs = list(daily.values())
        consolidated = reduce(lambda a, b: pd.merge(a, b, on="date", how="outer"), dfs)
        consolidated = consolidated.sort_values("date").reset_index(drop=True)
        consolidated.to_csv(PROCESSED_DIR / "apple_health_daily.csv", index=False)
        print(f"      ✓ apple_health_daily.csv salvo com {len(consolidated)} dias e {consolidated.shape[1]} métricas")

    # CSVs individuais por categoria
    for name, d in daily.items():
        d.to_csv(PROCESSED_DIR / f"apple_{name}.csv", index=False)

    print(f"\n{'='*50}")
    print(f"  PIPELINE APPLE HEALTH CONCLUÍDO!")
    print(f"  Métricas extraídas: {list(daily.keys())}")
    print(f"{'='*50}\n")

# ── Main ──────────────────────────────────────────────────────────────────────
def run(zip_path: str):
    xml_path = extract_zip(zip_path)
    df_raw   = parse_health_xml(xml_path)
    df_clean = clean_health(df_raw)
    daily    = aggregate_daily(df_clean)
    save_processed(df_clean, daily)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Apple Health")
    parser.add_argument("--zip", required=True, help="Caminho para o export.zip do Apple Health")
    args = parser.parse_args()
    run(args.zip)
