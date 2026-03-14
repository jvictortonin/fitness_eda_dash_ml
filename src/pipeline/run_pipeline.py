"""
Script unificado que processa Strava + Apple Health com dados reais.
"""
import re
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

RAW_STRAVA    = Path("data/raw/strava")
RAW_APPLE     = Path("data/raw/apple_health")
PROCESSED_DIR = Path("data/processed")
FEATURES_DIR  = Path("data/features")
for p in [RAW_STRAVA, RAW_APPLE, PROCESSED_DIR, FEATURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

MONTHS_PT = {'jan':1,'fev':2,'mar':3,'abr':4,'mai':5,'jun':6,
             'jul':7,'ago':8,'set':9,'out':10,'nov':11,'dez':12}

def parse_ptbr_date(s):
    try:
        m = re.search(r'(\d+) de (\w+)\.? de (\d{4}),?\s*([\d:]+)?', str(s).lower())
        if m:
            day = int(m.group(1))
            mon = MONTHS_PT.get(m.group(2)[:3], 1)
            year = int(m.group(3))
            time_str = m.group(4) or '00:00:00'
            return pd.Timestamp(f'{year}-{mon:02d}-{day:02d} {time_str}')
    except:
        pass
    return pd.NaT

def to_float(series):
    return (series.astype(str)
            .str.replace(',', '.', regex=False)
            .str.strip()
            .pipe(pd.to_numeric, errors='coerce'))

# ═══════════════════════════════════════════════════════════════════════════════
# STRAVA
# ═══════════════════════════════════════════════════════════════════════════════
def process_strava(zip_path: str):
    print("\n" + "="*55)
    print("  PIPELINE STRAVA")
    print("="*55)

    print(f"[1/4] Extraindo ZIP...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_STRAVA)

    csv_files = list(RAW_STRAVA.rglob("activities.csv"))
    df = pd.read_csv(csv_files[0])
    print(f"[2/4] Carregado: {len(df)} atividades | {df.shape[1]} colunas")

    # Renomeia colunas
    col_map = {
        "Data da atividade":           "activity_date",
        "Tipo de atividade":           "activity_type",
        "Nome da atividade":           "activity_name",
        "Distância":                   "distance_km",
        "Tempo de movimentação":       "moving_time_s",
        "Tempo decorrido":             "elapsed_time_s",
        "Ganho de elevação":           "elevation_gain",
        "Velocidade média":            "avg_speed",
        "Frequência cardíaca média":   "avg_heart_rate",
        "Frequência cardíaca máxima.1":"max_heart_rate",
        "Cadência média":              "avg_cadence",
        "Calorias":                    "calories",
        "Esforço relativo.1":          "suffer_score",
        "Carga de treinamento":        "training_load",
        "Total de passos":             "total_steps",
        "Temperatura média":           "avg_temperature",
        "Umidade":                     "humidity",
        "Hora de início":              "start_time",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    print("[3/4] Limpando dados...")
    # Data
    df["activity_date"] = df["activity_date"].apply(parse_ptbr_date)
    df = df.dropna(subset=["activity_date"]).sort_values("activity_date").reset_index(drop=True)

    # Numéricos com vírgula decimal
    for col in ["distance_km","moving_time_s","elapsed_time_s","elevation_gain",
                "avg_speed","avg_heart_rate","max_heart_rate","avg_cadence",
                "calories","suffer_score","training_load","total_steps",
                "avg_temperature","humidity"]:
        if col in df.columns:
            df[col] = to_float(df[col])

    # Distância: o export PT-BR já vem em km
    # median check
    if df["distance_km"].median() > 500:
        df["distance_km"] /= 1000

    print("[4/4] Criando features...")
    df["year"]        = df["activity_date"].dt.year
    df["month"]       = df["activity_date"].dt.month
    df["month_name"]  = df["activity_date"].dt.strftime("%b")
    df["weekday"]     = df["activity_date"].dt.day_name()
    df["week_num"]    = df["activity_date"].dt.isocalendar().week.astype(int)
    df["hour"]        = df["activity_date"].dt.hour
    df["is_morning"]  = df["hour"].between(5, 12)
    df["is_run"]      = df["activity_type"].str.contains("Corrida|Run", na=False, case=False)

    # Pace
    df["moving_time_min"] = df["moving_time_s"] / 60
    mask = df["distance_km"] > 0.1
    df["pace_min_km"] = np.nan
    df.loc[mask, "pace_min_km"] = df.loc[mask, "moving_time_min"] / df.loc[mask, "distance_km"]
    df["pace_min_km"] = df["pace_min_km"].clip(lower=2.5, upper=20)

    # Carga semanal
    df["weekly_distance_km"] = df.groupby(["year","week_num"])["distance_km"].transform("sum")

    # Dias desde o início
    df["days_since_start"] = (df["activity_date"] - df["activity_date"].min()).dt.days

    out = PROCESSED_DIR / "strava_processed.csv"
    df.to_csv(out, index=False)

    runs = df[df["is_run"]]
    print(f"\n✅ Strava processado!")
    print(f"   Total atividades : {len(df)}")
    print(f"   Corridas         : {len(runs)}")
    print(f"   Período          : {df['activity_date'].min().date()} → {df['activity_date'].max().date()}")
    print(f"   Tipos            : {df['activity_type'].value_counts().to_dict()}")
    if len(runs) > 0:
        print(f"   Pace médio       : {runs['pace_min_km'].dropna().mean():.2f} min/km")
        print(f"   Distância total  : {runs['distance_km'].sum():.0f} km")
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# APPLE HEALTH
# ═══════════════════════════════════════════════════════════════════════════════
def process_apple_health(zip_path: str):
    import xml.etree.ElementTree as ET

    print("\n" + "="*55)
    print("  PIPELINE APPLE HEALTH")
    print("="*55)

    print(f"[1/4] Extraindo ZIP...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_APPLE)

    xmls = list(RAW_APPLE.rglob("export.xml"))
    if not xmls:
        print("  ⚠️  export.xml não encontrado. Pulando Apple Health.")
        return None

    xml_path = xmls[0]
    print(f"[2/4] Parseando XML: {xml_path} ({xml_path.stat().st_size/1024/1024:.1f} MB)...")

    RECORD_TYPES = {
        "HKQuantityTypeIdentifierStepCount":               "steps",
        "HKQuantityTypeIdentifierHeartRate":               "heart_rate",
        "HKQuantityTypeIdentifierRestingHeartRate":        "resting_heart_rate",
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":"hrv",
        "HKQuantityTypeIdentifierActiveEnergyBurned":      "active_calories",
        "HKQuantityTypeIdentifierDistanceWalkingRunning":  "distance_walk_run",
        "HKQuantityTypeIdentifierVO2Max":                  "vo2max",
        "HKQuantityTypeIdentifierBodyMass":                "weight",
    }

    records = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    total = 0
    for record in root.iter("Record"):
        total += 1
        rtype = record.get("type", "")
        if rtype in RECORD_TYPES:
            records.append({
                "category":   RECORD_TYPES[rtype],
                "value":      record.get("value"),
                "start_date": record.get("startDate"),
                "source":     record.get("sourceName"),
            })

    df = pd.DataFrame(records)
    print(f"[3/4] {total} registros totais | {len(df)} relevantes extraídos")

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["value"]      = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["start_date","value"])
    df["date"] = df["start_date"].dt.normalize()

    print("[4/4] Agregando por dia...")
    daily_dfs = []
    agg_map = {
        "steps":            "sum",
        "active_calories":  "sum",
        "distance_walk_run":"sum",
        "heart_rate":       "mean",
        "resting_heart_rate":"mean",
        "hrv":              "mean",
        "vo2max":           "mean",
        "weight":           "mean",
    }
    for cat, agg in agg_map.items():
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        if agg == "sum":
            d = sub.groupby("date")["value"].sum().reset_index()
        else:
            d = sub.groupby("date")["value"].mean().reset_index()
        d.columns = ["date", cat]
        daily_dfs.append(d)
        print(f"   ✓ {cat}: {len(d)} dias")

    if not daily_dfs:
        print("   ⚠️  Nenhuma métrica encontrada.")
        return None

    from functools import reduce
    daily = reduce(lambda a, b: pd.merge(a, b, on="date", how="outer"), daily_dfs)
    daily = daily.sort_values("date").reset_index(drop=True)

    daily.to_csv(PROCESSED_DIR / "apple_health_daily.csv", index=False)
    print(f"\n✅ Apple Health processado! {len(daily)} dias | colunas: {list(daily.columns)}")
    return daily

# ═══════════════════════════════════════════════════════════════════════════════
# MERGE
# ═══════════════════════════════════════════════════════════════════════════════
def merge_datasets():
    print("\n" + "="*55)
    print("  MERGE FINAL")
    print("="*55)

    strava = pd.read_csv(PROCESSED_DIR / "strava_processed.csv", parse_dates=["activity_date"])
    strava["date"] = strava["activity_date"].dt.normalize()

    daily_strava = strava.groupby("date").agg(
        num_activities       = ("activity_date","count"),
        total_distance_km    = ("distance_km","sum"),
        total_moving_min     = ("moving_time_min","sum"),
        avg_pace_min_km      = ("pace_min_km","mean"),
        avg_heart_rate       = ("avg_heart_rate","mean"),
        total_elevation      = ("elevation_gain","sum"),
        weekly_distance_km   = ("weekly_distance_km","first"),
    ).reset_index()

    apple_path = PROCESSED_DIR / "apple_health_daily.csv"
    if apple_path.exists():
        apple = pd.read_csv(apple_path, parse_dates=["date"])
        combined = pd.merge(daily_strava, apple, on="date", how="outer")
    else:
        combined = daily_strava

    combined = combined.sort_values("date").reset_index(drop=True)
    combined["rolling_7d_km"]  = combined["total_distance_km"].fillna(0).rolling(7, min_periods=1).sum()
    combined["rolling_28d_km"] = combined["total_distance_km"].fillna(0).rolling(28, min_periods=1).sum()

    out = FEATURES_DIR / "combined_dataset.csv"
    combined.to_csv(out, index=False)
    print(f"✅ Dataset combinado: {combined.shape} → {out}")
    return combined

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    strava_zip = sys.argv[1]
    apple_zip  = sys.argv[2] if len(sys.argv) > 2 else None

    process_strava(strava_zip)
    if apple_zip:
        process_apple_health(apple_zip)
    merge_datasets()
    print("\n🎉 Todos os pipelines concluídos com sucesso!")
