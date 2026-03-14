"""
ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
=====================================
Gera visualizações e estatísticas descritivas
dos dados de treino Strava + Apple Health.

Como usar:
    python 04_eda.py

Outputs em: outputs/plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
FEATURES_DIR  = Path("data/features")
PROCESSED_DIR = Path("data/processed")
PLOTS_DIR     = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Paleta de cores
COLORS = {
    "primary":   "#FC4C02",   # laranja Strava
    "secondary": "#0D0D0D",
    "accent":    "#FF8C42",
    "soft":      "#FFF0E8",
    "blue":      "#2196F3",
    "green":     "#4CAF50",
    "gray":      "#9E9E9E",
}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "sans-serif",
})

# ── Carrega dados ──────────────────────────────────────────────────────────────
def load_data():
    strava_path = PROCESSED_DIR / "strava_processed.csv"
    combined_path = FEATURES_DIR / "combined_dataset.csv"

    strava = pd.read_csv(strava_path, parse_dates=["activity_date"]) if strava_path.exists() else None
    combined = pd.read_csv(combined_path, parse_dates=["date"]) if combined_path.exists() else None

    if strava is None:
        raise FileNotFoundError("Execute os pipelines 01, 02 e 03 primeiro!")

    # Filtra apenas corridas para análise de pace
    runs = strava[strava.get("activity_type_clean", pd.Series()).str.contains("Run", na=False)].copy() \
           if "activity_type_clean" in strava.columns else strava.copy()

    print(f"✓ Strava: {len(strava)} atividades | {len(runs)} corridas")
    return strava, runs, combined

# ── Plot 1: Visão Geral ────────────────────────────────────────────────────────
def plot_overview(strava: pd.DataFrame, runs: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("📊 Visão Geral dos Treinos", fontsize=18, fontweight="bold", y=1.02)

    # 1a. Atividades por tipo
    ax = axes[0, 0]
    if "activity_type_clean" in strava.columns:
        counts = strava["activity_type_clean"].value_counts().head(8)
        bars = ax.barh(counts.index, counts.values, color=COLORS["primary"])
        ax.set_title("Atividades por Tipo", fontweight="bold")
        ax.set_xlabel("Quantidade")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=9)

    # 1b. Distância por mês
    ax = axes[0, 1]
    if "distance_km" in runs.columns and "activity_date" in runs.columns:
        monthly = runs.groupby(runs["activity_date"].dt.to_period("M"))["distance_km"].sum()
        monthly.index = monthly.index.astype(str)
        ax.bar(range(len(monthly)), monthly.values, color=COLORS["primary"], alpha=0.8)
        ax.set_xticks(range(0, len(monthly), max(1, len(monthly)//12)))
        ax.set_xticklabels([monthly.index[i] for i in range(0, len(monthly), max(1, len(monthly)//12))],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title("Distância Mensal (km)", fontweight="bold")
        ax.set_ylabel("km")

    # 1c. Distribuição de distâncias
    ax = axes[1, 0]
    if "distance_km" in runs.columns:
        data = runs["distance_km"].dropna()
        data = data[(data > 0) & (data < 100)]
        ax.hist(data, bins=30, color=COLORS["primary"], edgecolor="white", alpha=0.85)
        ax.axvline(data.median(), color=COLORS["blue"], linestyle="--", label=f"Mediana: {data.median():.1f} km")
        ax.axvline(data.mean(),   color=COLORS["green"], linestyle="--", label=f"Média: {data.mean():.1f} km")
        ax.set_title("Distribuição de Distâncias", fontweight="bold")
        ax.set_xlabel("km")
        ax.legend()

    # 1d. Treinos por dia da semana
    ax = axes[1, 1]
    if "weekday" in strava.columns:
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pt = ["Segunda","Terça","Quarta","Quinta","Sexta","Sábado","Domingo"]
        wd = strava["weekday"].value_counts().reindex(order, fill_value=0)
        ax.bar(pt, wd.values, color=[COLORS["primary"] if v == wd.max() else COLORS["accent"] for v in wd.values])
        ax.set_title("Treinos por Dia da Semana", fontweight="bold")
        ax.set_ylabel("Quantidade")

    plt.tight_layout()
    path = PLOTS_DIR / "01_overview.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Salvo: {path}")

# ── Plot 2: Evolução de Pace ───────────────────────────────────────────────────
def plot_pace_evolution(runs: pd.DataFrame):
    if "pace_min_km" not in runs.columns or "activity_date" not in runs.columns:
        print("  [!] pace_min_km não encontrado, pulando gráfico de pace")
        return

    df = runs[["activity_date", "pace_min_km", "distance_km"]].dropna()
    df = df[(df["pace_min_km"] > 3) & (df["pace_min_km"] < 15)]

    fig, ax = plt.subplots(figsize=(16, 6))

    sc = ax.scatter(df["activity_date"], df["pace_min_km"],
                    c=df["distance_km"], cmap="YlOrRd", alpha=0.6, s=50, zorder=3)

    # Linha de tendência (rolling 10 treinos)
    df_sorted = df.sort_values("activity_date")
    roll = df_sorted["pace_min_km"].rolling(10, min_periods=3).mean()
    ax.plot(df_sorted["activity_date"], roll,
            color=COLORS["primary"], linewidth=2.5, label="Média móvel (10 treinos)", zorder=4)

    plt.colorbar(sc, ax=ax, label="Distância (km)")
    ax.set_title("📈 Evolução do Pace ao Longo do Tempo", fontsize=15, fontweight="bold")
    ax.set_ylabel("Pace (min/km)")
    ax.set_xlabel("")
    ax.invert_yaxis()  # pace menor = mais rápido = melhor (fica em cima)
    ax.legend()

    plt.tight_layout()
    path = PLOTS_DIR / "02_pace_evolution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Salvo: {path}")

# ── Plot 3: Heatmap de Atividade Anual (estilo GitHub) ─────────────────────────
def plot_activity_heatmap(strava: pd.DataFrame):
    if "activity_date" not in strava.columns:
        return

    # Agrupa por data
    daily = strava.groupby(strava["activity_date"].dt.date).agg(
        count=("activity_date", "count"),
        dist=("distance_km", "sum") if "distance_km" in strava.columns else ("activity_date", "count")
    ).reset_index()
    daily.columns = ["date", "count", "dist"]
    daily["date"] = pd.to_datetime(daily["date"])

    years = daily["date"].dt.year.unique()
    for year in sorted(years)[-2:]:  # últimos 2 anos
        year_data = daily[daily["date"].dt.year == year]
        start = pd.Timestamp(f"{year}-01-01")
        end   = pd.Timestamp(f"{year}-12-31")
        all_dates = pd.date_range(start, end)
        date_df = pd.DataFrame({"date": all_dates})
        date_df = date_df.merge(year_data, on="date", how="left").fillna(0)
        date_df["week"] = date_df["date"].dt.isocalendar().week.astype(int)
        date_df["dow"]  = date_df["date"].dt.dayofweek  # 0=Mon

        # Pivot
        pivot = date_df.pivot_table(index="dow", columns="week", values="dist", aggfunc="sum").fillna(0)

        fig, ax = plt.subplots(figsize=(20, 4))
        cmap = plt.cm.YlOrRd
        cmap.set_under("whitesmoke")

        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                       vmin=0.01, interpolation="nearest")

        ax.set_yticks(range(7))
        ax.set_yticklabels(["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"])
        ax.set_xticks(range(0, 53, 4))
        ax.set_xticklabels([f"Sem {i+1}" for i in range(0, 53, 4)], rotation=45, ha="right")
        ax.set_title(f"🗓️ Heatmap de Atividade — {year} (distância em km)", fontsize=13, fontweight="bold")

        plt.colorbar(im, ax=ax, label="km por dia", shrink=0.6)
        plt.tight_layout()
        path = PLOTS_DIR / f"03_heatmap_{year}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Salvo: {path}")

# ── Plot 4: Correlações ───────────────────────────────────────────────────────
def plot_correlations(runs: pd.DataFrame):
    cols = ["distance_km", "pace_min_km", "moving_time_min",
            "average_heart_rate", "elevation_gain", "suffer_score"]
    available = [c for c in cols if c in runs.columns]
    if len(available) < 3:
        print("  [!] Poucas colunas para correlação, pulando")
        return

    corr = runs[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, mask=mask, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("🔗 Correlações entre Variáveis de Treino", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = PLOTS_DIR / "04_correlations.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Salvo: {path}")

# ── Plot 5: Carga Semanal ─────────────────────────────────────────────────────
def plot_weekly_load(strava: pd.DataFrame):
    if not all(c in strava.columns for c in ["activity_date", "distance_km"]):
        return

    df = strava.copy()
    df["week"] = df["activity_date"].dt.to_period("W")
    weekly = df.groupby("week").agg(
        distance=("distance_km", "sum"),
        sessions=("activity_date", "count")
    ).reset_index()
    weekly["week_str"] = weekly["week"].astype(str)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Barras de distância
    ax1.bar(range(len(weekly)), weekly["distance"], color=COLORS["primary"], alpha=0.8)
    ax1.plot(range(len(weekly)), weekly["distance"].rolling(4).mean(),
             color=COLORS["blue"], linewidth=2, label="Média 4 semanas")
    ax1.set_title("📦 Carga de Treino Semanal", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Distância (km)")
    ax1.legend()

    # Nº de sessões
    ax2.bar(range(len(weekly)), weekly["sessions"], color=COLORS["accent"], alpha=0.8)
    ax2.set_ylabel("Nº de Sessões")

    step = max(1, len(weekly) // 20)
    ax2.set_xticks(range(0, len(weekly), step))
    ax2.set_xticklabels(weekly["week_str"].iloc[::step], rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    path = PLOTS_DIR / "05_weekly_load.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Salvo: {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("=" * 50)
    print("  ANÁLISE EXPLORATÓRIA DE DADOS")
    print("=" * 50)

    strava, runs, combined = load_data()

    print("\nEstatísticas descritivas - Corridas:")
    desc_cols = [c for c in ["distance_km","pace_min_km","moving_time_min","average_heart_rate"] if c in runs.columns]
    print(runs[desc_cols].describe().round(2))

    print("\nGerando gráficos...")
    plot_overview(strava, runs)
    plot_pace_evolution(runs)
    plot_activity_heatmap(strava)
    plot_correlations(runs)
    plot_weekly_load(strava)

    print(f"\n✅ EDA concluída! Gráficos em: {PLOTS_DIR}")

if __name__ == "__main__":
    run()
