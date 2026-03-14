"""
ANÁLISE ESTATÍSTICA
====================
Testa hipóteses sobre evolução de performance,
sazonalidade e correlações significativas.

Como usar:
    python 05_statistical_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR  = Path("data/features")
PLOTS_DIR     = Path("outputs/plots")
REPORTS_DIR   = Path("outputs/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_runs() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "strava_processed.csv", parse_dates=["activity_date"])
    if "activity_type_clean" in df.columns:
        df = df[df["activity_type_clean"].str.contains("Run", na=False)]
    return df

# ── Teste 1: Evolução do Pace (há melhora significativa?) ─────────────────────
def test_pace_improvement(runs: pd.DataFrame) -> dict:
    if "pace_min_km" not in runs.columns:
        return {}

    df = runs[["activity_date", "pace_min_km"]].dropna()
    df = df[(df["pace_min_km"] > 3) & (df["pace_min_km"] < 15)]
    df = df.sort_values("activity_date").reset_index(drop=True)

    if len(df) < 20:
        print("  [!] Poucos dados para teste de melhora")
        return {}

    # Divide em primeira metade e segunda metade do período
    mid = len(df) // 2
    first_half  = df.iloc[:mid]["pace_min_km"]
    second_half = df.iloc[mid:]["pace_min_km"]

    t_stat, p_value = stats.ttest_ind(first_half, second_half)
    improved = second_half.mean() < first_half.mean()  # pace menor = mais rápido

    # Correlação de Spearman com o tempo (tendência)
    df["days"] = (df["activity_date"] - df["activity_date"].min()).dt.days
    spearman_r, spearman_p = stats.spearmanr(df["days"], df["pace_min_km"])

    result = {
        "test": "Evolução do Pace",
        "media_1a_metade_min_km": round(first_half.mean(), 3),
        "media_2a_metade_min_km": round(second_half.mean(), 3),
        "melhorou": improved,
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "significativo_95": p_value < 0.05,
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 4),
        "tendencia": "melhora" if spearman_r < 0 and spearman_p < 0.05 else
                     "piora"   if spearman_r > 0 and spearman_p < 0.05 else "sem tendência clara"
    }

    print(f"\n📊 TESTE: Evolução do Pace")
    print(f"   Pace médio (1ª metade): {result['media_1a_metade_min_km']} min/km")
    print(f"   Pace médio (2ª metade): {result['media_2a_metade_min_km']} min/km")
    print(f"   T-test p-value: {result['p_value']} {'✅ significativo' if result['significativo_95'] else '❌ não significativo'}")
    print(f"   Tendência Spearman: {result['tendencia']} (r={result['spearman_r']}, p={result['spearman_p']})")

    return result

# ── Teste 2: Sazonalidade por dia da semana ────────────────────────────────────
def test_weekday_performance(runs: pd.DataFrame) -> dict:
    if "pace_min_km" not in runs.columns or "weekday" not in runs.columns:
        return {}

    df = runs[["weekday", "pace_min_km"]].dropna()
    df = df[(df["pace_min_km"] > 3) & (df["pace_min_km"] < 15)]

    groups = [group["pace_min_km"].values for _, group in df.groupby("weekday")]
    if len(groups) < 2:
        return {}

    f_stat, p_value = stats.f_oneway(*groups)
    weekday_means = df.groupby("weekday")["pace_min_km"].mean().sort_values()

    result = {
        "test": "Sazonalidade por Dia da Semana (ANOVA)",
        "f_statistic": round(f_stat, 4),
        "p_value": round(p_value, 4),
        "significativo_95": p_value < 0.05,
        "melhor_dia": weekday_means.index[0],
        "pior_dia": weekday_means.index[-1],
    }

    print(f"\n📊 TESTE: Pace por Dia da Semana (ANOVA)")
    print(f"   F-statistic: {result['f_statistic']} | p-value: {result['p_value']}")
    print(f"   Melhor dia (pace mais rápido): {result['melhor_dia']}")
    print(f"   Pior dia (pace mais lento): {result['pior_dia']}")
    print(f"   {'✅ Diferenças significativas entre dias' if result['significativo_95'] else '❌ Sem diferença significativa entre dias'}")

    return result

# ── Teste 3: Volume vs Performance ───────────────────────────────────────────
def test_volume_vs_pace(runs: pd.DataFrame) -> dict:
    if not all(c in runs.columns for c in ["pace_min_km", "weekly_distance_km"]):
        return {}

    df = runs[["pace_min_km", "weekly_distance_km"]].dropna()
    df = df[(df["pace_min_km"] > 3) & (df["pace_min_km"] < 15)]

    if len(df) < 10:
        return {}

    r, p = stats.pearsonr(df["weekly_distance_km"], df["pace_min_km"])
    result = {
        "test": "Volume Semanal vs Pace",
        "pearson_r": round(r, 4),
        "p_value": round(p, 4),
        "significativo_95": p < 0.05,
        "interpretacao": "Mais volume → pace mais rápido" if r < -0.1 and p < 0.05 else
                         "Mais volume → pace mais lento (fadiga?)" if r > 0.1 and p < 0.05 else
                         "Volume não correlaciona com pace"
    }

    print(f"\n📊 TESTE: Volume Semanal vs Pace")
    print(f"   Pearson r: {result['pearson_r']} | p-value: {result['p_value']}")
    print(f"   Interpretação: {result['interpretacao']}")

    return result

# ── Plot: Boxplot de Pace por Mês ─────────────────────────────────────────────
def plot_pace_by_month(runs: pd.DataFrame):
    if not all(c in runs.columns for c in ["pace_min_km", "activity_date"]):
        return

    df = runs[["activity_date", "pace_min_km"]].dropna()
    df = df[(df["pace_min_km"] > 3) & (df["pace_min_km"] < 15)]
    df["month"] = df["activity_date"].dt.strftime("%b")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_order = [m for m in month_order if m in df["month"].unique()]

    fig, ax = plt.subplots(figsize=(14, 6))
    data_by_month = [df[df["month"] == m]["pace_min_km"].values for m in month_order]
    bp = ax.boxplot(data_by_month, labels=month_order, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))

    for patch in bp["boxes"]:
        patch.set_facecolor("#FC4C02")
        patch.set_alpha(0.7)

    ax.set_title("📦 Distribuição de Pace por Mês", fontsize=14, fontweight="bold")
    ax.set_ylabel("Pace (min/km)")
    ax.invert_yaxis()

    plt.tight_layout()
    path = PLOTS_DIR / "06_pace_by_month.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Salvo: {path}")

# ── Salva relatório ───────────────────────────────────────────────────────────
def save_report(results: list):
    lines = ["RELATÓRIO DE ANÁLISE ESTATÍSTICA", "=" * 50, ""]
    for r in results:
        if not r:
            continue
        lines.append(f"TESTE: {r.get('test','')}")
        for k, v in r.items():
            if k != "test":
                lines.append(f"  {k}: {v}")
        lines.append("")

    path = REPORTS_DIR / "statistical_report.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Relatório salvo em {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("=" * 50)
    print("  ANÁLISE ESTATÍSTICA")
    print("=" * 50)

    runs = load_runs()
    print(f"✓ {len(runs)} corridas carregadas")

    results = []
    results.append(test_pace_improvement(runs))
    results.append(test_weekday_performance(runs))
    results.append(test_volume_vs_pace(runs))

    plot_pace_by_month(runs)
    save_report(results)

    print("\n✅ Análise estatística concluída!")

if __name__ == "__main__":
    run()
