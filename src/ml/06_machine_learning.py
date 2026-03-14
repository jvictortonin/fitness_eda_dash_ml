"""
MACHINE LEARNING
=================
Dois modelos:
  1. Regressão → Prever pace com base em features do treino
  2. Classificação → Classificar intensidade do treino (leve/moderado/intenso)

Como usar:
    python 06_machine_learning.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR  = Path("data/features")
PLOTS_DIR     = Path("outputs/plots")
REPORTS_DIR   = Path("outputs/reports")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"primary": "#FC4C02", "blue": "#2196F3", "green": "#4CAF50", "gray": "#9E9E9E"}

# ── Prepara dados ─────────────────────────────────────────────────────────────
def prepare_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "strava_processed.csv", parse_dates=["activity_date"])

    # Filtra corridas
    if "activity_type_clean" in df.columns:
        df = df[df["activity_type_clean"].str.contains("Run", na=False)].copy()

    # Features temporais
    df["days_since_start"] = (df["activity_date"] - df["activity_date"].min()).dt.days
    df["month"]   = df["activity_date"].dt.month
    df["weekday_num"] = df["activity_date"].dt.dayofweek

    # Remove outliers de pace
    if "pace_min_km" in df.columns:
        df = df[(df["pace_min_km"] > 3) & (df["pace_min_km"] < 15)]

    print(f"✓ Dataset ML: {len(df)} corridas")
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# MODELO 1: REGRESSÃO — Prever Pace
# ═══════════════════════════════════════════════════════════════════════════════
def model_pace_regression(df: pd.DataFrame):
    print("\n" + "="*50)
    print("  MODELO 1: REGRESSÃO DE PACE")
    print("="*50)

    target = "pace_min_km"
    feature_candidates = [
        "distance_km", "moving_time_min", "elevation_gain",
        "average_heart_rate", "weekly_distance_km",
        "days_since_start", "month", "weekday_num", "suffer_score"
    ]
    features = [c for c in feature_candidates if c in df.columns]

    X = df[features]
    y = df[target]
    mask = y.notna() & X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    if len(X) < 30:
        print("  [!] Dados insuficientes para regressão (mínimo: 30 corridas)")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]),
    }

    results = {}
    print(f"\nFeatures usadas: {features}")
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        results[name] = {"mae": mae, "rmse": rmse, "r2": r2, "pred": y_pred, "model": model}
        print(f"  {name}:")
        print(f"    MAE:  {mae:.3f} min/km  (erro médio absoluto)")
        print(f"    RMSE: {rmse:.3f} min/km")
        print(f"    R²:   {r2:.3f}")

    # Melhor modelo
    best_name = min(results, key=lambda k: results[k]["mae"])
    best      = results[best_name]
    print(f"\n🏆 Melhor modelo: {best_name} (MAE: {best['mae']:.3f} min/km)")

    # Feature importance (Random Forest)
    rf = results["Random Forest"]["model"]
    importances = rf.named_steps["model"].feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("🤖 Modelo 1: Predição de Pace", fontsize=15, fontweight="bold")

    # Predito vs Real
    ax = axes[0]
    y_pred_best = best["pred"]
    ax.scatter(y_test, y_pred_best, alpha=0.5, color=COLORS["primary"], s=40)
    mn, mx = min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.5, label="Predição perfeita")
    ax.set_xlabel("Pace Real (min/km)")
    ax.set_ylabel("Pace Predito (min/km)")
    ax.set_title(f"Real vs Predito — {best_name}\nR²={best['r2']:.3f} | MAE={best['mae']:.3f} min/km")
    ax.legend()

    # Feature importance
    ax = axes[1]
    feat_imp.head(8).plot(kind="barh", ax=ax, color=COLORS["primary"], alpha=0.8)
    ax.invert_yaxis()
    ax.set_title("Importância das Features (Random Forest)")
    ax.set_xlabel("Importância")

    plt.tight_layout()
    path = PLOTS_DIR / "07_ml_regression.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Gráfico salvo: {path}")

    return best

# ═══════════════════════════════════════════════════════════════════════════════
# MODELO 2: CLASSIFICAÇÃO — Intensidade do Treino
# ═══════════════════════════════════════════════════════════════════════════════
def model_intensity_classification(df: pd.DataFrame):
    print("\n" + "="*50)
    print("  MODELO 2: CLASSIFICAÇÃO DE INTENSIDADE")
    print("="*50)

    # Cria label de intensidade baseado no suffer_score ou FC média
    if "suffer_score" in df.columns and df["suffer_score"].notna().sum() > 30:
        score_col = "suffer_score"
    elif "average_heart_rate" in df.columns and df["average_heart_rate"].notna().sum() > 30:
        score_col = "average_heart_rate"
    else:
        print("  [!] Sem suffer_score nem FC média suficiente para classificação")
        return

    df_cls = df[df[score_col].notna()].copy()
    q33    = df_cls[score_col].quantile(0.33)
    q66    = df_cls[score_col].quantile(0.66)

    def classify(v):
        if v <= q33:   return "Leve"
        elif v <= q66: return "Moderado"
        else:          return "Intenso"

    df_cls["intensity"] = df_cls[score_col].apply(classify)
    print(f"\nDistribuição de intensidade:\n{df_cls['intensity'].value_counts()}")

    feature_candidates = [
        "distance_km", "moving_time_min", "elevation_gain",
        "weekly_distance_km", "days_since_start", "month", "weekday_num"
    ]
    features = [c for c in feature_candidates if c in df_cls.columns]
    X = df_cls[features]
    y = df_cls["intensity"]
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    if len(X) < 30:
        print("  [!] Dados insuficientes para classificação")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Cross validation
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"Acurácia Cross-Validation (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Plot matriz de confusão
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("🤖 Modelo 2: Classificação de Intensidade", fontsize=15, fontweight="bold")

    cm = confusion_matrix(y_test, y_pred, labels=["Leve", "Moderado", "Intenso"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Leve", "Moderado", "Intenso"])
    disp.plot(ax=axes[0], colorbar=False, cmap="YlOrRd")
    axes[0].set_title("Matriz de Confusão")

    # Feature importance
    importances = clf.named_steps["model"].feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    feat_imp.plot(kind="barh", ax=axes[1], color=COLORS["blue"], alpha=0.8)
    axes[1].invert_yaxis()
    axes[1].set_title("Importância das Features")
    axes[1].set_xlabel("Importância")

    plt.tight_layout()
    path = PLOTS_DIR / "08_ml_classification.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Gráfico salvo: {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    df = prepare_data()
    model_pace_regression(df)
    model_intensity_classification(df)
    print("\n✅ Machine Learning concluído!")

if __name__ == "__main__":
    run()
