# Fitness EDA, Dashboard & ML Project

[🇧🇷 Português](#-português) | [🇺🇸 English](#-english)

---

## 🇧🇷 Português

Este projeto é uma solução completa em Python para análise e modelagem de dados de atividades físicas, integrando informações extraídas do **Strava** e do **Apple Health**.

### 🚀 Funcionalidades

O projeto é dividido em diferentes módulos dentro da pasta `src/`:

- **🧩 Pipeline de Dados (`src/pipeline/`)**: 
  - Scripts de ETL para extrair dados brutos (ZIPs exportados) do **Strava** (corridas, elevação, tempo, pace, FC) e do **Apple Health** (HRV, VO2Max, passos, calorias).
  - Limpeza e mesclagem automatizada dos dados, gerando features avançadas e agregadas dia a dia.
  
- **📊 Análise Exploratória e Estatística (`src/eda/`)**:
  - Geração de gráficos complexos como Mapas de Calor (estilo contribuições do GitHub), evolução do Pace, carga de treino semanal e correlações.
  - Testes estatísticos estruturados. As visualizações são salvas automaticamente na pasta `outputs/plots/`.

- **🤖 Machine Learning (`src/ml/`)**:
  - **Modelo de Regressão**: Treinamento de diversos algoritmos (Ridge, Random Forest, Gradient Boosting) para **prever o Pace** com base em características dos treinos (distância, elevação, heart rate, carga das últimas semanas, etc).
  - **Modelo de Classificação**: Pipeline com Random Forest focado em classificar a **Intensidade do Treino** (Leve, Moderado, Intenso) usando o *suffer_score* e métricas de esforço.
  
- **📈 Dashboard (`src/dashboard/`)**:
  - Módulo reservado para a interface visual interativa de visualização de resultados de Machine Learning e EDA.

### 📂 Estrutura de Diretórios

```text
fitness_eda_dash_ml/
├── data/                  # Dados brutos das fontes, intermediários e features geradas
├── outputs/
│   ├── plots/             # Gráficos da EDA e avaliação dos modelos ML salvos automaticamente
│   └── reports/           # Relatórios gerados pelos modelos
├── src/
│   ├── pipeline/          # Processamento de Dados ETL (01_strava, 02_apple_health)
│   ├── eda/               # Scripts de Análise Exploratória
│   ├── ml/                # Pipeline preditivo (Regressão e Classificação)
│   └── dashboard/         # Código da aplicação Dash
└── README.md              # Este documento
```

> **⚠️ Aviso sobre os Dados**: A pasta `data/` encontra-se ignorada no controle de versão (`.gitignore`) para proteger a privacidade dos seus treinos. Para executar o projeto localmente de maneira completa, você deve exportar os seus ZIPs do Strava e do Apple Health.

### 🛠️ Como Executar

**1. Processamento Inicial dos Dados (Pipeline ETL):**
O fluxo inicia extraindo tudo a partir dos seus zips exportados de suas contas.
```bash
python src/pipeline/run_pipeline.py caminho/para/strava.zip caminho/para/apple_health.zip
```
*(O Apple Health ZIP é opcional caso queira rodar apenas as informações fornecidas pelo Strava).*

**2. Geração da Análise Exploratória (EDA):**
Execute este script para processar a base combinada e plotar visões detalhadas das suas atividades.
```bash
python src/eda/04_eda.py
```
*Os gráficos (visão geral, correlações, heatmap anual, carga de treino) ficarão salvos na pasta `outputs/plots/`.*

**3. Treinamento dos Modelos Predicionais (ML):**
Execute a pipeline de ML para treinar a regressão (previsão de pace) e a classificação de intensidade.
```bash
python src/ml/06_machine_learning.py
```
*Gera relatórios de acurácia, R², MAE e plota resultados de predições VS real e Matriz de Confusão no diretório de `outputs/plots/`.*

### 📋 Requisitos

- Python 3.8+
- Pandas, NumPy
- Scikit-Learn
- Matplotlib, Seaborn, Plotly

---

## 🇺🇸 English

This project is a comprehensive Python solution for analyzing and modeling physical activity data, integrating information extracted from **Strava** and **Apple Health**.

### 🚀 Features

The project is divided into different modules within the `src/` folder:

- **🧩 Data Pipeline (`src/pipeline/`)**: 
  - ETL scripts processing raw data (exported ZIPs) from **Strava** (runs, elevation, time, pace, HR) and **Apple Health** (HRV, VO2Max, steps, calories).
  - Automated cleaning, merging and structuring the data for daily analyses running advanced feature engineering.
  
- **📊 Exploratory Data Analysis & Statistics (`src/eda/`)**:
  - Plotting rich visualizations such as GitHub-style Activity Heatmaps, Pace evolution trends, weekly load bars, and variable correlations.
  - Automatically exports all generated plot images into the `outputs/plots/` folder.

- **🤖 Machine Learning (`src/ml/`)**:
  - **Regression Model**: Trains several algorithms (Ridge, Random Forest, Gradient Boosting) to **predict Pace** based on workouts details (distance, elevation, heart rate, historical training load, etc).
  - **Classification Model**: Uses Random Forest Pipeline focusing on categorizing **Workout Intensity** (Light, Moderate, Intense) according to *suffer_score* and exertion metrics.
  
- **📈 Dashboard (`src/dashboard/`)**:
  - Module reserved for the interactive visual interface tracking the results of the EDA and ML models.

### 📂 Directory Structure

```text
fitness_eda_dash_ml/
├── data/                  # Raw source data, processed datasets, and model features
├── outputs/
│   ├── plots/             # Charts generated from EDA & ML automatically saved here
│   └── reports/           # Generated metrics and text reports
├── src/
│   ├── pipeline/          # Complete ETL Data Pipeline
│   ├── eda/               # Scripts for Exploratory analysis plotting
│   ├── ml/                # AI and ML components (Pace Predictor and Clustering)
│   └── dashboard/         # Interactive App
└── README.md              # This document
```

> **⚠️ Data Privacy Notice**: The `data/` folder is intentionally excluded via `.gitignore` to preserve your privacy regarding workouts. To execute the application locally, you must provide your own valid exports of Strava and Apple Health. 

### 🛠️ How to Run

**1. Initial ETL Processing Pipeline:**
Run the initial script providing your exported ZIPs.
```bash
python src/pipeline/run_pipeline.py path/to/strava.zip path/to/apple_health.zip
```
*(Passing the Apple Health ZIP is optional; it defaults to running Strava logic solo).*

**2. Executing EDA (Charts Generation):**
Run this script to plot insights from your generated combined dataset.
```bash
python src/eda/04_eda.py
```
*Graphs (overview, heatmap, load evolution, correlations) will be generated inside the `outputs/plots/` tree.*

**3. Train Predictive Models (ML):**
Execute learning scripts to fit random forests and ridge models for Pace predictions and Intensity classifications.
```bash
python src/ml/06_machine_learning.py
```
*Will output evaluation metrics (R², MAE, Accuracy) and generate confusion matrices + True/Pred relation plots to `outputs/plots/`.*

### 📋 Requirements

- Python 3.8+
- Pandas, NumPy
- Scikit-Learn
- Matplotlib, Seaborn, Plotly
