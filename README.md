# Fitness EDA, Dashboard & ML Project

[🇧🇷 Português](#-português) | [🇺🇸 English](#-english)

---

## 🇧🇷 Português

Este projeto é uma solução completa em Python para análise e modelagem de dados de atividades físicas, integrando informações extraídas do **Strava** e do **Apple Health**.

### 🚀 Funcionalidades

O projeto é dividido em diferentes módulos dentro da pasta `src/`:

- **🧩 Pipeline de Dados (`src/pipeline/`)**: 
  - Scripts para processamento e extração de dados brutos do Strava (arquivos ZIP exportados).
  - Scripts para processamento do XML exportado do Apple Health (passos, calorias, batimentos cardíacos, etc.).
  - Mescla e estruturação dos dados para análises diárias e criação de features avançadas.
  
- **📊 Análise Exploratória e Estatística (`src/eda/`)**:
  - Análise exploratória de dados (EDA) detalhada das atividades.
  - Testes estatísticos para entender correlações e padrões de desempenho (como evolução de pace, impacto do volume de treino, etc.).

- **🤖 Machine Learning (`src/ml/`)**:
  - Modelos preditivos treinados com os dados históricos de treino.
  - Previsões de desempenho, categorização de esforço ou outras métricas alvo definidas nas features extraídas.

- **📈 Dashboard (`src/dashboard/`)**:
  - Módulo reservado para a interface visual interativa para visualização dos resultados da EDA e do Machine Learning.

### 📂 Estrutura de Diretórios

```text
fitness_eda_dash_ml/
├── data/                  # Dados brutos, processados e features (não versionados no Git)
├── src/
│   ├── pipeline/          # Scripts ETL (Strava e Apple Health)
│   ├── eda/               # Scripts de Análise Exploratória e Estatística
│   ├── ml/                # Pipeline de Machine Learning
│   └── dashboard/         # Código da aplicação do Dashboard interativo
├── fitness_project_final/ # Artefatos finais e consolidado do projeto
└── README.md              # Documentação principal
```

### 🛠️ Como Executar

O processamento inicial dos dados começa pelo pipeline. Se você tiver os arquivos ZIP exportados das suas contas:

```bash
python src/pipeline/run_pipeline.py caminho/para/strava.zip caminho/para/apple_health.zip
```
*(O Apple Health ZIP é opcional, caso queira processar apenas dados do Strava).*

Em seguida, explore os scripts nas pastas de `eda` e `ml` para geração de insights e modelos.

### 📋 Requisitos e Tecnologias

- Python 3.8+
- Pandas, NumPy
- Scikit-Learn
- Matplotlib, Seaborn, Plotly (para visualização de dados)
- (Demais bibliotecas específicas podem estar definidas nos scripts individuais)

---

## 🇺🇸 English

This project is a comprehensive Python solution for analyzing and modeling physical activity data, integrating information extracted from **Strava** and **Apple Health**.

### 🚀 Features

The project is divided into different modules within the `src/` folder:

- **🧩 Data Pipeline (`src/pipeline/`)**: 
  - Scripts for processing and extracting raw data from Strava (exported ZIP files).
  - Scripts for processing the XML exported from Apple Health (steps, calories, heart rate, etc.).
  - Merging and structuring the data for daily analyses and advanced feature creation.
  
- **📊 Exploratory Data Analysis & Statistics (`src/eda/`)**:
  - Detailed exploratory data analysis (EDA) of the activities.
  - Statistical tests to understand correlations and performance patterns (such as pace evolution, impact of training volume, etc.).

- **🤖 Machine Learning (`src/ml/`)**:
  - Predictive models trained with historical workout data.
  - Performance predictions, effort categorization, or other target metrics defined in the extracted features.

- **📈 Dashboard (`src/dashboard/`)**:
  - Module reserved for the interactive visual interface to track the results of the EDA and Machine Learning models.

### 📂 Directory Structure

```text
fitness_eda_dash_ml/
├── data/                  # Raw and processed data, features (not versioned in Git)
├── src/
│   ├── pipeline/          # ETL Scripts (Strava and Apple Health)
│   ├── eda/               # Exploratory and Statistical Analysis scripts
│   ├── ml/                # Machine Learning pipeline
│   └── dashboard/         # Interactive Dashboard application code
├── fitness_project_final/ # Final artifacts and consolidated project files
└── README.md              # Main documentation
```

### 🛠️ How to Run

The initial data processing starts with the pipeline. If you have the exported ZIP files from your accounts:

```bash
python src/pipeline/run_pipeline.py path/to/strava.zip path/to/apple_health.zip
```
*(The Apple Health ZIP is optional if you only want to process Strava data).*

Afterward, explore the scripts in the `eda` and `ml` folders to generate insights and models.

### 📋 Requirements and Technologies

- Python 3.8+
- Pandas, NumPy
- Scikit-Learn
- Matplotlib, Seaborn, Plotly (for data visualization)
- (Other specific libraries may be defined in individual scripts)

---

Developed for continuous tracking and deep analysis of fitness conditioning metrics.
