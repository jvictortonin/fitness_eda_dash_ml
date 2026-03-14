# Fitness EDA, Dashboard & ML Project

Este projeto é uma solução completa em Python para análise e modelagem de dados de atividades físicas, integrando informações extraídas do **Strava** e do **Apple Health**.

## 🚀 Funcionalidades

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

## 📂 Estrutura de Diretórios

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

## 🛠️ Como Executar

O processamento inicial dos dados começa pelo pipeline. Se você tiver os arquivos ZIP exportados das suas contas:

```bash
python src/pipeline/run_pipeline.py caminho/para/strava.zip caminho/para/apple_health.zip
```
*(O Apple Health ZIP é opcional, caso queira processar apenas dados do Strava).*

Em seguida, explore os scripts nas pastas de `eda` e `ml` para geração de insights e modelos.

## 📋 Requisitos e Tecnologias

- Python 3.8+
- Pandas, NumPy
- Scikit-Learn
- Matplotlib, Seaborn, Plotly (para visualização de dados)
- (Demais bibliotecas específicas podem estar definidas nos scripts individuais)

---

Desenvolvido para acompanhamento contínuo e análise profunda de métricas de condicionamento físico.
