# Tech Challenge 05 — Datathon (Fase 5) — Predição de risco

Este repositório implementa um fluxo completo de dados e modelagem para estimar a **probabilidade de um aluno entrar em risco de defasagem no próximo ano (T+1)** a partir dos indicadores do ano atual (T).
O pipeline inclui: ingestão e padronização do dado (ABT), EDA orientado às questões do enunciado, treinamento de modelo (Random Forest), checagens de overfitting/leakage, construção opcional de features históricas e um app Streamlit para consumo do modelo.

---

## Visão geral do problema

- **Entrada (Ano T):** indicadores do aluno no ano atual (ex.: INDE, IDA, IEG, IAA, IPS, IPV, Pedra etc.).
- **Saída (Ano T+1):** probabilidade de risco e recomendação operacional para priorização.
- **Estratégia de modelagem:** abordagem longitudinal **T→T+1**.
  - O modelo não aprende “risco do próprio ano” (snapshot).
  - O modelo aprende padrões em T que antecipam risco em T+1, alinhando com a necessidade de alerta precoce.

---

## Requisitos

- Python 3.10+ (ambiente utilizado no desenvolvimento: Python 3.14)
- Dependências principais:
  - pandas, numpy
  - pyarrow (parquet)
  - scikit-learn, joblib
  - matplotlib, seaborn (EDA)
  - streamlit (app)

## Fluxo recomendado de execução

### 1) Ingestão e criação do ABT

Responsável por ler o dado bruto, padronizar colunas e criar o dataset principal:

```bash
python scripts/01_data_ingestion.py
```

Saída esperada:

- `data/02_processed/abt.parquet`

Observação: o ABT contém `Ano` e `RA` como chaves principais e o target derivado (`Target_Risco`).

---

### 2) Treino do modelo (T→T+1)

Treinamento do modelo longitudinal e exportação de artefatos:

```bash
python scripts/02_machine_learning.py
```

Saídas esperadas (modelo base):

- `outputs/ml_t_to_t1/base/modelo_rf_t_to_t1_base.joblib`
- `outputs/ml_t_to_t1/base/predicoes_2024_usando_2023_thr045_base.csv`
- `outputs/ml_t_to_t1/base/metadata_t_to_t1_thr045_base.json`

O arquivo de predição contém:

- `Target_Risco_2024_real`: classe real no ano T+1 (ground truth do teste)
- `Prob_Risco_2024`: probabilidade (score) de risco no ano T+1
- `Flag_Acionar_thr045`: decisão operacional no threshold diagnóstico 0.45
- `Target_Risco_2024_pred_thr045`: mesma decisão, como rótulo previsto

Observação: o threshold 0.45 foi tratado como corte de referência

---

### 3) Checagens de overfitting / leakage

Validações estruturais e diagnósticos (inclui teste com label permutation):

```bash
python scripts/03_checks_overfitting_leakage.py
```

Objetivos típicos:

- verificação de schema e colunas suspeitas
- overlap de linhas idênticas entre treino e teste
- gap train/test (AUC)
- permutation importance
- label permutation test (AUC ~0.5 como evidência contra leakage estrutural)

---

## App Streamlit (entrega final)

O app expõe duas funcionalidades:

1) **Aluno**: seleção de Ano T e RA, exibindo probabilidade de risco em T+1 e recomendação.
2) **Priorizar**: ranking Top N por probabilidade para um Ano T, com export CSV.

### Executar

Na raiz do projeto:

```bash
python -m streamlit run app.py
```

---

## Notas sobre interpretação do resultado

- **Probabilidade (score)**: valor contínuo 0–1 indicando risco no ano T+1.
- **Recomendação/Flag**: decisão operacional baseada em threshold:
  - “Priorizar acompanhamento/intervenção para risco no próximo ano” quando score ≥ threshold.
  - “Manter acompanhamento padrão (sem prioridade adicional)” caso contrário.

O threshold pode ser calibrado dependendo da capacidade de intervenção (trade-off entre falsos positivos e falsos negativos).

---

## Notas gerais

Alguns documentos como **datathon.pdf** e **relatorio.pdf** podem ser encontrados no diretório **/files**K. Recomenda-se a leitura dos mesmos para melhor entendimento do problema e solução.
