## `DICIONÁRIO DE DADOS`
---

### 1) `Target_Risco_2024_real`: a “verdade”

É o que **de fato aconteceu em 2024** (0 ou 1).

* 1 = aluno em risco em 2024
* 0 = aluno não em risco em 2024

Serve para **avaliar** o modelo (comparar contra o que o modelo “disse”).

---

### 2) `Prob_Risco_2024`: “score” do modelo

É um número entre 0 e 1:

> “com base em 2023, qual a chance do aluno estar em risco em 2024?”

Exemplos:

* `Prob_Risco_2024 = 0.92` → alta chance
* `Prob_Risco_2024 = 0.12` → baixa chance

Esse valor não é “sim/não”. É uma régua contínua.

---

### 3) `Flag_Acionar`: decisão operacional (após threshold)

Transforma a probabilidade em decisão com:

> **Flag_Acionar = 1 se Prob_Risco_2024 ≥ 0.45**

---

### 4) `Pred_2024_thr045`: predição binária

No script, é definido:

```python
y_pred_thr = (y_proba >= THRESHOLD_FINAL).astype(int)
```

e salva esse mesmo `y_pred_thr` em duas colunas:

* `Flag_Acionar`
* `Pred_2024_thr045`

Ou seja, no arquivo **predicoes_2024_usando_2023_thr045.csv:**

`Flag_Acionar` **==** `Pred_2024_thr045` (sempre)

A diferença é semântica:

* `Flag_Acionar` = linguagem de negócio (“acionar intervenção?”)
* `Pred_2024_thr045` = linguagem de ML (“classe predita”)

---

# Exemplo:

Suponha um aluno:

* `Target_Risco_2024_real = 0` (não estava em risco em 2024)
* `Prob_Risco_2024 = 0.60`

Com threshold 0.45:

* `Flag_Acionar = 1`
* `Pred_2024_thr045 = 1`

Interpretação:

* o modelo “achou” que era risco
* foi acionado
* mas na realidade não era → **falso positivo (FP)**

Outro aluno:

* `Target_Risco_2024_real = 1`

* `Prob_Risco_2024 = 0.30`

* `Flag_Acionar = 0`

* `Pred_2024_thr045 = 0`

Interpretação:

* modelo não acionou
* mas era risco → **falso negativo (FN)** (o pior tipo, no seu caso)

---

### Como isso conecta com a matriz de confusão

Comparando:

* `Target_Risco_2024_real` (verdade)
  vs
* `Pred_2024_thr045` (predição binária = Flag_Acionar)

Exitem quatro casos:

* real=1 e pred=1 → **TP** (acertou e acionou)
* real=0 e pred=0 → **TN** (acertou e não acionou)
* real=0 e pred=1 → **FP** (acionou sem necessidade)
* real=1 e pred=0 → **FN** (não acionou quando precisava)