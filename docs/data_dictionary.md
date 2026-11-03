# AI Ops Copilot — Dicionário de Dados, Regras de Atraso e Geração Sintética (Fase 0)

> **Status:** Fase 0 — Especificação de Dados Consolidada  
> **Versão:** 1.0.0  
> **Unidade de Análise:** 1 registro (linha) = 1 pedido de delivery (`order_id`)  
> **Formato de Armazenamento:** Arquivo CSV UTF-8 (`data/generated/` e `data/samples/`)

---

## 1. Definição Formal e Matemática de Atraso

No domínio do **AI Ops Copilot**, cada pedido possui tempos estimados no momento em que o pedido é criado e tempos reais observados apenas após a conclusão da operação.

### 1.1. Equações de Tempo e Critério de Atraso

Sejam:
- $T_{\text{prep\_est}}$ (`estimated_prep_minutes`): tempo estimado de preparo pelo estabelecimento (em minutos).
- $T_{\text{ent\_est}}$ (`estimated_delivery_minutes`): tempo estimado de deslocamento/entrega (em minutos).
- $T_{\text{prep\_real}}$ (`actual_prep_minutes`): tempo real observado de preparo (em minutos).
- $T_{\text{ent\_real}}$ (`actual_delivery_minutes`): tempo real observado de entrega (em minutos).
- $\tau$ (`late_tolerance_minutes`): tolerância operacional configurável (valor padrão no MVP: $\tau = 5\text{ minutos}$).

O **tempo total prometido** ao cliente no instante da criação do pedido é:

$$
T_{\text{prometido}} = T_{\text{prep\_est}} + T_{\text{ent\_est}}
$$

O **tempo total real** decorrido até a conclusão do pedido é:

$$
T_{\text{real}} = T_{\text{prep\_real}} + T_{\text{ent\_real}}
$$

Um pedido é classificado como **atrasado** (`is_late = True`) se, e somente se, o tempo total real superar o tempo total prometido acrescido da tolerância $\tau$:

$$
\text{is\_late} = \begin{cases} 
\text{True}, & \text{se } T_{\text{real}} > T_{\text{prometido}} + \tau \\
\text{False}, & \text{caso contrário } (T_{\text{real}} \le T_{\text{prometido}} + \tau)
\end{cases}
$$

> [!IMPORTANT]
> O parâmetro $\tau$ (`late_tolerance_minutes`, padrão `5`) é armazenado nos metadados do dataset e do experimento. Qualquer alteração em $\tau$ altera a distribuição da variável-alvo `is_late` e deve ser registrada no `ExperimentRun`.

---

## 2. Esquema do Dataset Principal (CSV)

A tabela abaixo especifica todas as colunas obrigatórias do dataset bruto de pedidos:

| Coluna | Tipo Lógico | Tipo Python / Pandas | Domínio / Restrições | Obrigatório | Uso no Modelo de ML? | Descrição |
|---|---|---|---|---|---|---|
| `order_id` | `string` | `str` / `object` | Único no dataset (ex.: `"ORD-000001"`) | Sim | **Não** (Identificador) | Identificador único fictício do pedido. |
| `created_at` | `datetime` | `datetime64[ns]` | ISO-8601 (`YYYY-MM-DD HH:MM:SS`) | Sim | **Indireto** (Extrair hora/dia) | Timestamp de criação do pedido. Define o instante $t_0$ da predição. |
| `region` | `category` | `str` / `category` | `{"north", "south", "east", "west", "central"}` | Sim | **Sim** (Feature categórica) | Região geográfica fictícia de entrega. |
| `order_category` | `category` | `str` / `category` | `{"restaurant", "fast_food", "market", "pharmacy", "bakery"}` | Sim | **Sim** (Feature categórica) | Categoria do estabelecimento. |
| `order_value` | `decimal` | `float64` | $> 0.0$ (ex.: `10.00` a `500.00`) | Sim | **Sim** (Feature numérica) | Valor monetário total do pedido (em moeda fictícia). |
| `estimated_prep_minutes` | `integer` | `int64` | Inteiro $\in [1, 180]$ | Sim | **Sim** (Feature numérica) | Tempo estimado de preparo informado na criação do pedido (minutos). |
| `estimated_delivery_minutes` | `integer` | `int64` | Inteiro $\in [1, 180]$ | Sim | **Sim** (Feature numérica) | Tempo estimado de entrega informado na criação do pedido (minutos). |
| `actual_prep_minutes` | `integer` | `int64` | Inteiro $\in [1, 360]$ | Sim* | **NÃO (Proibido — Leakage)** | Tempo real observado de preparo após conclusão (minutos). |
| `actual_delivery_minutes` | `integer` | `int64` | Inteiro $\in [1, 360]$ | Sim* | **NÃO (Proibido — Leakage)** | Tempo real observado de entrega após conclusão (minutos). |
| `is_late` | `boolean` | `bool` | `{True, False}` | Sim* | **Alvo ($y$)** | Indicador binário de atraso segundo a regra $T_{\text{real}} > T_{\text{prometido}} + \tau$. |

*\*Nota: As colunas `actual_prep_minutes`, `actual_delivery_minutes` e `is_late` são obrigatórias em datasets históricos usados para análise operacional e treinamento/avaliação de modelos, mas **não existem** no payload de inferência de novos pedidos (`POST /api/v1/ml/predict`).*

---

## 3. Atributos Derivados (Feature Engineering & Analytics)

A partir das colunas brutas, os módulos `data` e `ml.preprocessing` derivam atributos determinísticos adicionais:

| Atributo Derivado | Tipo | Fórmula / Regra de Derivação | Domínio | Uso Permitido |
|---|---|---|---|---|
| `hour_of_day` | `int` | `created_at.dt.hour` | $0 \dots 23$ | Análise operacional (`compare_groups`) e Feature de ML ($X$). |
| `day_of_week` | `int` | `created_at.dt.dayofweek` (Segunda = `0`, Domingo = `6`) | $0 \dots 6$ | Análise operacional e Feature de ML ($X$). |
| `is_weekend` | `bool` | `day_of_week in {5, 6}` | `{True, False}` | Análise operacional e Feature de ML ($X$). |
| `promised_total_minutes` | `int` | `estimated_prep_minutes + estimated_delivery_minutes` | $\ge 2$ | Análise operacional e Feature opcional de ML ($X$). |
| `actual_total_minutes` | `int` | `actual_prep_minutes + actual_delivery_minutes` | $\ge 2$ | **Apenas** análise operacional histórica (Proibido em $X$). |
| `delay_minutes` | `int` | `(actual_prep_minutes + actual_delivery_minutes) - (estimated_prep_minutes + estimated_delivery_minutes)` | $\mathbb{Z}$ | **Apenas** análise operacional histórica (Proibido em $X$). |

---

## 4. Política Estrita de Prevenção de *Data Leakage*

> [!CAUTION]
> **Regra de Ouro do Pipeline de ML:** O modelo preditivo é acionado no instante $t_0$ (`created_at`), imediatamente após o pedido ser registrado. Nenhuma informação gerada em $t > t_0$ pode entrar na matriz de atributos $X$.

### 4.1. Particionamento de Variáveis para Machine Learning

| Conjunto | Variáveis | Regra de Segurança |
|---|---|---|
| **Features de Entrada Permitidas ($X$)** | `hour_of_day`, `day_of_week`, `is_weekend`, `region`, `order_category`, `order_value`, `estimated_prep_minutes`, `estimated_delivery_minutes` | Validadas pelo contrato `OrderPredictionFeatures` no treino e na inferência. |
| **Variável-Alvo ($y$)** | `is_late` | Usada exclusivamente como vetor de rótulos supervisionados no treino e avaliação. |
| **Colunas Bloqueadas em ML (Pós-Evento / Identificação)** | `order_id`, `created_at` (bruto), `actual_prep_minutes`, `actual_delivery_minutes`, `actual_total_minutes`, `delay_minutes` | Descartadas explicitamente antes do `ColumnTransformer`. Testes unitários verificam que a presença ou ausência de colunas `actual_*` não altera a predição. |

### 4.2. Ordem Estrita de Pré-processamento
1. **Divisão dos Dados (*Split*) Antes de Qualquer Ajuste:** O dataset válido é dividido em conjuntos de **Treino** (ex.: 70%), **Validação** (ex.: 15%) e **Teste** (ex.: 15%) antes de calcular médias, desvios-padrão ou categorias.
2. **Ajuste Exclusivo no Treino:** Transformadores numéricos (`StandardScaler` / imputadores) e categóricos (`OneHotEncoder(handle_unknown="ignore")`) executam `.fit()` **somente** sobre $X_{\text{train}}$.
3. **Transformação Independente:** As partições $X_{\text{val}}$ e $X_{\text{test}}$ passam exclusivamente por `.transform()`.

---

## 5. Regras de Validação e Qualidade de Dados (`data.validation`)

Quando um dataset é submetido à validação (`POST /api/v1/datasets/{dataset_id}/validate`), o validador aplica três níveis de verificação:

### 5.1. Nível 1 — Validação Estrutural de Arquivo (Bloqueante)
- O arquivo deve existir, não estar vazio e ser decodificável em UTF-8.
- Todas as 10 colunas obrigatórias do esquema bruto devem estar presentes no cabeçalho.
- Se falhar neste nível, a validação retorna erro estrutural (`INVALID_FILE_FORMAT`) e o dataset recebe status `FAILED`.

### 5.2. Nível 2 — Validação de Registros e Domínio (Por Linha)
Um registro individual é classificado como **inválido** (`invalid_records`) se apresentar qualquer uma das violações abaixo:

| Código da Violação (`issue`) | Campo(s) Afetado(s) | Condição de Disparo |
|---|---|---|
| `missing_values` | Qualquer coluna obrigatória | Valor `null`, `NaN` ou string vazia. |
| `duplicate_order_id` | `order_id` | `order_id` repetido dentro do mesmo dataset. |
| `invalid_datetime` | `created_at` | String que não pode ser convertida para `datetime`. |
| `unknown_category` | `region`, `order_category` | Valor fora do conjunto de categorias válidas documentadas. |
| `out_of_range` | `order_value`, `*_minutes` | `order_value <= 0`, tempos estimados $\notin [1, 180]$ ou tempos reais $\notin [1, 360]$. |
| `inconsistent_late_flag` | `is_late` | Valor de `is_late` divergente da fórmula oficial $T_{\text{real}} > T_{\text{prometido}} + \tau$. |

### 5.3. Nível 3 — Critério de Aprovação do Dataset
- O relatório de qualidade contabiliza `total_records`, `valid_records`, `invalid_records` e agrupa as ocorrências em `warnings` por `(field, issue, count)`.
- As análises operacionais e o treinamento de ML operam sobre os registros válidos (`valid_records`), desde que o dataset atinja o limiar mínimo de registros válidos (padrão: $\ge 100$ registros válidos e taxa de registros inválidos $\le 20\%$).

---

## 6. Especificação do Gerador de Dados Sintéticos (`data.generator`)

O gerador sintético (`RF-01`) tem como objetivo produzir dados realistas, estatisticamente ricos e totalmente livres de dados pessoais, permitindo testar análises e treinar classificadores sem que a relação entre atributos e atraso seja trivial (determinística) nem puramente aleatória (ruído branco).

### 6.1. Parâmetros de Configuração do Gerador

| Parâmetro | Tipo | Valor Padrão | Descrição |
|---|---|---|---|
| `number_of_orders` | `int` | `10000` | Quantidade total de pedidos gerados. |
| `random_seed` | `int` | `42` | Semente do gerador pseudoaleatório (`numpy.random.default_rng(seed)`). |
| `late_tolerance_minutes` | `int` | `5` | Tolerância operacional $\tau$ em minutos para definição de `is_late`. |
| `noise_level` | `float` | `1.0` | Multiplicador da variância do ruído estocástico nos tempos reais. |
| `inject_quality_issues_rate` | `float` | `0.0` (ou `0.002` em modo teste) | Fração opcional de registros com anomalias controladas para testar o validador. |

### 6.2. Dinâmica Simulada e Fatores de Risco de Atraso
Para que o classificador de ML e as ferramentas analíticas do agente encontrem padrões operacionais interpretáveis, o gerador modela os seguintes efeitos:

1. **Distribuição Temporal (`created_at`):**
   - Concentração maior de pedidos nas janelas de pico de **almoço (11h–14h)** e **jantar (18h–22h)**, além de maior volume às sextas, sábados e domingos.
2. **Distribuição Regional (`region`) e de Categoria (`order_category`):**
   - Regiões (`north`, `south`, `east`, `west`, `central`) possuem distâncias médias distintas que afetam `estimated_delivery_minutes` e a variância do trânsito.
   - Categorias (`restaurant`, `fast_food`, `market`, `pharmacy`, `bakery`) possuem perfis distintos de `order_value` e `estimated_prep_minutes` (ex.: `restaurant` e `market` têm maior tempo de preparo/separação que `pharmacy` ou `bakery`).
3. **Formação dos Tempos Reais (`actual_prep_minutes` e `actual_delivery_minutes`):**
   - Os tempos reais partem dos tempos estimados somados a desvios estocásticos influenciados por:
     - **Sobrecarga de horário de pico** (especialmente 12h–13h e 19h–21h);
     - **Efeito de final de semana** (`is_weekend`);
     - **Complexidade do pedido** (pedidos com `order_value` elevado em `restaurant` ou `market` sofrem maior variância de preparo);
     - **Penalidade logística regional** (ex.: regiões mais congestionadas em horários de pico);
     - **Ruído aleatório irreducível** (simulando imprevistos não observados nas features, garantindo que nenhum modelo alcance 100% de acurácia artificial).
4. **Balanceamento de Classes:**
   - A calibração padrão produz uma taxa global de atraso (`late_rate`) plausível (tipicamente entre **20% e 30%**), permitindo avaliar métricas sensíveis a desbalanceamento moderado (Precision, Recall, F1-score e PR-AUC).

---

## 7. Dicionário das Entidades de Metadados (SQLite)

Além do CSV de pedidos, o banco SQLite local armazena seis tabelas de metadados e rastreabilidade:

### 7.1. Tabela `datasets` (`Dataset`)
| Campo | Tipo SQLite | Descrição |
|---|---|---|
| `id` | `TEXT PRIMARY KEY` | Identificador único do dataset (ex.: `"ds_001"`). |
| `name` | `TEXT NOT NULL` | Nome amigável ou nome do arquivo de origem. |
| `source_type` | `TEXT NOT NULL` | Origem: `"synthetic"` ou `"imported"`. |
| `file_path` | `TEXT NOT NULL` | Caminho relativo para o CSV armazenado localmente. |
| `file_hash` | `TEXT NOT NULL` | Hash SHA-256 hexadecimal do arquivo CSV. |
| `record_count` | `INTEGER NOT NULL` | Total de linhas no arquivo. |
| `validation_status` | `TEXT NOT NULL` | Status: `"CREATED"`, `"VALIDATED"`, `"INVALID"`. |
| `quality_report_json` | `TEXT` | Relatório JSON detalhado da última validação. |
| `created_at` | `TEXT NOT NULL` | Timestamp ISO-8601 UTC de criação. |

### 7.2. Tabela `experiment_runs` (`ExperimentRun`)
| Campo | Tipo SQLite | Descrição |
|---|---|---|
| `id` | `TEXT PRIMARY KEY` | Identificador único do experimento (ex.: `"exp_001"`). |
| `dataset_id` | `TEXT NOT NULL` | FK referenciando `datasets(id)`. |
| `experiment_type` | `TEXT NOT NULL` | Tipo do experimento (ex.: `"delay_classification"`). |
| `model_type` | `TEXT NOT NULL` | Algoritmo avaliado (`"baseline_majority"`, `"logistic_regression"`, `"random_forest"`). |
| `config_json` | `TEXT NOT NULL` | Hiperparâmetros, proporções de split e `late_tolerance_minutes`. |
| `metrics_json` | `TEXT NOT NULL` | Métricas de treino, validação e teste (Precision, Recall, F1, PR-AUC, Confusion Matrix). |
| `random_seed` | `INTEGER NOT NULL` | Semente aleatória utilizada no split e no estimador. |
| `status` | `TEXT NOT NULL` | Status: `"RUNNING"`, `"COMPLETED"`, `"FAILED"`. |
| `created_at` | `TEXT NOT NULL` | Timestamp ISO-8601 UTC. |

### 7.3. Tabela `model_artifacts` (`ModelArtifact`)
| Campo | Tipo SQLite | Descrição |
|---|---|---|
| `id` | `TEXT PRIMARY KEY` | Identificador lógico do artefato (ex.: `"delay_classifier_v1"`). |
| `experiment_id` | `TEXT NOT NULL` | FK referenciando `experiment_runs(id)`. |
| `name` | `TEXT NOT NULL` | Nome descritivo do modelo. |
| `version` | `TEXT NOT NULL` | Versão semântica (ex.: `"1.0.0"`). |
| `artifact_path` | `TEXT NOT NULL` | Caminho relativo do arquivo serializado em `artifacts/models/`. |
| `feature_schema_json` | `TEXT NOT NULL` | Esquema JSON das features esperadas, tipos e categorias válidas. |
| `status` | `TEXT NOT NULL` | Status: `"ACTIVE"`, `"ARCHIVED"`. |
| `created_at` | `TEXT NOT NULL` | Timestamp ISO-8601 UTC. |

### 7.4. Tabela `agent_runs` (`AgentRun`)
| Campo | Tipo SQLite | Descrição |
|---|---|---|
| `id` | `TEXT PRIMARY KEY` | Identificador único da execução do agente (ex.: `"run_001"`). |
| `dataset_id` | `TEXT NOT NULL` | FK referenciando `datasets(id)`. |
| `question` | `TEXT NOT NULL` | Pergunta original enviada pelo usuário. |
| `answer_json` | `TEXT` | Payload completo validado da resposta (`answer`, `evidence`, `recommendations`, `limitations`). |
| `model_name` | `TEXT NOT NULL` | Identificador do modelo local no runtime (ex.: `"qwen2.5:3b"`). |
| `status` | `TEXT NOT NULL` | Status: `"RUNNING"`, `"COMPLETED"`, `"FAILED"`. |
| `duration_ms` | `INTEGER` | Tempo total de execução em milissegundos. |
| `created_at` | `TEXT NOT NULL` | Timestamp ISO-8601 UTC. |

### 7.5. Tabela `tool_calls` (`ToolCall`)
| Campo | Tipo SQLite | Descrição |
|---|---|---|
| `id` | `TEXT PRIMARY KEY` | Identificador único da chamada / referência de evidência (ex.: `"evidence_001"`). |
| `agent_run_id` | `TEXT NOT NULL` | FK referenciando `agent_runs(id)`. |
| `tool_name` | `TEXT NOT NULL` | Nome da ferramenta (`get_overview`, `compare_groups`, `predict_delay_risk`, `get_model_evaluation`). |
| `arguments_json` | `TEXT NOT NULL` | Argumentos validados enviados para a ferramenta. |
| `result_json` | `TEXT` | Resultado determinístico retornado pela ferramenta. |
| `status` | `TEXT NOT NULL` | Status: `"SUCCESS"`, `"ERROR"`. |
| `duration_ms` | `INTEGER` | Latência de execução da ferramenta em milissegundos. |

### 7.6. Tabela `evaluation_runs` (`EvaluationRun`)
| Campo | Tipo SQLite | Descrição |
|---|---|---|
| `id` | `TEXT PRIMARY KEY` | Identificador único da rodada de avaliação (ex.: `"eval_001"`). |
| `evaluation_type` | `TEXT NOT NULL` | Tipo: `"ml_benchmark"` ou `"agent_benchmark"`. |
| `dataset_id` | `TEXT NOT NULL` | FK referenciando `datasets(id)`. |
| `model_reference` | `TEXT NOT NULL` | Identificador do modelo de ML ou LLM avaliado. |
| `config_json` | `TEXT NOT NULL` | Configuração da suíte de benchmark utilizada. |
| `results_json` | `TEXT NOT NULL` | Resultados detalhados por pergunta/categoria e métricas agregadas. |
| `status` | `TEXT NOT NULL` | Status: `"RUNNING"`, `"COMPLETED"`, `"FAILED"`. |
| `created_at` | `TEXT NOT NULL` | Timestamp ISO-8601 UTC. |

