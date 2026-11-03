# AI Ops Copilot — Especificação de Requisitos e Rastreabilidade (Fase 0)

> **Status:** Fase 0 — Especificação Consolidada do MVP  
> **Versão:** 1.0.0  
> **Domínio:** Análise Operacional de Delivery (Dados Sintéticos)  
> **Execução:** 100% Local (Sem envio de dados a provedores externos)

---

## 1. Visão do Produto e Problema de Negócio

O **AI Ops Copilot** é uma aplicação local de análise operacional e apoio à decisão voltada para operações fictícias de delivery. O sistema combina **engenharia de dados determinística**, **modelos preditivos de Machine Learning** (classificação de risco de atraso) e um **agente de IA baseado em LLM local** integrado a ferramentas analíticas estritamente controladas.

### 1.1. Problema de Negócio
Operações de delivery geram continuamente registros sobre pedidos, regiões, categorias de estabelecimentos, horários de pico, tempos estimados e tempos reais de preparo e entrega. Analisar manualmente esses dados para investigar gargalos operacionais exige consultas repetitivas, cálculos agregados e interpretação contextual.

O **AI Ops Copilot** resolve esse problema permitindo que um **Analista de Operações**:
1. Gere ou importe datasets operacionais de pedidos e valide sua qualidade estrutural.
2. Consulte indicadores operacionais determinísticos (gerais e segmentados por faixa horária, região ou categoria).
3. Treine, avalie e utilize modelos de Machine Learning para estimar a probabilidade de atraso de novos pedidos no momento de sua criação (sem *data leakage*).
4. Interaja em linguagem natural com um agente de IA local que aciona ferramentas analíticas determinísticas, retornando respostas fundamentadas em evidências rastreáveis e explicitando hipóteses, recomendações e limitações.
5. Avalie cientificamente tanto os modelos de ML quanto o comportamento do agente por meio de benchmarks reproduzíveis.

### 1.2. Escopo do MVP

| Incluído no MVP | Fora do Escopo do MVP |
|---|---|
| Execução 100% local (aplicação, banco e LLM) | Deploy em nuvem, Kubernetes ou arquitetura distribuída |
| Um dataset ativo por contexto de análise | Gestão multiempresa (*multi-tenant*) ou RBAC complexo |
| Dados sintéticos gerados por script reproduzível e importação CSV | Integração com APIs reais (ex.: iFood) ou dados pessoais reais |
| Classificação supervisionada binária de risco de atraso (`is_late`) | Regressão de tempo exato de entrega no núcleo inicial |
| Agente único com 4 ferramentas analíticas determinísticas | Sistemas multiagentes autônomos ou execução de ações externas |
| Consultas analíticas parametrizadas e validadas via Pydantic | Execução de SQL arbitrário ou `eval`/`exec` gerado pelo LLM |
| Benchmark reproduzível de ML e do Agente (30 perguntas) | *Fine-tuning* de modelos de linguagem (LLMs) |
| Backend em FastAPI e interface em Streamlit | Frontend SPA complexo em React |
| Persistência local híbrida (SQLite + sistema de arquivos) | Bancos distribuídos ou *vector stores* dedicados no MVP |
| Logs técnicos estruturados e rastreabilidade local | Plataformas externas pagas de observabilidade |

---

## 2. Regras de Negócio e Restrições de Domínio

As regras de negócio (RNs) abaixo governam o comportamento de todos os módulos do sistema:

| ID | Regra | Descrição e Implicação Técnica |
|---|---|---|
| **RN-01** | **Definição Formal de Atraso** | Um pedido é considerado atrasado (`is_late = True`) exclusivamente quando o tempo total real excede o tempo total prometido somado a uma tolerância operacional configurável (padrão: $5\text{ min}$): $T_{\text{real}} > T_{\text{prometido}} + \text{tolerância}$. |
| **RN-02** | **Prevenção Estrita de *Data Leakage*** | A inferência e o treinamento do modelo de risco de atraso devem utilizar apenas variáveis conhecidas no instante de criação do pedido (`created_at`). Os campos `actual_prep_minutes` e `actual_delivery_minutes` existem apenas para apuração histórica e cálculo do rótulo `is_late`, sendo proibido seu uso como *features* preditivas. |
| **RN-03** | **Validação Explícita de Dataset** | Nenhum arquivo CSV importado é considerado automaticamente apto para análise ou treinamento. A validação estrutural e de domínio é uma etapa obrigatória que gera um relatório formal de qualidade (`validation_status`). |
| **RN-04** | **Cálculo Determinístico Fora do LLM** | O LLM nunca calcula métricas, agregações ou probabilidades por conta própria. Todo valor numérico reportado pelo agente deve originar-se da execução de uma ferramenta determinística do backend. |
| **RN-05** | **Sem Evidência, Sem Afirmação** | Respostas quantitativas do agente devem referenciar o resultado real produzido pela ferramenta (`result_reference`). Se os dados forem insuficientes ou a ferramenta falhar, o agente deve declarar a limitação sem inventar valores. |
| **RN-06** | **Separação Epistemológica na Resposta** | O agente deve distinguir explicitamente entre: **Fato** (métrica calculada pelo backend), **Interpretação** (leitura direta do dado), **Hipótese** (possível causa não comprovada), **Recomendação** (investigação sugerida) e **Limitação** (o que os dados não permitem concluir, ex.: natureza sintética dos dados). Correlação não deve ser apresentada como causalidade. |
| **RN-07** | **Isolamento entre Treinamento e Inferência** | A aplicação não retreina modelos durante perguntas do usuário ou chamadas de inferência. A inferência consome exclusivamente artefatos previamente treinados, avaliados e versionados (`ModelArtifact`). |
| **RN-08** | **Imutabilidade de Experimentos e Artefatos** | Execuções de experimentos (`ExperimentRun`), artefatos de modelos (`ModelArtifact`) e rodadas de avaliação (`EvaluationRun`) nunca sobrescrevem registros ou arquivos anteriores, garantindo comparabilidade histórica. |
| **RN-09** | **Privacidade e Execução Local** | Nenhum prompt, pergunta ou registro de dataset pode ser transmitido a provedores externos de IA. O adaptador de LLM deve operar exclusivamente contra um runtime local (ex.: `localhost` via Ollama). |
| **RN-10** | **Degradação Graciosa (*Graceful Degradation*)** | A indisponibilidade do runtime local de LLM não pode impedir o funcionamento da importação/geração de dados, das consultas analíticas determinísticas nem da inferência ou treinamento de modelos de ML. |

---

## 3. Requisitos Funcionais (RF)

Os requisitos funcionais estão classificados em:
- **P0 (Essencial):** Indispensável para a entrega e aprovação do MVP.
- **P1 (Evolução):** Melhoria importante planejada para incrementos subsequentes após estabilização do núcleo P0.

### 3.1. Gestão, Geração e Qualidade de Dados (`data`)

| ID | Prioridade | Requisito | Descrição Detalhada |
|---|---|---|---|
| **RF-01** | **P0** | Gerar dataset sintético de pedidos | Permitir gerar arquivos CSV sintéticos de pedidos com parâmetros configuráveis (quantidade de registros, `random_seed`, `late_tolerance_minutes`, distribuições e ruído), garantindo reprodutibilidade exata para a mesma semente. |
| **RF-02** | **P0** | Importar arquivos CSV | Permitir a importação de arquivos CSV externos contendo registros operacionais de pedidos, calculando hash SHA-256 e registrando metadados no banco local. |
| **RF-03** | **P0** | Validar estrutura e consistência dos dados | Verificar presença de colunas obrigatórias, tipos de dados, unicidade de `order_id`, valores ausentes, intervalos numéricos válidos (ex.: tempos e valores positivos) e consistência temporal/lógica. |
| **RF-04** | **P0** | Gerar relatório de qualidade do dataset | Produzir e persistir um relatório estruturado contendo `total_records`, `valid_records`, `invalid_records`, lista detalhada de `warnings`/erros por coluna e status final de validação. |
| **RF-05** | **P0** | Calcular indicadores operacionais | Calcular métricas globais (volume total, taxa de atraso geral, tempos médios estimados vs. reais, ticket médio) e métricas agrupadas por dimensão (`hour_of_day`, `day_of_week`, `region`, `order_category`). |

### 3.2. Machine Learning (`ml`)

| ID | Prioridade | Requisito | Descrição Detalhada |
|---|---|---|---|
| **RF-06** | **P0** | Treinar baseline e modelo candidato | Executar pipeline de treinamento supervisionado para classificação binária (`is_late`), incluindo pelo menos uma *baseline* (classe majoritária ou heurística simples) e um modelo candidato (Regressão Logística / Random Forest), ajustando pré-processadores exclusivamente no conjunto de treino. |
| **RF-07** | **P0** | Avaliar modelos em conjunto reservado | Avaliar modelos treinados em partição de teste estritamente separada, calculando Precision, Recall, F1-score, PR-AUC, matriz de confusão e latência de inferência. |
| **RF-08** | **P0** | Salvar e carregar artefatos de modelos | Serializar o pipeline completo (pré-processamento + estimador) com versão semântica, esquema de atributos de entrada (`feature_schema`) e vínculo ao experimento de origem. |
| **RF-09** | **P0** | Disponibilizar previsão de risco para novos pedidos | Receber os atributos de criação de um pedido, validar contra o `feature_schema` do modelo selecionado e retornar `is_late` (booleano), `risk_probability` (float) e `risk_level` (`low`, `medium`, `high`). |
| **RF-17** | **P1** | Comparar múltiplos modelos de ML | Expandir os experimentos para comparar sistematicamente múltiplos algoritmos e hiperparâmetros sob o mesmo protocolo experimental. |

### 3.3. Agente de IA e LLM Local (`agent`)

| ID | Prioridade | Requisito | Descrição Detalhada |
|---|---|---|---|
| **RF-10** | **P0** | Integrar LLM executado localmente | Comunicar-se com runtime local de LLM (inicialmente Ollama) por meio de uma interface abstrata (`LLMProvider`), sem dependência de provedores externos. |
| **RF-11** | **P0** | Disponibilizar ferramentas analíticas ao agente | Expor ao agente um catálogo fechado de 4 ferramentas determinísticas: `get_overview`, `compare_groups`, `predict_delay_risk` e `get_model_evaluation`. |
| **RF-12** | **P0** | Validar argumentos e saídas do agente | Validar com Pydantic todos os argumentos solicitados pelo LLM antes de executar qualquer ferramenta (com limite máximo de chamadas por interação, padrão: 3) e validar o contrato JSON da resposta final. |
| **RF-13** | **P0** | Gerar respostas com evidências e limitações | Estruturar a resposta final contendo texto explicativo (`answer`), lista de evidências verificáveis (`evidence` com `result_reference`), recomendações rotuladas (`recommendations`) e limitações explícitas (`limitations`). |
| **RF-18** | **P1** | Comparar diferentes modelos locais de LLM | Avaliar diferentes modelos locais quantizados (ex.: 3B a 4B parâmetros) comparando precisão na escolha de ferramentas, aderência ao schema JSON e latência no hardware-alvo. |

### 3.4. Avaliação Científica, Observabilidade e Interface (`evaluation`, `observability`, `api`, `app`)

| ID | Prioridade | Requisito | Descrição Detalhada |
|---|---|---|---|
| **RF-14** | **P0** | Executar benchmark do agente | Executar bateria automatizada de 30 perguntas de referência distribuídas em 5 categorias, comparando a saída do agente com gabaritos calculados deterministicamente pelo código. |
| **RF-15** | **P0** | Registrar execuções e erros relevantes | Persistir logs técnicos estruturados e registros de execução (`AgentRun`, `ToolCall`, `ExperimentRun`, `EvaluationRun`) com duração em milissegundos, versões, status e códigos de erro padronizados. |
| **RF-16** | **P0** | Disponibilizar interface web de análise | Prover dashboard em Streamlit integrado à API FastAPI cobrindo geração/importação de dataset, relatório de qualidade, painel de métricas, treinamento/inferência de ML, chat com o agente e visualização de benchmarks. |
| **RF-19** | **P1** | Exportar relatórios experimentais | Permitir a exportação de relatórios consolidados de experimentos de ML e benchmarks do agente nos formatos JSON e CSV. |
| **RF-20** | **P1** | Consultar histórico de análises e consultas | Disponibilizar visualização detalhada do histórico de execuções anteriores do agente e experimentos na interface. |

---

## 4. Requisitos Não Funcionais (RNF)

| ID | Categoria | Requisito | Critério Objetivo de Verificação |
|---|---|---|---|
| **RNF-01** | **Modularidade** | Separação estrita entre camadas (`domain`, `application`, `data`, `ml`, `agent`, `evaluation`, `persistence`, `api`, `app`). | Módulos de domínio e dados não importam `streamlit` nem dependem de rotas HTTP; a UI consome apenas a camada de API/contratos. |
| **RNF-02** | **Reprodutibilidade** | Execuções de geração de dados e treinamento de ML devem ser deterministicamente reproduzíveis. | Executar o gerador ou o pipeline de treino duas vezes com a mesma configuração e `random_seed` produz exatamente o mesmo hash de dataset e as mesmas métricas. |
| **RNF-03** | **Privacidade** | Dados, prompts e respostas permanecem estritamente locais. | Nenhuma requisição de rede externa é disparada durante ingestão, treino, inferência ou chat (verificável via inspeção de configuração e testes com rede isolada). |
| **RNF-04** | **Segurança** | Proteção contra execução arbitrária e *prompt injection*. | Ausência total de `eval`, `exec` ou SQL dinâmico não parametrizado; ferramentas restritas a um registro explícito (*allowlist*) com validação Pydantic e timeouts configurados. |
| **RNF-05** | **Robustez** | Isolamento de falhas de componentes externos ou arquivos inválidos. | Queda ou timeout do Ollama retorna erro estruturado `LLM_UNAVAILABLE` sem derrubar a API FastAPI nem afetar endpoints de `/datasets`, `/analytics` ou `/ml`. |
| **RNF-06** | **Testabilidade** | Cobertura automatizada de regras de negócio, contratos e fluxos críticos. | Suíte `pytest` executável localmente com testes unitários, de integração (usando *mocks* para LLM na suíte rápida) e ponta a ponta (E2E). |
| **RNF-07** | **Rastreabilidade** | Identificação completa de linhagem para previsões, experimentos e respostas. | Todo `ModelArtifact` aponta para um `ExperimentRun` e `Dataset` (com hash); todo `AgentRun` registra `ToolCall`s, argumentos, referências de evidência e modelo utilizado. |
| **RNF-08** | **Portabilidade** | Compatibilidade documentada para ambientes Windows e Linux. | Caminhos manipulados via `pathlib.Path`, variáveis de ambiente via `.env` e instruções de inicialização testadas para ambos os sistemas operacionais. |
| **RNF-09** | **Desempenho** | Medição empírica de latência por etapa no hardware de referência. | Todas as execuções de ferramentas, inferência de ML e chamadas ao LLM registram `duration_ms` para análise de gargalos em CPU/GPU. |
| **RNF-10** | **Manutenibilidade** | Desacoplamento de provedores e padronização de erros. | Troca de modelo local ou runtime requer apenas ajuste de configuração ou adaptador, mantendo inalterados os casos de uso e contratos da API. |

---

## 5. Critérios de Aceite Detalhados (P0)

### 5.1. Critérios de Dados (`CA-DATA`)
- **CA-DATA-01 (Reprodutibilidade Sintética):** Dado `number_of_orders = 10000`, `random_seed = 42` e `late_tolerance_minutes = 5`, quando o gerador for executado duas vezes, então os arquivos CSV gerados devem possuir conteúdo e hash SHA-256 idênticos.
- **CA-DATA-02 (Aplicação da Regra de Atraso):** Em 100% dos registros gerados ou validados, `is_late` deve ser `True` se e somente se `(actual_prep_minutes + actual_delivery_minutes) > (estimated_prep_minutes + estimated_delivery_minutes + late_tolerance_minutes)`.
- **CA-DATA-03 (Rejeição de Arquivo Inválido):** Dado um arquivo com formato inválido ou sem colunas obrigatórias, quando submetido a `POST /api/v1/datasets` ou `/validate`, então o sistema deve rejeitar a operação com código padronizado (`INVALID_FILE_FORMAT` ou relatório de falha) sem lançar exceção não tratada (HTTP 500).
- **CA-DATA-04 (Fidelidade das Métricas):** Os valores retornados por `/api/v1/analytics/overview` e `/api/v1/analytics/groups` devem coincidir exatamente (dentro de tolerância de ponto flutuante $10^{-6}$) com cálculos diretos de referência em `pandas` sobre os registros válidos do dataset.

### 5.2. Critérios de Machine Learning (`CA-ML`)
- **CA-ML-01 (Ausência de *Data Leakage*):** O pipeline de pré-processamento e o modelo de classificação devem rejeitar explicitamente ou ignorar qualquer coluna pós-evento (`actual_prep_minutes`, `actual_delivery_minutes`, `is_late`) nas *features* de entrada ($X$), e os transformadores (`StandardScaler`, `OneHotEncoder`) devem ser ajustados (`fit`) exclusivamente sobre a partição de treino.
- **CA-ML-02 (Comparação com Baseline):** Todo experimento (`POST /api/v1/ml/experiments`) deve registrar no mesmo protocolo de validação as métricas da *baseline* e do modelo candidato (Precision, Recall, F1-score, PR-AUC e Matriz de Confusão), distinguindo claramente métricas de treino, validação e teste.
- **CA-ML-03 (Contrato de Inferência):** Dado um artefato de modelo carregado, quando `POST /api/v1/ml/predict` receber um payload válido, deve retornar `is_late`, `risk_probability` $\in [0.0, 1.0]$, `risk_level` $\in \{\text{"low"}, \text{"medium"}, \text{"high"}\}$ e `model_version`. Se faltar qualquer atributo obrigatório, deve retornar erro `INVALID_FEATURES` (HTTP 422/400).

### 5.3. Critérios do Agente e LLM Local (`CA-AGENT`)
- **CA-AGENT-01 (Uso Obrigatório de Ferramentas Permitidas):** Diante de uma pergunta sobre métricas ou previsões, o agente deve selecionar apenas ferramentas contidas na *allowlist* (`get_overview`, `compare_groups`, `predict_delay_risk`, `get_model_evaluation`) com argumentos validados por Pydantic.
- **CA-AGENT-02 (Rastreabilidade de Evidências):** Toda resposta concluída (`status = "completed"`) para perguntas quantitativas deve conter ao menos um item em `evidence` cujo `result_reference` corresponda a um registro `ToolCall` persistido na mesma execução (`AgentRun`).
- **CA-AGENT-03 (Tratamento de Perguntas Sem Dados ou Inválidas):** Quando o usuário perguntar sobre variáveis inexistentes no dataset (ex.: "Qual foi o nome do entregador ou a avaliação do cliente?"), o agente não deve inventar números e deve informar explicitamente a limitação do esquema de dados.
- **CA-AGENT-04 (Resiliência a Falha do LLM):** Quando o serviço Ollama estiver desligado ou inacessível, `GET /api/v1/agent/health` deve reportar indisponibilidade e `POST /api/v1/agent/chat` deve retornar erro controlado `LLM_UNAVAILABLE`, mantendo todos os demais módulos operacionais.

### 5.4. Critérios do Benchmark Científico (`CA-BENCH`)
- **CA-BENCH-01 (Cobertura do Conjunto de Referência):** O benchmark do agente deve conter no mínimo **30 perguntas** distribuídas nas 5 categorias definidas:
  1. Indicadores gerais (6 perguntas)
  2. Comparações entre grupos (8 perguntas)
  3. Perguntas sobre previsões de risco (6 perguntas)
  4. Perguntas com dados insuficientes / fora do escopo do dataset (5 perguntas)
  5. Entradas inválidas ou ambíguas (5 perguntas)
- **CA-BENCH-02 (Métricas Desagregadas por Categoria):** O relatório do benchmark deve apurar e reportar, globalmente e por categoria: taxa de seleção correta de ferramenta, validade de argumentos, exatidão numérica frente ao gabarito determinístico, fidelidade às evidências, reconhecimento de limitações e latência média/percentil.

---

## 6. Matriz de Rastreabilidade

A tabela abaixo conecta cada Requisito Funcional (RF) aos módulos arquiteturais, endpoints REST / ferramentas do agente, critérios de aceite e itens do backlog:

| Requisito | Módulo Principal | Endpoint REST / Ferramenta do Agente | Critério de Aceite / Teste | Tarefas do Backlog |
|---|---|---|---|---|
| **RF-01** | `data.generator` | `POST /api/v1/datasets/generate` | `CA-DATA-01`, `CA-DATA-02` (`tests/unit/test_generator.py`) | B01, B03, B04 |
| **RF-02** | `data.ingestion`, `persistence` | `POST /api/v1/datasets`, `GET /api/v1/datasets` | `CA-DATA-03` (`tests/integration/test_datasets_api.py`) | B04, B16, B17 |
| **RF-03** | `data.validation` | `POST /api/v1/datasets/{id}/validate` | `CA-DATA-02`, `CA-DATA-03` (`tests/unit/test_validation.py`) | B05 |
| **RF-04** | `data.validation`, `persistence` | `GET /api/v1/datasets/{id}/quality` | `CA-DATA-03` (`tests/unit/test_validation.py`) | B05, B16, B17 |
| **RF-05** | `data.analytics` | `GET /api/v1/analytics/overview`, `/late-rate`, `/groups` | `CA-DATA-04` (`tests/unit/test_analytics.py`) | B06, B17 |
| **RF-06** | `ml.preprocessing`, `ml.training` | `POST /api/v1/ml/experiments` | `CA-ML-01`, `CA-ML-02` (`tests/unit/test_ml_training.py`) | B07, B08 |
| **RF-07** | `ml.evaluation` | `GET /api/v1/ml/experiments/{id}/metrics`, `POST /api/v1/evaluation/ml` | `CA-ML-02` (`tests/unit/test_ml_evaluation.py`) | B09, B17 |
| **RF-08** | `ml.artifacts`, `persistence` | `GET /api/v1/ml/models` | `CA-ML-03` (`tests/integration/test_ml_artifacts.py`) | B09, B16 |
| **RF-09** | `ml.inference` | `POST /api/v1/ml/predict` | `CA-ML-03` (`tests/unit/test_ml_inference.py`) | B10, B17 |
| **RF-10** | `agent.runtime` | `GET /api/v1/agent/health` | `CA-AGENT-04` (`tests/unit/test_llm_runtime.py`) | B11 |
| **RF-11** | `agent.tools` | Tools: `get_overview`, `compare_groups`, `predict_delay_risk`, `get_model_evaluation` | `CA-AGENT-01` (`tests/unit/test_agent_tools.py`) | B12 |
| **RF-12** | `agent.validation`, `agent.orchestration` | `POST /api/v1/agent/chat` | `CA-AGENT-01`, `CA-AGENT-03` (`tests/unit/test_agent_validation.py`) | B13, B14 |
| **RF-13** | `agent.orchestration` | `POST /api/v1/agent/chat`, `GET /api/v1/agent/runs/{id}` | `CA-AGENT-02`, `CA-AGENT-03` (`tests/integration/test_agent_flow.py`) | B13, B14, B17 |
| **RF-14** | `evaluation.benchmarks` | `POST /api/v1/evaluation/agent`, `GET /api/v1/evaluation/runs/{id}/report` | `CA-BENCH-01`, `CA-BENCH-02` (`tests/integration/test_benchmark.py`) | B15, B17 |
| **RF-15** | `observability`, `persistence` | `GET /api/v1/agent/runs/{id}`, `GET /api/v1/evaluation/runs/{id}` | `RNF-07`, `RNF-09` (`tests/integration/test_persistence.py`) | B16, B19 |
| **RF-16** | `app.dashboard` | Interface Streamlit consumindo `/api/v1/*` | Teste E2E de fluxo completo (`tests/end_to_end/test_mvp_flow.py`) | B18, B19, B20 |
| **RF-17** | `ml.training`, `ml.evaluation` | `POST /api/v1/ml/experiments` | Comparação multi-modelo documentada | B21 |
| **RF-18** | `agent.runtime`, `evaluation.benchmarks` | `POST /api/v1/evaluation/agent` | Comparação multi-LLM documentada | B22 |
| **RF-19** | `evaluation.reports` | `GET /api/v1/evaluation/runs/{id}/report` | Exportação JSON/CSV validada | B23 |
| **RF-20** | `app.dashboard`, `api` | `GET /api/v1/agent/runs`, `GET /api/v1/ml/experiments` | Visualização de histórico na UI | B24 |

---

## 7. Backlog Priorizado e Roadmap de Execução

### 7.1. Tarefas P0 — Núcleo do MVP

| ID | Fase | Tarefa | Entrega Esperada | Dependências |
|---|---|---|---|---|
| **B01** | Fase 0/1 | Criar estrutura inicial do projeto | Árvore de diretórios e pacotes Python organizados | — |
| **B02** | Fase 1 | Definir dependências e ambiente | `pyproject.toml`, `.env.example`, `.gitignore` e ambiente virtual reproduzível | B01 |
| **B03** | Fase 0 | Documentar requisitos, arquitetura e dicionário de dados | `docs/requirements.md`, `docs/architecture.md`, `docs/data_dictionary.md` | — |
| **B04** | Fase 1 | Implementar gerador de dataset sintético | Script e módulo gerador de CSV reproduzível com seed fixa | B01, B02, B03 |
| **B05** | Fase 1 | Implementar ingestão e validação do dataset | Validador estrutural/domínio e gerador de relatório de qualidade | B04 |
| **B06** | Fase 1 | Implementar métricas operacionais | Consultas analíticas determinísticas (overview, late-rate, groups) | B05 |
| **B07** | Fase 2 | Criar baseline de classificação | Classificador baseline avaliado em partição de treino/validação/teste | B05 |
| **B08** | Fase 2 | Treinar modelo candidato de ML | Pipeline `scikit-learn` (pré-processamento + Regressão Logística / Random Forest) | B07 |
| **B09** | Fase 2 | Avaliar modelos e salvar artefatos versionados | Cálculo de métricas completas e serialização imutável de `ModelArtifact` | B08 |
| **B10** | Fase 2 | Implementar serviço de inferência | Carregamento de artefato, validação de *features* e predição de risco | B09 |
| **B11** | Fase 3 | Implementar adaptador local de LLM | Cliente HTTP local para Ollama implementando `LLMProvider` + *healthcheck* | B02 |
| **B12** | Fase 3 | Implementar ferramentas determinísticas do agente | Registro e adaptadores para as 4 ferramentas permitidas | B06, B10 |
| **B13** | Fase 3 | Implementar orquestração do agente | Loop controlado de *tool calling* (máx. 3 chamadas) e montagem de contexto | B11, B12 |
| **B14** | Fase 3 | Implementar validação das respostas do agente | Validação Pydantic de saída e verificação de vínculo com `result_reference` | B13 |
| **B15** | Fase 4 | Implementar benchmark científico do agente | Dataset de 30 perguntas, avaliador automático contra gabarito e relatório | B06, B10, B14 |
| **B16** | Fase 4 | Implementar camada de persistência SQLite | Repositórios para `Dataset`, `ExperimentRun`, `ModelArtifact`, `AgentRun`, `ToolCall`, `EvaluationRun` | B02 |
| **B17** | Fase 5 | Criar endpoints REST FastAPI | Rotas `/api/v1/*`, injeção de dependências e handler global de erros | B06, B10, B14, B15, B16 |
| **B18** | Fase 5 | Criar interface Streamlit | `app/dashboard.py` cobrindo todos os fluxos de usuário via API | B17 |
| **B19** | Fase 5 | Implementar suíte de testes de integração e E2E | Testes automatizados cobrindo o fluxo ponta a ponta do MVP | B17, B18 |
| **B20** | Fase 5 | Documentar execução, experimentos e arquitetura final | `README.md`, `docs/ml_experiments.md` e `docs/agent_evaluation.md` atualizados | B19 |

### 7.2. Tarefas P1 — Evoluções Pós-MVP

| ID | Tarefa | Entrega Esperada |
|---|---|---|
| **B21** | Comparar modelos de ML adicionais | Estudo comparativo expandido com calibração de probabilidades e análise por fatias (*slices*) |
| **B22** | Comparar modelos locais de LLM | Benchmark comparativo entre diferentes LLMs locais quantizados (3B–4B) em CPU/GPU |
| **B23** | Exportar relatórios experimentais | Exportação de resultados de experimentos e benchmarks em JSON e CSV |
| **B24** | Implementar histórico visual na UI | Tela dedicada no Streamlit para navegação em execuções e experimentos passados |
| **B25** | Expandir observabilidade técnica | Painel de métricas de latência, taxa de erro por ferramenta e consumo de recursos |
| **B26** | Publicar dataset sintético | Empacotamento e documentação do dataset sintético no Hugging Face Datasets |
| **B27** | Escrever relatório/artigo técnico | Documento científico analisando a hipótese central do benchmark |
| **B28** | Gravar demonstração em vídeo | Vídeo curto demonstrando o produto ponta a ponta para portfólio |

---

## 8. Definição de Pronto (Definition of Done — DoD) do MVP

O MVP do **AI Ops Copilot** será considerado concluído quando qualquer avaliador técnico puder, em ambiente local e seguindo exclusivamente as instruções do `README.md`:

1. Inicializar o ambiente e executar a suíte de testes automatizados (`pytest`) com aprovação integral.
2. Gerar um dataset sintético reproduzível ou importar um CSV e visualizar seu relatório de validação e qualidade.
3. Consultar indicadores operacionais gerais e agrupados com exatidão determinística.
4. Executar o treinamento e avaliação comparativa entre a *baseline* e o classificador de risco de atraso, gerando artefatos versionados sem vazamento de dados.
5. Realizar previsões de risco de atraso para novos pedidos utilizando um modelo carregado.
6. Fazer perguntas em linguagem natural ao agente com LLM local e inspecionar quais ferramentas foram acionadas e quais evidências numéricas sustentam a resposta.
7. Executar o benchmark científico (ML e Agente) e consultar os relatórios desagregados por categoria.
8. Confirmar que nenhuma chamada de inferência ou dados operacionais são enviados para fora da máquina local.

