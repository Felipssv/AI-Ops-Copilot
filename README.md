# AI Ops Copilot

Aplicação local de **análise operacional, previsão de risco de atraso com Machine Learning e agente analítico com LLM local** para operações fictícias de delivery.

> **Status Atual:** Fase 0 Concluída (Preparação Técnica, Requisitos, Arquitetura e Dicionário de Dados)  
> **Próxima Fase:** Fase 1 — Fundação do Repositório e Pipeline de Dados Determinístico (`B01`–`B06`)

---

## Visão Geral

O **AI Ops Copilot** é um projeto de engenharia de IA construído sob o princípio de que **modelos probabilísticos (ML e LLMs) devem operar dentro de sistemas de software governados por regras determinísticas, contratos explícitos e validação empírica**.

Principais capacidades do MVP:
1. **Engenharia e Qualidade de Dados:** Geração reproduzível de datasets sintéticos de pedidos (`random_seed = 42`), importação de CSVs, validação estrutural/domínio e cálculo determinístico de indicadores operacionais.
2. **Machine Learning Supervisionado:** Pipeline de classificação binária de risco de atraso (`is_late`) estritamente livre de *data leakage* (utilizando apenas variáveis disponíveis no instante de criação do pedido), comparando *baseline* e modelos candidatos em partições isoladas.
3. **Agente de IA com LLM Local:** Integração com runtime local (Ollama) restrita a um catálogo fechado de 4 ferramentas determinísticas (`get_overview`, `compare_groups`, `predict_delay_risk`, `get_model_evaluation`), exigindo respostas fundamentadas em evidências (`result_reference`) e separação entre fatos, hipóteses e limitações.
4. **Avaliação Científica (Benchmark):** Bateria reproduzível de avaliação de ML e benchmark de 30 perguntas para o agente, comparando respostas contra gabaritos calculados por código.

---

## Documentação de Engenharia (Fase 0)

Toda a base conceitual, contratos da API, modelo de dados e regras de negócio foram consolidados antes da implementação de código:

- [**Especificação de Requisitos e Rastreabilidade (`docs/requirements.md`)**](docs/requirements.md)  
  Escopo do MVP, Regras de Negócio (`RN-01` a `RN-10`), Requisitos Funcionais (`RF-01` a `RF-20`), Requisitos Não Funcionais (`RNF-01` a `RNF-10`), Critérios de Aceite, Matriz de Rastreabilidade e Backlog Priorizado (`B01` a `B28`).
- [**Arquitetura de Software e Contratos Técnicos (`docs/architecture.md`)**](docs/architecture.md)  
  Monólito modular em Python, diagramas de camadas/sequência/ML/ER, estrutura de diretórios, arquitetura do LLM local, contratos REST (`/api/v1`), catálogo de ferramentas do agente e Registro de Decisões Arquiteturais (`ADR-01` a `ADR-07`).
- [**Dicionário de Dados e Regras de Geração (`docs/data_dictionary.md`)**](docs/data_dictionary.md)  
  Definição formal de atraso ($T_{\text{real}} > T_{\text{prometido}} + 5$), esquema completo do dataset, atributos derivados, prevenção de *data leakage*, regras de validação, especificação do gerador sintético e esquema físico SQLite.

---

## Stack Tecnológica Definida

| Componente | Tecnologia |
|---|---|
| **Arquitetura** | Monólito Modular (`src/ai_ops_copilot/`) |
| **Linguagem** | Python 3.11+ |
| **API Backend** | FastAPI + Pydantic |
| **Interface Analítica** | Streamlit (`app/dashboard.py`) |
| **Processamento e ML** | pandas, NumPy, scikit-learn |
| **Persistência Local** | SQLite (metadados/execuções) + Arquivos Locais (CSVs/artefatos) |
| **Runtime de LLM Local** | Ollama (`localhost`, sem envio de dados a provedores externos) |
| **Testes Automatizados** | pytest (unitários, integração e E2E) |

---

## Roadmap de Desenvolvimento

- [x] **Fase 0 — Definição:** Especificação de requisitos, arquitetura, contratos e dicionário de dados (`B03`).
- [ ] **Fase 1 — Dados:** Scaffolding do projeto (`B01`, `B02`), gerador sintético (`B04`), validação de qualidade (`B05`) e métricas determinísticas (`B06`).
- [ ] **Fase 2 — Machine Learning:** Baseline (`B07`), modelo candidato (`B08`), avaliação/artefatos versionados (`B09`) e serviço de inferência (`B10`).
- [ ] **Fase 3 — Agente e LLM Local:** Adaptador local (`B11`), ferramentas determinísticas (`B12`), orquestração (`B13`) e validação de evidências (`B14`).
- [ ] **Fase 4 — Qualidade e Persistência:** Benchmark de 30 perguntas (`B15`) e repositórios SQLite (`B16`).
- [ ] **Fase 5 — Produto e Entrega:** Endpoints FastAPI (`B17`), interface Streamlit (`B18`), testes E2E (`B19`) e relatórios finais (`B20`).

