# RBIA Execution Planes Architecture - Task 6.1.2
## Clear Separation of Concerns Across Five Execution Planes

**Version**: 1.0  
**Last Updated**: October 8, 2025  
**Status**: Production Architecture

---

## Executive Summary

The RBIA platform is architected across **five distinct execution planes**, each with clearly defined responsibilities, ensuring scalability, maintainability, and governance compliance. This separation of concerns enables:

- **Independent scaling** of each plane
- **Clear boundaries** for security and compliance
- **Maintainable codebase** with minimal coupling
- **Audit-ready architecture** with traceable flows
- **Multi-tenant isolation** enforced at each plane

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CONTROL PLANE                             │
│  Orchestrator | Compiler | Policy Engine | Governance Service   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                       EXECUTION PLANE                            │
│  RBA Runtime | ML Inference Layer | Feature Store | QoS Manager │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                      GOVERNANCE LAYER                            │
│  Evidence DB | Override Ledger | Explainability | Trust Scoring │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                         DATA PLANE                               │
│  Fabric Bronze/Silver/Gold | Postgres | Redis | pgvector        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                      KNOWLEDGE PLANE                             │
│  Knowledge Graph | RAG Pipelines | Trace Ingestion | Lineage    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. CONTROL PLANE

### Purpose
The Control Plane is the **command center** of RBIA, responsible for routing all requests, compiling workflows, enforcing policies, and coordinating execution across all other planes.

### Core Principle
**"Single Source of Truth"** - All RBIA requests MUST route through the Control Plane. No direct access to Execution or Data planes is permitted.

### Components

#### 1.1 Orchestrator (`dsl/hub/routing_orchestrator.py`)
**Responsibilities**:
- Route all incoming requests to appropriate workflows
- Coordinate execution across multiple services
- Manage workflow lifecycle (start, pause, resume, cancel)
- Handle fallback scenarios (RBIA → RBA → Baseline)
- Enforce SLA tiers (T0, T1, T2)

**Key Functions**:
```python
async def route_request(user_input, tenant_id, user_id, context_data)
async def execute_routing_result(routing_result)
async def handle_fallback(workflow_id, reason)
```

**Location**: `dsl/hub/routing_orchestrator.py`, `dsl/integration_orchestrator.py`

---

#### 1.2 Compiler (`dsl/compiler/`)
**Responsibilities**:
- Parse DSL workflow definitions (YAML/JSON)
- Validate workflow structure and dependencies
- Generate execution plans with hash signatures
- Inject governance hooks
- Apply industry overlays

**Key Functions**:
```python
async def compile_workflow(workflow_definition)
async def validate_workflow(workflow_def)
def generate_plan_hash(workflow_plan)
```

**Location**: `dsl/compiler/compiler.py`, `dsl/compiler/validation.py`

---

#### 1.3 Policy Engine (`dsl/governance/policy_gate.py`)
**Responsibilities**:
- Evaluate policy packs before execution
- Enforce confidence thresholds
- Check bias/fairness requirements
- Validate compliance rules
- Block or allow execution based on policies

**Key Functions**:
```python
async def evaluate_policies(policy_pack_id, workflow_id, data)
async def check_trust_gates(trust_score, thresholds)
async def enforce_compliance(framework, tenant_id)
```

**Location**: `dsl/hub/policy_gate.py`, `dsl/governance/`

---

#### 1.4 Governance Service
**Responsibilities**:
- Manage approval workflows (CAB)
- Track override requests
- Generate evidence packs
- Coordinate audit trails

**Location**: `api/governance_approvals.py`, `src/rba/governance_layer.py`

---

### Control Plane Data Flow
```
External Request → API Gateway → Orchestrator
                                     ↓
                            Intent Parsing
                                     ↓
                            Workflow Loading
                                     ↓
                            Policy Evaluation
                                     ↓
                    Execution Plan Generation
                                     ↓
                        [Route to Execution Plane]
```

### Technology Stack
- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Communication**: HTTP/REST, WebSocket (real-time)
- **Config Storage**: YAML files, Postgres
- **Caching**: Redis (routing decisions)

### Security Controls
- ✅ All requests authenticated via JWT
- ✅ Tenant context validated
- ✅ Rate limiting per tenant tier
- ✅ Request/response logging
- ✅ Circuit breaker for downstream failures

---

## 2. EXECUTION PLANE

### Purpose
The Execution Plane is the **runtime environment** where workflows execute, ML models inference occurs, and business logic processes data.

### Core Principle
**"Fail-Safe Execution"** - Every execution must have a fallback path. No execution should fail without attempting RBA fallback.

### Components

#### 2.1 RBA Runtime (`dsl/execution/`)
**Responsibilities**:
- Execute deterministic rule-based workflows
- Process business logic operators
- Handle data transformations
- Provide fallback for RBIA failures

**Operators**:
- Filter, Transform, Aggregate
- Join, Group, Sort
- Calculate, Validate
- Custom business rules

**Location**: `dsl/execution/`, `dsl/operators/rba/`

---

#### 2.2 ML Inference Layer (`dsl/operators/ml_*.py`)
**Responsibilities**:
- Execute ML model predictions
- Score entities based on ML models
- Classify data using ML models
- Generate explainability (SHAP/LIME)

**ML Operators**:
```python
MLPredictOperator  # dsl/operators/ml_predict.py
MLScoreOperator    # dsl/operators/ml_score.py
MLClassifyOperator # dsl/operators/ml_classify.py
MLExplainOperator  # dsl/operators/ml_explain.py
```

**Location**: `dsl/operators/ml_*.py`

---

#### 2.3 Feature Store (`dsl/intelligence/feature_store.py`)
**Responsibilities**:
- Serve features for online inference (Redis)
- Store features for offline training (Fabric/Postgres)
- Manage feature versioning
- Track feature lineage

**Storage Layers**:
- **Online**: Redis cache (low latency)
- **Offline**: Postgres + Azure Fabric (historical)

**Location**: `dsl/intelligence/feature_store.py`, `dsl/database/feature_store.sql`

---

#### 2.4 SLA/QoS Manager (`dsl/orchestration/sla_tier_manager.py`)
**Responsibilities**:
- Enforce SLA tiers (T0: 500ms, T1: 1000ms, T2: 2000ms)
- Manage execution priority queues
- Throttle low-priority requests
- Monitor execution performance

**Location**: `dsl/orchestration/sla_tier_manager.py`

---

#### 2.5 Drift & Bias Monitor (`dsl/operators/drift_bias_monitor.py`)
**Responsibilities**:
- Detect data drift (PSI, KS tests)
- Monitor prediction drift
- Check fairness metrics (demographic parity, equalized odds)
- Trigger auto-fallback on threshold breach

**Location**: `dsl/operators/drift_bias_monitor.py`, `rbia-drift-monitor/`

---

### Execution Plane Data Flow
```
Execution Plan from Control → Runtime Executor
                                     ↓
                        Feature Retrieval (Online)
                                     ↓
                         ML Model Inference
                                     ↓
                        Drift/Bias Checking
                                     ↓
                        Confidence Validation
                                     ↓
                     [Success] → Results to Governance
                     [Fail] → Fallback to RBA
```

### Technology Stack
- **Language**: Python 3.11+
- **ML Framework**: scikit-learn, PyTorch (future)
- **Inference**: Local (scalable to remote)
- **Caching**: Redis (features, predictions)
- **Monitoring**: Prometheus metrics

### Performance Guarantees
| Tier | Max Latency | Availability | Concurrency |
|------|-------------|--------------|-------------|
| T0   | 500ms       | 99.9%        | 1000        |
| T1   | 1000ms      | 99%          | 500         |
| T2   | 2000ms      | 95%          | 100         |

---

## 3. GOVERNANCE LAYER

### Purpose
The Governance Layer ensures **auditability, transparency, and compliance** for every RBIA execution through immutable evidence capture.

### Core Principle
**"Evidence-First"** - Every action produces auditable evidence. No execution without evidence capture.

### Components

#### 3.1 Evidence Database (`dsl/database/multi_tenant_schema.sql`)
**Responsibilities**:
- Store execution traces
- Maintain audit trails
- Capture input/output snapshots
- Track performance metrics

**Schema**:
```sql
dsl_execution_traces (
    trace_id, tenant_id, workflow_id, run_id,
    inputs, outputs, governance_metadata,
    execution_time_ms, trust_score,
    created_at, entities_affected
)
```

**Location**: `dsl/database/multi_tenant_schema.sql`

---

#### 3.2 Override Ledger (`dsl/operators/override_ledger.py`)
**Responsibilities**:
- Record all manual overrides
- Hash-chain for immutability
- Track approval workflows
- Maintain chain of custody

**Features**:
- SHA-256 hash chaining
- Append-only (no updates/deletes)
- Multi-approver support
- Justification required

**Location**: `dsl/operators/override_ledger.py`, `dsl/operators/override_service.py`

---

#### 3.3 Explainability Service (`dsl/operators/explainability_service.py`)
**Responsibilities**:
- Store SHAP/LIME explanations
- Generate feature attributions
- Provide narrative explanations
- Support regulatory queries

**Methods Supported**:
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance
- Counterfactual explanations

**Location**: `dsl/operators/explainability_service.py`, `api/explainability_service.py`

---

#### 3.4 Trust Scoring Engine (`dsl/intelligence/trust_scoring_engine.py`)
**Responsibilities**:
- Calculate multi-factor trust scores
- Combine: accuracy, explainability, drift, bias, SLA
- Enforce trust gates
- Track trust trends

**Scoring Formula**:
```
Trust Score = 
    (Accuracy × 0.25) +
    (Explainability × 0.20) +
    (Drift Stability × 0.20) +
    (Bias Fairness × 0.20) +
    (SLA Compliance × 0.15)
```

**Location**: `api/trust_score_service.py`, `dsl/intelligence/trust_scoring_engine.py`

---

#### 3.5 Evidence Pack Generator (`dsl/intelligence/evidence_pack_generator.py`)
**Responsibilities**:
- Aggregate evidence for audits
- Generate regulator reports
- Export for eDiscovery
- Package for legal holds

**Location**: `dsl/intelligence/evidence_pack_generator.py`, `api/legal_hold_service.py`

---

### Governance Layer Data Flow
```
Execution Result → Evidence Capture
                         ↓
                  Explainability Generation
                         ↓
                  Trust Score Calculation
                         ↓
                  Override Check (if applicable)
                         ↓
                  Ledger Write (immutable)
                         ↓
                  [Evidence Pack Available]
```

### Technology Stack
- **Storage**: PostgreSQL (JSONB for flexibility)
- **Integrity**: SHA-256 hash chains
- **Access**: Row Level Security (RLS)
- **Retention**: Policy-based purging
- **Export**: JSON, PDF, XML (legal formats)

### Compliance Features
- ✅ Immutable audit trail
- ✅ Hash-chained integrity
- ✅ Multi-tenant isolation (RLS)
- ✅ Data residency enforcement
- ✅ GDPR/SOX/HIPAA ready
- ✅ eDiscovery export

---

## 4. DATA PLANE

### Purpose
The Data Plane provides **persistent storage** for operational data, metrics, and historical records across the platform.

### Core Principle
**"Data Sovereignty"** - Tenant data never crosses boundaries. Residency rules strictly enforced.

### Components

#### 4.1 Azure Fabric (Bronze/Silver/Gold)
**Responsibilities**:
- Bronze: Raw data ingestion (CRM, billing, activity logs)
- Silver: Cleaned and transformed data
- Gold: Business-ready aggregates and features

**Data Sources**:
- Salesforce (opportunities, accounts, contacts)
- Billing systems (invoices, subscriptions)
- Activity logs (user behavior, engagement)

**Location**: External (Azure Fabric Lakehouse)

---

#### 4.2 PostgreSQL (Operational Database)
**Responsibilities**:
- Store workflow definitions
- Manage tenant metadata
- Persist execution traces
- Track governance events

**Key Tables**:
```sql
tenant_metadata
dsl_workflows
dsl_execution_traces
feature_metadata
feature_values_offline
override_ledger
trust_scores
bias_check_results
```

**Location**: Postgres instance (Azure Database for PostgreSQL)

---

#### 4.3 Redis (Cache Layer)
**Responsibilities**:
- Cache routing decisions (semantic similarity)
- Store online features (Feature Store)
- Cache inference results (TTL-based)
- Session management

**Namespacing**:
- `{tenant_id}:{key}` - Strict tenant isolation
- TTL: 1 hour (features), 5 min (predictions)

**Location**: Redis instance (Azure Cache for Redis)

---

#### 4.4 pgvector (Vector Embeddings)
**Responsibilities**:
- Store entity embeddings for RAG
- Semantic search for knowledge graph
- Similarity-based routing cache
- Feature embeddings

**Location**: PostgreSQL with pgvector extension

---

### Data Plane Data Flow
```
Data Sources → Fabric Bronze (Raw)
                     ↓
              Fabric Silver (Cleaned)
                     ↓
              Fabric Gold (Aggregated)
                     ↓
              Feature Store (Online/Offline)
                     ↓
              [Available for Execution Plane]

Governance Events → Postgres (Audit Trail)
                     ↓
              Evidence Database
                     ↓
              [Available for Compliance Queries]
```

### Technology Stack
- **OLTP**: PostgreSQL 15+
- **Cache**: Redis 7+
- **Lakehouse**: Azure Fabric
- **Vectors**: pgvector extension
- **Backup**: Automated daily snapshots

### Data Residency Enforcement
```yaml
# Example residency rules
tenants:
  acme_eu:
    region: EU
    data_center: eu-central-1
    allow_cross_border: false
  
  globalco_us:
    region: US
    data_center: us-east-1
    allow_cross_border: true
    allowed_regions: [EU]
```

**Enforcement**: `rbia/orchestrator_middleware.py`, `dsl/governance/multi_tenant_taxonomy.py`

---

## 5. KNOWLEDGE PLANE

### Purpose
The Knowledge Plane captures **organizational learning** from executions, enabling continuous improvement and intelligent querying.

### Core Principle
**"Learn from Every Execution"** - Every trace enriches the knowledge graph. Patterns emerge, insights compound.

### Components

#### 5.1 Knowledge Graph (`dsl/knowledge/kg_store.py`)
**Responsibilities**:
- Store execution traces as graph nodes
- Link workflows, models, policies, outcomes
- Enable graph-based queries (Cypher)
- Support lineage tracking

**Ontology**:
```
Entities: Workflow, Model, Policy, User, Tenant, Opportunity
Relationships: EXECUTED_BY, APPLIED_TO, RESULTED_IN, OVERRIDDEN_BY
```

**Technology**: PostgreSQL (graph queries via recursive CTEs) or Neo4j (future)

**Location**: `dsl/knowledge/kg_store.py`

---

#### 5.2 Trace Ingestion Pipeline (`dsl/knowledge/trace_ingestion.py`)
**Responsibilities**:
- Ingest execution traces from Governance Layer
- Extract entities and relationships
- Enrich with business context
- Write to Knowledge Graph

**Pipeline**:
```
Execution Trace → Parse & Extract
                       ↓
                Entity Extraction
                       ↓
              Relationship Inference
                       ↓
              KG Write (tenant-isolated)
```

**Location**: `dsl/knowledge/trace_ingestion.py`, `dsl/knowledge/rba_execution_tracer.py`

---

#### 5.3 RAG Pipelines (`scripts/implement_embeddings_rag.py`)
**Responsibilities**:
- Policy-aware retrieval (respects tenant boundaries)
- Semantic search over execution history
- Context-aware query answering
- Evidence retrieval for compliance

**Features**:
- pgvector for embeddings
- Tenant-scoped search
- Compliance filtering
- Source attribution

**Location**: `scripts/implement_embeddings_rag.py`, `src/services/comprehensive_rag_service.py`

---

#### 5.4 Lineage Explorer (`api/lineage_explorer_service.py`)
**Responsibilities**:
- Track data → feature → model → output → decision
- Visualize dependency chains
- Support impact analysis
- Enable root cause investigation

**Lineage Chain**:
```
Dataset → Feature → Model → Prediction → Decision → Override → Evidence
```

**Location**: `api/lineage_explorer_service.py`, `frontend_integration/lineage-explorer.tsx`

---

#### 5.5 SLM Export Pipeline (`dsl/intelligence/lineage_to_slm_exporter.py`)
**Responsibilities**:
- Export traces to SLM training format
- Prepare data for model fine-tuning
- Support vertical SLM development
- Enable ML flywheel

**Location**: `dsl/intelligence/lineage_to_slm_exporter.py`

---

### Knowledge Plane Data Flow
```
Execution Traces → Trace Ingestion
                         ↓
                  Entity Extraction
                         ↓
                  Knowledge Graph Write
                         ↓
              Vector Embeddings (pgvector)
                         ↓
              [Available for RAG Queries]

KG Queries → Cypher/SQL
                ↓
           RAG Retrieval
                ↓
         [Insights & Patterns]
```

### Technology Stack
- **Graph DB**: PostgreSQL (recursive CTEs) or Neo4j
- **Vectors**: pgvector
- **Embeddings**: SentenceTransformer (local)
- **RAG**: Custom pipeline with policy filtering
- **Export**: JSON, Parquet (ML training)

### Knowledge Plane Benefits
- ✅ Pattern recognition across executions
- ✅ Root cause analysis
- ✅ Impact prediction
- ✅ Intelligent querying (natural language)
- ✅ Continuous learning loop
- ✅ SLM training data generation

---

## Cross-Plane Interactions

### Request Flow (Happy Path)
```
User Request
    ↓
[CONTROL] Orchestrator routes to workflow
    ↓
[CONTROL] Policy Engine validates
    ↓
[CONTROL] Compiler generates execution plan
    ↓
[EXECUTION] Runtime fetches features from Data Plane
    ↓
[EXECUTION] ML model inference
    ↓
[EXECUTION] Drift/bias checking
    ↓
[GOVERNANCE] Evidence capture
    ↓
[GOVERNANCE] Trust score calculation
    ↓
[DATA] Write results to Postgres
    ↓
[KNOWLEDGE] Trace ingestion to KG
    ↓
Response to User
```

### Fallback Flow (RBIA → RBA)
```
[EXECUTION] ML inference fails OR confidence < threshold
    ↓
[CONTROL] Orchestrator triggers fallback
    ↓
[EXECUTION] Switch to RBA deterministic rules
    ↓
[GOVERNANCE] Log fallback event
    ↓
[GOVERNANCE] Evidence pack includes fallback reason
    ↓
Response with fallback result
```

---

## Tenant Isolation Enforcement

Each plane enforces multi-tenancy:

### Control Plane
- ✅ JWT validation (tenant_id in token)
- ✅ Tenant context propagation
- ✅ Rate limiting per tenant

### Execution Plane
- ✅ Tenant-scoped feature retrieval
- ✅ Separate model versions per tenant
- ✅ Isolated execution contexts

### Governance Layer
- ✅ RLS on all evidence tables
- ✅ Tenant-scoped override ledger
- ✅ Encrypted data at rest (tenant keys)

### Data Plane
- ✅ Database RLS policies
- ✅ Redis namespace isolation (`{tenant_id}:*`)
- ✅ Fabric partitions by tenant

### Knowledge Plane
- ✅ Graph queries filtered by tenant_id
- ✅ RAG search scoped to tenant
- ✅ Lineage isolated per tenant

---

## Scalability Strategy

### Horizontal Scaling
- **Control**: Multiple orchestrator instances (load balanced)
- **Execution**: Worker pool with queue (Celery/RabbitMQ)
- **Governance**: Read replicas for evidence queries
- **Data**: Postgres read replicas, Redis cluster
- **Knowledge**: Graph sharding by tenant (future)

### Vertical Scaling
- **Control**: CPU-optimized instances
- **Execution**: GPU instances for ML inference
- **Data**: Memory-optimized for Postgres/Redis
- **Knowledge**: Storage-optimized for graph data

---

## Monitoring & Observability

### Per-Plane Metrics

**Control Plane**:
- Request rate, latency (P50, P95, P99)
- Policy evaluation time
- Routing decision accuracy

**Execution Plane**:
- ML inference latency
- Fallback rate (RBIA → RBA)
- Feature retrieval latency
- Drift detection events

**Governance Layer**:
- Evidence capture rate
- Override frequency
- Trust score distribution
- Explainability generation time

**Data Plane**:
- Database connection pool utilization
- Cache hit rate
- Query performance
- Storage growth rate

**Knowledge Plane**:
- KG write throughput
- RAG query latency
- Lineage query performance
- Embedding generation time

### Alerting Thresholds
- Control: P99 latency > 2s
- Execution: Fallback rate > 10%
- Governance: Evidence write failures
- Data: Cache hit rate < 70%
- Knowledge: KG write lag > 5min

---

## Security & Compliance

### Authentication & Authorization
- **Control**: JWT tokens, OAuth2
- **Execution**: Service-to-service auth (mTLS)
- **Governance**: RBAC for evidence access
- **Data**: Database credentials rotation
- **Knowledge**: Query-level permissions

### Data Protection
- **At Rest**: AES-256 encryption (tenant keys)
- **In Transit**: TLS 1.3
- **Backups**: Encrypted, geo-replicated
- **Audit Logs**: Immutable, WORM storage

### Compliance Features
- ✅ SOX: Audit trail, SoD, approval workflows
- ✅ GDPR: Right to erasure, data portability, residency
- ✅ HIPAA: PHI protection, access logs, encryption
- ✅ RBI: Data localization, audit readiness
- ✅ DPDP: Consent management, purpose binding

---

## Future Enhancements

### Control Plane
- Auto-scaling based on load
- A/B testing of workflows
- Blue-green deployments

### Execution Plane
- GPU acceleration for ML
- Model serving via Triton/TorchServe
- Auto-retraining pipelines

### Governance Layer
- Blockchain for ledger (future)
- Zero-knowledge proofs (privacy)
- Federated learning support

### Data Plane
- Time-series DB (InfluxDB)
- Data lake optimization
- Cold storage tiering

### Knowledge Plane
- Neo4j migration (graph-native)
- Graph neural networks
- Automated insight generation

---

## Conclusion

The RBIA platform's **five-plane architecture** provides:

1. **Clear Separation**: Each plane has distinct responsibilities
2. **Scalability**: Independent scaling per plane
3. **Security**: Defense-in-depth across planes
4. **Auditability**: Evidence captured at every layer
5. **Flexibility**: Easy to extend without affecting others

This architecture supports **enterprise-scale deployments** with **regulatory compliance** baked in from the ground up.

---

**Document Owner**: RBIA Architecture Team  
**Review Cycle**: Quarterly  
**Next Review**: January 2026  
**Version**: 1.0 (October 2025)

