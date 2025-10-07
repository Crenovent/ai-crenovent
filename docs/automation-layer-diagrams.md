# Automation Layer Architecture Diagrams

This document contains the architecture diagrams for the RevAI Pro automation layer.

## 1. Overall Automation Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[UI Mode]
        Hybrid[Hybrid Mode]
        Agent[Agent Mode]
    end
    
    subgraph "AI Orchestrator"
        Router[Routing Orchestrator]
        Confidence[Confidence Thresholds]
        Policy[Policy Engine]
    end
    
    subgraph "Core Services"
        Calendar[Calendar Service]
        LetsMeet[Let's Meet Service]
        Cruxx[Cruxx Service]
    end
    
    subgraph "AI Microservices"
        AgentReg[Agent Registry]
        ModelAudit[Model Audit]
        KPIs[KPI Exporter]
        Traces[Run Trace Schema]
    end
    
    subgraph "Automation Hooks"
        CalAuto[Calendar Automation]
        MeetAuto[Let's Meet Automation]
        CruxxAuto[Cruxx Automation]
    end
    
    subgraph "Infrastructure"
        EventBus[Event Bus]
        DLQ[DLQ + Replay]
        Metrics[Metrics Exporter]
        Schema[Schema Registry]
    end
    
    UI --> Router
    Hybrid --> Router
    Agent --> Router
    
    Router --> Confidence
    Router --> Policy
    Router --> AgentReg
    
    Confidence --> Calendar
    Confidence --> LetsMeet
    Confidence --> Cruxx
    
    Calendar --> CalAuto
    LetsMeet --> MeetAuto
    Cruxx --> CruxxAuto
    
    CalAuto --> EventBus
    MeetAuto --> EventBus
    CruxxAuto --> EventBus
    
    EventBus --> DLQ
    EventBus --> Schema
    
    ModelAudit --> Traces
    KPIs --> Metrics
    
    Traces --> DLQ
    Metrics --> DLQ
```

## 2. Event-Driven Automation Flow

```mermaid
sequenceDiagram
    participant User
    participant Router
    participant Calendar
    participant EventBus
    participant LetsMeet
    participant Cruxx
    participant DLQ
    
    User->>Router: Create Calendar Event
    Router->>Router: Evaluate Confidence
    Router->>Calendar: Create Event (Agent Mode)
    Calendar->>EventBus: Publish calendar.event.created
    
    EventBus->>LetsMeet: Trigger Meeting Capture
    LetsMeet->>LetsMeet: Auto-transcribe Audio
    LetsMeet->>EventBus: Publish letsmeet.transcription.completed
    
    EventBus->>LetsMeet: Trigger Summary Generation
    LetsMeet->>EventBus: Publish letsmeet.summary.generated
    
    EventBus->>Cruxx: Trigger Action Extraction
    Cruxx->>Cruxx: Create Cruxx Actions
    Cruxx->>EventBus: Publish cruxx.action.created
    
    EventBus->>Calendar: Trigger Overlay Creation
    Calendar->>Calendar: Create Reminder Overlays
    
    alt Error Handling
        EventBus->>DLQ: Dead Letter Failed Events
        DLQ->>DLQ: Retry with Backoff
    end
```

## 3. Agent Registry and Routing

```mermaid
graph LR
    subgraph "Agent Registry"
        Agent1[Calendar Agent]
        Agent2[Meeting Agent]
        Agent3[Cruxx Agent]
        Agent4[CRM Agent]
    end
    
    subgraph "Routing Logic"
        Query[Agent Query]
        Filter[Capability Filter]
        Select[Agent Selection]
    end
    
    subgraph "Policy Engine"
        Rules[Policy Rules]
        Thresholds[Confidence Thresholds]
        Fallback[Fallback Logic]
    end
    
    Query --> Filter
    Filter --> Agent1
    Filter --> Agent2
    Filter --> Agent3
    Filter --> Agent4
    
    Rules --> Select
    Thresholds --> Select
    Fallback --> Select
    
    Select --> Agent1
    Select --> Agent2
    Select --> Agent3
    Select --> Agent4
```

## 4. Trust and Confidence Management

```mermaid
graph TB
    subgraph "Input Processing"
        Request[User Request]
        Context[Context Analysis]
        History[User History]
    end
    
    subgraph "Confidence Evaluation"
        ModelConf[Model Confidence]
        TrustScore[Trust Score]
        PolicyCheck[Policy Check]
    end
    
    subgraph "Decision Engine"
        Threshold[Threshold Check]
        Fallback[Fallback Decision]
        Override[Override Detection]
    end
    
    subgraph "Mode Selection"
        UIMode[UI Mode]
        HybridMode[Hybrid Mode]
        AgentMode[Agent Mode]
    end
    
    Request --> Context
    Context --> History
    History --> ModelConf
    
    ModelConf --> TrustScore
    TrustScore --> PolicyCheck
    
    PolicyCheck --> Threshold
    Threshold --> Fallback
    Fallback --> Override
    
    Override --> UIMode
    Override --> HybridMode
    Override --> AgentMode
```

## 5. Run Trace and Audit Flow

```mermaid
graph TB
    subgraph "Trace Collection"
        Input[Trace Input]
        Decision[Trace Decision]
        Output[Trace Output]
        Span[Trace Span]
    end
    
    subgraph "Audit Processing"
        Validation[Trace Validation]
        Aggregation[Trace Aggregation]
        Analysis[Trust Analysis]
    end
    
    subgraph "Evidence Management"
        Evidence[Evidence Pack]
        Override[Override Ledger]
        Compliance[Compliance Check]
    end
    
    subgraph "Reporting"
        Metrics[Metrics Export]
        Alerts[Alert Generation]
        Dashboard[Dashboard Update]
    end
    
    Input --> Validation
    Decision --> Validation
    Output --> Validation
    Span --> Validation
    
    Validation --> Aggregation
    Aggregation --> Analysis
    
    Analysis --> Evidence
    Analysis --> Override
    Analysis --> Compliance
    
    Evidence --> Metrics
    Override --> Alerts
    Compliance --> Dashboard
```

## 6. DLQ and Replay Architecture

```mermaid
graph TB
    subgraph "Event Publishing"
        Publisher[Event Publisher]
        Schema[Schema Validation]
        Topic[Topic Routing]
    end
    
    subgraph "Event Consumption"
        Consumer[Event Consumer]
        Handler[Message Handler]
        Processing[Event Processing]
    end
    
    subgraph "Error Handling"
        Failure[Processing Failure]
        Retry[Retry Logic]
        DLQ[Dead Letter Queue]
    end
    
    subgraph "Replay System"
        Replay[Replay Engine]
        Scheduler[Scheduler]
        Monitor[SLO Monitor]
    end
    
    Publisher --> Schema
    Schema --> Topic
    Topic --> Consumer
    
    Consumer --> Handler
    Handler --> Processing
    
    Processing --> Failure
    Failure --> Retry
    Retry --> DLQ
    
    DLQ --> Replay
    Replay --> Scheduler
    Scheduler --> Monitor
    
    Monitor --> Consumer
```

## 7. Microservices Communication

```mermaid
graph TB
    subgraph "API Gateway"
        Gateway[API Gateway]
        Auth[Authentication]
        RateLimit[Rate Limiting]
    end
    
    subgraph "Core Services"
        Orchestrator[AI Orchestrator]
        AgentReg[Agent Registry]
        Router[Routing Orchestrator]
        Confidence[Confidence Thresholds]
    end
    
    subgraph "Automation Services"
        CalAuto[Calendar Automation]
        MeetAuto[Let's Meet Automation]
        CruxxAuto[Cruxx Automation]
    end
    
    subgraph "Infrastructure Services"
        EventBus[Event Bus]
        DLQ[DLQ Service]
        Metrics[Metrics Exporter]
        Traces[Trace Schema]
    end
    
    Gateway --> Auth
    Auth --> RateLimit
    RateLimit --> Orchestrator
    
    Orchestrator --> AgentReg
    Orchestrator --> Router
    Orchestrator --> Confidence
    
    Router --> CalAuto
    Router --> MeetAuto
    Router --> CruxxAuto
    
    CalAuto --> EventBus
    MeetAuto --> EventBus
    CruxxAuto --> EventBus
    
    EventBus --> DLQ
    EventBus --> Metrics
    EventBus --> Traces
```

## 8. Data Flow Architecture

```mermaid
graph LR
    subgraph "Data Sources"
        Calendar[Calendar Data]
        Meetings[Meeting Data]
        Cruxx[Cruxx Data]
        CRM[CRM Data]
    end
    
    subgraph "Processing Layer"
        ETL[ETL Pipeline]
        Transform[Data Transform]
        Enrich[Data Enrichment]
    end
    
    subgraph "Storage Layer"
        PostgreSQL[(PostgreSQL)]
        Redis[(Redis Cache)]
        Vector[(pgVector)]
    end
    
    subgraph "Analytics Layer"
        Metrics[Metrics Engine]
        KPIs[KPI Calculator]
        Reports[Report Generator]
    end
    
    Calendar --> ETL
    Meetings --> ETL
    Cruxx --> ETL
    CRM --> ETL
    
    ETL --> Transform
    Transform --> Enrich
    
    Enrich --> PostgreSQL
    Enrich --> Redis
    Enrich --> Vector
    
    PostgreSQL --> Metrics
    Redis --> KPIs
    Vector --> Reports
```

## 9. Security and Compliance Architecture

```mermaid
graph TB
    subgraph "Security Layer"
        Auth[Authentication]
        Authz[Authorization]
        RBAC[RBAC Engine]
        RLS[Row Level Security]
    end
    
    subgraph "Compliance Layer"
        Audit[Audit Logging]
        PII[PII Protection]
        Consent[Consent Management]
        DLP[Data Loss Prevention]
    end
    
    subgraph "Monitoring Layer"
        SIEM[SIEM Integration]
        Alerts[Security Alerts]
        Incident[Incident Response]
        Forensics[Digital Forensics]
    end
    
    subgraph "Data Protection"
        Encryption[Data Encryption]
        Backup[Backup & Recovery]
        Retention[Data Retention]
        Residency[Data Residency]
    end
    
    Auth --> Authz
    Authz --> RBAC
    RBAC --> RLS
    
    RLS --> Audit
    Audit --> PII
    PII --> Consent
    Consent --> DLP
    
    DLP --> SIEM
    SIEM --> Alerts
    Alerts --> Incident
    Incident --> Forensics
    
    Forensics --> Encryption
    Encryption --> Backup
    Backup --> Retention
    Retention --> Residency
```

## 10. Deployment Architecture

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Load Balancer]
        SSL[SSL Termination]
        Health[Health Checks]
    end
    
    subgraph "Application Tier"
        App1[App Instance 1]
        App2[App Instance 2]
        App3[App Instance N]
    end
    
    subgraph "Microservices Tier"
        MS1[Microservice 1]
        MS2[Microservice 2]
        MS3[Microservice N]
    end
    
    subgraph "Data Tier"
        Primary[(Primary DB)]
        Replica[(Read Replica)]
        Cache[(Redis Cache)]
    end
    
    subgraph "Infrastructure"
        Queue[Message Queue]
        Storage[Object Storage]
        CDN[CDN]
    end
    
    LB --> SSL
    SSL --> Health
    Health --> App1
    Health --> App2
    Health --> App3
    
    App1 --> MS1
    App2 --> MS2
    App3 --> MS3
    
    MS1 --> Primary
    MS2 --> Replica
    MS3 --> Cache
    
    Primary --> Queue
    Replica --> Storage
    Cache --> CDN
```

---

## Diagram Usage

These diagrams should be embedded in the relevant documentation chapters:

- **Chapter 2.1**: Overall Automation Architecture
- **Chapter 2.2**: Event-Driven Automation Flow
- **Chapter 2.3**: Agent Registry and Routing
- **Chapter 2.4**: Trust and Confidence Management
- **Chapter 2.5**: Run Trace and Audit Flow
- **Chapter 2.6**: DLQ and Replay Architecture
- **Chapter 2.7**: Microservices Communication
- **Chapter 2.8**: Data Flow Architecture
- **Chapter 2.9**: Security and Compliance Architecture
- **Chapter 2.10**: Deployment Architecture

Each diagram provides a visual representation of the system components and their interactions, making it easier to understand the complex automation layer architecture.
