# Internal RevAI Pro Module Integration Hooks

**Task 4.5.7**: Build internal RevAI Pro module integration hooks (Forecast, Compensation, Dashboard)

## Overview

This document defines the integration contracts, event schemas, and orchestration patterns for internal RevAI Pro modules to seamlessly integrate with RBIA workflows.

## Module Categories

### 1. Core Revenue Modules
- **Forecasting**: Revenue forecasting and prediction
- **Pipeline**: Sales pipeline management
- **Planning**: Territory and quota planning
- **Compensation**: Commission and incentive calculations

### 2. Customer-Facing Modules
- **My Customers**: Customer health and engagement
- **Customer Success**: CSM workflows and health scores
- **DealDesk**: Deal approval and pricing workflows

### 3. Operations Modules
- **Dashboard**: Analytics and reporting
- **Partner Management**: Channel partner workflows
- **Market Intelligence**: Competitive intelligence

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RBIA Orchestrator                         │
│                  (Integration Layer)                         │
└───────────┬─────────────────────────────────┬───────────────┘
            │                                 │
            │                                 │
    ┌───────▼────────┐              ┌────────▼────────┐
    │  Module Event  │              │  Module State   │
    │     Bus        │              │    Registry     │
    └───────┬────────┘              └────────┬────────┘
            │                                 │
            │                                 │
┌───────────▼─────────────────────────────────▼──────────────┐
│                    Internal Modules                         │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │Forecast  │  │Pipeline  │  │Planning  │  │Comp      │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │Customer  │  │DealDesk  │  │Partner   │  │Dashboard │  │
│  │Success   │  │          │  │Mgmt      │  │          │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Module-to-Module Event Contracts

### Standard Event Format

All inter-module events follow this schema:

```json
{
  "event_id": "evt_uuid",
  "event_type": "module.action.status",
  "source_module": "forecast",
  "target_module": "compensation",
  "tenant_id": "tenant_001",
  "user_id": "user_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "payload": {
    "action": "forecast_submitted",
    "data": { ... },
    "metadata": { ... }
  },
  "correlation_id": "corr_workflow_001",
  "requires_response": true,
  "timeout_seconds": 30
}
```

### Event Types by Module

#### Forecasting Module Events

**Outbound Events:**
```yaml
forecast.submitted:
  description: "Forecast submitted for approval"
  payload:
    forecast_id: string
    forecast_period: string
    forecast_amount: number
    rep_id: string
    manager_id: string
  triggers:
    - compensation.recalculate
    - pipeline.validate
    - dashboard.refresh

forecast.approved:
  description: "Forecast approved by manager"
  payload:
    forecast_id: string
    approved_by: string
    approved_at: datetime
    final_amount: number
  triggers:
    - compensation.commit
    - dashboard.update

forecast.revised:
  description: "Forecast revised after RBIA analysis"
  payload:
    forecast_id: string
    old_amount: number
    new_amount: number
    revision_reason: string
    confidence_score: float
  triggers:
    - pipeline.reanalyze
    - customer_success.alert
```

**Inbound Events:**
```yaml
pipeline.closed_won:
  action: "Update forecast with actual closed deal"
  
customer_success.churn_risk:
  action: "Adjust forecast downward for at-risk accounts"
  
compensation.attainment_check:
  action: "Provide forecast attainment data"
```

#### Pipeline Module Events

**Outbound Events:**
```yaml
pipeline.deal_created:
  description: "New deal added to pipeline"
  payload:
    deal_id: string
    account_id: string
    amount: number
    close_date: date
    stage: string
  triggers:
    - forecast.include
    - customer_success.monitor
    - market_intelligence.track

pipeline.stage_changed:
  description: "Deal moved to new stage"
  payload:
    deal_id: string
    old_stage: string
    new_stage: string
    probability: float
  triggers:
    - forecast.recalculate
    - compensation.preview

pipeline.deal_won:
  description: "Deal closed/won"
  payload:
    deal_id: string
    final_amount: number
    close_date: date
    rep_id: string
  triggers:
    - compensation.calculate
    - forecast.validate
    - customer_success.onboard

pipeline.deal_lost:
  description: "Deal lost"
  payload:
    deal_id: string
    lost_reason: string
    competitor: string
  triggers:
    - forecast.adjust
    - market_intelligence.analyze
```

**Inbound Events:**
```yaml
forecast.capacity_check:
  action: "Validate pipeline against capacity"
  
deal_desk.pricing_approved:
  action: "Update deal with approved pricing"
  
customer_success.expansion_opportunity:
  action: "Create expansion deal in pipeline"
```

#### Compensation Module Events

**Outbound Events:**
```yaml
compensation.calculated:
  description: "Commission calculated for rep"
  payload:
    rep_id: string
    period: string
    base_commission: number
    accelerators: number
    total_commission: number
  triggers:
    - dashboard.update_metrics
    - forecast.validate_incentives

compensation.threshold_reached:
  description: "Rep reached compensation threshold/tier"
  payload:
    rep_id: string
    threshold_name: string
    new_rate: float
  triggers:
    - dashboard.celebrate
    - pipeline.notify
```

**Inbound Events:**
```yaml
pipeline.deal_won:
  action: "Calculate commission for closed deal"
  
forecast.approved:
  action: "Preview commission based on forecast"
  
planning.quota_changed:
  action: "Recalculate attainment percentage"
```

#### Customer Success Module Events

**Outbound Events:**
```yaml
customer_success.health_score_changed:
  description: "Customer health score updated"
  payload:
    account_id: string
    old_score: number
    new_score: number
    risk_level: string
  triggers:
    - forecast.adjust_retention
    - pipeline.flag_at_risk

customer_success.churn_risk_high:
  description: "Customer flagged as high churn risk"
  payload:
    account_id: string
    churn_probability: float
    risk_factors: array
  triggers:
    - forecast.downgrade
    - pipeline.intervention

customer_success.expansion_opportunity:
  description: "Expansion opportunity identified"
  payload:
    account_id: string
    estimated_value: number
    opportunity_type: string
  triggers:
    - pipeline.create_opportunity
    - forecast.include_upside
```

#### Dashboard Module Events

**Outbound Events:**
```yaml
dashboard.threshold_alert:
  description: "Metric crossed threshold"
  payload:
    metric_name: string
    current_value: number
    threshold: number
    severity: string
  triggers:
    - forecast.investigate
    - pipeline.review

dashboard.anomaly_detected:
  description: "Anomalous pattern detected"
  payload:
    metric_name: string
    expected_range: object
    actual_value: number
  triggers:
    - forecast.validate
    - customer_success.investigate
```

**Inbound Events:**
```yaml
*.* (all events):
  action: "Update dashboards with latest data"
```

## Module Integration Patterns

### Pattern 1: Request-Response

Used for synchronous data queries between modules.

```python
# Forecast module requests pipeline data
from dsl.hub.module_integration import ModuleClient

module_client = ModuleClient()

response = await module_client.request(
    source_module="forecast",
    target_module="pipeline",
    action="get_pipeline_summary",
    payload={
        "rep_id": "rep_123",
        "time_period": "Q1_2024"
    },
    timeout_seconds=10
)

pipeline_data = response["data"]
```

### Pattern 2: Event Publish-Subscribe

Used for asynchronous event notifications.

```python
# Pipeline module publishes deal won event
from dsl.hub.module_integration import ModuleEventBus

event_bus = ModuleEventBus()

await event_bus.publish(
    event_type="pipeline.deal_won",
    source_module="pipeline",
    payload={
        "deal_id": "deal_456",
        "amount": 100000,
        "rep_id": "rep_123"
    },
    tenant_id="tenant_001"
)

# Compensation and Forecast modules automatically notified
```

### Pattern 3: State Synchronization

Used for maintaining consistent state across modules.

```python
# Customer Success updates account health
from dsl.hub.module_integration import ModuleStateRegistry

state_registry = ModuleStateRegistry()

await state_registry.update_state(
    module="customer_success",
    entity_type="account",
    entity_id="acct_789",
    state={
        "health_score": 85,
        "risk_level": "low",
        "last_updated": datetime.utcnow()
    },
    tenant_id="tenant_001"
)

# Other modules can query this state
account_state = await state_registry.get_state(
    module="customer_success",
    entity_type="account",
    entity_id="acct_789",
    tenant_id="tenant_001"
)
```

### Pattern 4: Workflow Orchestration

Used for multi-module workflows.

```python
# RBIA orchestrates cross-module workflow
from dsl.hub.module_integration import ModuleOrchestrator

orchestrator = ModuleOrchestrator()

workflow_result = await orchestrator.execute_workflow(
    workflow_id="deal_closure_workflow",
    tenant_id="tenant_001",
    steps=[
        {
            "module": "pipeline",
            "action": "close_deal",
            "payload": {"deal_id": "deal_456"}
        },
        {
            "module": "forecast",
            "action": "update_actuals",
            "payload": {"deal_id": "deal_456"},
            "depends_on": ["pipeline"]
        },
        {
            "module": "compensation",
            "action": "calculate_commission",
            "payload": {"deal_id": "deal_456"},
            "depends_on": ["pipeline"]
        },
        {
            "module": "customer_success",
            "action": "begin_onboarding",
            "payload": {"deal_id": "deal_456"},
            "depends_on": ["pipeline"]
        }
    ]
)
```

## Implementation Files

### Core Integration Infrastructure

1. **`dsl/hub/module_integration.py`** (to be created)
   - `ModuleClient` - Request-response client
   - `ModuleEventBus` - Event pub/sub
   - `ModuleStateRegistry` - State management
   - `ModuleOrchestrator` - Workflow orchestration

2. **`dsl/operators/rba/centralized_module_assignment.py`** (exists)
   - Module definitions and assignments
   - Role-based module access

3. **`dsl/integration_orchestrator.py`** (exists)
   - High-level integration orchestration
   - Component initialization

4. **`src/rba/complete_integration.py`** (exists)
   - Complete RBA system integration
   - Module coordination

### Module-Specific Integrations

Each module should implement:

```
api/modules/{module_name}/
├── events.py           # Event definitions
├── handlers.py         # Event handlers
├── integration.py      # Integration endpoints
└── schemas.py          # Data schemas
```

## Security & Governance

### Tenant Isolation

All module-to-module communication enforces tenant isolation:

```python
# Every request includes tenant context
async def handle_module_request(request: ModuleRequest):
    # Validate tenant access
    if request.tenant_id != current_tenant_id:
        raise PermissionError("Cross-tenant access denied")
    
    # Process request with tenant isolation
    await process_with_tenant_context(request)
```

### Audit Trail

All inter-module events are logged:

```python
# Automatic audit logging
await audit_logger.log_module_event(
    event_type="module_integration",
    source_module=source,
    target_module=target,
    tenant_id=tenant_id,
    payload=payload,
    outcome="success"
)
```

### Policy Enforcement

Module integrations respect governance policies:

```python
# Check if integration is allowed
policy_result = await policy_engine.check_integration(
    source_module=source,
    target_module=target,
    tenant_id=tenant_id,
    user_id=user_id
)

if not policy_result.allowed:
    raise PolicyViolation(policy_result.reason)
```

## Testing

### Integration Tests

```python
# Test cross-module workflow
async def test_deal_closure_workflow():
    # Create test deal in Pipeline
    deal = await pipeline_module.create_deal({
        "amount": 100000,
        "rep_id": "rep_test"
    })
    
    # Close deal
    await pipeline_module.close_deal(deal.id)
    
    # Verify Compensation was updated
    commission = await compensation_module.get_commission(
        rep_id="rep_test",
        deal_id=deal.id
    )
    assert commission.amount > 0
    
    # Verify Forecast was updated
    forecast = await forecast_module.get_forecast("rep_test")
    assert deal.id in forecast.closed_deals
    
    # Verify Customer Success was notified
    onboarding = await customer_success_module.get_onboarding(deal.account_id)
    assert onboarding.status == "initiated"
```

## Monitoring & Observability

### Key Metrics

1. **Event Delivery Rate**: % of events successfully delivered
2. **Request Latency**: Time for module-to-module requests
3. **Error Rate**: % of failed integrations
4. **Event Backlog**: Number of unprocessed events

### Dashboards

Use `api/cross_integration_trust_dashboard.py` to monitor:
- Module health status
- Integration success rates
- Latency metrics
- Error tracking

## Migration Guide

### Adding a New Module

1. Define module events in `events.py`
2. Implement event handlers in `handlers.py`
3. Register module in `ModuleRegistry`
4. Add integration tests
5. Document in this file

### Example: Adding Marketing Module

```python
# api/modules/marketing/events.py
class MarketingEvents:
    CAMPAIGN_LAUNCHED = "marketing.campaign_launched"
    LEAD_GENERATED = "marketing.lead_generated"
    LEAD_QUALIFIED = "marketing.lead_qualified"

# api/modules/marketing/handlers.py
@event_handler("pipeline.deal_won")
async def on_deal_won(event: ModuleEvent):
    # Attribute deal to marketing campaign
    await marketing_attribution.attribute_deal(event.payload)

# Register in ModuleRegistry
module_registry.register(
    module_name="marketing",
    events=MarketingEvents,
    handlers=marketing_handlers
)
```

## Related Documentation

- `docs/COMPILER_FALLBACK_OVERRIDE_README.md` - Compiler and fallback logic
- `docs/INTELLIGENT_CSV_PROCESSING.md` - Data processing workflows
- `schemas/integration_event_schema.json` - Standard event schema
- `api/cross_integration_trust_dashboard.py` - Integration monitoring

## Support

For questions or issues with module integrations:
- Review this documentation
- Check integration logs in Knowledge Graph
- Consult cross-integration trust dashboard
- Contact RevOps platform team

