# Compiler Hooks, Fallback Logic, and Override Hooks Implementation

This document outlines the implementation of three critical tasks for the Risk-Based Intelligent Automation (RBIA) framework:

- **Task 4.1.3**: Compiler hooks for intelligent nodes
- **Task 4.1.5**: Fallback logic (ML error → rule-only path)
- **Task 4.1.6**: Override hooks for ML nodes (manual justifications)

## Table of Contents

1. [Overview](#overview)
2. [Task 4.1.3: Compiler Hooks](#task-413-compiler-hooks)
3. [Task 4.1.5: Fallback Logic](#task-415-fallback-logic)
4. [Task 4.1.6: Override Hooks](#task-416-override-hooks)
5. [Integration Example](#integration-example)
6. [How to Run](#how-to-run)

---

## Overview

These three tasks work together to create a robust, enterprise-grade ML infrastructure that can:

- **Compile intelligent nodes** with proper execution plans and governance hooks
- **Handle ML failures gracefully** with rule-based fallback mechanisms
- **Allow manual overrides** with full audit trails and compliance tracking

## Task 4.1.3: Compiler Hooks

### Purpose
Extend the workflow compiler to execute intelligent ML nodes like traditional rules, with proper execution planning and governance integration.

### Implementation

**File**: `dsl/compiler/intelligent_node_compiler.py`

**Key Components**:

1. **IntelligentNodeConfig**: Configuration for ML nodes
2. **CompilationResult**: Result of node compilation
3. **IntelligentNodeCompiler**: Main compiler class

**Features**:
- ML node compilation and optimization
- Execution plan generation
- Model registry integration
- Fallback logic compilation
- Governance rule enforcement
- Explainability hooks

**Example Usage**:
```python
# Create intelligent node configuration
node_config = IntelligentNodeConfig(
    node_id="churn_prediction_node",
    node_type="ml_predict",
    model_id="saas_churn_predictor_v2",
    input_mapping={
        "mrr": "customer_mrr",
        "usage_frequency": "usage_frequency"
    },
    output_mapping={
        "churn_probability": "churn_probability"
    },
    confidence_threshold=0.7,
    fallback_enabled=True,
    explainability_enabled=True
)

# Compile the intelligent node
compiler = IntelligentNodeCompiler(node_registry)
result = await compiler.compile_intelligent_node(node_config, workflow_context)
```

## Task 4.1.5: Fallback Logic

### Purpose
Implement resilient workflows with fallback to deterministic rules when ML operations fail or produce low-confidence results.

### Implementation

**File**: `dsl/operators/fallback_service.py`

**Key Components**:

1. **FallbackRule**: Configuration for fallback rules
2. **FallbackExecution**: Record of fallback execution
3. **FallbackService**: Main service class

**Features**:
- Rule-based fallback execution
- ML error handling and recovery
- Fallback logging and monitoring
- Integration with governance systems
- Performance tracking and analytics
- Multiple fallback strategies

**Example Usage**:
```python
# Create fallback service
fallback_service = FallbackService()

# Create fallback rule
rule = FallbackRule(
    rule_id="low_confidence_fallback",
    rule_name="Low Confidence Fallback",
    rule_type="threshold",
    condition="confidence < 0.5",
    action={
        "prediction": "unknown",
        "confidence": 0.3,
        "reason": "Low confidence fallback"
    }
)

# Execute fallback
result = await fallback_service.execute_fallback(
    workflow_id="workflow_123",
    step_id="ml_step",
    model_id="model_123",
    trigger_reason="confidence_low",
    original_input=input_data,
    context=context
)
```

## Task 4.1.6: Override Hooks

### Purpose
Enable governance-first workflows with manual override capabilities, including approval workflows and immutable audit trails.

### Implementation

**Files**: 
- `dsl/operators/override_service.py` - Override request management
- `dsl/operators/override_ledger.py` - Immutable audit ledger

**Key Components**:

1. **OverrideRequest**: Override request configuration
2. **OverrideApproval**: Approval record
3. **OverrideLedger**: Immutable ledger entry
4. **OverrideService**: Main service class
5. **OverrideLedger**: Ledger management

**Features**:
- Override request management
- Approval workflow
- Immutable override ledger
- Hash-chained integrity verification
- Cryptographic tamper detection
- Compliance-ready audit records
- Historical query capabilities

**Example Usage**:
```python
# Create override service
override_service = OverrideService()

# Create override request
override_id = await override_service.create_override_request(
    workflow_id="workflow_123",
    step_id="ml_step",
    model_id="model_123",
    node_id="node_123",
    original_prediction={"churn_probability": 0.8},
    override_prediction={"churn_probability": 0.3},
    justification="Special circumstances",
    override_type="manual",
    requested_by=456,
    tenant_id="tenant_123"
)

# Approve override
await override_service.approve_override(
    override_id=override_id,
    approver_id=789,
    approval_status="approved",
    approval_reason="Business context override",
    tenant_id="tenant_123"
)
```

## Integration Example

The `examples/compiler_fallback_override_example.py` file demonstrates all three tasks working together in a comprehensive RBIA workflow.

### Key Features Demonstrated:

1. **Compiler Hooks**:
   - Intelligent node compilation
   - Execution plan generation
   - Governance integration

2. **Fallback Logic**:
   - Rule-based fallback execution
   - Low confidence handling
   - Fallback analytics

3. **Override Hooks**:
   - Override request creation
   - Approval workflow
   - Ledger integrity verification

### Example Workflow:

```yaml
workflow_id: comprehensive_rbia_workflow
name: Comprehensive RBIA Workflow
automation_type: RBIA
governance:
  trust_score_threshold: 0.75
  evidence_pack_required: true

steps:
  - id: predict_churn_risk
    type: ml_predict
    params:
      model_id: "saas_churn_predictor_v2"
      confidence_threshold: 0.7
      explainability_enabled: true
      fallback_enabled: true
    governance:
      explainability_required: true
      drift_bias_monitoring_enabled: true
      override_enabled: true
```

## How to Run

### Prerequisites

1. Python 3.8+
2. Required dependencies:
   ```bash
   pip install pyyaml jsonschema sqlite3
   ```

### Running the Example

1. **Run the comprehensive example**:
   ```bash
   python examples/compiler_fallback_override_example.py
   ```

2. **Run individual components**:
   ```python
   # Compiler hooks
   from dsl.compiler.intelligent_node_compiler import IntelligentNodeCompiler
   
   # Fallback logic
   from dsl.operators.fallback_service import FallbackService
   
   # Override hooks
   from dsl.operators.override_service import OverrideService
   from dsl.operators.override_ledger import OverrideLedger
   ```

### Database Files

The implementation creates several SQLite databases:

- `fallback_service.db` - Fallback rules and execution logs
- `override_service.db` - Override requests and approvals
- `override_ledger.db` - Immutable override ledger

## Key Benefits

### 1. **Robust ML Operations**
- Graceful handling of ML failures
- Rule-based fallback mechanisms
- Confidence threshold management

### 2. **Governance and Compliance**
- Immutable audit trails
- Hash-chained integrity verification
- Approval workflows for overrides

### 3. **Enterprise Readiness**
- Multi-tenant support
- Performance monitoring
- Analytics and reporting

### 4. **Developer Experience**
- Clear APIs and interfaces
- Comprehensive error handling
- Extensive logging and debugging

## Architecture Integration

These three tasks integrate seamlessly with the existing RBIA framework:

- **Compiler Hooks** extend the DSL parser and workflow execution
- **Fallback Logic** integrates with ML operators and governance systems
- **Override Hooks** provide compliance and audit capabilities

The implementation follows the existing patterns and conventions, ensuring consistency and maintainability.

---

## Summary

The implementation of Tasks 4.1.3, 4.1.5, and 4.1.6 provides a complete foundation for enterprise-grade ML operations within the RBIA framework. These components work together to create a robust, compliant, and maintainable system that can handle the complexities of production ML workflows while maintaining strict governance and audit requirements.
