# RBIA DSL v2 Grammar Examples
# Task 6.2.1 - Demonstrates new hybrid syntax with ml_node, threshold, confidence, policy_pack, fallback[], explainability

## Example 1: Basic RBIA Workflow with ML Node
```yaml
workflow_id: "customer_churn_prediction_v2"
name: "Customer Churn Prediction with ML"
module: "customer_success"
automation_type: "RBIA"
version: "2.1.0"
policy_pack: "customer_data_governance_v1"

governance:
  policy_id: "customer_ml_policy"
  evidence_capture: true
  evidence_pack_required: true
  region_id: "us-east-1"
  sla_budget_ms: 5000
  sla_tier: "T1"

steps:
  - id: "fetch_customer_data"
    type: "query"
    params:
      data_source: "customer_db"
      query: "SELECT * FROM customers WHERE customer_id = {{ customer_id }}"
    outputs:
      customer_data: "$.result"
    
  - id: "churn_prediction"
    type: "ml_node"
    params:
      model_id: "saas_churn_predictor_v2"
      model_version: "1.2.3"
      input_data: "{{ customer_data }}"
      confidence:
        min_confidence: 0.7
        auto_execute_above: 0.85
        assisted_mode_below: 0.6
      threshold:
        confidence_threshold: 0.7
        score_threshold: 0.8
      explainability:
        enabled: true
        method: "shap"
        params:
          background_samples: 100
          max_evals: 1000
    outputs:
      churn_prediction: "$.prediction"
      churn_confidence: "$.confidence"
      explanation: "$.explanation"
    governance:
      evidence_capture: true
      approval_required: false
    fallback:
      enabled: true
      fallback:
        - type: "rba_rule"
          target: "traditional_churn_rules"
          condition: "confidence < 0.7"
        - type: "human_escalation"
          target: "customer_success_team"
          condition: "error OR timeout > 3000"
      trigger_conditions:
        - "confidence < 0.7"
        - "timeout > 3000"
    
  - id: "traditional_churn_rules"
    type: "decision"
    params:
      rules:
        - condition: "{{ customer_data.days_since_login }} > 30"
          action: "mark_at_risk"
        - condition: "{{ customer_data.support_tickets }} > 5"
          action: "mark_at_risk"
        - default: "mark_healthy"
    outputs:
      fallback_prediction: "$.action"
```

## Example 2: Multi-ML Node Workflow with Policy Pack
```yaml
workflow_id: "fraud_detection_pipeline_v2"
name: "Advanced Fraud Detection with Multiple ML Models"
module: "risk_management"
automation_type: "RBIA"
version: "2.0.0"
policy_pack: "financial_ml_governance_v2"

governance:
  policy_id: "fraud_detection_policy"
  evidence_pack_required: true
  region_id: "eu-west-1"
  data_residency: "GDPR_COMPLIANT"
  sla_budget_ms: 2000
  sla_tier: "T0"

steps:
  - id: "transaction_scoring"
    type: "ml_node"
    params:
      model_id: "transaction_risk_scorer"
      model_version: "2.1.0"
      confidence:
        min_confidence: 0.8
        auto_execute_above: 0.9
        assisted_mode_below: 0.7
      threshold:
        score_threshold: 75
      explainability:
        enabled: true
        method: "lime"
    fallback:
      enabled: true
      fallback:
        - type: "rba_rule"
          target: "rule_based_scoring"
        - type: "default_action"
          target: "allow_transaction"
    
  - id: "behavioral_analysis"
    type: "ml_node"
    params:
      model_id: "behavioral_anomaly_detector"
      confidence:
        min_confidence: 0.75
      explainability:
        enabled: true
        method: "shap"
        params:
          shap:
            background_samples: 50
    fallback:
      enabled: true
      fallback:
        - type: "previous_step"
          target: "transaction_scoring"
          condition: "confidence < 0.75"
```

## Example 3: Backward Compatible RBA Workflow
```yaml
workflow_id: "traditional_approval_workflow"
name: "Traditional Rule-Based Approval"
module: "approvals"
automation_type: "RBA"
version: "1.5.0"

# Note: No policy_pack, ml_node, or fallback[] - fully backward compatible
steps:
  - id: "check_amount"
    type: "decision"
    params:
      condition: "{{ amount }} > 10000"
      true_action: "require_approval"
      false_action: "auto_approve"
    
  - id: "require_approval"
    type: "governance"
    params:
      approval_required: true
      approver_roles: ["finance_manager", "cfo"]
```

## Key DSL v2 Features Demonstrated:

### 1. **ml_node** Type
- Unified ML node type that can handle predict, score, classify, explain
- Replaces separate ml_predict, ml_score, etc. types

### 2. **threshold** Parameter
- confidence_threshold: minimum confidence for execution
- score_threshold: minimum score threshold
- Flexible threshold configuration

### 3. **confidence** Block
- min_confidence: baseline confidence requirement
- auto_execute_above: confidence level for automatic execution
- assisted_mode_below: confidence level requiring human assistance

### 4. **policy_pack** Workflow-Level Setting
- References governance policy collections
- Enables governance-by-design at compile time

### 5. **fallback[]** Array Syntax
- Multiple fallback strategies per ML node
- Conditional fallback triggers
- Supports RBA rule fallback, human escalation, default actions

### 6. **explainability** Configuration
- Multiple explanation methods (SHAP, LIME, gradient, attention, counterfactual)
- Method-specific parameters
- Required for ML nodes in RBIA workflows

### 7. **Backward Compatibility**
- Traditional RBA workflows work unchanged
- Gradual migration path from RBA → RBIA
- No breaking changes to existing DSL v1 syntax
