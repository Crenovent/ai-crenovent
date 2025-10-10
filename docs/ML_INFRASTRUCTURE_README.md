# RBIA ML Infrastructure

This document describes the Core ML Infrastructure implementation for RBIA (Risk-Based Intelligent Automation), covering ML primitives, node registry, explainability, drift/bias monitoring, and industry overlays.

## Overview

The ML Infrastructure provides a comprehensive foundation for embedding machine learning capabilities into deterministic RBA workflows, enabling the transition from Rule-Based Automation (RBA) to Risk-Based Intelligent Automation (RBIA).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RBIA ML Infrastructure                   │
├─────────────────────────────────────────────────────────────┤
│  ML Primitives    │  Node Registry  │  Explainability      │
│  • predict        │  • Model Mgmt   │  • SHAP/LIME         │
│  • score          │  • Discovery    │  • Feature Importance│
│  • classify       │  • Versioning   │  • Audit Logging     │
│  • explain        │  • Lifecycle    │  • Query Interface   │
├─────────────────────────────────────────────────────────────┤
│  Drift/Bias       │  Industry       │  Governance          │
│  Monitoring       │  Overlays       │  & Compliance        │
│  • Data Drift     │  • SaaS         │  • Evidence Packs    │
│  • Concept Drift  │  • Banking      │  • Audit Trails      │
│  • Prediction     │  • Insurance    │  • Policy Enforcement│
│  • Bias Detection │  • E-commerce   │  • Regulatory        │
│  • Alert System   │  • Financial    │  • Compliance        │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. ML Primitives (Task 4.1.1)

Four core ML operators that extend the workflow DSL:

#### ML Predict Operator (`ml_predict.py`)
- **Purpose**: Model inference for predictions
- **Supports**: Classification, regression, anomaly detection
- **Features**: Confidence scoring, fallback rules, explainability
- **Usage**:
```yaml
- id: "predict_churn"
  type: "ml_predict"
  params:
    model_id: "saas_churn_predictor_v2"
    input_data: "{{ customer_data }}"
    confidence_threshold: 0.8
    explainability_enabled: true
    fallback_enabled: true
```

#### ML Score Operator (`ml_score.py`)
- **Purpose**: Model scoring with thresholds
- **Supports**: Risk scoring, probability scoring, custom ranges
- **Features**: Feature contributions, threshold categorization
- **Usage**:
```yaml
- id: "score_risk"
  type: "ml_score"
  params:
    model_id: "banking_credit_scorer_v2"
    input_data: "{{ application_data }}"
    score_range: [300, 850]
    thresholds:
      excellent: 750
      good: 700
      fair: 650
```

#### ML Classify Operator (`ml_classify.py`)
- **Purpose**: Multi-class classification
- **Supports**: Binary and multi-class classification
- **Features**: Class probabilities, confidence scoring
- **Usage**:
```yaml
- id: "classify_risk_tier"
  type: "ml_classify"
  params:
    model_id: "insurance_risk_assessor_v1"
    input_data: "{{ policy_data }}"
    class_mapping:
      low_risk: "Low Risk"
      high_risk: "High Risk"
```

#### ML Explain Operator (`ml_explain.py`)
- **Purpose**: Model explainability
- **Supports**: SHAP, LIME, gradient, attention, counterfactual
- **Features**: Feature importance, reasoning explanations
- **Usage**:
```yaml
- id: "explain_prediction"
  type: "ml_explain"
  params:
    model_id: "saas_churn_predictor_v2"
    input_data: "{{ customer_data }}"
    explanation_type: "shap"
    explanation_params:
      shap:
        background_samples: 100
        max_evals: 1000
```

### 2. ML Decision Node Schema (Task 4.1.2)

Comprehensive schema defining ML decision nodes:

```json
{
  "node_id": "churn_predictor_node",
  "node_type": "predict",
  "model_id": "saas_churn_predictor_v2",
  "input_schema": {
    "features": [
      {
        "name": "days_since_last_login",
        "type": "integer",
        "required": true,
        "validation_rules": {"min_value": 0}
      }
    ]
  },
  "output_schema": {
    "prediction": {"type": "string", "enum": ["low_risk", "high_risk"]},
    "confidence": {"type": "float", "min": 0.0, "max": 1.0}
  },
  "confidence_threshold": 0.8,
  "fallback_config": {
    "enabled": true,
    "fallback_rules": [...]
  }
}
```

### 3. Node Registry Service (Task 4.1.4)

Central catalog for reusable ML blocks:

#### Features
- Model registration and discovery
- Industry-specific model templates
- Version control and lifecycle management
- Integration with workflow DSL

#### Usage
```python
# Initialize registry
node_registry = NodeRegistryService()

# Register a model
model_config = MLModelConfig(
    model_id="saas_churn_predictor_v2",
    model_name="SaaS Churn Predictor Pro",
    model_type="classification",
    industry="saas",
    input_features=["days_since_last_login", "mrr", ...],
    output_schema={...}
)
await node_registry.register_model(model_config)

# Get model for operator
model = await node_registry.get_model("saas_churn_predictor_v2")
```

### 4. Explainability Service (Task 4.1.8)

Comprehensive explainability logging and querying:

#### Features
- SHAP/LIME value storage
- Feature importance tracking
- Query and retrieval interface
- Integration with governance systems

#### Usage
```python
# Log explanation
log_id = await explainability_service.log_explanation(
    workflow_id="churn_analysis",
    model_id="saas_churn_predictor_v2",
    explanation_type="shap",
    input_features=customer_data,
    prediction="high_risk",
    confidence=0.85,
    explanation_data=shap_data,
    feature_importance=feature_contributions
)

# Query explanations
explanations = await explainability_service.get_explanations_by_workflow(
    workflow_id="churn_analysis",
    tenant_id="saas_company_001"
)
```

### 5. Drift/Bias Monitor (Task 4.1.9)

Real-time model health monitoring:

#### Features
- Data drift detection (Kolmogorov-Smirnov tests)
- Concept drift detection (performance degradation)
- Prediction drift detection (output distribution changes)
- Bias detection (demographic parity, equalized odds, disparate impact)
- Alert generation and management

#### Usage
```python
# Check for data drift
drift_alerts = await drift_bias_monitor.check_data_drift(
    model_id="saas_churn_predictor_v2",
    current_data=current_batch,
    reference_data=reference_batch,
    tenant_id="saas_company_001"
)

# Check for bias
bias_alerts = await drift_bias_monitor.check_bias(
    model_id="saas_churn_predictor_v2",
    predictions=predictions,
    labels=labels,
    protected_attributes={"gender": ["M", "F"], "age_group": ["18-25", "26-35"]},
    tenant_id="saas_company_001"
)
```

### 6. Industry Overlays (Task 4.1.10)

Industry-specific ML models and governance:

#### Supported Industries
- **SaaS**: Churn prediction, usage analytics
- **Banking**: Credit scoring, fraud detection
- **Insurance**: Fraud detection, risk assessment
- **E-commerce**: Recommendations, pricing optimization
- **Financial Services**: Risk assessment, market analysis

#### Usage
```python
# Initialize industry overlays
industry_overlays = IndustryOverlayService(node_registry)

# Register all industry models
await industry_overlays.register_all_overlay_models()

# Get SaaS overlay
saas_overlay = await industry_overlays.get_overlay("saas")

# Validate model compliance
compliance = await industry_overlays.validate_model_compliance(
    model_id="saas_churn_predictor_v2",
    industry="saas"
)
```

## Workflow Integration

### DSL Extension

The workflow DSL has been extended to support ML primitives:

```yaml
workflow_id: "saas_churn_analysis"
automation_type: "RBIA"
steps:
  - id: "predict_churn_risk"
    type: "ml_predict"
    params:
      model_id: "saas_churn_predictor_v2"
      input_data: "{{ customer_data }}"
      confidence_threshold: 0.8
      explainability_enabled: true
    outputs:
      prediction: "churn_prediction"
      confidence: "prediction_confidence"
```

### Governance Integration

All ML operations include comprehensive governance:

- **Evidence Capture**: All predictions, explanations, and decisions logged
- **Audit Trails**: Complete traceability for regulatory compliance
- **Bias Monitoring**: Real-time fairness checks
- **Drift Detection**: Proactive model health monitoring
- **Override Logging**: Manual intervention tracking

## Example Workflow

See `workflows/ml_workflow_example.yaml` for a complete example demonstrating:

1. Data collection and preparation
2. ML prediction with churn risk scoring
3. ML classification for risk tiers
4. ML explanation generation (SHAP)
5. Decision logic based on ML results
6. Action execution with evidence logging

## Integration Example

See `examples/ml_integration_example.py` for a comprehensive demonstration of:

- Service initialization and configuration
- ML operator execution
- Explainability logging and querying
- Drift and bias monitoring
- Industry overlay validation
- Compliance checking

## Database Schema

### Explainability Logs
```sql
CREATE TABLE explainability_logs (
    log_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    explanation_type TEXT NOT NULL,
    input_features TEXT NOT NULL,
    prediction TEXT,
    confidence REAL NOT NULL,
    explanation_data TEXT NOT NULL,
    feature_importance TEXT NOT NULL,
    shap_values TEXT,
    lime_values TEXT,
    tenant_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Drift/Bias Alerts
```sql
CREATE TABLE drift_alerts (
    alert_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    drift_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    drift_score REAL NOT NULL,
    affected_features TEXT NOT NULL,
    description TEXT NOT NULL,
    detected_at TIMESTAMP NOT NULL,
    tenant_id TEXT NOT NULL
);
```

## Configuration

### Model Registry Configuration
```python
# Database path
node_registry = NodeRegistryService(db_path="node_registry.db")

# Model configuration
model_config = MLModelConfig(
    model_id="custom_model_v1",
    model_name="Custom Model",
    model_type="classification",
    industry="custom",
    input_features=["feature1", "feature2"],
    output_schema={"prediction": {"type": "string"}},
    confidence_threshold=0.75
)
```

### Drift/Bias Thresholds
```python
# Customize thresholds
drift_monitor = DriftBiasMonitor()
drift_monitor._drift_thresholds = {
    'data_drift': 0.1,
    'concept_drift': 0.05,
    'prediction_drift': 0.15
}
drift_monitor._bias_thresholds = {
    'demographic_parity': 0.8,
    'equalized_odds': 0.1,
    'disparate_impact': 0.8
}
```

## Security and Compliance

### Data Protection
- All PII encrypted at rest and in transit
- Tenant isolation enforced at database level
- Audit logs for all data access

### Regulatory Compliance
- GDPR: Data minimization and right to explanation
- CCPA: Consumer privacy rights
- SOX: Financial reporting compliance
- HIPAA: Healthcare data protection (Insurance overlay)

### Governance Features
- Evidence pack generation for audits
- Override logging with justifications
- Model performance monitoring
- Bias and fairness tracking

## Performance Considerations

### Optimization
- Model caching for frequently used models
- Batch processing for drift detection
- Asynchronous execution for all operations
- Database indexing for query performance

### Monitoring
- Execution time tracking
- Memory usage monitoring
- Database performance metrics
- Alert thresholds for performance degradation

## Future Enhancements

### Planned Features
- Real-time model serving
- A/B testing framework
- Model versioning and rollback
- Advanced explainability methods
- Cross-industry model sharing

### Integration Points
- MLOps pipeline integration
- CI/CD for model deployment
- External model registry support
- Cloud provider integrations

## Troubleshooting

### Common Issues

1. **Model Not Found**: Check model registration in node registry
2. **Low Confidence**: Adjust confidence thresholds or improve model
3. **Drift Alerts**: Review data quality and model performance
4. **Bias Detection**: Examine protected attributes and fairness metrics

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for troubleshooting
logger = logging.getLogger("dsl.operators")
logger.setLevel(logging.DEBUG)
```

## Support

For technical support and questions:
- Review the integration example
- Check the workflow examples
- Examine the database schemas
- Review the operator documentation

## License

This ML Infrastructure is part of the RBIA system and follows the same licensing terms.
