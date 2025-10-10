"""
ML Integration Example - Demonstrates Core ML Infrastructure
Shows how to use the new ML primitives in RBIA workflows
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

# Import the ML operators and services
from dsl.operators.ml_predict import MLPredictOperator
from dsl.operators.ml_score import MLScoreOperator
from dsl.operators.ml_classify import MLClassifyOperator
from dsl.operators.ml_explain import MLExplainOperator
from dsl.operators.node_registry_service import NodeRegistryService, MLModelConfig
from dsl.operators.explainability_service import ExplainabilityService
from dsl.operators.drift_bias_monitor import DriftBiasMonitor
from dsl.operators.industry_overlays import IndustryOverlayService
from dsl.operators.base import OperatorContext

async def main():
    """Main example demonstrating ML infrastructure integration"""
    
    print("🚀 RBIA ML Infrastructure Integration Example")
    print("=" * 50)
    
    # Initialize services
    print("\n1. Initializing Services...")
    node_registry = NodeRegistryService()
    explainability_service = ExplainabilityService()
    drift_bias_monitor = DriftBiasMonitor()
    industry_overlays = IndustryOverlayService(node_registry)
    
    # Register industry overlay models
    print("\n2. Registering Industry Overlay Models...")
    overlay_results = await industry_overlays.register_all_overlay_models()
    for industry, success in overlay_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {industry.title()} overlay: {'Success' if success else 'Failed'}")
    
    # Get model registry for operators
    model_registry = await node_registry.get_model_registry_dict()
    
    # Initialize ML operators
    print("\n3. Initializing ML Operators...")
    predict_operator = MLPredictOperator()
    score_operator = MLScoreOperator()
    classify_operator = MLClassifyOperator()
    explain_operator = MLExplainOperator()
    
    # Inject model registry into operators
    predict_operator.set_model_registry(model_registry)
    score_operator.set_model_registry(model_registry)
    classify_operator.set_model_registry(model_registry)
    explain_operator.set_model_registry(model_registry)
    
    print("   ✅ All ML operators initialized")
    
    # Create execution context
    context = OperatorContext(
        user_id=123,
        tenant_id="saas_company_001",
        workflow_id="churn_analysis_workflow",
        step_id="ml_prediction_step",
        execution_id="exec_001",
        input_data={
            "customer_id": "CUST_001",
            "days_since_last_login": 5,
            "monthly_recurring_revenue": 2500,
            "support_tickets_count": 2,
            "feature_usage_score": 0.75,
            "contract_length_months": 12,
            "payment_history_score": 0.9,
            "engagement_score": 0.8,
            "product_usage_frequency": 0.7,
            "customer_success_score": 0.85
        },
        trust_threshold=0.75,
        evidence_required=True
    )
    
    # Example 1: ML Prediction
    print("\n4. Running ML Prediction...")
    prediction_config = {
        "model_id": "saas_churn_predictor_v2",
        "input_data": context.input_data,
        "confidence_threshold": 0.8,
        "explainability_enabled": True,
        "fallback_enabled": True
    }
    
    prediction_result = await predict_operator.execute(context, prediction_config)
    if prediction_result.success:
        print(f"   ✅ Prediction: {prediction_result.output_data['prediction']}")
        print(f"   📊 Confidence: {prediction_result.output_data['confidence']:.3f}")
        print(f"   🔄 Fallback Used: {prediction_result.output_data['fallback_used']}")
    else:
        print(f"   ❌ Prediction failed: {prediction_result.error_message}")
    
    # Example 2: ML Scoring
    print("\n5. Running ML Scoring...")
    scoring_config = {
        "model_id": "saas_churn_predictor_v2",
        "input_data": context.input_data,
        "score_range": [0, 100],
        "thresholds": {"low": 30, "medium": 60, "high": 80},
        "explainability_enabled": True
    }
    
    scoring_result = await score_operator.execute(context, scoring_config)
    if scoring_result.success:
        print(f"   ✅ Score: {scoring_result.output_data['score']:.1f}")
        print(f"   📈 Percentile: {scoring_result.output_data['percentile']:.1f}%")
        print(f"   🏷️  Risk Category: {scoring_result.output_data['threshold_categories']['primary_category']}")
    else:
        print(f"   ❌ Scoring failed: {scoring_result.error_message}")
    
    # Example 3: ML Classification
    print("\n6. Running ML Classification...")
    classification_config = {
        "model_id": "saas_churn_predictor_v2",
        "input_data": context.input_data,
        "confidence_threshold": 0.75,
        "class_mapping": {
            "low_risk": "Low Risk",
            "medium_risk": "Medium Risk", 
            "high_risk": "High Risk",
            "critical_risk": "Critical Risk"
        },
        "explainability_enabled": True
    }
    
    classification_result = await classify_operator.execute(context, classification_config)
    if classification_result.success:
        print(f"   ✅ Predicted Class: {classification_result.output_data['predicted_class']}")
        print(f"   📊 Confidence: {classification_result.output_data['confidence']:.3f}")
        print(f"   🎯 Class Probabilities: {classification_result.output_data['class_probabilities']}")
    else:
        print(f"   ❌ Classification failed: {classification_result.error_message}")
    
    # Example 4: ML Explanation
    print("\n7. Running ML Explanation...")
    explanation_config = {
        "model_id": "saas_churn_predictor_v2",
        "input_data": context.input_data,
        "explanation_type": "shap",
        "explanation_params": {
            "shap": {"background_samples": 100, "max_evals": 1000}
        }
    }
    
    explanation_result = await explain_operator.execute(context, explanation_config)
    if explanation_result.success:
        print(f"   ✅ Explanation Method: {explanation_result.output_data['explanation_type']}")
        print(f"   📊 Confidence: {explanation_result.output_data['confidence']:.3f}")
        print(f"   🔍 Top Features: {list(explanation_result.output_data['feature_importance'].keys())[:3]}")
    else:
        print(f"   ❌ Explanation failed: {explanation_result.error_message}")
    
    # Example 5: Log Explainability Data
    print("\n8. Logging Explainability Data...")
    if explanation_result.success:
        log_id = await explainability_service.log_explanation(
            workflow_id=context.workflow_id,
            step_id=context.step_id,
            execution_id=context.execution_id,
            model_id="saas_churn_predictor_v2",
            model_version="2.0.0",
            explanation_type="shap",
            input_features=context.input_data,
            prediction=prediction_result.output_data.get('prediction'),
            confidence=explanation_result.output_data['confidence'],
            explanation_data=explanation_result.output_data['explanation_data'],
            feature_importance=explanation_result.output_data['feature_importance'],
            tenant_id=context.tenant_id,
            user_id=context.user_id
        )
        print(f"   ✅ Explainability logged with ID: {log_id}")
    
    # Example 6: Drift and Bias Monitoring
    print("\n9. Running Drift and Bias Checks...")
    
    # Simulate some test data for drift detection
    test_data = [
        {"days_since_last_login": 3, "mrr": 2000, "support_tickets_count": 1},
        {"days_since_last_login": 7, "mrr": 3000, "support_tickets_count": 0},
        {"days_since_last_login": 2, "mrr": 1500, "support_tickets_count": 3}
    ]
    
    # Check for data drift
    drift_alerts = await drift_bias_monitor.check_data_drift(
        model_id="saas_churn_predictor_v2",
        current_data=test_data,
        tenant_id=context.tenant_id,
        workflow_id=context.workflow_id,
        step_id=context.step_id
    )
    
    if drift_alerts:
        print(f"   ⚠️  {len(drift_alerts)} drift alerts detected")
        for alert in drift_alerts:
            print(f"      - {alert.drift_type}: {alert.severity} severity")
    else:
        print("   ✅ No drift detected")
    
    # Check for bias (simulate with dummy data)
    predictions = ["high_risk", "low_risk", "medium_risk"]
    labels = ["high_risk", "low_risk", "medium_risk"]
    protected_attributes = {
        "customer_segment": ["enterprise", "smb", "startup"],
        "region": ["us", "eu", "apac"]
    }
    
    bias_alerts = await drift_bias_monitor.check_bias(
        model_id="saas_churn_predictor_v2",
        predictions=predictions,
        labels=labels,
        protected_attributes=protected_attributes,
        tenant_id=context.tenant_id,
        workflow_id=context.workflow_id,
        step_id=context.step_id
    )
    
    if bias_alerts:
        print(f"   ⚠️  {len(bias_alerts)} bias alerts detected")
        for alert in bias_alerts:
            print(f"      - {alert.bias_type}: {alert.severity} severity")
    else:
        print("   ✅ No bias detected")
    
    # Example 7: Industry Overlay Validation
    print("\n10. Validating Model Compliance...")
    compliance_result = await industry_overlays.validate_model_compliance(
        model_id="saas_churn_predictor_v2",
        industry="saas"
    )
    
    if compliance_result["compliant"]:
        print("   ✅ Model is compliant with SaaS industry requirements")
    else:
        print("   ❌ Model compliance issues:")
        for error in compliance_result["errors"]:
            print(f"      - {error}")
    
    if compliance_result["warnings"]:
        print("   ⚠️  Compliance warnings:")
        for warning in compliance_result["warnings"]:
            print(f"      - {warning}")
    
    # Example 8: Query Explainability Data
    print("\n11. Querying Explainability Data...")
    explanations = await explainability_service.get_explanations_by_workflow(
        workflow_id=context.workflow_id,
        tenant_id=context.tenant_id,
        limit=5
    )
    
    print(f"   📊 Found {len(explanations)} explanations for workflow")
    
    # Get feature importance statistics
    feature_stats = await explainability_service.get_feature_importance_stats(
        model_id="saas_churn_predictor_v2",
        tenant_id=context.tenant_id,
        days=30
    )
    
    if feature_stats:
        print(f"   📈 Feature importance analysis:")
        print(f"      - Total explanations: {feature_stats['total_explanations']}")
        print(f"      - Top features: {feature_stats['top_features'][:3]}")
    
    # Example 9: Get Active Alerts
    print("\n12. Checking Active Alerts...")
    active_alerts = await drift_bias_monitor.get_active_alerts(
        tenant_id=context.tenant_id,
        limit=10
    )
    
    if active_alerts:
        print(f"   ⚠️  {len(active_alerts)} active alerts found")
        for alert in active_alerts[:3]:  # Show first 3
            alert_type = type(alert).__name__.replace("Alert", "")
            print(f"      - {alert_type}: {alert.severity} - {alert.description[:50]}...")
    else:
        print("   ✅ No active alerts")
    
    print("\n" + "=" * 50)
    print("🎉 ML Infrastructure Integration Example Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ ML Primitives: predict, score, classify, explain")
    print("✅ Node Registry: Model management and discovery")
    print("✅ Explainability: SHAP/LIME logging and querying")
    print("✅ Drift/Bias Monitoring: Real-time model health checks")
    print("✅ Industry Overlays: SaaS-specific models and governance")
    print("✅ Compliance Validation: Industry requirement checking")
    print("✅ Evidence Logging: Full audit trail for governance")

if __name__ == "__main__":
    asyncio.run(main())
