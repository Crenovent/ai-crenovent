"""
Comprehensive Example: Compiler Hooks, Fallback Logic, and Override Hooks
Demonstrates Tasks 4.1.3, 4.1.5, and 4.1.6
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime, timedelta

# Import DSL components
from dsl.parser import DSLParser, DSLWorkflow
from dsl.operators.base import OperatorContext, OperatorResult
from dsl.operators.ml_predict import MLPredictOperator
from dsl.operators.ml_score import MLScoreOperator
from dsl.operators.ml_classify import MLClassifyOperator
from dsl.operators.ml_explain import MLExplainOperator
from dsl.operators.node_registry_service import NodeRegistryService
from dsl.operators.fallback_service import FallbackService, FallbackRule
from dsl.operators.override_service import OverrideService
from dsl.operators.override_ledger import OverrideLedger
from dsl.compiler.intelligent_node_compiler import IntelligentNodeCompiler, IntelligentNodeConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveRBIAExample:
    """
    Comprehensive example demonstrating:
    - Task 4.1.3: Compiler hooks for intelligent nodes
    - Task 4.1.5: Fallback logic (ML error → rule-only path)
    - Task 4.1.6: Override hooks for ML nodes (manual justifications)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.node_registry = NodeRegistryService()
        self.fallback_service = FallbackService()
        self.override_service = OverrideService()
        self.override_ledger = OverrideLedger()
        self.compiler = IntelligentNodeCompiler(self.node_registry)
        
        # Initialize operators
        self.operators = {
            "ml_predict": MLPredictOperator(),
            "ml_score": MLScoreOperator(),
            "ml_classify": MLClassifyOperator(),
            "ml_explain": MLExplainOperator()
        }
        
        self.logger.info("Comprehensive RBIA Example initialized")
    
    async def setup_example_data(self):
        """Set up example data and configurations"""
        try:
            # Register example ML models
            await self.node_registry.register_model(
                model_id="saas_churn_predictor_v2",
                model_config={
                    "name": "SaaS Churn Predictor v2",
                    "type": "predict",
                    "version": "2.0",
                    "input_features": ["mrr", "usage_frequency", "support_tickets", "last_login_days"],
                    "output_features": ["churn_probability", "churn_segment"],
                    "confidence_threshold": 0.7,
                    "explainability_enabled": True,
                    "fallback_enabled": True
                }
            )
            
            await self.node_registry.register_model(
                model_id="saas_engagement_scorer_v1",
                model_config={
                    "name": "SaaS Engagement Scorer v1",
                    "type": "score",
                    "version": "1.0",
                    "input_features": ["feature_usage", "login_frequency", "support_interactions"],
                    "output_features": ["engagement_score"],
                    "confidence_threshold": 0.6,
                    "explainability_enabled": False,
                    "fallback_enabled": True
                }
            )
            
            # Create fallback rules
            await self.fallback_service.create_fallback_rule(FallbackRule(
                rule_id="saas_low_mrr_fallback",
                rule_name="SaaS Low MRR Fallback",
                rule_type="field_check",
                condition="mrr < 100",
                action={
                    "prediction": "high_risk",
                    "confidence": 0.6,
                    "reason": "Low MRR fallback rule",
                    "method": "rule_based"
                },
                priority=1,
                model_id="saas_churn_predictor_v2"
            ))
            
            await self.fallback_service.create_fallback_rule(FallbackRule(
                rule_id="saas_low_confidence_fallback",
                rule_name="SaaS Low Confidence Fallback",
                rule_type="threshold",
                condition="confidence < 0.5",
                action={
                    "prediction": "unknown",
                    "confidence": 0.3,
                    "reason": "Low confidence fallback",
                    "method": "default_fallback"
                },
                priority=1
            ))
            
            self.logger.info("Example data setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup example data: {e}")
            raise
    
    async def demonstrate_compiler_hooks(self):
        """Demonstrate Task 4.1.3: Compiler hooks for intelligent nodes"""
        self.logger.info("=== Demonstrating Compiler Hooks (Task 4.1.3) ===")
        
        try:
            # Create intelligent node configuration
            node_config = IntelligentNodeConfig(
                node_id="churn_prediction_node",
                node_type="ml_predict",
                model_id="saas_churn_predictor_v2",
                input_mapping={
                    "mrr": "customer_mrr",
                    "usage_frequency": "usage_frequency",
                    "support_tickets": "support_tickets_last_30d",
                    "last_login_days": "last_login_days_ago"
                },
                output_mapping={
                    "churn_probability": "churn_probability",
                    "churn_segment": "churn_segment"
                },
                confidence_threshold=0.7,
                fallback_enabled=True,
                explainability_enabled=True,
                governance_config={
                    "trust_threshold": 0.75,
                    "audit_logging": True
                }
            )
            
            # Compile the intelligent node
            compilation_result = await self.compiler.compile_intelligent_node(
                node_config, 
                {
                    "tenant_id": "tenant_123",
                    "user_id": 456,
                    "workflow_id": "customer_churn_workflow"
                }
            )
            
            if compilation_result.success:
                self.logger.info(f"✅ Successfully compiled intelligent node: {node_config.node_id}")
                self.logger.info(f"   - Compiled operators: {len(compilation_result.compiled_operators)}")
                self.logger.info(f"   - Execution plan steps: {len(compilation_result.execution_plan)}")
                
                # Show execution plan
                for i, step in enumerate(compilation_result.execution_plan):
                    self.logger.info(f"   Step {i+1}: {step['step_id']} ({step['operator_type']})")
            else:
                self.logger.error(f"❌ Failed to compile intelligent node: {compilation_result.error_message}")
            
        except Exception as e:
            self.logger.error(f"Failed to demonstrate compiler hooks: {e}")
    
    async def demonstrate_fallback_logic(self):
        """Demonstrate Task 4.1.5: Fallback logic (ML error → rule-only path)"""
        self.logger.info("=== Demonstrating Fallback Logic (Task 4.1.5) ===")
        
        try:
            # Create operator context
            context = OperatorContext(
                user_id=456,
                tenant_id="tenant_123",
                workflow_id="customer_churn_workflow",
                step_id="churn_prediction_step",
                execution_id="exec_001",
                input_data={
                    "customer_mrr": 50,  # Low MRR to trigger fallback
                    "usage_frequency": 0.3,
                    "support_tickets_last_30d": 5,
                    "last_login_days_ago": 15
                },
                trust_threshold=0.7
            )
            
            # Simulate ML prediction failure (low confidence)
            original_input = {
                "mrr": 50,
                "usage_frequency": 0.3,
                "support_tickets": 5,
                "last_login_days": 15
            }
            
            # Execute fallback
            fallback_result = await self.fallback_service.execute_fallback(
                workflow_id="customer_churn_workflow",
                step_id="churn_prediction_step",
                model_id="saas_churn_predictor_v2",
                trigger_reason="confidence_low",
                original_input=original_input,
                context=context
            )
            
            if fallback_result.success:
                self.logger.info(f"✅ Fallback executed successfully")
                self.logger.info(f"   - Fallback prediction: {fallback_result.output_data.get('prediction')}")
                self.logger.info(f"   - Fallback confidence: {fallback_result.output_data.get('confidence')}")
                self.logger.info(f"   - Fallback reason: {fallback_result.output_data.get('reason')}")
            else:
                self.logger.error(f"❌ Fallback execution failed: {fallback_result.error_message}")
            
            # Get fallback analytics
            analytics = await self.fallback_service.get_fallback_analytics("tenant_123", days=30)
            self.logger.info(f"📊 Fallback Analytics: {analytics}")
            
        except Exception as e:
            self.logger.error(f"Failed to demonstrate fallback logic: {e}")
    
    async def demonstrate_override_hooks(self):
        """Demonstrate Task 4.1.6: Override hooks for ML nodes (manual justifications)"""
        self.logger.info("=== Demonstrating Override Hooks (Task 4.1.6) ===")
        
        try:
            # Create override request
            override_id = await self.override_service.create_override_request(
                workflow_id="customer_churn_workflow",
                step_id="churn_prediction_step",
                model_id="saas_churn_predictor_v2",
                node_id="churn_prediction_node",
                original_prediction={
                    "churn_probability": 0.8,
                    "churn_segment": "high_risk"
                },
                override_prediction={
                    "churn_probability": 0.3,
                    "churn_segment": "low_risk"
                },
                justification="Customer has special circumstances - recent contract renewal and high engagement",
                override_type="manual",
                requested_by=456,
                tenant_id="tenant_123",
                expires_in_hours=24
            )
            
            self.logger.info(f"✅ Created override request: {override_id}")
            
            # Get pending overrides
            pending_overrides = await self.override_service.get_pending_overrides("tenant_123")
            self.logger.info(f"📋 Pending overrides: {len(pending_overrides)}")
            
            # Approve the override
            approval_success = await self.override_service.approve_override(
                override_id=override_id,
                approver_id=789,
                approval_status="approved",
                approval_reason="Override approved based on business context",
                tenant_id="tenant_123"
            )
            
            if approval_success:
                self.logger.info(f"✅ Override approved successfully")
            else:
                self.logger.error(f"❌ Override approval failed")
            
            # Get override ledger
            ledger_entries = await self.override_ledger.get_ledger_entries("tenant_123", limit=10)
            self.logger.info(f"📚 Ledger entries: {len(ledger_entries)}")
            
            # Verify ledger integrity
            integrity_report = await self.override_ledger.verify_ledger_integrity("tenant_123")
            self.logger.info(f"🔒 Ledger integrity: {'✅ Valid' if integrity_report.is_valid else '❌ Invalid'}")
            
            # Get override analytics
            analytics = await self.override_service.get_override_analytics("tenant_123", days=30)
            self.logger.info(f"📊 Override Analytics: {analytics}")
            
        except Exception as e:
            self.logger.error(f"Failed to demonstrate override hooks: {e}")
    
    async def demonstrate_integrated_workflow(self):
        """Demonstrate all three tasks working together in an integrated workflow"""
        self.logger.info("=== Demonstrating Integrated Workflow ===")
        
        try:
            # Create a comprehensive workflow that uses all three features
            workflow_yaml = """
workflow_id: comprehensive_rbia_workflow
name: Comprehensive RBIA Workflow
module: customer_management
automation_type: RBIA
version: 1.0.0
governance:
  policy_pack_id: comprehensive_governance_v1
  trust_score_threshold: 0.75
  evidence_pack_required: true
  industry_overlay: SaaS
metadata:
  description: "Comprehensive workflow demonstrating compiler hooks, fallback logic, and override capabilities"
  owner: "AI Development Team"
  created_date: "2023-10-27"

steps:
  - id: fetch_customer_data
    type: query
    params:
      source: "crm_db"
      query: "SELECT * FROM customers WHERE customer_id = {{ customer_id }}"
      output_fields: ["customer_id", "mrr", "usage_frequency", "support_tickets_last_30d", "last_login_days_ago"]
    outputs:
      customer_data: "query_result"
    next_steps: ["predict_churn_risk"]

  - id: predict_churn_risk
    type: ml_predict
    params:
      model_id: "saas_churn_predictor_v2"
      input_data:
        mrr: "{{ customer_data.mrr }}"
        usage_frequency: "{{ customer_data.usage_frequency }}"
        support_tickets: "{{ customer_data.support_tickets_last_30d }}"
        last_login_days: "{{ customer_data.last_login_days_ago }}"
      confidence_threshold: 0.7
      explainability_enabled: true
      fallback_enabled: true
    outputs:
      churn_probability: "churn_probability"
      churn_segment: "churn_segment"
    governance:
      explainability_required: true
      drift_bias_monitoring_enabled: true
      override_enabled: true
    next_steps: ["score_engagement"]

  - id: score_engagement
    type: ml_score
    params:
      model_id: "saas_engagement_scorer_v1"
      input_data:
        feature_usage: "{{ customer_data.feature_usage }}"
        login_frequency: "{{ customer_data.login_frequency }}"
        support_interactions: "{{ customer_data.support_tickets_last_30d }}"
      confidence_threshold: 0.6
      fallback_enabled: true
    outputs:
      engagement_score: "engagement_score"
    next_steps: ["decide_action"]

  - id: decide_action
    type: decision
    params:
      expression: "churn_probability > 0.7 and engagement_score < 0.5"
      on_true: "trigger_retention_campaign"
      on_false: "log_and_monitor"
    outputs:
      decision_path: "next_step"
    next_steps: []

  - id: trigger_retention_campaign
    type: notify
    params:
      channel: "email"
      recipient: "customer_success_manager@example.com"
      subject: "High Churn Risk Alert for Customer {{ customer_id }}"
      body: "Customer {{ customer_id }} has a high churn risk ({{ churn_probability | round(2) }}). Engagement score: {{ engagement_score | round(2) }}. Please initiate retention campaign."
    next_steps: []

  - id: log_and_monitor
    type: governance
    params:
      action: "log_event"
      event_type: "churn_risk_monitored"
      details:
        customer_id: "{{ customer_id }}"
        churn_probability: "{{ churn_probability }}"
        engagement_score: "{{ engagement_score }}"
    next_steps: []
"""
            
            # Parse the workflow
            parser = DSLParser()
            workflow = parser.parse_yaml(workflow_yaml)
            
            self.logger.info(f"✅ Parsed comprehensive workflow: {workflow.workflow_id}")
            
            # Compile intelligent nodes
            compilation_results = await self.compiler.compile_workflow_intelligent_nodes(
                workflow.steps,
                {
                    "tenant_id": "tenant_123",
                    "user_id": 456,
                    "workflow_id": workflow.workflow_id
                }
            )
            
            self.logger.info(f"✅ Compiled {len(compilation_results)} intelligent nodes")
            
            # Show compilation results
            for step_id, result in compilation_results.items():
                if result.success:
                    self.logger.info(f"   ✅ {step_id}: {len(result.execution_plan)} execution steps")
                else:
                    self.logger.error(f"   ❌ {step_id}: {result.error_message}")
            
        except Exception as e:
            self.logger.error(f"Failed to demonstrate integrated workflow: {e}")
    
    async def run_comprehensive_example(self):
        """Run the complete comprehensive example"""
        try:
            self.logger.info("🚀 Starting Comprehensive RBIA Example")
            
            # Setup
            await self.setup_example_data()
            
            # Demonstrate each task
            await self.demonstrate_compiler_hooks()
            await self.demonstrate_fallback_logic()
            await self.demonstrate_override_hooks()
            await self.demonstrate_integrated_workflow()
            
            self.logger.info("✅ Comprehensive RBIA Example completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive example failed: {e}")

async def main():
    """Main execution function"""
    example = ComprehensiveRBIAExample()
    await example.run_comprehensive_example()

if __name__ == "__main__":
    asyncio.run(main())
