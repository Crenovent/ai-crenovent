"""
Comprehensive Template System Example
Demonstrates all implemented features for tasks 4.2.3-4.2.27
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Import all template components
from dsl.templates.industry_template_registry import IndustryTemplateRegistry
from dsl.templates.template_explainability_system import TemplateExplainabilitySystem
from dsl.templates.template_confidence_manager import TemplateConfidenceManager
from dsl.templates.conversational_deployment import ConversationalTemplateDeployment
from dsl.templates.shadow_mode_system import ShadowModeSystem, ShadowMode
from dsl.templates.template_data_simulator import TemplateDataSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveTemplateExample:
    """
    Comprehensive example demonstrating all template system features:
    
    ✅ 4.2.3-4.2.12: Industry-specific templates with ML models
    ✅ 4.2.13: Explainability hooks (SHAP/LIME)
    ✅ 4.2.14: Confidence thresholds enforcement
    ✅ 4.2.17: Conversational Mode support
    ✅ 4.2.22: Shadow mode (ML vs RBA comparison)
    ✅ 4.2.27: Sample data simulators
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize all systems
        self.template_registry = IndustryTemplateRegistry()
        self.explainability_system = TemplateExplainabilitySystem()
        self.confidence_manager = TemplateConfidenceManager()
        self.conversational_deployment = ConversationalTemplateDeployment()
        self.shadow_mode_system = ShadowModeSystem()
        self.data_simulator = TemplateDataSimulator()
        
        self.logger.info("Comprehensive Template Example initialized")
    
    async def demonstrate_industry_templates(self):
        """Demonstrate Task 4.2.3-4.2.12: Industry-specific templates"""
        
        self.logger.info("=== Demonstrating Industry-Specific Templates ===")
        
        # Get industry summary
        industry_summary = self.template_registry.get_industry_summary()
        
        self.logger.info(f"📊 Industry Template Summary:")
        for industry, data in industry_summary.items():
            self.logger.info(f"  🏢 {industry}: {data['template_count']} templates")
            for template in data['templates']:
                self.logger.info(f"    • {template['template_name']} ({template['template_type']})")
        
        # Demonstrate specific templates
        templates_to_demo = [
            "saas_churn_risk_alert",
            "banking_credit_scoring_check", 
            "insurance_claim_fraud_anomaly",
            "ecommerce_checkout_fraud_scoring",
            "fs_liquidity_risk_early_warning"
        ]
        
        for template_id in templates_to_demo:
            template = self.template_registry.get_template(template_id)
            if template:
                self.logger.info(f"\n📋 Template: {template.template_name}")
                self.logger.info(f"   Industry: {template.industry}")
                self.logger.info(f"   Type: {template.template_type}")
                self.logger.info(f"   ML Models: {len(template.ml_models)}")
                self.logger.info(f"   Workflow Steps: {len(template.workflow_steps)}")
                self.logger.info(f"   Confidence Thresholds: {template.confidence_thresholds}")
        
        # Generate workflow YAML for a template
        churn_workflow = self.template_registry.generate_workflow_yaml("saas_churn_risk_alert")
        self.logger.info(f"\n🔧 Generated Workflow YAML (first 500 chars):")
        self.logger.info(churn_workflow[:500] + "...")
    
    async def demonstrate_explainability_hooks(self):
        """Demonstrate Task 4.2.13: Explainability hooks (SHAP/LIME)"""
        
        self.logger.info("\n=== Demonstrating Explainability Hooks ===")
        
        # Generate explanation for SaaS churn template
        explanation = await self.explainability_system.generate_template_explanation(
            template_id="saas_churn_risk_alert",
            workflow_id="workflow_123",
            step_id="predict_churn_risk",
            model_id="saas_churn_predictor_v3",
            input_features={
                "customer_tenure_months": 6,
                "mrr": 150,
                "usage_frequency": 0.2,
                "support_tickets_last_30d": 8,
                "last_login_days_ago": 45,
                "feature_adoption_score": 0.3
            },
            prediction_output={
                "churn_probability": 0.85,
                "churn_segment": "high_risk",
                "churn_reason_codes": ["Low usage", "High support tickets", "Recent inactivity"]
            },
            confidence_score=0.82,
            tenant_id="tenant_123"
        )
        
        self.logger.info(f"🔍 Generated Explanation:")
        self.logger.info(f"   ID: {explanation.explanation_id}")
        self.logger.info(f"   Methods: {explanation.explanation_method}")
        self.logger.info(f"   Confidence: {explanation.confidence_score}")
        self.logger.info(f"   Top Features: {list(explanation.feature_importance.keys())[:5]}")
        self.logger.info(f"   Text: {explanation.explanation_text[:200]}...")
        
        if explanation.customer_summary:
            self.logger.info(f"   Customer Summary: {explanation.customer_summary[:150]}...")
        
        if explanation.regulatory_summary:
            self.logger.info(f"   Regulatory Summary: {explanation.regulatory_summary[:150]}...")
        
        # Generate explanation report
        report = await self.explainability_system.generate_explanation_report(
            template_id="saas_churn_risk_alert",
            workflow_id="workflow_123",
            format="json"
        )
        
        if not isinstance(report, dict) or "error" not in report:
            self.logger.info(f"📊 Explanation Report: {len(report.get('explanations', []))} explanations")
    
    async def demonstrate_confidence_thresholds(self):
        """Demonstrate Task 4.2.14: Confidence thresholds enforcement"""
        
        self.logger.info("\n=== Demonstrating Confidence Thresholds ===")
        
        # Test different confidence scenarios
        test_scenarios = [
            {"confidence": 0.85, "description": "High confidence - should pass"},
            {"confidence": 0.65, "description": "Medium confidence - may trigger fallback"},
            {"confidence": 0.45, "description": "Low confidence - should trigger action"}
        ]
        
        for scenario in test_scenarios:
            self.logger.info(f"\n🎯 Testing: {scenario['description']}")
            
            # Create mock context
            from dsl.operators.base import OperatorContext
            context = OperatorContext(
                user_id=123,
                tenant_id="tenant_123",
                workflow_id="workflow_123",
                step_id="predict_churn_risk",
                execution_id="exec_123"
            )
            
            evaluation = await self.confidence_manager.evaluate_confidence_threshold(
                template_id="saas_churn_risk_alert",
                step_id="predict_churn_risk",
                model_id="saas_churn_predictor_v3",
                confidence_score=scenario["confidence"],
                prediction_output={"churn_probability": scenario["confidence"]},
                input_features={"mrr": 500, "usage_frequency": 0.6},
                workflow_context={"tenant_id": "tenant_123", "workflow_id": "workflow_123"}
            )
            
            self.logger.info(f"   Threshold Met: {evaluation.threshold_met}")
            self.logger.info(f"   Action Taken: {evaluation.action_taken.value}")
            self.logger.info(f"   Fallback Used: {evaluation.fallback_used}")
            self.logger.info(f"   Human Review: {evaluation.human_review_required}")
        
        # Get performance summary
        performance = await self.confidence_manager.get_performance_summary("saas_churn_risk_alert")
        if performance and "saas_churn_risk_alert" in performance:
            perf_data = performance["saas_churn_risk_alert"]
            self.logger.info(f"\n📈 Performance Summary:")
            self.logger.info(f"   Success Rate: {perf_data['success_rate']:.2%}")
            self.logger.info(f"   Fallback Rate: {perf_data['fallback_rate']:.2%}")
            self.logger.info(f"   Avg Confidence: {perf_data['average_confidence']:.3f}")
    
    async def demonstrate_conversational_mode(self):
        """Demonstrate Task 4.2.17: Conversational Mode support"""
        
        self.logger.info("\n=== Demonstrating Conversational Mode ===")
        
        # Start a conversation
        response = await self.conversational_deployment.start_conversation(
            user_id=123,
            tenant_id="tenant_123",
            initial_message="I need fraud detection for banking"
        )
        
        self.logger.info(f"🤖 Assistant: {response.message[:200]}...")
        self.logger.info(f"   Input Required: {response.input_required}")
        self.logger.info(f"   Next State: {response.next_state}")
        
        conversation_id = response.metadata.get("conversation_id")
        
        # Continue conversation - select template
        if conversation_id:
            response = await self.conversational_deployment.continue_conversation(
                conversation_id=conversation_id,
                user_message="I want the Banking Fraudulent Disbursal Detector"
            )
            
            self.logger.info(f"🤖 Assistant: {response.message[:200]}...")
            
            # Continue with parameter collection
            response = await self.conversational_deployment.continue_conversation(
                conversation_id=conversation_id,
                user_message="Database: postgresql://user:pass@localhost:5432/bank_db, Review Queue: fraud_review"
            )
            
            self.logger.info(f"🤖 Assistant: {response.message[:200]}...")
            
            # Get conversation context
            context = self.conversational_deployment.get_conversation_context(conversation_id)
            if context:
                self.logger.info(f"📝 Conversation State: {context.state}")
                self.logger.info(f"   Selected Template: {context.selected_template}")
                self.logger.info(f"   Collected Parameters: {len(context.collected_parameters)}")
    
    async def demonstrate_shadow_mode(self):
        """Demonstrate Task 4.2.22: Shadow mode (ML vs RBA comparison)"""
        
        self.logger.info("\n=== Demonstrating Shadow Mode ===")
        
        # Configure shadow mode for a template
        self.shadow_mode_system.configure_shadow_mode(
            template_id="saas_churn_risk_alert",
            shadow_mode=ShadowMode.COMPARATIVE,
            traffic_percentage=100.0
        )
        
        # Simulate RBA logic
        async def mock_rba_logic(input_data: Dict[str, Any], context) -> Dict[str, Any]:
            # Simple rule-based churn detection
            mrr = input_data.get("mrr", 0)
            usage = input_data.get("usage_frequency", 1.0)
            last_login = input_data.get("last_login_days_ago", 0)
            
            risk_score = 0
            if mrr < 200:
                risk_score += 0.3
            if usage < 0.3:
                risk_score += 0.4
            if last_login > 30:
                risk_score += 0.3
            
            return {
                "success": True,
                "result": {
                    "churn_risk": "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low",
                    "risk_score": risk_score
                },
                "execution_time_ms": 50
            }
        
        # Create mock context
        from dsl.operators.base import OperatorContext
        context = OperatorContext(
            user_id=123,
            tenant_id="tenant_123",
            workflow_id="workflow_123",
            step_id="predict_churn_risk",
            execution_id="exec_123"
        )
        
        # Execute shadow mode
        test_inputs = [
            {"mrr": 150, "usage_frequency": 0.2, "last_login_days_ago": 45},  # High risk
            {"mrr": 800, "usage_frequency": 0.8, "last_login_days_ago": 2},   # Low risk
            {"mrr": 400, "usage_frequency": 0.5, "last_login_days_ago": 15}   # Medium risk
        ]
        
        for i, input_data in enumerate(test_inputs):
            self.logger.info(f"\n🔄 Shadow Execution {i+1}:")
            
            result = await self.shadow_mode_system.execute_shadow_mode(
                template_id="saas_churn_risk_alert",
                workflow_id="workflow_123",
                step_id="predict_churn_risk",
                input_data=input_data,
                context=context,
                rba_logic=mock_rba_logic
            )
            
            self.logger.info(f"   Result: {result.get('result', {})}")
            self.logger.info(f"   Shadow Mode: {result.get('shadow_mode', {})}")
        
        # Analyze shadow performance
        metrics = await self.shadow_mode_system.analyze_shadow_performance("saas_churn_risk_alert", days=7)
        
        self.logger.info(f"\n📊 Shadow Mode Metrics:")
        self.logger.info(f"   Total Executions: {metrics.total_executions}")
        self.logger.info(f"   Agreement Rate: {metrics.agreement_rate:.2%}")
        self.logger.info(f"   ML Error Rate: {metrics.ml_error_rate:.2%}")
        self.logger.info(f"   Performance Ratio: {metrics.performance_ratio:.2f}")
        self.logger.info(f"   Recommendation: {metrics.recommendation}")
    
    async def demonstrate_sample_data_simulators(self):
        """Demonstrate Task 4.2.27: Sample data simulators"""
        
        self.logger.info("\n=== Demonstrating Sample Data Simulators ===")
        
        # Simulate data for different templates and scenarios
        templates_to_simulate = [
            ("saas_churn_risk_alert", ["default", "high_churn_risk", "healthy_customers"]),
            ("banking_credit_scoring_check", ["default", "high_creditworthy", "risky_applicants"]),
            ("ecommerce_checkout_fraud_scoring", ["default", "normal_shopping", "fraud_attempts"])
        ]
        
        for template_id, scenarios in templates_to_simulate:
            self.logger.info(f"\n📊 Simulating data for: {template_id}")
            
            for scenario in scenarios:
                try:
                    simulation = await self.data_simulator.simulate_data(
                        template_id=template_id,
                        record_count=100,
                        scenario=scenario,
                        include_quality_issues=True
                    )
                    
                    self.logger.info(f"   Scenario '{scenario}':")
                    self.logger.info(f"     Records: {simulation.record_count}")
                    self.logger.info(f"     Quality Score: {simulation.quality_metrics['overall_quality_score']:.3f}")
                    self.logger.info(f"     Missing Rate: {simulation.quality_metrics['missing_rate']:.3f}")
                    
                    # Show sample record
                    if simulation.data:
                        sample_record = simulation.data[0]
                        sample_fields = list(sample_record.keys())[:5]
                        self.logger.info(f"     Sample Fields: {sample_fields}")
                    
                    # Validate quality
                    validation = await self.data_simulator.validate_simulation_quality(simulation)
                    self.logger.info(f"     Quality Check: {'✅ Passed' if validation['passed'] else '❌ Failed'}")
                    
                except Exception as e:
                    self.logger.error(f"   Failed to simulate {scenario}: {e}")
        
        # Demonstrate data export
        try:
            simulation = await self.data_simulator.simulate_data("saas_churn_risk_alert", 10, "default")
            json_export = self.data_simulator.export_simulation_data(simulation, "json")
            csv_export = self.data_simulator.export_simulation_data(simulation, "csv")
            
            self.logger.info(f"\n📤 Data Export:")
            self.logger.info(f"   JSON Export: {len(json_export)} characters")
            self.logger.info(f"   CSV Export: {len(csv_export)} characters")
            
        except Exception as e:
            self.logger.error(f"Failed to demonstrate data export: {e}")
    
    async def demonstrate_integrated_workflow(self):
        """Demonstrate all systems working together"""
        
        self.logger.info("\n=== Demonstrating Integrated Workflow ===")
        
        # 1. Generate sample data
        self.logger.info("1️⃣ Generating sample data...")
        simulation = await self.data_simulator.simulate_data(
            template_id="saas_churn_risk_alert",
            record_count=50,
            scenario="high_churn_risk"
        )
        
        # 2. Process data through template with all features
        sample_record = simulation.data[0] if simulation.data else {}
        
        self.logger.info("2️⃣ Processing through template with full feature set...")
        
        # Create context
        from dsl.operators.base import OperatorContext
        context = OperatorContext(
            user_id=123,
            tenant_id="tenant_123",
            workflow_id="integrated_workflow_123",
            step_id="predict_churn_risk",
            execution_id="exec_integrated_123"
        )
        
        # 3. Evaluate confidence threshold
        confidence_evaluation = await self.confidence_manager.evaluate_confidence_threshold(
            template_id="saas_churn_risk_alert",
            step_id="predict_churn_risk", 
            model_id="saas_churn_predictor_v3",
            confidence_score=0.78,
            prediction_output={"churn_probability": 0.78, "churn_segment": "high_risk"},
            input_features=sample_record,
            workflow_context={"tenant_id": "tenant_123", "workflow_id": "integrated_workflow_123"}
        )
        
        self.logger.info(f"   Confidence Evaluation: {confidence_evaluation.action_taken.value}")
        
        # 4. Generate explainability
        if confidence_evaluation.explanation_generated:
            explanation = await self.explainability_system.generate_template_explanation(
                template_id="saas_churn_risk_alert",
                workflow_id="integrated_workflow_123",
                step_id="predict_churn_risk",
                model_id="saas_churn_predictor_v3",
                input_features=sample_record,
                prediction_output={"churn_probability": 0.78, "churn_segment": "high_risk"},
                confidence_score=0.78,
                tenant_id="tenant_123"
            )
            
            self.logger.info(f"   Explanation Generated: {explanation.explanation_id}")
        
        # 5. Show integration summary
        self.logger.info("3️⃣ Integration Summary:")
        self.logger.info(f"   ✅ Industry Templates: 5 industries, 10 templates")
        self.logger.info(f"   ✅ Explainability: SHAP/LIME/Reason codes")
        self.logger.info(f"   ✅ Confidence Management: Dynamic thresholds")
        self.logger.info(f"   ✅ Conversational Deployment: Natural language")
        self.logger.info(f"   ✅ Shadow Mode: ML vs RBA comparison")
        self.logger.info(f"   ✅ Data Simulation: Realistic test data")
    
    async def run_comprehensive_demo(self):
        """Run the complete comprehensive demonstration"""
        
        try:
            self.logger.info("🚀 Starting Comprehensive Template System Demo")
            self.logger.info("=" * 60)
            
            # Demonstrate each feature
            await self.demonstrate_industry_templates()
            await self.demonstrate_explainability_hooks()
            await self.demonstrate_confidence_thresholds()
            await self.demonstrate_conversational_mode()
            await self.demonstrate_shadow_mode()
            await self.demonstrate_sample_data_simulators()
            await self.demonstrate_integrated_workflow()
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("✅ Comprehensive Template System Demo Completed Successfully!")
            self.logger.info("\n🎯 All Tasks Implemented:")
            self.logger.info("   ✅ 4.2.3-4.2.12: Industry-specific templates with ML models")
            self.logger.info("   ✅ 4.2.13: Explainability hooks (SHAP/LIME)")
            self.logger.info("   ✅ 4.2.14: Confidence thresholds enforcement")
            self.logger.info("   ✅ 4.2.17: Conversational Mode support")
            self.logger.info("   ✅ 4.2.22: Shadow mode (ML vs RBA comparison)")
            self.logger.info("   ✅ 4.2.27: Sample data simulators")
            
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {e}", exc_info=True)

async def main():
    """Main execution function"""
    demo = ComprehensiveTemplateExample()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main())
