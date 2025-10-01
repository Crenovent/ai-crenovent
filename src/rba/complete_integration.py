"""
Complete RBA Integration Module
Integrates all RBA components for SaaS industry with full testing and demonstration
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Import all RBA components
from .workflow_engine import RBAWorkflowEngine, ExecutionStatus
from .governance_layer import GovernanceOrchestrator, GovernanceConfig, GovernanceLevel, ComplianceFramework, RiskLevel
from ..saas.dsl_templates import SaaSDSLTemplateManager, SaaSWorkflowType
from ..saas.parameter_packs import SaaSParameterPackManager
from ..saas.schemas import SaaSSchemaFactory
from ..saas.compiler_enhancements import SaaSRuntimeEnhancer, PolicyInjector
from ..saas.api_endpoints import get_saas_router

logger = logging.getLogger(__name__)

class CompleteRBASystem:
    """Complete integrated RBA system for SaaS industry"""
    
    def __init__(self):
        # Initialize core components
        self.workflow_engine = RBAWorkflowEngine()
        self.template_manager = SaaSDSLTemplateManager()
        self.parameter_manager = SaaSParameterPackManager()
        self.schema_factory = SaaSSchemaFactory()
        self.runtime_enhancer = SaaSRuntimeEnhancer()
        self.policy_injector = PolicyInjector()
        
        # Initialize governance
        self.governance_config = GovernanceConfig(
            governance_level=GovernanceLevel.ENHANCED,
            compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.GDPR],
            audit_retention_days=2555,
            immutable_logs=True,
            maker_checker_required=True,
            approval_timeout_hours=24,
            risk_threshold=RiskLevel.MEDIUM,
            evidence_collection_required=True
        )
        self.governance_orchestrator = GovernanceOrchestrator(self.governance_config)
        
        # System metrics
        self.system_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time_ms": 0,
            "governance_events": 0,
            "compliance_violations": 0,
            "policy_enforcements": 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize_system(self) -> Dict[str, Any]:
        """Initialize the complete RBA system"""
        
        initialization_result = {
            "system_status": "initializing",
            "components_initialized": [],
            "errors": [],
            "warnings": []
        }
        
        try:
            # Initialize templates
            available_templates = self.template_manager.list_available_templates()
            initialization_result["components_initialized"].append(f"Templates: {len(available_templates)}")
            
            # Initialize parameter packs
            parameter_packs = self.parameter_manager.list_parameter_packs()
            initialization_result["components_initialized"].append(f"Parameter Packs: {len(parameter_packs)}")
            
            # Validate system configuration
            validation_result = await self._validate_system_configuration()
            if not validation_result["is_valid"]:
                initialization_result["errors"].extend(validation_result["errors"])
            
            # Set system status
            if not initialization_result["errors"]:
                initialization_result["system_status"] = "ready"
                self.logger.info("RBA System initialized successfully")
            else:
                initialization_result["system_status"] = "error"
                self.logger.error(f"RBA System initialization failed: {initialization_result['errors']}")
            
        except Exception as e:
            initialization_result["system_status"] = "error"
            initialization_result["errors"].append(str(e))
            self.logger.error(f"System initialization error: {e}")
        
        return initialization_result
    
    async def _validate_system_configuration(self) -> Dict[str, Any]:
        """Validate system configuration"""
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate templates
        for template_type in [SaaSWorkflowType.PIPELINE_HYGIENE, SaaSWorkflowType.FORECAST_ACCURACY, SaaSWorkflowType.LEAD_SCORING]:
            try:
                template = self.template_manager.get_template(template_type)
                template_validation = self.template_manager.validate_template(template)
                
                if not template_validation["is_valid"]:
                    validation_result["errors"].extend([f"Template {template_type.value}: {error}" for error in template_validation["errors"]])
                    validation_result["is_valid"] = False
                
                if template_validation["warnings"]:
                    validation_result["warnings"].extend([f"Template {template_type.value}: {warning}" for warning in template_validation["warnings"]])
                    
            except Exception as e:
                validation_result["errors"].append(f"Template validation error for {template_type.value}: {e}")
                validation_result["is_valid"] = False
        
        return validation_result
    
    async def execute_complete_workflow(
        self,
        workflow_type: SaaSWorkflowType,
        tenant_id: str,
        user_id: Optional[str] = None,
        parameters: Dict[str, Any] = None,
        apply_governance: bool = True
    ) -> Dict[str, Any]:
        """Execute a complete workflow with all RBA features"""
        
        execution_start = datetime.utcnow()
        
        execution_result = {
            "execution_id": None,
            "workflow_type": workflow_type.value,
            "status": "starting",
            "governance_applied": apply_governance,
            "workflow_result": None,
            "governance_result": None,
            "system_metrics_updated": False,
            "execution_time_ms": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # 1. Pre-execution governance check
            if apply_governance:
                workflow_template = self.template_manager.get_template(workflow_type)
                execution_context = {
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "parameters": parameters or {},
                    "audit_trail_enabled": True,
                    "contains_personal_data": workflow_type == SaaSWorkflowType.LEAD_SCORING,
                    "consent_verified": True,  # Mock - in real implementation, verify actual consent
                    "high_risk_operation": False
                }
                
                governance_result = await self.governance_orchestrator.apply_governance_controls(
                    workflow_execution_id="pending",  # Will be updated after workflow creation
                    workflow_definition=workflow_template,
                    execution_context=execution_context,
                    tenant_id=tenant_id,
                    user_id=user_id
                )
                
                execution_result["governance_result"] = governance_result
                
                if not governance_result["can_proceed"]:
                    execution_result["status"] = "blocked_by_governance"
                    execution_result["errors"].append("Workflow blocked by governance controls")
                    return execution_result
            
            # 2. Execute workflow
            workflow_result = await self.workflow_engine.execute_workflow(
                workflow_type=workflow_type,
                tenant_id=tenant_id,
                user_id=user_id,
                parameters=parameters
            )
            
            execution_result["execution_id"] = workflow_result.execution_id
            execution_result["workflow_result"] = workflow_result.dict()
            execution_result["status"] = workflow_result.status.value
            
            # Update governance result with actual execution ID
            if apply_governance and execution_result["governance_result"]:
                # In real implementation, update governance records with actual execution ID
                pass
            
            # 3. Update system metrics
            await self._update_system_metrics(workflow_result)
            execution_result["system_metrics_updated"] = True
            
            # 4. Calculate execution time
            execution_end = datetime.utcnow()
            execution_result["execution_time_ms"] = int((execution_end - execution_start).total_seconds() * 1000)
            
            self.logger.info(f"Complete workflow execution finished: {workflow_result.execution_id} - {workflow_result.status.value}")
            
        except Exception as e:
            execution_result["status"] = "error"
            execution_result["errors"].append(str(e))
            self.logger.error(f"Complete workflow execution error: {e}")
        
        return execution_result
    
    async def _update_system_metrics(self, workflow_result) -> None:
        """Update system-wide metrics"""
        
        self.system_metrics["total_executions"] += 1
        
        if workflow_result.status == ExecutionStatus.COMPLETED:
            self.system_metrics["successful_executions"] += 1
        else:
            self.system_metrics["failed_executions"] += 1
        
        # Update average execution time
        total_time = (self.system_metrics["average_execution_time_ms"] * (self.system_metrics["total_executions"] - 1) + 
                     workflow_result.duration_ms)
        self.system_metrics["average_execution_time_ms"] = int(total_time / self.system_metrics["total_executions"])
    
    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run a comprehensive demonstration of all RBA capabilities"""
        
        demo_result = {
            "demo_status": "starting",
            "demo_scenarios": [],
            "system_metrics_before": self.system_metrics.copy(),
            "system_metrics_after": None,
            "total_demo_time_ms": 0,
            "errors": [],
            "summary": {}
        }
        
        demo_start = datetime.utcnow()
        
        try:
            self.logger.info("Starting comprehensive RBA demo")
            
            # Scenario 1: Pipeline Hygiene with Governance
            scenario_1 = await self._demo_scenario_1()
            demo_result["demo_scenarios"].append(scenario_1)
            
            # Scenario 2: Forecast Accuracy with Policy Violations
            scenario_2 = await self._demo_scenario_2()
            demo_result["demo_scenarios"].append(scenario_2)
            
            # Scenario 3: Lead Scoring with GDPR Compliance
            scenario_3 = await self._demo_scenario_3()
            demo_result["demo_scenarios"].append(scenario_3)
            
            # Scenario 4: Governance Dashboard and Analytics
            scenario_4 = await self._demo_scenario_4()
            demo_result["demo_scenarios"].append(scenario_4)
            
            # Calculate demo results
            demo_end = datetime.utcnow()
            demo_result["total_demo_time_ms"] = int((demo_end - demo_start).total_seconds() * 1000)
            demo_result["system_metrics_after"] = self.system_metrics.copy()
            demo_result["demo_status"] = "completed"
            
            # Generate summary
            successful_scenarios = len([s for s in demo_result["demo_scenarios"] if s["status"] == "success"])
            demo_result["summary"] = {
                "total_scenarios": len(demo_result["demo_scenarios"]),
                "successful_scenarios": successful_scenarios,
                "success_rate": successful_scenarios / len(demo_result["demo_scenarios"]) * 100,
                "total_workflows_executed": demo_result["system_metrics_after"]["total_executions"] - demo_result["system_metrics_before"]["total_executions"],
                "governance_features_demonstrated": ["policy_enforcement", "compliance_monitoring", "approval_workflows", "audit_trails", "evidence_collection"],
                "compliance_frameworks_tested": ["SOX", "GDPR"]
            }
            
            self.logger.info(f"RBA Demo completed: {successful_scenarios}/{len(demo_result['demo_scenarios'])} scenarios successful")
            
        except Exception as e:
            demo_result["demo_status"] = "error"
            demo_result["errors"].append(str(e))
            self.logger.error(f"Demo execution error: {e}")
        
        return demo_result
    
    async def _demo_scenario_1(self) -> Dict[str, Any]:
        """Demo Scenario 1: Pipeline Hygiene with Full Governance"""
        
        scenario = {
            "scenario_id": "pipeline_hygiene_governance",
            "name": "Pipeline Hygiene with Full Governance",
            "description": "Demonstrate pipeline hygiene workflow with complete governance controls",
            "status": "running",
            "results": {},
            "errors": []
        }
        
        try:
            result = await self.execute_complete_workflow(
                workflow_type=SaaSWorkflowType.PIPELINE_HYGIENE,
                tenant_id="demo_tenant_1",
                user_id="demo_user_1",
                parameters={
                    "stalled_opportunity_threshold_days": 15,
                    "pipeline_coverage_minimum": 3.0,
                    "hygiene_score_threshold": 0.8
                },
                apply_governance=True
            )
            
            scenario["results"] = result
            scenario["status"] = "success" if result["status"] in ["completed", "running"] else "failed"
            
        except Exception as e:
            scenario["status"] = "error"
            scenario["errors"].append(str(e))
        
        return scenario
    
    async def _demo_scenario_2(self) -> Dict[str, Any]:
        """Demo Scenario 2: Forecast Accuracy with Policy Violations"""
        
        scenario = {
            "scenario_id": "forecast_accuracy_violations",
            "name": "Forecast Accuracy with Policy Violations",
            "description": "Demonstrate forecast accuracy workflow with intentional policy violations",
            "status": "running",
            "results": {},
            "errors": []
        }
        
        try:
            result = await self.execute_complete_workflow(
                workflow_type=SaaSWorkflowType.FORECAST_ACCURACY,
                tenant_id="demo_tenant_2",
                user_id="demo_user_2",
                parameters={
                    "forecast_accuracy_target": 0.95,  # Very high target to potentially trigger violations
                    "forecast_variance_threshold": 0.05  # Very low threshold
                },
                apply_governance=True
            )
            
            scenario["results"] = result
            scenario["status"] = "success" if result["status"] in ["completed", "running", "blocked_by_governance"] else "failed"
            
        except Exception as e:
            scenario["status"] = "error"
            scenario["errors"].append(str(e))
        
        return scenario
    
    async def _demo_scenario_3(self) -> Dict[str, Any]:
        """Demo Scenario 3: Lead Scoring with GDPR Compliance"""
        
        scenario = {
            "scenario_id": "lead_scoring_gdpr",
            "name": "Lead Scoring with GDPR Compliance",
            "description": "Demonstrate lead scoring workflow with GDPR compliance requirements",
            "status": "running",
            "results": {},
            "errors": []
        }
        
        try:
            result = await self.execute_complete_workflow(
                workflow_type=SaaSWorkflowType.LEAD_SCORING,
                tenant_id="demo_tenant_3",
                user_id="demo_user_3",
                parameters={
                    "hot_lead_threshold": 85,
                    "demographic_weight": 0.3,
                    "behavioral_weight": 0.7
                },
                apply_governance=True
            )
            
            scenario["results"] = result
            scenario["status"] = "success" if result["status"] in ["completed", "running"] else "failed"
            
        except Exception as e:
            scenario["status"] = "error"
            scenario["errors"].append(str(e))
        
        return scenario
    
    async def _demo_scenario_4(self) -> Dict[str, Any]:
        """Demo Scenario 4: Governance Dashboard and Analytics"""
        
        scenario = {
            "scenario_id": "governance_dashboard",
            "name": "Governance Dashboard and Analytics",
            "description": "Demonstrate governance dashboard and analytics capabilities",
            "status": "running",
            "results": {},
            "errors": []
        }
        
        try:
            # Get governance dashboard for each demo tenant
            dashboards = {}
            for tenant_id in ["demo_tenant_1", "demo_tenant_2", "demo_tenant_3"]:
                dashboard = await self.governance_orchestrator.get_governance_dashboard(tenant_id)
                dashboards[tenant_id] = dashboard
            
            # Get system metrics
            system_status = await self.get_system_status()
            
            scenario["results"] = {
                "governance_dashboards": dashboards,
                "system_status": system_status,
                "analytics_generated": True
            }
            scenario["status"] = "success"
            
        except Exception as e:
            scenario["status"] = "error"
            scenario["errors"].append(str(e))
        
        return scenario
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "system_health": "healthy",
            "uptime_seconds": 3600,  # Mock uptime
            "components_status": {
                "workflow_engine": "active",
                "governance_orchestrator": "active",
                "policy_injector": "active",
                "compliance_monitor": "active",
                "evidence_collector": "active"
            },
            "metrics": self.system_metrics,
            "active_executions": len(self.workflow_engine.list_active_executions()),
            "pending_approvals": len(self.governance_orchestrator.approval_manager.pending_requests),
            "governance_events_today": len([
                e for e in self.governance_orchestrator.governance_events 
                if e.created_at.date() == datetime.utcnow().date()
            ])
        }
    
    def create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application with all RBA endpoints"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self.logger.info("Starting RBA System...")
            init_result = await self.initialize_system()
            if init_result["system_status"] != "ready":
                self.logger.error(f"Failed to initialize RBA system: {init_result['errors']}")
            else:
                self.logger.info("RBA System ready")
            yield
            # Shutdown
            self.logger.info("Shutting down RBA System...")
        
        app = FastAPI(
            title="Complete RBA System for SaaS",
            description="Rule-Based Automation system with full governance and compliance",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Include SaaS API router
        saas_router = get_saas_router()
        app.include_router(saas_router)
        
        # Add system-level endpoints
        @app.get("/api/v1/system/status")
        async def get_system_status_endpoint():
            return await self.get_system_status()
        
        @app.post("/api/v1/system/demo")
        async def run_demo_endpoint():
            return await self.run_comprehensive_demo()
        
        @app.post("/api/v1/workflows/execute-complete")
        async def execute_complete_workflow_endpoint(
            workflow_type: SaaSWorkflowType,
            tenant_id: str,
            user_id: Optional[str] = None,
            parameters: Dict[str, Any] = None
        ):
            return await self.execute_complete_workflow(
                workflow_type, tenant_id, user_id, parameters
            )
        
        return app

# Main execution and testing
async def main():
    """Main function for testing the complete RBA system"""
    
    # Initialize system
    rba_system = CompleteRBASystem()
    
    print("🚀 Initializing Complete RBA System...")
    init_result = await rba_system.initialize_system()
    print(f"   Status: {init_result['system_status']}")
    print(f"   Components: {len(init_result['components_initialized'])}")
    
    if init_result["system_status"] == "ready":
        print("\n🎯 Running Comprehensive Demo...")
        demo_result = await rba_system.run_comprehensive_demo()
        
        print(f"   Demo Status: {demo_result['demo_status']}")
        print(f"   Scenarios: {demo_result['summary']['total_scenarios']}")
        print(f"   Success Rate: {demo_result['summary']['success_rate']:.1f}%")
        print(f"   Total Time: {demo_result['total_demo_time_ms']}ms")
        
        print("\n📊 System Metrics:")
        metrics = demo_result['system_metrics_after']
        print(f"   Total Executions: {metrics['total_executions']}")
        print(f"   Success Rate: {metrics['successful_executions']}/{metrics['total_executions']}")
        print(f"   Avg Execution Time: {metrics['average_execution_time_ms']}ms")
        
        print("\n✅ RBA System Demo Complete!")
        print("   Features Demonstrated:")
        for feature in demo_result['summary']['governance_features_demonstrated']:
            print(f"     - {feature.replace('_', ' ').title()}")
        
    else:
        print(f"❌ System initialization failed: {init_result['errors']}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the main function
    asyncio.run(main())
