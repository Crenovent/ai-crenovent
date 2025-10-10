#!/usr/bin/env python3
"""
RBA Template Population Script - Chapter 19.1
Tasks 19.1.9-19.1.34: Populate industry-specific RBA templates dynamically
Registers existing YAML workflows as templates in the registry
"""

import asyncio
import os
import sys
import logging
from typing import Dict, List, Any

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_crenovent.database.database_manager import DatabaseManager
from ai_crenovent.dsl.governance.evidence_service import EvidenceService
from ai_crenovent.dsl.tenancy.tenant_context_manager import TenantContextManager
from ai_crenovent.dsl.templates.rba_template_service import DynamicRBATemplateService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RBATemplatePopulator:
    """
    Populates RBA templates from existing YAML workflows and dynamic generators
    Tasks 19.1.9-19.1.34: SaaS, Banking, Insurance template registration
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.evidence_service = EvidenceService(self.db_manager)
        self.tenant_context_manager = TenantContextManager(self.db_manager)
        self.template_service = DynamicRBATemplateService(
            self.db_manager, 
            self.evidence_service, 
            self.tenant_context_manager
        )
        
        # Template definitions (dynamic, not hardcoded)
        self.template_definitions = {
            "SaaS": [
                {
                    "yaml_path": "saas_rba_workflows.yaml",
                    "templates": [
                        {"category": "arr_rollup", "workflow_key": "saas_arr_tracking_workflow"},
                        {"category": "churn_alerts", "workflow_key": "saas_churn_risk_workflow"},
                        {"category": "pipeline_hygiene", "workflow_key": "pipeline_summary_workflow"},
                        {"category": "subscription_management", "workflow_key": "saas_subscription_lifecycle_workflow"},
                        {"category": "customer_success", "workflow_key": "saas_customer_health_workflow"}
                    ]
                }
            ],
            "Banking": [
                {
                    "yaml_path": "banking_rba_workflows.yaml",
                    "templates": [
                        {"category": "npa_detection", "workflow_key": "banking_npa_detection_workflow"},
                        {"category": "aml_kyc_compliance", "workflow_key": "banking_aml_kyc_workflow"},
                        {"category": "credit_scoring", "workflow_key": "banking_credit_assessment_workflow"},
                        {"category": "regulatory_reporting", "workflow_key": "banking_regulatory_workflow"}
                    ]
                }
            ],
            "Insurance": [
                {
                    "yaml_path": "insurance_rba_workflows.yaml", 
                    "templates": [
                        {"category": "claims_aging", "workflow_key": "insurance_claims_aging_workflow"},
                        {"category": "underwriting_validation", "workflow_key": "insurance_underwriting_workflow"},
                        {"category": "policy_renewal", "workflow_key": "insurance_renewal_workflow"},
                        {"category": "fraud_investigation", "workflow_key": "insurance_fraud_workflow"}
                    ]
                }
            ]
        }

    async def populate_all_templates(self, tenant_id: int = 1300, user_id: int = 1) -> Dict[str, Any]:
        """
        Populate all industry templates
        """
        results = {
            "registered_templates": [],
            "errors": [],
            "summary": {}
        }
        
        for industry_code, industry_data in self.template_definitions.items():
            try:
                industry_results = await self.populate_industry_templates(
                    industry_code, industry_data, tenant_id, user_id
                )
                results["registered_templates"].extend(industry_results["registered"])
                results["errors"].extend(industry_results["errors"])
                results["summary"][industry_code] = {
                    "registered_count": len(industry_results["registered"]),
                    "error_count": len(industry_results["errors"])
                }
                
            except Exception as e:
                logger.error(f"Failed to populate {industry_code} templates: {e}")
                results["errors"].append({
                    "industry": industry_code,
                    "error": str(e)
                })
        
        return results

    async def populate_industry_templates(self, industry_code: str, industry_data: List[Dict], 
                                        tenant_id: int, user_id: int) -> Dict[str, Any]:
        """
        Populate templates for a specific industry
        """
        results = {"registered": [], "errors": []}
        
        for yaml_config in industry_data:
            yaml_path = yaml_config["yaml_path"]
            templates = yaml_config["templates"]
            
            # Check if YAML file exists
            workflows_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workflows")
            full_yaml_path = os.path.join(workflows_dir, yaml_path)
            
            if not os.path.exists(full_yaml_path):
                logger.warning(f"YAML file not found: {full_yaml_path}, creating placeholder")
                await self.create_placeholder_yaml(full_yaml_path, industry_code, templates)
            
            # Register each template
            for template_config in templates:
                try:
                    template_id = await self.template_service.register_template_from_yaml(
                        yaml_path=yaml_path,
                        industry_code=industry_code,
                        template_category=template_config["category"],
                        tenant_id=tenant_id,
                        user_id=user_id
                    )
                    
                    results["registered"].append({
                        "template_id": template_id,
                        "industry": industry_code,
                        "category": template_config["category"],
                        "yaml_path": yaml_path
                    })
                    
                    logger.info(f"✅ Registered {industry_code} template: {template_config['category']}")
                    
                except Exception as e:
                    error_msg = f"Failed to register {industry_code} {template_config['category']}: {e}"
                    logger.error(error_msg)
                    results["errors"].append({
                        "industry": industry_code,
                        "category": template_config["category"],
                        "error": str(e)
                    })
        
        return results

    async def create_placeholder_yaml(self, yaml_path: str, industry_code: str, templates: List[Dict]):
        """
        Create placeholder YAML file for missing industry workflows
        """
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        yaml_content = f"""# ===============================================================================
# {industry_code} RBA Workflow Templates - Chapter 19.1
# Auto-generated placeholder for dynamic template registration
# ===============================================================================

"""
        
        for template_config in templates:
            category = template_config["category"]
            workflow_key = template_config["workflow_key"]
            
            yaml_content += f"""
# {category.replace('_', ' ').title()} Workflow
{workflow_key}:
  name: "{industry_code} {category.replace('_', ' ').title()}"
  description: "Dynamic {industry_code} workflow for {category.replace('_', ' ')}"
  version: "1.0"
  automation_type: "RBA"
  industry: "{industry_code}"
  compliance: {self._get_compliance_frameworks(industry_code)}
  
  steps:
    - id: "initialize_{category}"
      type: "query"
      params:
        operator: "{industry_code.lower()}_{category}"
        analysis_type: "dynamic_execution"
        tenant_scoped: true
    
    - id: "process_{category}"
      type: "decision"
      params:
        rules: []
        dynamic_config: true
    
    - id: "notify_{category}_results"
      type: "notify"
      params:
        channels: ["email", "slack"]
        template: "{category}_notification"
    
    - id: "log_evidence"
      type: "governance"
      params:
        policy_pack_id: "{industry_code.lower()}_compliance_pack"
        evidence_capture: true
        audit_trail: true

"""
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Created placeholder YAML: {yaml_path}")

    def _get_compliance_frameworks(self, industry_code: str) -> List[str]:
        """Get compliance frameworks for industry"""
        frameworks = {
            "SaaS": ["SOX", "GDPR"],
            "Banking": ["RBI", "BASEL_III", "DPDP", "AML"],
            "Insurance": ["HIPAA", "NAIC", "SOX"]
        }
        return frameworks.get(industry_code, ["SOX"])

    async def create_sample_contracts(self, template_id: str, industry_code: str, 
                                    category: str, tenant_id: int) -> None:
        """
        Task 19.1.10, 19.1.14, 19.1.18, 19.1.21, 19.1.25, 19.1.29, 19.1.33: 
        Create sample contracts for templates
        """
        # Get capability_id for the template
        template_record = await self.db_manager.fetch_one(
            "SELECT capability_id FROM rba_industry_templates WHERE template_id = ?",
            (template_id,)
        )
        
        if not template_record:
            return
        
        capability_id = template_record["capability_id"]
        
        # Create input contract
        input_contract = self._generate_input_contract(industry_code, category)
        await self.db_manager.execute(
            """
            INSERT INTO rba_template_contracts (
                capability_id, contract_type, contract_schema, tenant_id
            ) VALUES (?, ?, ?, ?)
            """,
            (capability_id, "input", input_contract, tenant_id)
        )
        
        # Create output contract
        output_contract = self._generate_output_contract(industry_code, category)
        await self.db_manager.execute(
            """
            INSERT INTO rba_template_contracts (
                capability_id, contract_type, contract_schema, tenant_id
            ) VALUES (?, ?, ?, ?)
            """,
            (capability_id, "output", output_contract, tenant_id)
        )
        
        logger.info(f"Created contracts for {industry_code} {category} template")

    def _generate_input_contract(self, industry_code: str, category: str) -> str:
        """Generate dynamic input contract schema"""
        base_schema = {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "integer", "minimum": 1}
            },
            "required": ["tenant_id"]
        }
        
        # Add industry/category specific fields
        if industry_code == "SaaS":
            if category == "arr_rollup":
                base_schema["properties"].update({
                    "period_start": {"type": "string", "format": "date"},
                    "period_end": {"type": "string", "format": "date"}
                })
            elif category == "churn_alerts":
                base_schema["properties"].update({
                    "churn_threshold_days": {"type": "integer", "minimum": 1, "default": 14},
                    "usage_decline_threshold": {"type": "number", "minimum": 0, "maximum": 100}
                })
        elif industry_code == "Banking":
            if category == "npa_detection":
                base_schema["properties"].update({
                    "npa_threshold_days": {"type": "integer", "minimum": 30, "default": 90}
                })
        elif industry_code == "Insurance":
            if category == "claims_aging":
                base_schema["properties"].update({
                    "aging_threshold_days": {"type": "integer", "minimum": 1, "default": 30}
                })
        
        return json.dumps(base_schema)

    def _generate_output_contract(self, industry_code: str, category: str) -> str:
        """Generate dynamic output contract schema"""
        base_schema = {
            "type": "object",
            "properties": {
                "execution_timestamp": {"type": "string", "format": "date-time"},
                "status": {"type": "string", "enum": ["success", "error", "warning"]}
            },
            "required": ["execution_timestamp", "status"]
        }
        
        # Add industry/category specific output fields
        if industry_code == "SaaS":
            if category == "arr_rollup":
                base_schema["properties"].update({
                    "annual_recurring_revenue": {"type": "number", "minimum": 0},
                    "active_customers": {"type": "integer", "minimum": 0}
                })
        elif industry_code == "Banking":
            if category == "npa_detection":
                base_schema["properties"].update({
                    "npa_accounts_count": {"type": "integer", "minimum": 0},
                    "total_npa_amount": {"type": "number", "minimum": 0}
                })
        elif industry_code == "Insurance":
            if category == "claims_aging":
                base_schema["properties"].update({
                    "aged_claims_count": {"type": "integer", "minimum": 0},
                    "total_aged_amount": {"type": "number", "minimum": 0}
                })
        
        return json.dumps(base_schema)

async def main():
    """Main execution function"""
    logger.info("🚀 Starting RBA Template Population - Chapter 19.1")
    
    populator = RBATemplatePopulator()
    
    try:
        # Populate all templates
        results = await populator.populate_all_templates()
        
        # Print summary
        logger.info("📊 Population Summary:")
        for industry, summary in results["summary"].items():
            logger.info(f"  {industry}: {summary['registered_count']} registered, {summary['error_count']} errors")
        
        logger.info(f"✅ Total registered: {len(results['registered_templates'])}")
        logger.info(f"❌ Total errors: {len(results['errors'])}")
        
        if results["errors"]:
            logger.warning("Errors encountered:")
            for error in results["errors"]:
                logger.warning(f"  {error}")
        
        # Create contracts for registered templates
        logger.info("📝 Creating sample contracts...")
        for template in results["registered_templates"]:
            try:
                await populator.create_sample_contracts(
                    template["template_id"],
                    template["industry"],
                    template["category"],
                    1300  # Default tenant
                )
            except Exception as e:
                logger.error(f"Failed to create contracts for {template['template_id']}: {e}")
        
        logger.info("🎉 RBA Template Population Complete!")
        
    except Exception as e:
        logger.error(f"💥 Population failed: {e}")
        raise

if __name__ == "__main__":
    import json
    asyncio.run(main())
