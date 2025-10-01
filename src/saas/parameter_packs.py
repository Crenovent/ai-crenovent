"""
SaaS Industry Parameter Packs
Industry-specific configuration parameters for SaaS RBA workflows
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field, validator
import json

class SaaSParameterCategory(Enum):
    PIPELINE_HYGIENE = "pipeline_hygiene"
    FORECAST_ACCURACY = "forecast_accuracy"
    LEAD_SCORING = "lead_scoring"
    CHURN_PREVENTION = "churn_prevention"
    REVENUE_RECOGNITION = "revenue_recognition"
    SALES_PERFORMANCE = "sales_performance"

class ComplianceFramework(Enum):
    SOX = "sox"
    GDPR = "gdpr"
    DSAR = "dsar"
    CCPA = "ccpa"

class SaaSParameterPack(BaseModel):
    """Base model for SaaS parameter packs"""
    pack_id: str
    name: str
    description: str
    category: SaaSParameterCategory
    version: str = "1.0.0"
    compliance_frameworks: List[ComplianceFramework] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    parameters: Dict[str, Any]
    governance_rules: Dict[str, Any] = {}
    
    class Config:
        use_enum_values = True

class PipelineHygieneParameterPack(SaaSParameterPack):
    """Parameter pack for SaaS Pipeline Hygiene workflows"""
    
    def __init__(self, **data):
        if 'parameters' not in data:
            data['parameters'] = self._default_parameters()
        if 'governance_rules' not in data:
            data['governance_rules'] = self._default_governance_rules()
        super().__init__(**data)
    
    @staticmethod
    def _default_parameters() -> Dict[str, Any]:
        return {
            # Opportunity Management
            "stalled_opportunity_threshold_days": 15,
            "long_stalled_threshold_days": 30,
            "max_stage_duration_days": {
                "qualification": 7,
                "needs_analysis": 14,
                "proposal": 21,
                "negotiation": 30,
                "closed_won": 0,
                "closed_lost": 0
            },
            
            # Pipeline Coverage
            "pipeline_coverage_minimum": 3.0,
            "pipeline_coverage_warning": 2.5,
            "pipeline_coverage_critical": 2.0,
            
            # Hygiene Scoring
            "hygiene_score_threshold": 0.8,
            "hygiene_score_warning": 0.7,
            "hygiene_score_critical": 0.6,
            
            # Activity Tracking
            "minimum_activities_per_week": 2,
            "activity_types_required": ["call", "email", "meeting"],
            "last_activity_threshold_days": 7,
            
            # Deal Size Thresholds
            "small_deal_threshold": 10000,
            "medium_deal_threshold": 50000,
            "large_deal_threshold": 100000,
            "enterprise_deal_threshold": 500000,
            
            # Notification Settings
            "notification_cooldown_hours": 24,
            "escalation_threshold_days": 30,
            "executive_alert_threshold": 0.5,  # Coverage ratio
            
            # Data Quality
            "required_fields": [
                "opportunity_name",
                "account_name", 
                "amount",
                "close_date",
                "stage",
                "owner"
            ],
            "data_quality_threshold": 0.9,
            
            # Forecasting
            "forecast_categories": ["commit", "best_case", "pipeline"],
            "forecast_lock_day": 25,  # Day of month
            "forecast_variance_threshold": 0.15,
            
            # Sales Velocity
            "average_sales_cycle_days": 90,
            "velocity_warning_multiplier": 1.5,
            "velocity_critical_multiplier": 2.0
        }
    
    @staticmethod
    def _default_governance_rules() -> Dict[str, Any]:
        return {
            "approval_required_for": [
                "pipeline_coverage_override",
                "hygiene_threshold_change",
                "notification_disable"
            ],
            "audit_trail_required": True,
            "retention_days": 2555,  # 7 years for SOX
            "encryption_required": True,
            "immutable_logs": True,
            "maker_checker_required": True,
            "override_approval_chain": [
                "sales_manager",
                "sales_director", 
                "cro"
            ],
            "compliance_checks": {
                "sox_validation": True,
                "data_privacy_check": True,
                "audit_readiness": True
            }
        }

class ForecastAccuracyParameterPack(SaaSParameterPack):
    """Parameter pack for SaaS Forecast Accuracy workflows"""
    
    def __init__(self, **data):
        if 'parameters' not in data:
            data['parameters'] = self._default_parameters()
        if 'governance_rules' not in data:
            data['governance_rules'] = self._default_governance_rules()
        super().__init__(**data)
    
    @staticmethod
    def _default_parameters() -> Dict[str, Any]:
        return {
            # Accuracy Targets
            "forecast_accuracy_target": 0.90,  # 90%
            "forecast_accuracy_warning": 0.85,  # 85%
            "forecast_accuracy_critical": 0.75,  # 75%
            
            # Variance Thresholds
            "forecast_variance_threshold": 0.15,  # 15%
            "variance_warning_threshold": 0.10,   # 10%
            "variance_critical_threshold": 0.25,  # 25%
            
            # Time Periods
            "historical_periods_to_analyze": 6,
            "forecast_horizon_months": 3,
            "rolling_average_periods": 4,
            
            # Forecast Categories
            "forecast_categories": {
                "commit": {"weight": 0.9, "confidence": 0.95},
                "best_case": {"weight": 0.7, "confidence": 0.75},
                "pipeline": {"weight": 0.3, "confidence": 0.50}
            },
            
            # Timing
            "forecast_submission_deadline": 25,  # Day of month
            "forecast_lock_buffer_days": 3,
            "early_warning_days": 7,
            
            # Team Performance
            "individual_accuracy_threshold": 0.85,
            "team_accuracy_threshold": 0.90,
            "rep_warning_threshold": 0.80,
            
            # Adjustment Rules
            "max_adjustment_percentage": 0.20,  # 20%
            "adjustment_approval_threshold": 0.10,  # 10%
            "late_adjustment_penalty": 0.05,
            
            # Reporting
            "accuracy_report_frequency": "monthly",
            "trend_analysis_periods": 12,
            "benchmark_comparison": True
        }
    
    @staticmethod
    def _default_governance_rules() -> Dict[str, Any]:
        return {
            "approval_required_for": [
                "forecast_override",
                "accuracy_threshold_change",
                "category_weight_adjustment"
            ],
            "audit_trail_required": True,
            "sox_compliance": True,
            "immutable_forecasts": True,
            "version_control": True,
            "approval_chain": [
                "sales_manager",
                "finance_director",
                "cfo"
            ]
        }

class LeadScoringParameterPack(SaaSParameterPack):
    """Parameter pack for SaaS Lead Scoring workflows"""
    
    def __init__(self, **data):
        if 'parameters' not in data:
            data['parameters'] = self._default_parameters()
        if 'governance_rules' not in data:
            data['governance_rules'] = self._default_governance_rules()
        super().__init__(**data)
    
    @staticmethod
    def _default_parameters() -> Dict[str, Any]:
        return {
            # Scoring Thresholds
            "hot_lead_threshold": 80,
            "warm_lead_threshold": 60,
            "cold_lead_threshold": 30,
            "disqualified_threshold": 20,
            
            # Scoring Weights
            "demographic_weight": 0.4,
            "behavioral_weight": 0.6,
            "firmographic_weight": 0.3,
            "engagement_weight": 0.7,
            
            # Demographic Scoring
            "demographic_criteria": {
                "job_title_scores": {
                    "ceo": 100, "cto": 95, "cfo": 90,
                    "vp": 85, "director": 75, "manager": 65,
                    "individual_contributor": 40
                },
                "company_size_scores": {
                    "enterprise": 100,    # 1000+ employees
                    "mid_market": 80,     # 100-999 employees
                    "smb": 60,           # 10-99 employees
                    "startup": 40        # <10 employees
                },
                "industry_match_scores": {
                    "technology": 100,
                    "financial_services": 90,
                    "healthcare": 85,
                    "manufacturing": 70,
                    "retail": 60,
                    "other": 40
                }
            },
            
            # Behavioral Scoring
            "behavioral_criteria": {
                "website_visit_score": 10,
                "pricing_page_visit_score": 25,
                "demo_request_score": 50,
                "trial_signup_score": 75,
                "whitepaper_download_score": 15,
                "webinar_attendance_score": 30,
                "email_open_score": 5,
                "email_click_score": 15
            },
            
            # Engagement Decay
            "engagement_decay_days": 30,
            "decay_rate": 0.1,  # 10% per month
            
            # Lead Lifecycle
            "lead_aging_threshold_days": 90,
            "nurture_cycle_days": 30,
            "follow_up_frequency_days": 7,
            
            # Assignment Rules
            "auto_assignment_threshold": 70,
            "sdr_assignment_threshold": 50,
            "marketing_nurture_threshold": 30,
            
            # Data Quality
            "required_lead_fields": [
                "first_name", "last_name", "email",
                "company", "job_title"
            ],
            "data_quality_minimum": 0.8
        }
    
    @staticmethod
    def _default_governance_rules() -> Dict[str, Any]:
        return {
            "gdpr_compliance": True,
            "ccpa_compliance": True,
            "consent_required": True,
            "data_retention_days": 1095,  # 3 years
            "right_to_deletion": True,
            "data_portability": True,
            "audit_trail_required": True,
            "approval_required_for": [
                "scoring_model_changes",
                "threshold_adjustments",
                "weight_modifications"
            ]
        }

class SaaSParameterPackManager:
    """Manager for SaaS parameter packs"""
    
    def __init__(self):
        self.parameter_packs = {}
        self._initialize_default_packs()
    
    def _initialize_default_packs(self):
        """Initialize default parameter packs"""
        # Pipeline Hygiene Pack
        pipeline_pack = PipelineHygieneParameterPack(
            pack_id="saas_pipeline_hygiene_v1.0",
            name="SaaS Pipeline Hygiene Parameters",
            description="Standard parameter pack for SaaS pipeline hygiene monitoring",
            category=SaaSParameterCategory.PIPELINE_HYGIENE,
            compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.DSAR]
        )
        self.parameter_packs[pipeline_pack.pack_id] = pipeline_pack
        
        # Forecast Accuracy Pack
        forecast_pack = ForecastAccuracyParameterPack(
            pack_id="saas_forecast_accuracy_v1.0",
            name="SaaS Forecast Accuracy Parameters",
            description="Standard parameter pack for SaaS forecast accuracy monitoring",
            category=SaaSParameterCategory.FORECAST_ACCURACY,
            compliance_frameworks=[ComplianceFramework.SOX]
        )
        self.parameter_packs[forecast_pack.pack_id] = forecast_pack
        
        # Lead Scoring Pack
        lead_scoring_pack = LeadScoringParameterPack(
            pack_id="saas_lead_scoring_v1.0",
            name="SaaS Lead Scoring Parameters",
            description="Standard parameter pack for SaaS lead scoring automation",
            category=SaaSParameterCategory.LEAD_SCORING,
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA]
        )
        self.parameter_packs[lead_scoring_pack.pack_id] = lead_scoring_pack
    
    def get_parameter_pack(self, pack_id: str) -> Optional[SaaSParameterPack]:
        """Get parameter pack by ID"""
        return self.parameter_packs.get(pack_id)
    
    def list_parameter_packs(self, category: Optional[SaaSParameterCategory] = None) -> List[SaaSParameterPack]:
        """List parameter packs, optionally filtered by category"""
        packs = list(self.parameter_packs.values())
        
        if category:
            packs = [pack for pack in packs if pack.category == category]
        
        return packs
    
    def create_custom_parameter_pack(
        self,
        pack_id: str,
        name: str,
        description: str,
        category: SaaSParameterCategory,
        parameters: Dict[str, Any],
        compliance_frameworks: List[ComplianceFramework] = None,
        governance_rules: Dict[str, Any] = None
    ) -> SaaSParameterPack:
        """Create a custom parameter pack"""
        
        custom_pack = SaaSParameterPack(
            pack_id=pack_id,
            name=name,
            description=description,
            category=category,
            parameters=parameters,
            compliance_frameworks=compliance_frameworks or [],
            governance_rules=governance_rules or {}
        )
        
        self.parameter_packs[pack_id] = custom_pack
        return custom_pack
    
    def validate_parameters(self, pack_id: str, workflow_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow parameters against parameter pack"""
        pack = self.get_parameter_pack(pack_id)
        if not pack:
            return {"is_valid": False, "errors": [f"Parameter pack {pack_id} not found"]}
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "missing_parameters": [],
            "invalid_values": []
        }
        
        # Check for missing required parameters
        pack_params = pack.parameters
        for key, default_value in pack_params.items():
            if key not in workflow_parameters:
                validation_result["missing_parameters"].append(key)
                validation_result["warnings"].append(f"Missing parameter '{key}', using default: {default_value}")
        
        # Validate parameter types and ranges
        for key, value in workflow_parameters.items():
            if key in pack_params:
                expected_type = type(pack_params[key])
                if not isinstance(value, expected_type):
                    validation_result["invalid_values"].append({
                        "parameter": key,
                        "expected_type": expected_type.__name__,
                        "actual_type": type(value).__name__,
                        "value": value
                    })
                    validation_result["is_valid"] = False
        
        return validation_result
    
    def merge_parameters(self, pack_id: str, workflow_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Merge workflow parameters with parameter pack defaults"""
        pack = self.get_parameter_pack(pack_id)
        if not pack:
            return workflow_parameters
        
        merged_parameters = pack.parameters.copy()
        merged_parameters.update(workflow_parameters)
        
        return merged_parameters
    
    def export_parameter_pack(self, pack_id: str, file_path: str, format: str = "json"):
        """Export parameter pack to file"""
        pack = self.get_parameter_pack(pack_id)
        if not pack:
            raise ValueError(f"Parameter pack {pack_id} not found")
        
        pack_data = pack.dict()
        
        if format.lower() == "json":
            with open(file_path, 'w') as f:
                json.dump(pack_data, f, indent=2, default=str)
        elif format.lower() == "yaml":
            import yaml
            with open(file_path, 'w') as f:
                yaml.dump(pack_data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_compliance_requirements(self, pack_id: str) -> Dict[str, Any]:
        """Get compliance requirements for a parameter pack"""
        pack = self.get_parameter_pack(pack_id)
        if not pack:
            return {}
        
        return {
            "compliance_frameworks": [cf.value for cf in pack.compliance_frameworks],
            "governance_rules": pack.governance_rules,
            "audit_requirements": {
                "audit_trail_required": pack.governance_rules.get("audit_trail_required", False),
                "retention_days": pack.governance_rules.get("retention_days", 365),
                "encryption_required": pack.governance_rules.get("encryption_required", False),
                "immutable_logs": pack.governance_rules.get("immutable_logs", False)
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize parameter pack manager
    manager = SaaSParameterPackManager()
    
    # Get pipeline hygiene parameter pack
    pipeline_pack = manager.get_parameter_pack("saas_pipeline_hygiene_v1.0")
    print(f"Pipeline Pack: {pipeline_pack.name}")
    print(f"Parameters: {list(pipeline_pack.parameters.keys())}")
    
    # Test parameter validation
    test_params = {
        "stalled_opportunity_threshold_days": 20,
        "pipeline_coverage_minimum": 2.5
    }
    
    validation_result = manager.validate_parameters("saas_pipeline_hygiene_v1.0", test_params)
    print(f"Validation Result: {validation_result}")
    
    # Test parameter merging
    merged_params = manager.merge_parameters("saas_pipeline_hygiene_v1.0", test_params)
    print(f"Merged Parameters Count: {len(merged_params)}")
    
    # Get compliance requirements
    compliance_reqs = manager.get_compliance_requirements("saas_pipeline_hygiene_v1.0")
    print(f"Compliance Frameworks: {compliance_reqs['compliance_frameworks']}")
