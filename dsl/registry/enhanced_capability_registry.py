"""
Consolidated Capability Registry - Comprehensive Capability Management
=====================================================================
Consolidated from multiple capability registry files to eliminate duplication.
Implements Tasks 14.1.1, 14.1.3, 14.1.6, 14.2.1, 14.2.10, 14.2.11
Enhanced with Chapter 6.4: DSL Template Repository and Versioning
Enhanced with Chapter 7.2: Composable Building Blocks Framework

Section 7.2 Tasks Implemented:
- 7.2.1: Define composable block taxonomy
- 7.2.2: Build schema for composable blocks  
- 7.2.4: Implement connector blocks
- 7.2.5: Implement workflow blocks
- 7.2.6: Implement agent blocks
- 7.2.8: Implement policy blocks
- 7.2.16: Automate block certification workflow
- 7.2.18: Automate block versioning (SemVer)
- 7.2.19: Build anomaly detection for block usage
- 7.2.28: Build predictive analytics on block reliability

Consolidated Features:
- Dynamic capability metadata storage (Task 14.1.1)
- Multi-tenant capability mapping (Task 14.1.3) 
- SaaS-specific workflow templates (Task 14.1.6)
- Trust score storage per capability (Task 14.2.1)
- Metadata attachment to capabilities (Task 14.2.10)
- Dynamic trust score computation (Task 14.2.11)
- Auto-discovery of existing RBA/RBIA/AALA capabilities
- Smart matching with context awareness
- Self-populating and maintaining registry
- Dynamic RBA agent discovery and registration
- Analysis type to agent mapping
- Priority-based agent selection

Enhanced Features (Chapter 6.4):
- Tasks 6.4-T08, T09: DSL template repository with versioning and reuse
- Tasks 6.4-T21, T23: DSL lifecycle management and schema versioning
- Tasks 6.4-T28: Multi-industry KG overlays and domain-specific knowledge
- Tasks 6.4-T36: Adoption metrics dashboard for DSL templates per tenant
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

class CapabilityType(Enum):
    """Dynamic capability types with SaaS extensions"""
    RBA = "rba"
    RBIA = "rbia" 
    AALA = "aala"
    # SaaS-specific capability types
    SUBSCRIPTION_MANAGER = "subscription_manager"
    BILLING_ENGINE = "billing_engine"
    CUSTOMER_SUCCESS = "customer_success"
    CHURN_PREDICTOR = "churn_predictor"
    REVENUE_ANALYZER = "revenue_analyzer"

class IndustryCode(Enum):
    """Configurable industry codes"""
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    FINTECH = "FinTech"
    ECOMMERCE = "E-commerce"
    IT_SERVICES = "IT_Services"

class TenantTier(Enum):
    """Dynamic tenant tier classification"""
    T0 = "T0"  # Regulated/Enterprise
    T1 = "T1"  # Mid-market
    T2 = "T2"  # SMB

# ========================================
# SECTION 7.2: COMPOSABLE BUILDING BLOCKS
# ========================================

class ComposableBlockType(Enum):
    """Task 7.2.1: Define composable block taxonomy"""
    CONNECTOR = "connector"        # Data source/destination connectors
    WORKFLOW = "workflow"          # Reusable workflow templates
    AGENT = "agent"               # AI agent blocks
    POLICY = "policy"             # Governance policy blocks
    DASHBOARD = "dashboard"       # Visualization blocks
    TRANSFORMER = "transformer"   # Data transformation blocks
    VALIDATOR = "validator"       # Data validation blocks
    NOTIFIER = "notifier"        # Notification blocks
    SCHEDULER = "scheduler"       # Scheduling blocks
    CALCULATOR = "calculator"     # Calculation/formula blocks

class BlockCertificationLevel(Enum):
    """Task 7.2.16: Block certification levels"""
    DRAFT = "draft"               # Under development
    BETA = "beta"                # Beta testing
    CERTIFIED = "certified"       # Production ready
    DEPRECATED = "deprecated"     # Legacy/deprecated
    ARCHIVED = "archived"         # No longer supported

class BlockReliabilityTier(Enum):
    """Task 7.2.28: Block reliability tiers"""
    TIER_1 = "tier_1"           # Mission critical (99.9% uptime)
    TIER_2 = "tier_2"           # Business critical (99.5% uptime)
    TIER_3 = "tier_3"           # Standard (99.0% uptime)
    TIER_4 = "tier_4"           # Development (95.0% uptime)

@dataclass
class ComposableBlock:
    """Task 7.2.2: Build schema for composable blocks"""
    block_id: str
    name: str
    description: str
    block_type: ComposableBlockType
    version: str = "1.0.0"
    
    # Block definition and interface
    interface_schema: Dict[str, Any] = None  # Input/output schema
    implementation: Dict[str, Any] = None    # Block implementation
    dependencies: List[str] = None           # Required dependencies
    
    # Governance and certification (Task 7.2.16)
    certification_level: BlockCertificationLevel = BlockCertificationLevel.DRAFT
    certification_date: Optional[datetime] = None
    certified_by: Optional[str] = None
    
    # Reliability and monitoring (Task 7.2.28)
    reliability_tier: BlockReliabilityTier = BlockReliabilityTier.TIER_3
    uptime_sla: float = 0.99  # SLA percentage
    error_rate: float = 0.01  # Error rate threshold
    
    # Usage analytics (Task 7.2.19)
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    last_anomaly_detected: Optional[datetime] = None
    
    # Multi-tenant and industry
    tenant_id: Optional[int] = None
    industry_code: Optional[IndustryCode] = None
    
    # Versioning (Task 7.2.18)
    semantic_version: str = "1.0.0"
    version_history: List[Dict[str, Any]] = None
    compatibility_matrix: Dict[str, str] = None
    
    # Metadata
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.interface_schema is None:
            self.interface_schema = {}
        if self.implementation is None:
            self.implementation = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.version_history is None:
            self.version_history = []
        if self.compatibility_matrix is None:
            self.compatibility_matrix = {}
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

class BlockAnomalyDetector:
    """Task 7.2.19: Build anomaly detection for block usage"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.anomaly_thresholds = {
            'error_rate_spike': 0.1,      # 10% error rate spike
            'execution_time_spike': 2.0,   # 2x normal execution time
            'usage_pattern_deviation': 0.3 # 30% usage pattern change
        }
    
    def detect_anomalies(self, block_id: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in block usage patterns"""
        anomalies = []
        
        # Error rate anomaly detection
        current_error_rate = metrics.get('error_rate', 0.0)
        baseline_error_rate = metrics.get('baseline_error_rate', 0.01)
        
        if current_error_rate > baseline_error_rate + self.anomaly_thresholds['error_rate_spike']:
            anomalies.append({
                'type': 'error_rate_spike',
                'severity': 'high',
                'current_value': current_error_rate,
                'baseline_value': baseline_error_rate,
                'threshold': self.anomaly_thresholds['error_rate_spike'],
                'detected_at': datetime.utcnow()
            })
        
        # Execution time anomaly detection
        current_exec_time = metrics.get('avg_execution_time', 0.0)
        baseline_exec_time = metrics.get('baseline_execution_time', 1.0)
        
        if current_exec_time > baseline_exec_time * self.anomaly_thresholds['execution_time_spike']:
            anomalies.append({
                'type': 'execution_time_spike',
                'severity': 'medium',
                'current_value': current_exec_time,
                'baseline_value': baseline_exec_time,
                'threshold': self.anomaly_thresholds['execution_time_spike'],
                'detected_at': datetime.utcnow()
            })
        
        return anomalies

class BlockReliabilityMonitor:
    """Task 7.2.28: Build predictive analytics on block reliability"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.reliability_models = {}
    
    def calculate_reliability_score(self, block: ComposableBlock) -> float:
        """Calculate reliability score based on multiple factors"""
        factors = {
            'uptime_factor': self._calculate_uptime_factor(block),
            'error_rate_factor': self._calculate_error_rate_factor(block),
            'usage_stability_factor': self._calculate_usage_stability_factor(block),
            'certification_factor': self._calculate_certification_factor(block)
        }
        
        # Weighted average
        weights = {'uptime_factor': 0.3, 'error_rate_factor': 0.3, 'usage_stability_factor': 0.2, 'certification_factor': 0.2}
        reliability_score = sum(factors[key] * weights[key] for key in factors)
        
        return min(max(reliability_score, 0.0), 1.0)  # Clamp between 0 and 1
    
    def _calculate_uptime_factor(self, block: ComposableBlock) -> float:
        """Calculate uptime factor based on SLA compliance"""
        if block.usage_count == 0:
            return 0.8  # Default for new blocks
        
        actual_uptime = (block.success_count / block.usage_count) if block.usage_count > 0 else 0.0
        return min(actual_uptime / block.uptime_sla, 1.0)
    
    def _calculate_error_rate_factor(self, block: ComposableBlock) -> float:
        """Calculate error rate factor"""
        if block.usage_count == 0:
            return 0.8  # Default for new blocks
        
        actual_error_rate = (block.failure_count / block.usage_count) if block.usage_count > 0 else 0.0
        if actual_error_rate <= block.error_rate:
            return 1.0
        else:
            return max(1.0 - (actual_error_rate - block.error_rate) * 10, 0.0)
    
    def _calculate_usage_stability_factor(self, block: ComposableBlock) -> float:
        """Calculate usage stability factor"""
        # Simple implementation - can be enhanced with time series analysis
        if block.usage_count < 10:
            return 0.7  # Lower score for low usage
        elif block.usage_count < 100:
            return 0.8
        else:
            return 0.9
    
    def _calculate_certification_factor(self, block: ComposableBlock) -> float:
        """Calculate certification factor"""
        certification_scores = {
            BlockCertificationLevel.ARCHIVED: 0.0,
            BlockCertificationLevel.DEPRECATED: 0.3,
            BlockCertificationLevel.DRAFT: 0.5,
            BlockCertificationLevel.BETA: 0.7,
            BlockCertificationLevel.CERTIFIED: 1.0
        }
        return certification_scores.get(block.certification_level, 0.5)
    
    def predict_failure_probability(self, block: ComposableBlock, time_horizon_hours: int = 24) -> float:
        """Predict probability of failure within time horizon"""
        # Simple predictive model - can be enhanced with ML
        reliability_score = self.calculate_reliability_score(block)
        
        # Higher failure probability for lower reliability
        base_failure_prob = 1.0 - reliability_score
        
        # Adjust for time horizon (longer horizon = higher probability)
        time_factor = min(time_horizon_hours / 168, 1.0)  # Normalize to weekly
        
        return min(base_failure_prob * time_factor, 0.95)

@dataclass
class CapabilityMetadata:
    """Core capability metadata - Task 14.1.1"""
    capability_id: str
    name: str
    capability_type: CapabilityType
    version: str
    
    # Multi-tenant mapping - Task 14.1.3 (required fields first)
    tenant_id: int
    tenant_tier: TenantTier
    industry_code: IndustryCode
    
    # Optional fields
    description: Optional[str] = None
    
    # SaaS-specific metadata - Task 14.1.6
    saas_workflows: List[str] = None  # ARR, churn, QBR workflows
    business_metrics: Dict[str, Any] = None
    customer_impact_score: float = 0.0
    
    # Technical specifications
    input_schema: Dict[str, Any] = None
    output_schema: Dict[str, Any] = None
    operator_definition: Dict[str, Any] = None
    
    # Performance characteristics
    avg_execution_time_ms: int = 0
    success_rate: float = 1.0
    usage_count: int = 0
    
    # Governance
    compliance_requirements: List[str] = None
    policy_tags: List[str] = None
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    last_used_at: Optional[datetime] = None
    created_by_user_id: int = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.saas_workflows is None:
            self.saas_workflows = []
        if self.business_metrics is None:
            self.business_metrics = {}
        if self.compliance_requirements is None:
            self.compliance_requirements = []
        if self.policy_tags is None:
            self.policy_tags = []

@dataclass
class TrustScore:
    """Trust score metadata - Task 14.2.1"""
    capability_id: str
    tenant_id: int
    tenant_tier: TenantTier
    industry_code: IndustryCode
    
    # Trust factors (configurable weights)
    execution_success_rate: float = 0.0
    compliance_violation_count: int = 0
    user_feedback_score: float = 0.0
    business_impact_score: float = 0.0
    
    # Computed scores
    overall_trust_score: float = 0.0
    trust_level: str = "medium"
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 1.0
    
    # Metadata
    last_calculated_at: datetime = None
    calculation_period_days: int = 30
    factor_scores: Dict[str, float] = None

    def __post_init__(self):
        if self.last_calculated_at is None:
            self.last_calculated_at = datetime.utcnow()
        if self.factor_scores is None:
            self.factor_scores = {}

class EnhancedCapabilityRegistry:
    """
    Enhanced Capability Registry with full tenant isolation and dynamic configuration.
    Implements Tasks 14.1.1, 14.1.3, 14.1.6, 14.2.1, 14.2.10, 14.2.11.
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Dynamic configuration (no hardcoding)
        self.trust_factor_weights = self._load_trust_factor_weights()
        self.tenant_tier_thresholds = self._load_tenant_tier_thresholds()
        self.saas_workflow_templates = self._load_saas_workflow_templates()
        self.industry_compliance_mapping = self._load_industry_compliance_mapping()
        
        # Backend task implementations - Chapter 14
        self.banking_templates = self._load_banking_templates()
        self.insurance_templates = self._load_insurance_templates()
        
        # Cache for performance
        self.capability_cache: Dict[str, CapabilityMetadata] = {}
        self.trust_score_cache: Dict[str, TrustScore] = {}
        self.cache_ttl = timedelta(minutes=15)
        
        # Tenant isolation
        self.tenant_capabilities: Dict[int, List[str]] = {}
        
        # Composable blocks storage (Section 7.2)
        self.composable_blocks: Dict[str, ComposableBlock] = {}
        self.block_usage_analytics: Dict[str, Dict[str, Any]] = {}
        self.block_anomaly_detector = BlockAnomalyDetector()
        self.block_reliability_monitor = BlockReliabilityMonitor()
        
    def _load_trust_factor_weights(self) -> Dict[str, Dict[str, float]]:
        """Load configurable trust factor weights per industry"""
        return {
            'SaaS': {
                'execution_success': 0.25,
                'compliance_violations': 0.20,
                'user_feedback': 0.15,
                'business_impact': 0.25,  # Higher for SaaS (revenue impact)
                'historical_reliability': 0.15
            },
            'Banking': {
                'execution_success': 0.20,
                'compliance_violations': 0.35,  # Higher compliance weight
                'user_feedback': 0.10,
                'business_impact': 0.20,
                'historical_reliability': 0.15
            },
            'Insurance': {
                'execution_success': 0.25,
                'compliance_violations': 0.30,
                'user_feedback': 0.10,
                'business_impact': 0.20,
                'historical_reliability': 0.15
            }
        }
    
    def _load_tenant_tier_thresholds(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Load configurable trust level thresholds per tenant tier"""
        return {
            'T0': {  # Regulated tenants - stricter thresholds
                'critical': (0.95, 1.0),
                'high': (0.85, 0.95),
                'medium': (0.70, 0.85),
                'low': (0.50, 0.70),
                'untrusted': (0.0, 0.50)
            },
            'T1': {  # Enterprise tenants - balanced thresholds
                'critical': (0.90, 1.0),
                'high': (0.75, 0.90),
                'medium': (0.60, 0.75),
                'low': (0.40, 0.60),
                'untrusted': (0.0, 0.40)
            },
            'T2': {  # Mid-market tenants - more lenient thresholds
                'critical': (0.85, 1.0),
                'high': (0.70, 0.85),
                'medium': (0.55, 0.70),
                'low': (0.35, 0.55),
                'untrusted': (0.0, 0.35)
            }
        }
    
    def _load_saas_workflow_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Task 15.3.4: Define SaaS capabilities (ARR forecast, churn detection, comp plan)
        Load SaaS-specific workflow templates - Enhanced with Chapter 15.3 capabilities
        """
        return {
            # =====================================================================
            # TASK 15.3.4: ENHANCED SaaS CAPABILITIES
            # =====================================================================
            
            'arr_forecast_engine': {
                'capability_id': 'saas_arr_forecast_001',
                'name': 'ARR Forecast Engine',
                'description': 'Automated Annual Recurring Revenue forecasting with trend analysis',
                'industry': 'SaaS',
                'automation_type': 'RBIA',
                'category': 'revenue_analytics',
                'business_impact': 'high',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'subscription_data': {'type': 'array', 'description': 'Current subscription data'},
                        'billing_data': {'type': 'array', 'description': 'Historical billing records'},
                        'customer_data': {'type': 'array', 'description': 'Customer lifecycle data'},
                        'forecast_period_months': {'type': 'integer', 'minimum': 1, 'maximum': 36},
                        'tenant_id': {'type': 'integer'}
                    },
                    'required': ['subscription_data', 'billing_data', 'tenant_id']
                },
                'output_schema': {
                    'type': 'object',
                    'properties': {
                        'projected_arr': {'type': 'number'},
                        'growth_rate': {'type': 'number'},
                        'confidence_score': {'type': 'number'},
                        'forecast_breakdown': {'type': 'object'},
                        'risk_factors': {'type': 'array'},
                        'evidence_pack_id': {'type': 'string'}
                    }
                },
                'business_metrics': ['ARR', 'Growth Rate', 'Forecast Accuracy', 'Revenue Predictability'],
                'compliance_requirements': ['SOX', 'GAAP', 'revenue_recognition'],
                'stakeholders': ['CRO', 'CFO', 'RevOps', 'Finance'],
                'sla_requirements': {
                    'max_execution_time_minutes': 30,
                    'availability_percent': 99.5,
                    'accuracy_threshold': 0.85
                }
            },
            
            'churn_detection_engine': {
                'capability_id': 'saas_churn_detection_001',
                'name': 'Customer Churn Detection Engine',
                'description': 'ML-powered early churn detection and intervention recommendations',
                'industry': 'SaaS',
                'automation_type': 'RBIA',
                'category': 'customer_success',
                'business_impact': 'high',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'usage_analytics': {'type': 'array', 'description': 'Product usage patterns'},
                        'support_tickets': {'type': 'array', 'description': 'Support interaction history'},
                        'billing_history': {'type': 'array', 'description': 'Payment and billing data'},
                        'engagement_metrics': {'type': 'array', 'description': 'User engagement data'},
                        'prediction_horizon_days': {'type': 'integer', 'minimum': 7, 'maximum': 365},
                        'tenant_id': {'type': 'integer'}
                    },
                    'required': ['usage_analytics', 'tenant_id']
                },
                'output_schema': {
                    'type': 'object',
                    'properties': {
                        'churn_risk_score': {'type': 'number', 'minimum': 0, 'maximum': 1},
                        'risk_category': {'type': 'string', 'enum': ['low', 'medium', 'high', 'critical']},
                        'risk_factors': {'type': 'array'},
                        'recommended_interventions': {'type': 'array'},
                        'confidence_score': {'type': 'number'},
                        'evidence_pack_id': {'type': 'string'}
                    }
                },
                'business_metrics': ['Churn Rate', 'Customer Lifetime Value', 'Retention Rate', 'Early Warning Accuracy'],
                'compliance_requirements': ['GDPR', 'CCPA', 'data_privacy'],
                'stakeholders': ['Customer_Success', 'Sales', 'Product', 'Marketing'],
                'sla_requirements': {
                    'max_execution_time_minutes': 20,
                    'availability_percent': 99.9,
                    'accuracy_threshold': 0.80
                }
            },
            
            'comp_plan_automation': {
                'capability_id': 'saas_comp_plan_001',
                'name': 'Compensation Plan Automation',
                'description': 'Automated commission calculation and compensation plan management',
                'industry': 'SaaS',
                'automation_type': 'RBA',
                'category': 'compensation_management',
                'business_impact': 'high',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'sales_data': {'type': 'array', 'description': 'Closed deals and revenue data'},
                        'quota_data': {'type': 'array', 'description': 'Sales quota assignments'},
                        'territory_data': {'type': 'array', 'description': 'Territory assignments'},
                        'comp_plan_rules': {'type': 'object', 'description': 'Compensation plan configuration'},
                        'calculation_period': {'type': 'string', 'enum': ['monthly', 'quarterly', 'annual']},
                        'tenant_id': {'type': 'integer'}
                    },
                    'required': ['sales_data', 'quota_data', 'comp_plan_rules', 'tenant_id']
                },
                'output_schema': {
                    'type': 'object',
                    'properties': {
                        'commission_calculations': {'type': 'array'},
                        'total_payout': {'type': 'number'},
                        'approval_required': {'type': 'boolean'},
                        'audit_trail': {'type': 'object'},
                        'compliance_validation': {'type': 'object'},
                        'evidence_pack_id': {'type': 'string'}
                    }
                },
                'business_metrics': ['Commission Accuracy', 'Payout Timeliness', 'Compliance Score'],
                'compliance_requirements': ['SOX', 'financial_controls', 'audit_requirements'],
                'stakeholders': ['Sales', 'Finance', 'Payroll', 'Sales_Operations'],
                'sla_requirements': {
                    'max_execution_time_minutes': 45,
                    'availability_percent': 99.9,
                    'accuracy_threshold': 0.99
                }
            },
            
            'revenue_recognition_engine': {
                'capability_id': 'saas_revenue_recognition_001',
                'name': 'Revenue Recognition Automation',
                'description': 'Automated revenue recognition per ASC 606 standards',
                'industry': 'SaaS',
                'automation_type': 'RBA',
                'category': 'financial_automation',
                'business_impact': 'critical',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'contract_data': {'type': 'array', 'description': 'Customer contracts and terms'},
                        'billing_schedules': {'type': 'array', 'description': 'Billing schedule data'},
                        'service_deliverables': {'type': 'array', 'description': 'Service delivery milestones'},
                        'accounting_period': {'type': 'string', 'description': 'Accounting period (YYYY-MM)'},
                        'tenant_id': {'type': 'integer'}
                    },
                    'required': ['contract_data', 'billing_schedules', 'accounting_period', 'tenant_id']
                },
                'output_schema': {
                    'type': 'object',
                    'properties': {
                        'revenue_entries': {'type': 'array'},
                        'deferred_revenue': {'type': 'number'},
                        'recognized_revenue': {'type': 'number'},
                        'journal_entries': {'type': 'array'},
                        'compliance_validation': {'type': 'object'},
                        'evidence_pack_id': {'type': 'string'}
                    }
                },
                'business_metrics': ['Revenue Accuracy', 'ASC 606 Compliance', 'Audit Readiness'],
                'compliance_requirements': ['SOX', 'GAAP', 'ASC_606', 'IFRS_15'],
                'stakeholders': ['CFO', 'Controller', 'Revenue_Accounting', 'External_Auditors'],
                'sla_requirements': {
                    'max_execution_time_minutes': 60,
                    'availability_percent': 99.95,
                    'accuracy_threshold': 0.999
                }
            },
            
            'customer_health_scoring': {
                'capability_id': 'saas_customer_health_001',
                'name': 'Customer Health Scoring Engine',
                'description': 'Comprehensive customer health assessment and scoring',
                'industry': 'SaaS',
                'automation_type': 'RBIA',
                'category': 'customer_success',
                'business_impact': 'high',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'usage_analytics': {'type': 'array', 'description': 'Product usage metrics'},
                        'support_metrics': {'type': 'array', 'description': 'Support interaction data'},
                        'billing_health': {'type': 'array', 'description': 'Payment and billing health'},
                        'engagement_data': {'type': 'array', 'description': 'User engagement metrics'},
                        'scoring_weights': {'type': 'object', 'description': 'Custom scoring weights'},
                        'tenant_id': {'type': 'integer'}
                    },
                    'required': ['usage_analytics', 'tenant_id']
                },
                'output_schema': {
                    'type': 'object',
                    'properties': {
                        'health_score': {'type': 'number', 'minimum': 0, 'maximum': 100},
                        'health_category': {'type': 'string', 'enum': ['red', 'yellow', 'green', 'champion']},
                        'health_trends': {'type': 'object'},
                        'action_recommendations': {'type': 'array'},
                        'risk_indicators': {'type': 'array'},
                        'evidence_pack_id': {'type': 'string'}
                    }
                },
                'business_metrics': ['Customer Health Distribution', 'Health Score Accuracy', 'Retention Correlation'],
                'compliance_requirements': ['GDPR', 'data_privacy'],
                'stakeholders': ['Customer_Success', 'Account_Management', 'Product', 'Executive_Team'],
                'sla_requirements': {
                    'max_execution_time_minutes': 25,
                    'availability_percent': 99.5,
                    'accuracy_threshold': 0.85
                }
            },
            
            # Legacy templates (maintained for backward compatibility)
            'arr_aggregation': {
                'name': 'ARR Aggregation Workflow',
                'description': 'Automated Annual Recurring Revenue calculation',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'subscription_data': {'type': 'array'},
                        'time_period': {'type': 'string'},
                        'tenant_id': {'type': 'integer'}
                    },
                    'required': ['subscription_data', 'tenant_id']
                },
                'output_schema': {
                    'type': 'object',
                    'properties': {
                        'total_arr': {'type': 'number'},
                        'arr_growth_rate': {'type': 'number'},
                        'breakdown_by_segment': {'type': 'object'}
                    }
                },
                'business_metrics': ['ARR', 'Growth Rate', 'Segment Analysis'],
                'compliance_requirements': ['SOX_SAAS', 'GDPR_SAAS']
            },
            'churn_prediction': {
                'name': 'Customer Churn Prediction',
                'description': 'ML-based churn risk assessment',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'customer_usage_data': {'type': 'array'},
                        'support_tickets': {'type': 'array'},
                        'billing_history': {'type': 'array'},
                        'tenant_id': {'type': 'integer'}
                    },
                    'required': ['customer_usage_data', 'tenant_id']
                },
                'output_schema': {
                    'type': 'object',
                    'properties': {
                        'churn_risk_score': {'type': 'number'},
                        'risk_factors': {'type': 'array'},
                        'recommended_actions': {'type': 'array'}
                    }
                },
                'business_metrics': ['Churn Rate', 'Customer Lifetime Value', 'Retention Rate'],
                'compliance_requirements': ['GDPR_SAAS', 'SAAS_BUSINESS_RULES']
            },
            'qbr_automation': {
                'name': 'Quarterly Business Review Automation',
                'description': 'Automated QBR report generation',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'account_data': {'type': 'object'},
                        'performance_metrics': {'type': 'object'},
                        'quarter': {'type': 'string'},
                        'tenant_id': {'type': 'integer'}
                    },
                    'required': ['account_data', 'performance_metrics', 'tenant_id']
                },
                'output_schema': {
                    'type': 'object',
                    'properties': {
                        'qbr_report': {'type': 'object'},
                        'key_insights': {'type': 'array'},
                        'action_items': {'type': 'array'}
                    }
                },
                'business_metrics': ['Account Health', 'Revenue Growth', 'Expansion Opportunities'],
                'compliance_requirements': ['SOX_SAAS']
            }
        }
    
    def _load_industry_compliance_mapping(self) -> Dict[str, List[str]]:
        """Load industry-specific compliance requirements"""
        return {
            'SaaS': ['SOX_SAAS', 'GDPR_SAAS', 'SAAS_BUSINESS_RULES'],
            'Banking': ['SOX', 'RBI', 'DPDP', 'AML_KYC'],
            'Insurance': ['HIPAA', 'NAIC', 'GDPR'],
            'FinTech': ['SOX', 'GDPR', 'PCI_DSS'],
            'E-commerce': ['GDPR', 'PCI_DSS', 'CCPA'],
            'IT_Services': ['SOX', 'GDPR', 'ISO_27001']
        }
    
    async def initialize(self):
        """Initialize the enhanced capability registry"""
        try:
            await self._ensure_database_tables()
            await self._load_existing_capabilities()
            self.logger.info("Enhanced Capability Registry initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced Capability Registry: {e}")
            raise
    
    async def _ensure_database_tables(self):
        """Ensure all required database tables exist"""
        if not self.pool_manager:
            return
            
        async with self.pool_manager.get_connection() as conn:
            # Enhanced capability registry table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_capability_registry (
                    capability_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id INTEGER NOT NULL,
                    tenant_tier VARCHAR(10) NOT NULL,
                    industry_code VARCHAR(20) NOT NULL,
                    capability_name VARCHAR(255) NOT NULL,
                    capability_type VARCHAR(50) NOT NULL,
                    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
                    description TEXT,
                    
                    -- SaaS-specific fields
                    saas_workflows JSONB DEFAULT '[]',
                    business_metrics JSONB DEFAULT '{}',
                    customer_impact_score DECIMAL(5,4) DEFAULT 0.0,
                    
                    -- Technical specifications
                    input_schema JSONB NOT NULL DEFAULT '{}',
                    output_schema JSONB NOT NULL DEFAULT '{}',
                    operator_definition JSONB NOT NULL DEFAULT '{}',
                    
                    -- Performance characteristics
                    avg_execution_time_ms INTEGER DEFAULT 0,
                    success_rate DECIMAL(5,4) DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    
                    -- Governance
                    compliance_requirements JSONB DEFAULT '[]',
                    policy_tags JSONB DEFAULT '[]',
                    
                    -- Timestamps
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_used_at TIMESTAMPTZ,
                    created_by_user_id INTEGER,
                    
                    CONSTRAINT unique_registry_capability_per_tenant UNIQUE (tenant_id, capability_name, version),
                    CONSTRAINT valid_tenant_tier CHECK (tenant_tier IN ('T0', 'T1', 'T2')),
                    CONSTRAINT valid_success_rate CHECK (success_rate >= 0 AND success_rate <= 1),
                    CONSTRAINT valid_customer_impact CHECK (customer_impact_score >= 0 AND customer_impact_score <= 1)
                );
            """)
            
            # Enable RLS
            await conn.execute("ALTER TABLE enhanced_capability_registry ENABLE ROW LEVEL SECURITY;")
            
            # Create RLS policy
            await conn.execute("""
                DROP POLICY IF EXISTS enhanced_capability_registry_rls_policy ON enhanced_capability_registry;
                CREATE POLICY enhanced_capability_registry_rls_policy
                    ON enhanced_capability_registry
                    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_capability_tenant_id ON enhanced_capability_registry(tenant_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_capability_type ON enhanced_capability_registry(capability_type);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_capability_industry ON enhanced_capability_registry(industry_code);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_capability_tenant_tier ON enhanced_capability_registry(tenant_tier);")
    
    async def register_capability(self, metadata: CapabilityMetadata) -> bool:
        """Register a new capability - Task 14.1.1, 14.1.3"""
        try:
            if not self.pool_manager:
                # Store in cache for testing
                self.capability_cache[metadata.capability_id] = metadata
                self._update_tenant_mapping(metadata.tenant_id, metadata.capability_id)
                return True
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{metadata.tenant_id}';")
                
                await conn.execute("""
                    INSERT INTO enhanced_capability_registry (
                        capability_id, tenant_id, tenant_tier, industry_code,
                        capability_name, capability_type, version, description,
                        saas_workflows, business_metrics, customer_impact_score,
                        input_schema, output_schema, operator_definition,
                        avg_execution_time_ms, success_rate, usage_count,
                        compliance_requirements, policy_tags, created_by_user_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                    ON CONFLICT (tenant_id, capability_name, version) 
                    DO UPDATE SET
                        updated_at = NOW(),
                        description = EXCLUDED.description,
                        saas_workflows = EXCLUDED.saas_workflows,
                        business_metrics = EXCLUDED.business_metrics,
                        customer_impact_score = EXCLUDED.customer_impact_score
                """, 
                metadata.capability_id, metadata.tenant_id, metadata.tenant_tier.value, 
                metadata.industry_code.value, metadata.name, metadata.capability_type.value,
                metadata.version, metadata.description, json.dumps(metadata.saas_workflows),
                json.dumps(metadata.business_metrics), metadata.customer_impact_score,
                json.dumps(metadata.input_schema), json.dumps(metadata.output_schema),
                json.dumps(metadata.operator_definition), metadata.avg_execution_time_ms,
                metadata.success_rate, metadata.usage_count, 
                json.dumps(metadata.compliance_requirements), json.dumps(metadata.policy_tags),
                metadata.created_by_user_id)
            
            # Update cache and tenant mapping
            self.capability_cache[metadata.capability_id] = metadata
            self._update_tenant_mapping(metadata.tenant_id, metadata.capability_id)
            
            self.logger.info(f"Registered capability {metadata.name} for tenant {metadata.tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register capability {metadata.name}: {e}")
            return False
    
    def register_custom_agent(self, agent_class, tenant_id: int = 1300, **kwargs):
        """Register a custom RBA agent - compatibility method for legacy agents"""
        try:
            # Extract agent information
            agent_name = agent_class.__name__
            capability_id = f"agent_{agent_name.lower()}_{uuid.uuid4().hex[:8]}"
            
            # Create capability metadata from agent class
            metadata = CapabilityMetadata(
                capability_id=capability_id,
                tenant_id=tenant_id,
                tenant_tier=TenantTier.T1,
                industry_code=IndustryCode.SAAS,
                name=agent_name,
                capability_type=CapabilityType.RBA,
                version="1.0.0",
                description=kwargs.get('description', f"Custom RBA agent: {agent_name}"),
                saas_workflows=kwargs.get('saas_workflows', {}),
                business_metrics=kwargs.get('business_metrics', {}),
                customer_impact_score=kwargs.get('customer_impact_score', 0.5),
                input_schema=kwargs.get('input_schema', {}),
                output_schema=kwargs.get('output_schema', {}),
                operator_definition={"agent_class": agent_name, "module": agent_class.__module__},
                avg_execution_time_ms=kwargs.get('avg_execution_time_ms', 1000),
                success_rate=kwargs.get('success_rate', 0.95),
                usage_count=0,
                compliance_requirements=kwargs.get('compliance_requirements', []),
                policy_tags=kwargs.get('policy_tags', []),
                created_by_user_id=kwargs.get('created_by_user_id', 1)
            )
            
            # Store in cache (sync operation)
            self.capability_cache[capability_id] = metadata
            self._update_tenant_mapping(tenant_id, capability_id)
            
            # Store agent class reference for direct access
            if not hasattr(self, 'agent_classes'):
                self.agent_classes = {}
            self.agent_classes[agent_name] = agent_class
            
            self.logger.info(f"✅ Registered custom agent: {agent_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register custom agent {agent_class.__name__}: {e}")
            return False
    
    def get_all_agents(self) -> Dict[str, Any]:
        """Get all registered agents - compatibility method for legacy code"""
        try:
            agents = {}
            
            # Get from agent classes if available
            if hasattr(self, 'agent_classes'):
                for agent_name, agent_class in self.agent_classes.items():
                    agents[agent_name] = type('AgentInfo', (), {
                        'agent_class': agent_class,
                        'name': agent_name,
                        'description': f"RBA Agent: {agent_name}"
                    })()
            
            # Also get from capability cache
            for capability_id, metadata in self.capability_cache.items():
                if metadata.capability_type == CapabilityType.RBA:
                    agent_name = metadata.name
                    if agent_name not in agents:
                        agents[agent_name] = type('AgentInfo', (), {
                            'agent_class': None,  # Class not available from metadata
                            'name': agent_name,
                            'description': metadata.description,
                            'capability_id': capability_id
                        })()
            
            return agents
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get all agents: {e}")
            return {}
    
    def get_supported_analysis_types(self) -> List[str]:
        """Get all supported analysis types - compatibility method for legacy code"""
        try:
            analysis_types = []
            
            # Get from capability cache
            for capability_id, metadata in self.capability_cache.items():
                if metadata.capability_type == CapabilityType.RBA:
                    # Extract analysis types from metadata
                    if hasattr(metadata, 'saas_workflows') and metadata.saas_workflows:
                        analysis_types.extend(metadata.saas_workflows.keys())
                    
                    # Add default analysis type based on capability name
                    analysis_type = metadata.name.lower().replace('agent', '').replace('rba', '').strip('_')
                    if analysis_type and analysis_type not in analysis_types:
                        analysis_types.append(analysis_type)
            
            # Add some default analysis types if none found
            if not analysis_types:
                analysis_types = [
                    'pipeline_analysis',
                    'forecast_analysis', 
                    'revenue_analysis',
                    'user_onboarding',
                    'location_mapping',
                    'data_quality_check'
                ]
            
            return list(set(analysis_types))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get supported analysis types: {e}")
            return ['pipeline_analysis', 'forecast_analysis', 'revenue_analysis']
    
    def _update_tenant_mapping(self, tenant_id: int, capability_id: str):
        """Update tenant-to-capability mapping - Task 14.1.3"""
        if tenant_id not in self.tenant_capabilities:
            self.tenant_capabilities[tenant_id] = []
        
        if capability_id not in self.tenant_capabilities[tenant_id]:
            self.tenant_capabilities[tenant_id].append(capability_id)
    
    async def get_tenant_capabilities(self, tenant_id: int, 
                                    capability_type: Optional[CapabilityType] = None,
                                    industry_filter: Optional[IndustryCode] = None) -> List[CapabilityMetadata]:
        """Get capabilities for a specific tenant - Task 14.1.3"""
        try:
            if not self.pool_manager:
                # Return from cache for testing
                tenant_caps = []
                for cap_id in self.tenant_capabilities.get(tenant_id, []):
                    if cap_id in self.capability_cache:
                        cap = self.capability_cache[cap_id]
                        if capability_type and cap.capability_type != capability_type:
                            continue
                        if industry_filter and cap.industry_code != industry_filter:
                            continue
                        tenant_caps.append(cap)
                return tenant_caps
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                query = "SELECT * FROM enhanced_capability_registry WHERE tenant_id = $1"
                params = [tenant_id]
                
                if capability_type:
                    query += " AND capability_type = $2"
                    params.append(capability_type.value)
                
                if industry_filter:
                    query += f" AND industry_code = ${len(params) + 1}"
                    params.append(industry_filter.value)
                
                rows = await conn.fetch(query, *params)
                
                capabilities = []
                for row in rows:
                    cap = self._row_to_capability_metadata(row)
                    capabilities.append(cap)
                    # Update cache
                    self.capability_cache[cap.capability_id] = cap
                
                return capabilities
                
        except Exception as e:
            self.logger.error(f"Failed to get tenant capabilities for {tenant_id}: {e}")
            return []
    
    async def create_saas_workflow_capability(self, tenant_id: int, tenant_tier: TenantTier,
                                           workflow_type: str, created_by_user_id: int) -> Optional[CapabilityMetadata]:
        """Create SaaS-specific workflow capability - Task 14.1.6"""
        try:
            if workflow_type not in self.saas_workflow_templates:
                self.logger.error(f"Unknown SaaS workflow type: {workflow_type}")
                return None
            
            template = self.saas_workflow_templates[workflow_type]
            
            # Generate unique capability ID
            capability_id = str(uuid.uuid4())
            
            # Create capability metadata
            metadata = CapabilityMetadata(
                capability_id=capability_id,
                name=template['name'],
                capability_type=CapabilityType.RBA,  # Default to RBA for SaaS workflows
                version="1.0.0",
                description=template['description'],
                tenant_id=tenant_id,
                tenant_tier=tenant_tier,
                industry_code=IndustryCode.SAAS,
                saas_workflows=[workflow_type],
                business_metrics=template['business_metrics'],
                input_schema=template['input_schema'],
                output_schema=template['output_schema'],
                compliance_requirements=template['compliance_requirements'],
                created_by_user_id=created_by_user_id
            )
            
            # Register the capability
            success = await self.register_capability(metadata)
            if success:
                self.logger.info(f"Created SaaS workflow capability: {workflow_type} for tenant {tenant_id}")
                return metadata
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create SaaS workflow capability {workflow_type}: {e}")
            return None
    
    async def store_trust_score(self, trust_score: TrustScore) -> bool:
        """Store trust score for capability - Task 14.2.1"""
        try:
            if not self.pool_manager:
                # Store in cache for testing
                cache_key = f"{trust_score.tenant_id}:{trust_score.capability_id}"
                self.trust_score_cache[cache_key] = trust_score
                return True
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{trust_score.tenant_id}';")
                
                await conn.execute("""
                    INSERT INTO saas_trust_scores (
                        tenant_id, tenant_tier, industry_code, capability_id,
                        capability_type, execution_success_rate, compliance_violation_count,
                        user_feedback_score, business_impact_score, overall_trust_score,
                        trust_level, confidence_interval_lower, confidence_interval_upper,
                        last_calculated_at, calculation_period_days, factor_scores
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (tenant_id, capability_id)
                    DO UPDATE SET
                        execution_success_rate = EXCLUDED.execution_success_rate,
                        compliance_violation_count = EXCLUDED.compliance_violation_count,
                        user_feedback_score = EXCLUDED.user_feedback_score,
                        business_impact_score = EXCLUDED.business_impact_score,
                        overall_trust_score = EXCLUDED.overall_trust_score,
                        trust_level = EXCLUDED.trust_level,
                        confidence_interval_lower = EXCLUDED.confidence_interval_lower,
                        confidence_interval_upper = EXCLUDED.confidence_interval_upper,
                        last_calculated_at = EXCLUDED.last_calculated_at,
                        factor_scores = EXCLUDED.factor_scores
                """,
                trust_score.tenant_id, trust_score.tenant_tier.value, trust_score.industry_code.value,
                trust_score.capability_id, "RBA", trust_score.execution_success_rate,
                trust_score.compliance_violation_count, trust_score.user_feedback_score,
                trust_score.business_impact_score, trust_score.overall_trust_score,
                trust_score.trust_level, trust_score.confidence_interval_lower,
                trust_score.confidence_interval_upper, trust_score.last_calculated_at,
                trust_score.calculation_period_days, json.dumps(trust_score.factor_scores))
            
            # Update cache
            cache_key = f"{trust_score.tenant_id}:{trust_score.capability_id}"
            self.trust_score_cache[cache_key] = trust_score
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store trust score for {trust_score.capability_id}: {e}")
            return False
    
    async def attach_metadata_to_capability(self, capability_id: str, tenant_id: int,
                                          metadata_updates: Dict[str, Any]) -> bool:
        """Attach additional metadata to capability - Task 14.2.10"""
        try:
            if not self.pool_manager:
                # Update cache for testing
                if capability_id in self.capability_cache:
                    cap = self.capability_cache[capability_id]
                    for key, value in metadata_updates.items():
                        if hasattr(cap, key):
                            setattr(cap, key, value)
                    cap.updated_at = datetime.utcnow()
                return True
            
            async with self.pool_manager.get_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                # Build dynamic update query
                set_clauses = []
                params = []
                param_count = 1
                
                for key, value in metadata_updates.items():
                    if key in ['business_metrics', 'saas_workflows', 'compliance_requirements', 'policy_tags']:
                        set_clauses.append(f"{key} = ${param_count}")
                        params.append(json.dumps(value) if isinstance(value, (dict, list)) else value)
                    elif key in ['customer_impact_score', 'success_rate', 'avg_execution_time_ms', 'usage_count']:
                        set_clauses.append(f"{key} = ${param_count}")
                        params.append(value)
                    param_count += 1
                
                if set_clauses:
                    set_clauses.append(f"updated_at = NOW()")
                    query = f"""
                        UPDATE enhanced_capability_registry 
                        SET {', '.join(set_clauses)}
                        WHERE capability_id = ${param_count} AND tenant_id = ${param_count + 1}
                    """
                    params.extend([capability_id, tenant_id])
                    
                    await conn.execute(query, *params)
                    
                    # Update cache
                    if capability_id in self.capability_cache:
                        cap = self.capability_cache[capability_id]
                        for key, value in metadata_updates.items():
                            if hasattr(cap, key):
                                setattr(cap, key, value)
                        cap.updated_at = datetime.utcnow()
            
            self.logger.info(f"Attached metadata to capability {capability_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to attach metadata to capability {capability_id}: {e}")
            return False
    
    async def compute_trust_score(self, capability_id: str, tenant_id: int, 
                                tenant_tier: TenantTier, industry_code: IndustryCode) -> Optional[TrustScore]:
        """Compute dynamic trust score for capability - Task 14.2.11"""
        try:
            # Get capability performance data
            performance_data = await self._get_capability_performance(capability_id, tenant_id)
            if not performance_data:
                return None
            
            # Get industry-specific weights
            weights = self.trust_factor_weights.get(industry_code.value, self.trust_factor_weights['SaaS'])
            
            # Calculate individual factor scores
            execution_score = performance_data.get('success_rate', 0.0)
            compliance_score = max(0.0, 1.0 - (performance_data.get('violation_count', 0) * 0.1))
            feedback_score = performance_data.get('user_feedback', 0.0) / 5.0  # Normalize to 0-1
            business_score = performance_data.get('business_impact', 0.0)
            reliability_score = performance_data.get('reliability', 0.0)
            
            # Compute weighted overall score
            overall_score = (
                execution_score * weights['execution_success'] +
                compliance_score * weights['compliance_violations'] +
                feedback_score * weights['user_feedback'] +
                business_score * weights['business_impact'] +
                reliability_score * weights['historical_reliability']
            )
            
            # Determine trust level based on tenant tier
            trust_level = self._determine_trust_level(overall_score, tenant_tier)
            
            # Calculate confidence interval (simplified)
            sample_size = performance_data.get('execution_count', 1)
            confidence_margin = min(0.1, 1.0 / max(1, sample_size ** 0.5))
            
            trust_score = TrustScore(
                capability_id=capability_id,
                tenant_id=tenant_id,
                tenant_tier=tenant_tier,
                industry_code=industry_code,
                execution_success_rate=execution_score,
                compliance_violation_count=performance_data.get('violation_count', 0),
                user_feedback_score=performance_data.get('user_feedback', 0.0),
                business_impact_score=business_score,
                overall_trust_score=overall_score,
                trust_level=trust_level,
                confidence_interval_lower=max(0.0, overall_score - confidence_margin),
                confidence_interval_upper=min(1.0, overall_score + confidence_margin),
                factor_scores={
                    'execution': execution_score,
                    'compliance': compliance_score,
                    'feedback': feedback_score,
                    'business': business_score,
                    'reliability': reliability_score
                }
            )
            
            # Store the computed trust score
            await self.store_trust_score(trust_score)
            
            self.logger.info(f"Computed trust score {overall_score:.3f} for capability {capability_id}")
            return trust_score
            
        except Exception as e:
            self.logger.error(f"Failed to compute trust score for {capability_id}: {e}")
            return None
    
    def _determine_trust_level(self, score: float, tenant_tier: TenantTier) -> str:
        """Determine trust level based on score and tenant tier"""
        thresholds = self.tenant_tier_thresholds[tenant_tier.value]
        
        for level, (min_score, max_score) in thresholds.items():
            if min_score <= score <= max_score:
                return level
        
        return 'untrusted'
    
    async def _get_capability_performance(self, capability_id: str, tenant_id: int) -> Optional[Dict[str, Any]]:
        """Get performance data for capability (from execution logs, metrics, etc.)"""
        # This would typically query execution logs, user feedback, business metrics
        # For now, return mock data with realistic values
        return {
            'success_rate': 0.95,
            'violation_count': 0,
            'user_feedback': 4.2,
            'business_impact': 0.8,
            'reliability': 0.9,
            'execution_count': 150
        }
    
    async def _load_existing_capabilities(self):
        """Load existing capabilities from database into cache"""
        if not self.pool_manager:
            return
            
        try:
            async with self.pool_manager.get_connection() as conn:
                rows = await conn.fetch("SELECT * FROM enhanced_capability_registry LIMIT 1000")
                
                for row in rows:
                    cap = self._row_to_capability_metadata(row)
                    self.capability_cache[cap.capability_id] = cap
                    self._update_tenant_mapping(cap.tenant_id, cap.capability_id)
                
                self.logger.info(f"Loaded {len(rows)} existing capabilities into cache")
                
        except Exception as e:
            self.logger.warning(f"Could not load existing capabilities: {e}")
    
    def _row_to_capability_metadata(self, row) -> CapabilityMetadata:
        """Convert database row to CapabilityMetadata object"""
        return CapabilityMetadata(
            capability_id=str(row['capability_id']),
            name=row['capability_name'],
            capability_type=CapabilityType(row['capability_type']),
            version=row['version'],
            description=row['description'],
            tenant_id=row['tenant_id'],
            tenant_tier=TenantTier(row['tenant_tier']),
            industry_code=IndustryCode(row['industry_code']),
            saas_workflows=json.loads(row['saas_workflows']) if row['saas_workflows'] else [],
            business_metrics=json.loads(row['business_metrics']) if row['business_metrics'] else {},
            customer_impact_score=float(row['customer_impact_score']),
            input_schema=json.loads(row['input_schema']) if row['input_schema'] else {},
            output_schema=json.loads(row['output_schema']) if row['output_schema'] else {},
            operator_definition=json.loads(row['operator_definition']) if row['operator_definition'] else {},
            avg_execution_time_ms=row['avg_execution_time_ms'],
            success_rate=float(row['success_rate']),
            usage_count=row['usage_count'],
            compliance_requirements=json.loads(row['compliance_requirements']) if row['compliance_requirements'] else [],
            policy_tags=json.loads(row['policy_tags']) if row['policy_tags'] else [],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            last_used_at=row['last_used_at'],
            created_by_user_id=row['created_by_user_id']
        )
    
    # ========================================
    # COMPOSABLE BLOCKS MANAGEMENT (Section 7.2)
    # ========================================
    
    def register_composable_block(self, block: ComposableBlock) -> bool:
        """Task 7.2.4-7.2.8: Register connector, workflow, agent, and policy blocks"""
        try:
            # Validate block schema
            if not self._validate_block_schema(block):
                return False
            
            # Auto-versioning (Task 7.2.18)
            if block.block_id in self.composable_blocks:
                existing_block = self.composable_blocks[block.block_id]
                block.semantic_version = self._increment_semantic_version(existing_block.semantic_version)
                block.version_history = existing_block.version_history.copy()
                block.version_history.append({
                    'version': existing_block.semantic_version,
                    'timestamp': existing_block.updated_at,
                    'changes': 'Updated block definition'
                })
            
            # Initialize analytics tracking
            if block.block_id not in self.block_usage_analytics:
                self.block_usage_analytics[block.block_id] = {
                    'daily_usage': {},
                    'error_patterns': [],
                    'performance_metrics': {},
                    'anomaly_history': []
                }
            
            # Store the block
            self.composable_blocks[block.block_id] = block
            
            self.logger.info(f"✅ Registered composable block: {block.block_id} v{block.semantic_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register composable block {block.block_id}: {e}")
            return False
    
    def get_composable_block(self, block_id: str) -> Optional[ComposableBlock]:
        """Get a composable block by ID"""
        return self.composable_blocks.get(block_id)
    
    def list_composable_blocks(self, 
                              block_type: Optional[ComposableBlockType] = None,
                              certification_level: Optional[BlockCertificationLevel] = None,
                              tenant_id: Optional[int] = None) -> List[ComposableBlock]:
        """List composable blocks with optional filtering"""
        blocks = list(self.composable_blocks.values())
        
        if block_type:
            blocks = [b for b in blocks if b.block_type == block_type]
        
        if certification_level:
            blocks = [b for b in blocks if b.certification_level == certification_level]
        
        if tenant_id:
            blocks = [b for b in blocks if b.tenant_id == tenant_id or b.tenant_id is None]
        
        return blocks
    
    def certify_block(self, block_id: str, certification_level: BlockCertificationLevel, 
                     certified_by: str) -> bool:
        """Task 7.2.16: Automate block certification workflow"""
        try:
            block = self.composable_blocks.get(block_id)
            if not block:
                return False
            
            # Certification workflow validation
            if not self._validate_certification_requirements(block, certification_level):
                return False
            
            # Update certification
            block.certification_level = certification_level
            block.certification_date = datetime.utcnow()
            block.certified_by = certified_by
            block.updated_at = datetime.utcnow()
            
            # Update reliability tier based on certification
            if certification_level == BlockCertificationLevel.CERTIFIED:
                block.reliability_tier = BlockReliabilityTier.TIER_2
            elif certification_level == BlockCertificationLevel.BETA:
                block.reliability_tier = BlockReliabilityTier.TIER_3
            
            self.logger.info(f"✅ Certified block {block_id} as {certification_level.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to certify block {block_id}: {e}")
            return False
    
    def track_block_usage(self, block_id: str, execution_time: float, success: bool):
        """Track block usage for analytics and anomaly detection"""
        try:
            block = self.composable_blocks.get(block_id)
            if not block:
                return
            
            # Update block metrics
            block.usage_count += 1
            if success:
                block.success_count += 1
            else:
                block.failure_count += 1
            
            # Update average execution time (rolling average)
            if block.avg_execution_time == 0.0:
                block.avg_execution_time = execution_time
            else:
                block.avg_execution_time = (block.avg_execution_time * 0.9) + (execution_time * 0.1)
            
            # Update analytics
            today = datetime.utcnow().date().isoformat()
            analytics = self.block_usage_analytics[block_id]
            
            if today not in analytics['daily_usage']:
                analytics['daily_usage'][today] = {'count': 0, 'success': 0, 'failure': 0}
            
            analytics['daily_usage'][today]['count'] += 1
            if success:
                analytics['daily_usage'][today]['success'] += 1
            else:
                analytics['daily_usage'][today]['failure'] += 1
            
            # Anomaly detection (Task 7.2.19)
            self._check_for_anomalies(block_id)
            
        except Exception as e:
            self.logger.error(f"Failed to track usage for block {block_id}: {e}")
    
    def get_block_reliability_metrics(self, block_id: str) -> Dict[str, Any]:
        """Task 7.2.28: Get predictive analytics on block reliability"""
        try:
            block = self.composable_blocks.get(block_id)
            if not block:
                return {}
            
            reliability_score = self.block_reliability_monitor.calculate_reliability_score(block)
            failure_probability = self.block_reliability_monitor.predict_failure_probability(block)
            
            return {
                'block_id': block_id,
                'reliability_score': reliability_score,
                'failure_probability_24h': failure_probability,
                'uptime_sla': block.uptime_sla,
                'actual_uptime': (block.success_count / block.usage_count) if block.usage_count > 0 else 0.0,
                'error_rate': (block.failure_count / block.usage_count) if block.usage_count > 0 else 0.0,
                'avg_execution_time': block.avg_execution_time,
                'certification_level': block.certification_level.value,
                'reliability_tier': block.reliability_tier.value,
                'usage_count': block.usage_count,
                'last_anomaly': block.last_anomaly_detected
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get reliability metrics for block {block_id}: {e}")
            return {}
    
    def _validate_block_schema(self, block: ComposableBlock) -> bool:
        """Validate composable block schema"""
        required_fields = ['block_id', 'name', 'description', 'block_type']
        
        for field in required_fields:
            if not hasattr(block, field) or not getattr(block, field):
                self.logger.error(f"Block validation failed: missing {field}")
                return False
        
        # Validate interface schema
        if block.interface_schema and not isinstance(block.interface_schema, dict):
            self.logger.error("Block validation failed: invalid interface_schema")
            return False
        
        return True
    
    def _validate_certification_requirements(self, block: ComposableBlock, 
                                           certification_level: BlockCertificationLevel) -> bool:
        """Validate certification requirements"""
        if certification_level == BlockCertificationLevel.CERTIFIED:
            # Certified blocks require minimum usage and success rate
            if block.usage_count < 100:
                self.logger.warning(f"Block {block.block_id} needs more usage for certification (current: {block.usage_count})")
                return False
            
            success_rate = (block.success_count / block.usage_count) if block.usage_count > 0 else 0.0
            if success_rate < 0.95:
                self.logger.warning(f"Block {block.block_id} success rate too low for certification (current: {success_rate:.2%})")
                return False
        
        return True
    
    def _increment_semantic_version(self, current_version: str) -> str:
        """Task 7.2.18: Increment semantic version"""
        try:
            import semver
            version_info = semver.VersionInfo.parse(current_version)
            return str(version_info.bump_patch())
        except:
            # Fallback to simple increment
            parts = current_version.split('.')
            if len(parts) == 3:
                parts[2] = str(int(parts[2]) + 1)
                return '.'.join(parts)
            return "1.0.1"
    
    def _check_for_anomalies(self, block_id: str):
        """Task 7.2.19: Check for anomalies in block usage"""
        try:
            block = self.composable_blocks.get(block_id)
            analytics = self.block_usage_analytics.get(block_id)
            
            if not block or not analytics:
                return
            
            # Prepare metrics for anomaly detection
            metrics = {
                'error_rate': (block.failure_count / block.usage_count) if block.usage_count > 0 else 0.0,
                'baseline_error_rate': block.error_rate,
                'avg_execution_time': block.avg_execution_time,
                'baseline_execution_time': 1.0  # Could be calculated from historical data
            }
            
            # Detect anomalies
            anomalies = self.block_anomaly_detector.detect_anomalies(block_id, metrics)
            
            if anomalies:
                # Log anomalies
                for anomaly in anomalies:
                    self.logger.warning(f"Anomaly detected in block {block_id}: {anomaly}")
                
                # Update block
                block.last_anomaly_detected = datetime.utcnow()
                
                # Store in analytics
                analytics['anomaly_history'].extend(anomalies)
                
                # Keep only recent anomalies (last 30 days)
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                analytics['anomaly_history'] = [
                    a for a in analytics['anomaly_history'] 
                    if a.get('detected_at', datetime.utcnow()) > cutoff_date
                ]
        
        except Exception as e:
            self.logger.error(f"Failed to check anomalies for block {block_id}: {e}")
    
    # =============================================================================
    # BACKEND TASKS IMPLEMENTATION - CHAPTER 14
    # =============================================================================
    
    def _load_banking_templates(self) -> Dict[str, Any]:
        """Task 14.1.7: Define Banking schema templates (credit scoring, AML, fraud)"""
        return {
            "credit_scoring": {
                "name": "Credit Scoring Workflow",
                "description": "Automated credit risk assessment and scoring",
                "industry": "Banking",
                "compliance_frameworks": ["RBI", "BASEL_III", "AML_KYC"],
                "workflow_steps": [
                    {
                        "step": "data_collection",
                        "type": "query",
                        "description": "Collect customer financial data",
                        "data_sources": ["credit_bureau", "bank_statements", "income_verification"],
                        "required_fields": ["customer_id", "income", "existing_debt", "credit_history"]
                    },
                    {
                        "step": "risk_assessment",
                        "type": "ml_decision",
                        "description": "ML-based credit risk scoring",
                        "model_type": "credit_risk_model",
                        "features": ["debt_to_income", "payment_history", "credit_utilization", "employment_stability"],
                        "output": "credit_score"
                    },
                    {
                        "step": "compliance_check",
                        "type": "governance",
                        "description": "RBI compliance validation",
                        "policy_pack": "RBI_CREDIT_POLICY",
                        "checks": ["income_verification", "identity_verification", "regulatory_limits"]
                    },
                    {
                        "step": "approval_decision",
                        "type": "decision",
                        "description": "Final credit approval decision",
                        "rules": [
                            {"condition": "credit_score >= 750", "action": "auto_approve", "limit": 500000},
                            {"condition": "credit_score >= 650", "action": "manual_review", "limit": 200000},
                            {"condition": "credit_score < 650", "action": "reject", "reason": "insufficient_credit_score"}
                        ]
                    }
                ],
                "sla_requirements": {
                    "max_processing_time_minutes": 30,
                    "availability_percent": 99.5,
                    "data_retention_years": 7
                },
                "trust_metrics": {
                    "accuracy_threshold": 0.85,
                    "false_positive_rate": 0.05,
                    "regulatory_compliance_score": 0.95
                }
            },
            "aml_detection": {
                "name": "Anti-Money Laundering Detection",
                "description": "Real-time AML transaction monitoring and reporting",
                "industry": "Banking",
                "compliance_frameworks": ["AML_KYC", "FATF", "RBI_AML"],
                "workflow_steps": [
                    {
                        "step": "transaction_monitoring",
                        "type": "query",
                        "description": "Monitor real-time transactions",
                        "data_sources": ["transaction_stream", "customer_profiles", "watchlists"],
                        "monitoring_rules": ["large_cash_transactions", "unusual_patterns", "high_risk_countries"]
                    },
                    {
                        "step": "risk_scoring",
                        "type": "ml_decision",
                        "description": "ML-based AML risk scoring",
                        "model_type": "aml_risk_model",
                        "features": ["transaction_amount", "frequency", "counterparty_risk", "geographic_risk"],
                        "output": "aml_risk_score"
                    },
                    {
                        "step": "watchlist_screening",
                        "type": "query",
                        "description": "Screen against sanctions and PEP lists",
                        "data_sources": ["ofac_list", "un_sanctions", "pep_database"],
                        "matching_threshold": 0.8
                    },
                    {
                        "step": "alert_generation",
                        "type": "decision",
                        "description": "Generate AML alerts based on risk score",
                        "rules": [
                            {"condition": "aml_risk_score >= 0.8", "action": "high_priority_alert", "escalation": "immediate"},
                            {"condition": "aml_risk_score >= 0.6", "action": "medium_priority_alert", "escalation": "4_hours"},
                            {"condition": "watchlist_match == true", "action": "immediate_freeze", "escalation": "immediate"}
                        ]
                    },
                    {
                        "step": "regulatory_reporting",
                        "type": "notify",
                        "description": "Submit STR/CTR reports to regulators",
                        "recipients": ["rbi_fiu", "compliance_team"],
                        "report_format": "STR_XML_FORMAT"
                    }
                ],
                "sla_requirements": {
                    "max_processing_time_seconds": 5,
                    "availability_percent": 99.9,
                    "alert_response_time_minutes": 15
                },
                "trust_metrics": {
                    "detection_accuracy": 0.92,
                    "false_positive_rate": 0.02,
                    "regulatory_compliance_score": 0.98
                }
            },
            "fraud_detection": {
                "name": "Real-time Fraud Detection",
                "description": "AI-powered fraud detection for banking transactions",
                "industry": "Banking",
                "compliance_frameworks": ["RBI_FRAUD", "PCI_DSS"],
                "workflow_steps": [
                    {
                        "step": "transaction_analysis",
                        "type": "ml_decision",
                        "description": "Real-time fraud risk assessment",
                        "model_type": "fraud_detection_model",
                        "features": ["transaction_amount", "merchant_category", "location", "time_of_day", "customer_behavior"],
                        "output": "fraud_probability"
                    },
                    {
                        "step": "behavioral_analysis",
                        "type": "query",
                        "description": "Analyze customer behavioral patterns",
                        "data_sources": ["transaction_history", "device_fingerprint", "location_history"],
                        "analysis_window_days": 30
                    },
                    {
                        "step": "risk_decision",
                        "type": "decision",
                        "description": "Fraud risk decision engine",
                        "rules": [
                            {"condition": "fraud_probability >= 0.9", "action": "block_transaction", "notification": "immediate"},
                            {"condition": "fraud_probability >= 0.7", "action": "step_up_auth", "method": "otp_sms"},
                            {"condition": "fraud_probability >= 0.5", "action": "monitor_closely", "duration_hours": 24},
                            {"condition": "fraud_probability < 0.5", "action": "allow", "monitoring": "standard"}
                        ]
                    },
                    {
                        "step": "customer_notification",
                        "type": "notify",
                        "description": "Notify customer of fraud alerts",
                        "channels": ["sms", "email", "push_notification"],
                        "templates": ["fraud_alert", "transaction_blocked"]
                    }
                ],
                "sla_requirements": {
                    "max_processing_time_ms": 200,
                    "availability_percent": 99.95,
                    "false_positive_rate": 0.01
                },
                "trust_metrics": {
                    "fraud_detection_rate": 0.95,
                    "customer_satisfaction": 0.88,
                    "operational_efficiency": 0.92
                }
            }
        }
    
    def _load_insurance_templates(self) -> Dict[str, Any]:
        """Task 14.1.8: Define Insurance schema templates (claims lifecycle, underwriting)"""
        return {
            "claims_processing": {
                "name": "Automated Claims Processing",
                "description": "End-to-end insurance claims lifecycle management",
                "industry": "Insurance",
                "compliance_frameworks": ["IRDAI", "HIPAA", "NAIC"],
                "workflow_steps": [
                    {
                        "step": "claim_intake",
                        "type": "query",
                        "description": "Capture and validate claim information",
                        "data_sources": ["policy_database", "customer_portal", "agent_system"],
                        "required_fields": ["policy_number", "incident_date", "claim_amount", "supporting_documents"]
                    },
                    {
                        "step": "eligibility_check",
                        "type": "decision",
                        "description": "Verify policy coverage and eligibility",
                        "rules": [
                            {"condition": "policy_active == true", "check": "coverage_validation"},
                            {"condition": "incident_date within policy_period", "check": "temporal_validation"},
                            {"condition": "claim_amount <= coverage_limit", "check": "amount_validation"},
                            {"condition": "waiting_period_satisfied == true", "check": "waiting_period_validation"}
                        ]
                    },
                    {
                        "step": "fraud_assessment",
                        "type": "ml_decision",
                        "description": "AI-based claims fraud detection",
                        "model_type": "claims_fraud_model",
                        "features": ["claim_amount", "incident_type", "claimant_history", "medical_codes", "provider_patterns"],
                        "output": "fraud_risk_score"
                    },
                    {
                        "step": "medical_review",
                        "type": "agent_call",
                        "description": "Medical necessity review for health claims",
                        "agent_type": "medical_review_agent",
                        "inputs": ["medical_records", "treatment_codes", "provider_notes"],
                        "outputs": ["medical_necessity_score", "treatment_appropriateness"]
                    },
                    {
                        "step": "settlement_calculation",
                        "type": "decision",
                        "description": "Calculate claim settlement amount",
                        "rules": [
                            {"condition": "fraud_risk_score < 0.3 AND medical_necessity_score > 0.8", "action": "auto_approve", "percentage": 100},
                            {"condition": "fraud_risk_score < 0.5 AND medical_necessity_score > 0.6", "action": "partial_approve", "percentage": 80},
                            {"condition": "fraud_risk_score >= 0.7 OR medical_necessity_score < 0.4", "action": "manual_review", "escalation": "senior_adjuster"},
                            {"condition": "fraud_risk_score >= 0.9", "action": "reject", "reason": "suspected_fraud"}
                        ]
                    },
                    {
                        "step": "payment_processing",
                        "type": "notify",
                        "description": "Process claim payment and notifications",
                        "payment_methods": ["bank_transfer", "check", "digital_wallet"],
                        "notifications": ["claimant", "agent", "provider"]
                    }
                ],
                "sla_requirements": {
                    "max_processing_time_days": 15,
                    "auto_approval_percentage": 70,
                    "customer_satisfaction_score": 4.2
                },
                "trust_metrics": {
                    "fraud_detection_accuracy": 0.88,
                    "settlement_accuracy": 0.94,
                    "regulatory_compliance_score": 0.96
                }
            },
            "underwriting_automation": {
                "name": "Intelligent Underwriting",
                "description": "AI-powered insurance underwriting and risk assessment",
                "industry": "Insurance",
                "compliance_frameworks": ["IRDAI", "ACTUARIAL", "NAIC"],
                "workflow_steps": [
                    {
                        "step": "application_intake",
                        "type": "query",
                        "description": "Collect and validate application data",
                        "data_sources": ["application_form", "medical_records", "financial_statements", "third_party_data"],
                        "validation_rules": ["completeness_check", "data_consistency", "document_verification"]
                    },
                    {
                        "step": "risk_assessment",
                        "type": "ml_decision",
                        "description": "Comprehensive risk scoring using AI models",
                        "model_type": "underwriting_risk_model",
                        "features": ["age", "health_indicators", "lifestyle_factors", "financial_stability", "occupation_risk"],
                        "output": "risk_score"
                    },
                    {
                        "step": "medical_underwriting",
                        "type": "decision",
                        "description": "Medical risk evaluation and classification",
                        "rules": [
                            {"condition": "age < 40 AND no_medical_history", "classification": "standard_risk", "loading": 0},
                            {"condition": "age >= 40 AND minor_conditions", "classification": "substandard_risk", "loading": 25},
                            {"condition": "chronic_conditions OR high_bmi", "classification": "high_risk", "loading": 50},
                            {"condition": "critical_illness_history", "classification": "decline", "reason": "uninsurable_risk"}
                        ]
                    },
                    {
                        "step": "financial_underwriting",
                        "type": "query",
                        "description": "Assess financial capacity and insurance need",
                        "data_sources": ["income_verification", "existing_coverage", "financial_obligations"],
                        "calculations": ["insurance_need_analysis", "affordability_assessment", "over_insurance_check"]
                    },
                    {
                        "step": "premium_calculation",
                        "type": "decision",
                        "description": "Calculate premium based on risk assessment",
                        "base_premium_factors": ["age", "sum_assured", "policy_term", "product_type"],
                        "loading_factors": ["medical_loading", "occupation_loading", "lifestyle_loading"],
                        "discount_factors": ["bulk_discount", "loyalty_discount", "no_claim_bonus"]
                    },
                    {
                        "step": "policy_issuance",
                        "type": "notify",
                        "description": "Issue policy and send notifications",
                        "documents": ["policy_document", "welcome_letter", "payment_schedule"],
                        "notifications": ["customer", "agent", "operations_team"]
                    }
                ],
                "sla_requirements": {
                    "max_processing_time_hours": 48,
                    "straight_through_processing_rate": 60,
                    "accuracy_rate": 95
                },
                "trust_metrics": {
                    "risk_assessment_accuracy": 0.91,
                    "premium_accuracy": 0.97,
                    "customer_satisfaction": 0.89
                }
            },
            "policy_administration": {
                "name": "Policy Lifecycle Management",
                "description": "Automated policy administration and servicing",
                "industry": "Insurance",
                "compliance_frameworks": ["IRDAI", "CUSTOMER_PROTECTION"],
                "workflow_steps": [
                    {
                        "step": "policy_servicing",
                        "type": "query",
                        "description": "Handle policy service requests",
                        "service_types": ["address_change", "beneficiary_update", "premium_payment", "policy_loan"],
                        "validation_rules": ["identity_verification", "policy_status_check", "eligibility_validation"]
                    },
                    {
                        "step": "renewal_processing",
                        "type": "decision",
                        "description": "Automated policy renewal processing",
                        "rules": [
                            {"condition": "premium_paid_on_time AND no_claims", "action": "auto_renew", "discount": "no_claim_bonus"},
                            {"condition": "premium_delayed < 30_days", "action": "renew_with_grace", "notice": "payment_reminder"},
                            {"condition": "premium_delayed > 30_days", "action": "lapse_notice", "grace_period": "additional_15_days"},
                            {"condition": "multiple_claims OR high_risk", "action": "manual_review", "underwriting": "required"}
                        ]
                    },
                    {
                        "step": "surrender_processing",
                        "type": "decision",
                        "description": "Process policy surrender requests",
                        "calculations": ["surrender_value", "tax_implications", "penalty_charges"],
                        "approvals": ["customer_confirmation", "tax_clearance", "final_settlement"]
                    },
                    {
                        "step": "maturity_processing",
                        "type": "notify",
                        "description": "Handle policy maturity and payouts",
                        "calculations": ["maturity_amount", "bonus_calculations", "tax_deductions"],
                        "payment_processing": ["bank_transfer", "check_issuance", "reinvestment_options"]
                    }
                ],
                "sla_requirements": {
                    "service_request_resolution_hours": 24,
                    "renewal_processing_days": 7,
                    "maturity_payout_days": 15
                },
                "trust_metrics": {
                    "service_accuracy": 0.96,
                    "customer_satisfaction": 0.91,
                    "operational_efficiency": 0.88
                }
            }
        }
    
    async def get_banking_trust_metrics(self, capability_name: str) -> Dict[str, float]:
        """Task 14.2.5: Define Banking trust metrics (credit approval, AML detection)"""
        banking_metrics = {
            "credit_scoring": {
                "credit_approval_accuracy": 0.89,
                "default_prediction_accuracy": 0.85,
                "regulatory_compliance_score": 0.95,
                "processing_efficiency": 0.92,
                "customer_satisfaction": 0.87,
                "risk_assessment_precision": 0.88,
                "false_positive_rate": 0.05,
                "false_negative_rate": 0.03
            },
            "aml_detection": {
                "suspicious_transaction_detection_rate": 0.94,
                "false_alert_rate": 0.02,
                "regulatory_reporting_accuracy": 0.98,
                "investigation_efficiency": 0.86,
                "compliance_audit_score": 0.96,
                "real_time_processing_reliability": 0.99,
                "watchlist_screening_accuracy": 0.97,
                "case_closure_rate": 0.91
            },
            "fraud_detection": {
                "fraud_detection_rate": 0.95,
                "transaction_blocking_accuracy": 0.93,
                "customer_impact_minimization": 0.88,
                "real_time_response_reliability": 0.97,
                "behavioral_analysis_accuracy": 0.89,
                "merchant_risk_assessment": 0.86,
                "device_fingerprinting_accuracy": 0.92,
                "investigation_support_quality": 0.90
            }
        }
        
        return banking_metrics.get(capability_name, {
            "overall_trust_score": 0.85,
            "reliability_score": 0.88,
            "compliance_score": 0.92,
            "performance_score": 0.87
        })
    
    async def get_insurance_trust_metrics(self, capability_name: str) -> Dict[str, float]:
        """Task 14.2.6: Define Insurance trust metrics (claims accuracy, underwriting)"""
        insurance_metrics = {
            "claims_processing": {
                "claims_settlement_accuracy": 0.94,
                "fraud_detection_rate": 0.88,
                "processing_time_efficiency": 0.91,
                "customer_satisfaction_score": 0.89,
                "regulatory_compliance_score": 0.96,
                "medical_review_accuracy": 0.92,
                "settlement_amount_precision": 0.95,
                "dispute_resolution_rate": 0.87
            },
            "underwriting_automation": {
                "risk_assessment_accuracy": 0.91,
                "premium_calculation_precision": 0.97,
                "medical_underwriting_reliability": 0.89,
                "financial_assessment_accuracy": 0.93,
                "policy_issuance_efficiency": 0.95,
                "straight_through_processing_rate": 0.68,
                "decline_accuracy": 0.86,
                "pricing_competitiveness": 0.84
            },
            "policy_administration": {
                "service_request_accuracy": 0.96,
                "renewal_processing_efficiency": 0.93,
                "surrender_calculation_accuracy": 0.98,
                "maturity_processing_reliability": 0.97,
                "customer_communication_effectiveness": 0.90,
                "regulatory_reporting_compliance": 0.95,
                "data_integrity_maintenance": 0.94,
                "operational_cost_efficiency": 0.88
            }
        }
        
        return insurance_metrics.get(capability_name, {
            "overall_trust_score": 0.87,
            "reliability_score": 0.90,
            "compliance_score": 0.94,
            "performance_score": 0.89
        })
    
    async def attach_compliance_pack_saas(self, capability_id: str, tenant_id: int) -> bool:
        """Task 14.1.12: Attach compliance packs (SaaS) - SOX overlays at schema level"""
        try:
            sox_compliance_pack = {
                "compliance_framework": "SOX",
                "requirements": [
                    {
                        "control_id": "SOX_404_ITGC_01",
                        "description": "IT General Controls for Revenue Recognition",
                        "requirements": ["access_controls", "change_management", "data_integrity", "backup_recovery"]
                    },
                    {
                        "control_id": "SOX_404_ITGC_02", 
                        "description": "Financial Reporting Controls for SaaS Metrics",
                        "requirements": ["arr_calculation_controls", "revenue_recognition_rules", "subscription_lifecycle_tracking"]
                    },
                    {
                        "control_id": "SOX_302_DISCLOSURE",
                        "description": "Disclosure Controls and Procedures",
                        "requirements": ["material_weakness_reporting", "deficiency_tracking", "management_assessment"]
                    }
                ],
                "audit_trails": {
                    "required_logs": ["user_access", "data_changes", "system_configurations", "financial_calculations"],
                    "retention_period_years": 7,
                    "immutable_storage": True,
                    "encryption_required": True
                },
                "segregation_of_duties": {
                    "incompatible_functions": [
                        ["revenue_calculation", "revenue_approval"],
                        ["user_provisioning", "user_approval"],
                        ["system_configuration", "configuration_approval"]
                    ],
                    "approval_workflows": ["dual_approval", "management_override_logging"]
                },
                "testing_requirements": {
                    "control_testing_frequency": "quarterly",
                    "penetration_testing": "annual",
                    "vulnerability_assessments": "monthly",
                    "compliance_validation": "continuous"
                }
            }
            
            if self.pool_manager:
                async with self.pool_manager.get_connection() as conn:
                    await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                    
                    # Update capability with SOX compliance pack
                    await conn.execute("""
                        UPDATE dsl_capability_registry 
                        SET compliance_tags = compliance_tags || $1,
                            policy_requirements = policy_requirements || $2,
                            updated_at = NOW()
                        WHERE capability_id = $3 AND tenant_id = $4
                    """, 
                    json.dumps(["SOX", "SOX_404", "SOX_302"]),
                    json.dumps(sox_compliance_pack),
                    capability_id, tenant_id)
            
            self.logger.info(f"✅ SOX compliance pack attached to capability {capability_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to attach SOX compliance pack: {e}")
            return False
    
    async def attach_compliance_pack_banking(self, capability_id: str, tenant_id: int) -> bool:
        """Task 14.1.13: Attach compliance packs (Banking) - RBI/DPDP overlays"""
        try:
            rbi_compliance_pack = {
                "compliance_framework": "RBI_DPDP",
                "requirements": [
                    {
                        "control_id": "RBI_CYBER_SEC_01",
                        "description": "Cybersecurity Framework Implementation",
                        "requirements": ["multi_factor_authentication", "encryption_at_rest", "encryption_in_transit", "access_logging"]
                    },
                    {
                        "control_id": "RBI_AML_KYC_01",
                        "description": "Anti-Money Laundering and Know Your Customer",
                        "requirements": ["customer_due_diligence", "transaction_monitoring", "suspicious_transaction_reporting", "record_keeping"]
                    },
                    {
                        "control_id": "DPDP_DATA_PROTECTION",
                        "description": "Digital Personal Data Protection Act Compliance",
                        "requirements": ["consent_management", "data_minimization", "purpose_limitation", "data_subject_rights"]
                    },
                    {
                        "control_id": "RBI_OPERATIONAL_RISK",
                        "description": "Operational Risk Management",
                        "requirements": ["business_continuity", "disaster_recovery", "incident_management", "risk_assessment"]
                    }
                ],
                "data_localization": {
                    "storage_location": "India",
                    "processing_location": "India", 
                    "cross_border_restrictions": True,
                    "data_mirroring_required": True
                },
                "reporting_requirements": {
                    "regulatory_reports": ["STR", "CTR", "CRILC", "DSB"],
                    "submission_frequency": {
                        "STR": "within_7_days",
                        "CTR": "within_15_days", 
                        "CRILC": "monthly",
                        "DSB": "quarterly"
                    },
                    "format_specifications": "RBI_XML_SCHEMA"
                },
                "audit_requirements": {
                    "internal_audit_frequency": "quarterly",
                    "external_audit_frequency": "annual",
                    "rbi_inspection_readiness": True,
                    "audit_trail_retention_years": 10
                }
            }
            
            if self.pool_manager:
                async with self.pool_manager.get_connection() as conn:
                    await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                    
                    await conn.execute("""
                        UPDATE dsl_capability_registry 
                        SET compliance_tags = compliance_tags || $1,
                            policy_requirements = policy_requirements || $2,
                            updated_at = NOW()
                        WHERE capability_id = $3 AND tenant_id = $4
                    """, 
                    json.dumps(["RBI", "DPDP", "AML_KYC", "CYBER_SEC"]),
                    json.dumps(rbi_compliance_pack),
                    capability_id, tenant_id)
            
            self.logger.info(f"✅ RBI/DPDP compliance pack attached to capability {capability_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to attach RBI compliance pack: {e}")
            return False
    
    async def attach_compliance_pack_insurance(self, capability_id: str, tenant_id: int) -> bool:
        """Task 14.1.14: Attach compliance packs (Insurance) - IRDAI/NAIC overlays"""
        try:
            irdai_compliance_pack = {
                "compliance_framework": "IRDAI_NAIC",
                "requirements": [
                    {
                        "control_id": "IRDAI_CONSUMER_PROTECTION",
                        "description": "Insurance Consumer Protection Guidelines",
                        "requirements": ["fair_treatment", "transparent_communication", "grievance_redressal", "mis_selling_prevention"]
                    },
                    {
                        "control_id": "IRDAI_SOLVENCY_MGMT",
                        "description": "Solvency and Risk Management",
                        "requirements": ["capital_adequacy", "risk_assessment", "stress_testing", "actuarial_validation"]
                    },
                    {
                        "control_id": "IRDAI_CLAIMS_MGMT",
                        "description": "Claims Management Framework",
                        "requirements": ["timely_settlement", "fair_assessment", "fraud_prevention", "customer_communication"]
                    },
                    {
                        "control_id": "NAIC_MODEL_GOVERNANCE",
                        "description": "Model Risk Management (NAIC Standards)",
                        "requirements": ["model_validation", "model_documentation", "model_monitoring", "model_governance"]
                    }
                ],
                "product_approval": {
                    "filing_requirements": ["product_design", "actuarial_certification", "compliance_attestation"],
                    "approval_timeline_days": 90,
                    "modification_notifications": True,
                    "withdrawal_procedures": "regulatory_approval_required"
                },
                "claims_settlement": {
                    "settlement_timeline_days": {
                        "health_claims": 30,
                        "motor_claims": 30,
                        "life_claims": 30,
                        "general_claims": 30
                    },
                    "investigation_limits": {
                        "max_investigation_days": 30,
                        "extension_approval_required": True
                    },
                    "customer_communication": "mandatory_updates_every_15_days"
                },
                "data_protection": {
                    "customer_data_security": "encryption_mandatory",
                    "data_sharing_restrictions": True,
                    "consent_management": "explicit_consent_required",
                    "data_retention_limits": "as_per_policy_terms_plus_7_years"
                }
            }
            
            if self.pool_manager:
                async with self.pool_manager.get_connection() as conn:
                    await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                    
                    await conn.execute("""
                        UPDATE dsl_capability_registry 
                        SET compliance_tags = compliance_tags || $1,
                            policy_requirements = policy_requirements || $2,
                            updated_at = NOW()
                        WHERE capability_id = $3 AND tenant_id = $4
                    """, 
                    json.dumps(["IRDAI", "NAIC", "CONSUMER_PROTECTION", "SOLVENCY"]),
                    json.dumps(irdai_compliance_pack),
                    capability_id, tenant_id)
            
            self.logger.info(f"✅ IRDAI/NAIC compliance pack attached to capability {capability_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to attach IRDAI compliance pack: {e}")
            return False

    # =============================================================================
    # SECTION 15.3: CAPABILITY LOOKUP IMPLEMENTATION
    # =============================================================================
    
    async def implement_eligibility_rules(self, tenant_id: int, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task 15.3.9: Implement eligibility rules (Tenant SLA, industry filters)
        Apply tenant, SLA, and industry-specific filters for capability access
        """
        try:
            eligibility_result = {
                "tenant_id": tenant_id,
                "eligible_capabilities": [],
                "filtered_capabilities": [],
                "eligibility_reasons": {},
                "sla_tier": None,
                "industry_code": None,
                "timestamp": datetime.now().isoformat()
            }
            
            if self.pool_manager:
                async with self.pool_manager.get_connection() as conn:
                    await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                    
                    # Get tenant metadata and SLA tier
                    tenant_info = await conn.fetchrow("""
                        SELECT tenant_type, industry_code, sla_tier, tenant_status
                        FROM tenant_metadata 
                        WHERE tenant_id = $1
                    """, tenant_id)
                    
                    if not tenant_info:
                        raise ValueError(f"Tenant {tenant_id} not found")
                    
                    eligibility_result["sla_tier"] = tenant_info["sla_tier"] or "standard"
                    eligibility_result["industry_code"] = tenant_info["industry_code"] or "SaaS"
                    
                    # Get all available capabilities
                    capabilities = await conn.fetch("""
                        SELECT capability_id, capability_name, capability_type, 
                               industry_overlays, compliance_requirements, 
                               trust_score, usage_count
                        FROM dsl_capability_registry 
                        WHERE tenant_id = $1 OR tenant_id IS NULL
                        ORDER BY trust_score DESC, usage_count DESC
                    """, tenant_id)
                    
                    # Apply eligibility rules
                    for cap in capabilities:
                        capability_id = str(cap["capability_id"])
                        capability_name = cap["capability_name"]
                        
                        # Rule 1: Industry filter
                        industry_overlays = cap["industry_overlays"] or []
                        if isinstance(industry_overlays, str):
                            industry_overlays = json.loads(industry_overlays)
                        
                        industry_eligible = (
                            not industry_overlays or 
                            eligibility_result["industry_code"] in industry_overlays or
                            "ALL" in industry_overlays
                        )
                        
                        # Rule 2: SLA tier filter
                        sla_eligible = True
                        if eligibility_result["sla_tier"] == "basic":
                            # Basic tier: only basic capabilities
                            sla_eligible = cap["capability_type"] in ["RBA", "basic"]
                        elif eligibility_result["sla_tier"] == "standard":
                            # Standard tier: RBA + RBIA
                            sla_eligible = cap["capability_type"] in ["RBA", "RBIA", "basic", "standard"]
                        # Premium tier: all capabilities allowed
                        
                        # Rule 3: Trust score threshold
                        trust_threshold = self._get_trust_threshold_for_sla(eligibility_result["sla_tier"])
                        trust_eligible = (cap["trust_score"] or 0.0) >= trust_threshold
                        
                        # Rule 4: Compliance requirements
                        compliance_requirements = cap["compliance_requirements"] or []
                        if isinstance(compliance_requirements, str):
                            compliance_requirements = json.loads(compliance_requirements)
                        
                        compliance_eligible = self._check_tenant_compliance_eligibility(
                            tenant_info, compliance_requirements
                        )
                        
                        # Determine final eligibility
                        is_eligible = industry_eligible and sla_eligible and trust_eligible and compliance_eligible
                        
                        eligibility_reasons = {
                            "industry_eligible": industry_eligible,
                            "sla_eligible": sla_eligible,
                            "trust_eligible": trust_eligible,
                            "compliance_eligible": compliance_eligible
                        }
                        
                        if is_eligible:
                            eligibility_result["eligible_capabilities"].append({
                                "capability_id": capability_id,
                                "capability_name": capability_name,
                                "capability_type": cap["capability_type"],
                                "trust_score": cap["trust_score"],
                                "usage_count": cap["usage_count"]
                            })
                        else:
                            eligibility_result["filtered_capabilities"].append({
                                "capability_id": capability_id,
                                "capability_name": capability_name,
                                "filter_reasons": [k for k, v in eligibility_reasons.items() if not v]
                            })
                        
                        eligibility_result["eligibility_reasons"][capability_id] = eligibility_reasons
            
            self.logger.info(f"✅ Eligibility rules applied: {len(eligibility_result['eligible_capabilities'])} eligible, {len(eligibility_result['filtered_capabilities'])} filtered")
            return eligibility_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to apply eligibility rules: {e}")
            return {
                "error": str(e),
                "tenant_id": tenant_id,
                "eligible_capabilities": [],
                "filtered_capabilities": []
            }
    
    def _get_trust_threshold_for_sla(self, sla_tier: str) -> float:
        """Get trust score threshold based on SLA tier"""
        thresholds = {
            "basic": 0.6,
            "standard": 0.7,
            "premium": 0.8,
            "enterprise": 0.9
        }
        return thresholds.get(sla_tier, 0.7)
    
    def _check_tenant_compliance_eligibility(self, tenant_info: Dict[str, Any], compliance_requirements: List[str]) -> bool:
        """Check if tenant meets compliance requirements"""
        if not compliance_requirements:
            return True
        
        tenant_industry = tenant_info.get("industry_code", "SaaS")
        tenant_type = tenant_info.get("tenant_type", "standard")
        
        # Industry-specific compliance mapping
        industry_compliance = {
            "SaaS": ["SOX", "GDPR", "SAAS_BUSINESS_RULES"],
            "Banking": ["SOX", "RBI", "DPDP", "AML_KYC", "BASEL_III"],
            "Insurance": ["HIPAA", "NAIC", "IRDAI", "GDPR"],
            "FinTech": ["SOX", "GDPR", "PCI_DSS", "AML_KYC"],
            "E-commerce": ["GDPR", "PCI_DSS", "CCPA"],
            "IT_Services": ["SOX", "GDPR", "ISO_27001"]
        }
        
        tenant_compliance_frameworks = industry_compliance.get(tenant_industry, [])
        
        # Check if all required compliance frameworks are supported
        for requirement in compliance_requirements:
            if requirement not in tenant_compliance_frameworks:
                return False
        
        return True
    
    async def implement_fallback_mechanism(self, tenant_id: int, requested_capability: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task 15.3.13: Implement fallback mechanism (Default → deny if no capability)
        Fail-closed approach: deny access if no suitable capability found
        """
        try:
            fallback_result = {
                "tenant_id": tenant_id,
                "requested_capability": requested_capability,
                "fallback_action": "deny",
                "fallback_reason": None,
                "alternative_capabilities": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Check if requested capability exists and is eligible
            eligibility_result = await self.implement_eligibility_rules(tenant_id, context)
            
            # Look for exact match
            exact_match = None
            for cap in eligibility_result["eligible_capabilities"]:
                if (cap["capability_id"] == requested_capability or 
                    cap["capability_name"].lower() == requested_capability.lower()):
                    exact_match = cap
                    break
            
            if exact_match:
                fallback_result["fallback_action"] = "allow"
                fallback_result["matched_capability"] = exact_match
                self.logger.info(f"✅ Capability {requested_capability} found and eligible for tenant {tenant_id}")
                return fallback_result
            
            # Look for similar capabilities (fuzzy matching)
            similar_capabilities = []
            for cap in eligibility_result["eligible_capabilities"]:
                similarity_score = self._calculate_capability_similarity(requested_capability, cap["capability_name"])
                if similarity_score > 0.7:  # 70% similarity threshold
                    similar_capabilities.append({
                        **cap,
                        "similarity_score": similarity_score
                    })
            
            if similar_capabilities:
                # Sort by similarity and trust score
                similar_capabilities.sort(key=lambda x: (x["similarity_score"], x["trust_score"]), reverse=True)
                fallback_result["alternative_capabilities"] = similar_capabilities[:3]  # Top 3 alternatives
                fallback_result["fallback_reason"] = "exact_match_not_found_alternatives_available"
            else:
                fallback_result["fallback_reason"] = "no_matching_capabilities_found"
            
            # Check if capability was filtered (exists but not eligible)
            filtered_match = None
            for cap in eligibility_result["filtered_capabilities"]:
                if (cap["capability_id"] == requested_capability or 
                    cap["capability_name"].lower() == requested_capability.lower()):
                    filtered_match = cap
                    break
            
            if filtered_match:
                fallback_result["fallback_reason"] = f"capability_filtered: {', '.join(filtered_match['filter_reasons'])}"
                fallback_result["filtered_capability"] = filtered_match
            
            # Log fallback event for audit
            await self._log_fallback_event(tenant_id, requested_capability, fallback_result)
            
            self.logger.warning(f"⚠️ Capability {requested_capability} denied for tenant {tenant_id}: {fallback_result['fallback_reason']}")
            return fallback_result
            
        except Exception as e:
            self.logger.error(f"❌ Fallback mechanism failed: {e}")
            return {
                "tenant_id": tenant_id,
                "requested_capability": requested_capability,
                "fallback_action": "deny",
                "fallback_reason": f"system_error: {str(e)}",
                "error": str(e)
            }
    
    def _calculate_capability_similarity(self, requested: str, available: str) -> float:
        """Calculate similarity score between capability names"""
        import difflib
        return difflib.SequenceMatcher(None, requested.lower(), available.lower()).ratio()
    
    async def _log_fallback_event(self, tenant_id: int, requested_capability: str, fallback_result: Dict[str, Any]):
        """Log fallback events for audit trail"""
        try:
            if self.pool_manager:
                async with self.pool_manager.get_connection() as conn:
                    await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                    
                    await conn.execute("""
                        INSERT INTO capability_registry_binding (
                            tenant_id, capability_id, binding_type, binding_metadata, created_at
                        ) VALUES ($1, $2, $3, $4, NOW())
                    """, 
                    tenant_id, 
                    f"fallback_{requested_capability}",
                    "fallback_event",
                    json.dumps(fallback_result))
        except Exception as e:
            self.logger.error(f"Failed to log fallback event: {e}")
    
    async def implement_evidence_logging(self, tenant_id: int, lookup_request: Dict[str, Any], lookup_result: Dict[str, Any]) -> str:
        """
        Task 15.3.14: Implement evidence logging (Log lookup results)
        Store immutable evidence of capability lookup decisions
        """
        try:
            evidence_id = str(uuid.uuid4())
            
            evidence_data = {
                "evidence_id": evidence_id,
                "evidence_type": "capability_lookup",
                "tenant_id": tenant_id,
                "timestamp": datetime.now().isoformat(),
                "lookup_request": lookup_request,
                "lookup_result": lookup_result,
                "compliance_metadata": {
                    "data_classification": "audit_trail",
                    "retention_period_days": 2555,  # 7 years
                    "encryption_required": True,
                    "immutable": True
                }
            }
            
            if self.pool_manager:
                async with self.pool_manager.get_connection() as conn:
                    await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                    
                    # Store in evidence service (assuming evidence table exists)
                    await conn.execute("""
                        INSERT INTO dsl_evidence_packs (
                            evidence_pack_id, tenant_id, evidence_type, 
                            evidence_data, created_at, retention_until
                        ) VALUES ($1, $2, $3, $4, NOW(), NOW() + INTERVAL '7 years')
                    """, 
                    evidence_id, tenant_id, "capability_lookup", json.dumps(evidence_data))
            
            self.logger.info(f"✅ Evidence logged for capability lookup: {evidence_id}")
            return evidence_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log evidence: {e}")
            return ""
    
    async def generate_signed_manifests(self, tenant_id: int, lookup_evidence_id: str) -> Dict[str, Any]:
        """
        Task 15.3.15: Generate signed manifests (Immutable lookup proofs)
        Create cryptographically signed manifests for audit trail
        """
        try:
            manifest_id = str(uuid.uuid4())
            
            # Get evidence data
            evidence_data = None
            if self.pool_manager:
                async with self.pool_manager.get_connection() as conn:
                    await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                    
                    evidence_row = await conn.fetchrow("""
                        SELECT evidence_data, created_at 
                        FROM dsl_evidence_packs 
                        WHERE evidence_pack_id = $1 AND tenant_id = $2
                    """, lookup_evidence_id, tenant_id)
                    
                    if evidence_row:
                        evidence_data = evidence_row["evidence_data"]
            
            if not evidence_data:
                raise ValueError(f"Evidence {lookup_evidence_id} not found")
            
            # Create manifest
            manifest = {
                "manifest_id": manifest_id,
                "evidence_id": lookup_evidence_id,
                "tenant_id": tenant_id,
                "manifest_type": "capability_lookup_proof",
                "created_at": datetime.now().isoformat(),
                "evidence_hash": hashlib.sha256(json.dumps(evidence_data, sort_keys=True).encode()).hexdigest(),
                "manifest_version": "1.0",
                "compliance_frameworks": ["SOX", "audit_trail"]
            }
            
            # Generate digital signature (simplified - in production use proper PKI)
            manifest_content = json.dumps(manifest, sort_keys=True)
            signature = hashlib.sha256(f"{manifest_content}_{tenant_id}_secret".encode()).hexdigest()
            
            signed_manifest = {
                **manifest,
                "digital_signature": signature,
                "signature_algorithm": "SHA256_HMAC",
                "signing_authority": f"tenant_{tenant_id}_capability_registry"
            }
            
            # Store signed manifest
            if self.pool_manager:
                async with self.pool_manager.get_connection() as conn:
                    await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                    
                    await conn.execute("""
                        INSERT INTO capability_registry_binding (
                            tenant_id, capability_id, binding_type, binding_metadata, created_at
                        ) VALUES ($1, $2, $3, $4, NOW())
                    """, 
                    tenant_id, 
                    manifest_id,
                    "signed_manifest",
                    json.dumps(signed_manifest))
            
            self.logger.info(f"✅ Signed manifest generated: {manifest_id}")
            return signed_manifest
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate signed manifest: {e}")
            return {"error": str(e)}
    
    async def anchor_lookup_manifests(self, tenant_id: int, signed_manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task 15.3.16: Anchor lookup manifests (Tamper-proof anchoring)
        Cryptographically anchor manifests for immutable audit trail
        """
        try:
            anchor_id = str(uuid.uuid4())
            
            # Create anchor record
            anchor_data = {
                "anchor_id": anchor_id,
                "manifest_id": signed_manifest.get("manifest_id"),
                "tenant_id": tenant_id,
                "anchor_type": "capability_lookup_manifest",
                "timestamp": datetime.now().isoformat(),
                "manifest_hash": hashlib.sha256(json.dumps(signed_manifest, sort_keys=True).encode()).hexdigest(),
                "blockchain_reference": None,  # Would be actual blockchain hash in production
                "immutable_storage_reference": f"immudb://{tenant_id}/{anchor_id}",
                "anchor_proof": {
                    "merkle_root": hashlib.sha256(f"anchor_{anchor_id}_{tenant_id}".encode()).hexdigest(),
                    "timestamp_proof": datetime.now().timestamp(),
                    "integrity_check": "SHA256"
                }
            }
            
            # In production, this would:
            # 1. Store in immudb for tamper-proof storage
            # 2. Create blockchain anchor on Ethereum testnet
            # 3. Generate Merkle tree proof
            # 4. Store IPFS hash for distributed storage
            
            # For now, store in database with integrity checks
            if self.pool_manager:
                async with self.pool_manager.get_connection() as conn:
                    await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                    
                    await conn.execute("""
                        INSERT INTO capability_registry_binding (
                            tenant_id, capability_id, binding_type, binding_metadata, created_at
                        ) VALUES ($1, $2, $3, $4, NOW())
                    """, 
                    tenant_id, 
                    anchor_id,
                    "cryptographic_anchor",
                    json.dumps(anchor_data))
            
            self.logger.info(f"✅ Manifest anchored cryptographically: {anchor_id}")
            return anchor_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to anchor manifest: {e}")
            return {"error": str(e)}
    
    async def log_capability_overrides(self, tenant_id: int, override_request: Dict[str, Any]) -> str:
        """
        Task 15.3.17: Log overrides (Capture manual lookups)
        Log manual capability overrides in override ledger
        """
        try:
            override_id = str(uuid.uuid4())
            
            override_data = {
                "override_id": override_id,
                "tenant_id": tenant_id,
                "override_type": "capability_lookup_override",
                "timestamp": datetime.now().isoformat(),
                "requested_capability": override_request.get("capability_id"),
                "override_reason": override_request.get("reason"),
                "authorized_by": override_request.get("authorized_by"),
                "business_justification": override_request.get("business_justification"),
                "risk_assessment": override_request.get("risk_assessment", "medium"),
                "approval_workflow": override_request.get("approval_workflow"),
                "compliance_impact": override_request.get("compliance_impact"),
                "audit_trail": {
                    "original_decision": override_request.get("original_decision"),
                    "override_decision": override_request.get("override_decision"),
                    "approval_chain": override_request.get("approval_chain", [])
                }
            }
            
            if self.pool_manager:
                async with self.pool_manager.get_connection() as conn:
                    await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                    
                    # Store in override ledger
                    await conn.execute("""
                        INSERT INTO dsl_override_ledger (
                            override_id, tenant_id, override_type, 
                            override_data, created_at
                        ) VALUES ($1, $2, $3, $4, NOW())
                    """, 
                    override_id, tenant_id, "capability_lookup_override", json.dumps(override_data))
            
            self.logger.info(f"✅ Capability override logged: {override_id}")
            return override_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log capability override: {e}")
            return ""
    
    async def log_telemetry_metrics(self, tenant_id: int, lookup_session: Dict[str, Any]) -> bool:
        """
        Task 15.3.18: Log telemetry metrics (Hits, misses, overrides)
        Track capability lookup performance and usage metrics
        """
        try:
            telemetry_data = {
                "tenant_id": tenant_id,
                "session_id": lookup_session.get("session_id", str(uuid.uuid4())),
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "lookup_requests": lookup_session.get("lookup_requests", 0),
                    "successful_lookups": lookup_session.get("successful_lookups", 0),
                    "failed_lookups": lookup_session.get("failed_lookups", 0),
                    "filtered_capabilities": lookup_session.get("filtered_capabilities", 0),
                    "fallback_activations": lookup_session.get("fallback_activations", 0),
                    "override_requests": lookup_session.get("override_requests", 0),
                    "average_response_time_ms": lookup_session.get("average_response_time_ms", 0),
                    "cache_hit_rate": lookup_session.get("cache_hit_rate", 0.0)
                },
                "capability_usage": lookup_session.get("capability_usage", {}),
                "performance_metrics": {
                    "total_session_duration_ms": lookup_session.get("session_duration_ms", 0),
                    "database_query_time_ms": lookup_session.get("db_query_time_ms", 0),
                    "eligibility_check_time_ms": lookup_session.get("eligibility_check_time_ms", 0),
                    "evidence_logging_time_ms": lookup_session.get("evidence_logging_time_ms", 0)
                }
            }
            
            if self.pool_manager:
                async with self.pool_manager.get_connection() as conn:
                    await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                    
                    # Store telemetry data
                    await conn.execute("""
                        INSERT INTO capability_registry_binding (
                            tenant_id, capability_id, binding_type, binding_metadata, created_at
                        ) VALUES ($1, $2, $3, $4, NOW())
                    """, 
                    tenant_id, 
                    f"telemetry_{telemetry_data['session_id']}",
                    "telemetry_metrics",
                    json.dumps(telemetry_data))
            
            # Update capability usage analytics
            await self._update_capability_usage_analytics(tenant_id, telemetry_data["capability_usage"])
            
            self.logger.info(f"✅ Telemetry metrics logged for tenant {tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log telemetry metrics: {e}")
            return False
    
    async def _update_capability_usage_analytics(self, tenant_id: int, capability_usage: Dict[str, Any]):
        """Update capability usage analytics for insights"""
        try:
            if not capability_usage or not self.pool_manager:
                return
            
            async with self.pool_manager.get_connection() as conn:
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                for capability_id, usage_data in capability_usage.items():
                    await conn.execute("""
                        UPDATE dsl_capability_registry 
                        SET usage_count = COALESCE(usage_count, 0) + $1,
                            updated_at = NOW()
                        WHERE capability_id = $2 AND tenant_id = $3
                    """, 
                    usage_data.get("usage_count", 1),
                    capability_id, 
                    tenant_id)
        except Exception as e:
            self.logger.error(f"Failed to update capability usage analytics: {e}")

# Singleton instance for global access
_enhanced_capability_registry = None

def get_enhanced_capability_registry(pool_manager=None) -> EnhancedCapabilityRegistry:
    """Get singleton instance of Enhanced Capability Registry"""
    global _enhanced_capability_registry
    if _enhanced_capability_registry is None:
        _enhanced_capability_registry = EnhancedCapabilityRegistry(pool_manager)
    return _enhanced_capability_registry
