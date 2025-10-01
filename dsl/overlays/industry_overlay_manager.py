"""
Industry Overlay Manager
========================
Implements Chapter 13: Industry Overlays (SaaS, Banking, Insurance)

Features:
- Task 13.1: SaaS overlays (ARR, churn, NDR, revenue recognition)
- Task 13.2: Banking overlays (Credit scoring, AML, NPA, RBI compliance)
- Task 13.3: Insurance overlays (Claims lifecycle, underwriting, HIPAA/NAIC)
- Task 13.4: Multi-tenant overlay binding
- Task 13.5: Overlay performance monitoring

Supported Industry Overlays:
- SaaS: Revenue operations, subscription metrics, customer success
- Banking: Credit risk, regulatory compliance, fraud detection
- Insurance: Claims processing, underwriting, regulatory reporting
- E-commerce: Customer journey, conversion optimization
- Financial Services: Investment compliance, risk management
- IT Services: Project delivery, resource optimization
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class IndustryCode(Enum):
    SAAS = "SaaS"
    BANKING = "BANK"
    INSURANCE = "INSUR"
    ECOMMERCE = "ECOMM"
    FINANCIAL_SERVICES = "FS"
    IT_SERVICES = "IT"

class OverlayType(Enum):
    WORKFLOW_TEMPLATE = "workflow_template"
    BUSINESS_RULE = "business_rule"
    COMPLIANCE_REQUIREMENT = "compliance_requirement"
    KPI_DEFINITION = "kpi_definition"
    DATA_MODEL = "data_model"

@dataclass
class IndustryOverlay:
    """Industry overlay definition"""
    overlay_id: str
    tenant_id: int
    industry_code: IndustryCode
    overlay_name: str
    overlay_type: OverlayType
    overlay_definition: Dict[str, Any]
    compliance_frameworks: List[str]
    version: str = "1.0.0"
    status: str = "active"
    created_at: datetime = None

@dataclass
class SaaSMetrics:
    """SaaS-specific metrics and KPIs"""
    arr: float = 0.0  # Annual Recurring Revenue
    mrr: float = 0.0  # Monthly Recurring Revenue
    ndr: float = 0.0  # Net Dollar Retention
    churn_rate: float = 0.0  # Customer churn rate
    ltv: float = 0.0  # Customer Lifetime Value
    cac: float = 0.0  # Customer Acquisition Cost
    revenue_per_customer: float = 0.0
    expansion_revenue: float = 0.0

@dataclass
class BankingMetrics:
    """Banking-specific metrics and KPIs"""
    npa_ratio: float = 0.0  # Non-Performing Assets ratio
    credit_loss_provision: float = 0.0
    loan_approval_rate: float = 0.0
    aml_alerts: int = 0  # Anti-Money Laundering alerts
    regulatory_violations: int = 0
    risk_weighted_assets: float = 0.0
    capital_adequacy_ratio: float = 0.0

@dataclass
class InsuranceMetrics:
    """Insurance-specific metrics and KPIs"""
    claims_ratio: float = 0.0  # Claims to premium ratio
    underwriting_profit: float = 0.0
    policy_renewal_rate: float = 0.0
    claims_settlement_time: float = 0.0  # Average days
    fraud_detection_rate: float = 0.0
    regulatory_compliance_score: float = 0.0
    reserve_adequacy: float = 0.0

class IndustryOverlayManager:
    """
    Industry Overlay Manager for multi-industry automation
    
    Features:
    - Industry-specific workflow templates
    - Compliance requirement mapping
    - KPI definitions and calculations
    - Business rule customization
    - Performance monitoring
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Overlay cache for performance
        self.overlay_cache: Dict[Tuple[int, str], IndustryOverlay] = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_refresh = {}
        
        # Built-in overlay definitions
        self.built_in_overlays = self._load_built_in_overlays()
    
    async def initialize(self) -> bool:
        """Initialize industry overlay manager"""
        try:
            self.logger.info("🏭 Initializing Industry Overlay Manager...")
            
            # Load overlays into cache
            await self._refresh_overlay_cache()
            
            # Create default overlays for existing tenants
            await self._create_default_overlays()
            
            self.logger.info("✅ Industry Overlay Manager initialized successfully")
            self.logger.info(f"📊 Loaded {len(self.overlay_cache)} industry overlays into cache")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Industry Overlay Manager initialization failed: {e}")
            return False
    
    async def apply_industry_overlay(self, tenant_id: int, workflow_data: Dict[str, Any], industry_code: IndustryCode) -> Dict[str, Any]:
        """
        Apply industry-specific overlay to workflow data (Task 13.1-13.3)
        
        Args:
            tenant_id: Tenant identifier
            workflow_data: Base workflow data
            industry_code: Industry to apply overlay for
            
        Returns:
            Enhanced workflow data with industry overlay applied
        """
        try:
            # Get industry overlays for tenant
            overlays = await self._get_tenant_industry_overlays(tenant_id, industry_code)
            
            enhanced_data = workflow_data.copy()
            
            # Apply each overlay type
            for overlay in overlays:
                if overlay.overlay_type == OverlayType.WORKFLOW_TEMPLATE:
                    enhanced_data = await self._apply_workflow_template_overlay(enhanced_data, overlay)
                elif overlay.overlay_type == OverlayType.BUSINESS_RULE:
                    enhanced_data = await self._apply_business_rule_overlay(enhanced_data, overlay)
                elif overlay.overlay_type == OverlayType.KPI_DEFINITION:
                    enhanced_data = await self._apply_kpi_overlay(enhanced_data, overlay)
                elif overlay.overlay_type == OverlayType.COMPLIANCE_REQUIREMENT:
                    enhanced_data = await self._apply_compliance_overlay(enhanced_data, overlay)
            
            # Add industry-specific metadata
            enhanced_data['industry_overlay'] = {
                'industry_code': industry_code.value,
                'overlays_applied': [o.overlay_name for o in overlays],
                'applied_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Applied {len(overlays)} {industry_code.value} overlays to workflow")
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to apply industry overlay: {e}")
            return workflow_data  # Return original data on error
    
    async def calculate_industry_metrics(self, tenant_id: int, industry_code: IndustryCode, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate industry-specific metrics and KPIs
        
        Args:
            tenant_id: Tenant identifier
            industry_code: Industry for metric calculation
            data: Input data for calculations
            
        Returns:
            Calculated metrics dictionary
        """
        try:
            if industry_code == IndustryCode.SAAS:
                return await self._calculate_saas_metrics(tenant_id, data)
            elif industry_code == IndustryCode.BANKING:
                return await self._calculate_banking_metrics(tenant_id, data)
            elif industry_code == IndustryCode.INSURANCE:
                return await self._calculate_insurance_metrics(tenant_id, data)
            else:
                return await self._calculate_generic_metrics(tenant_id, data)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate {industry_code.value} metrics: {e}")
            return {}
    
    async def create_industry_overlay(self, tenant_id: int, overlay_data: Dict[str, Any]) -> Optional[IndustryOverlay]:
        """
        Create new industry overlay (Task 13.4)
        
        Args:
            tenant_id: Tenant identifier
            overlay_data: Overlay definition data
            
        Returns:
            IndustryOverlay if successful, None otherwise
        """
        try:
            # Validate required fields
            required_fields = ['industry_code', 'overlay_name', 'overlay_type', 'overlay_definition']
            for field in required_fields:
                if field not in overlay_data:
                    raise ValueError(f"Required field missing: {field}")
            
            overlay_id = str(uuid.uuid4())
            
            overlay = IndustryOverlay(
                overlay_id=overlay_id,
                tenant_id=tenant_id,
                industry_code=IndustryCode(overlay_data['industry_code']),
                overlay_name=overlay_data['overlay_name'],
                overlay_type=OverlayType(overlay_data['overlay_type']),
                overlay_definition=overlay_data['overlay_definition'],
                compliance_frameworks=overlay_data.get('compliance_frameworks', []),
                version=overlay_data.get('version', '1.0.0'),
                status=overlay_data.get('status', 'active'),
                created_at=datetime.now()
            )
            
            # Store in database (would be stored in industry_overlays table)
            # For now, we'll use the existing workflow templates table
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                await conn.execute("""
                    INSERT INTO dsl_workflow_templates (
                        tenant_id, template_name, template_type, industry_overlay, category,
                        template_definition, description, tags, compliance_frameworks,
                        created_by_user_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (tenant_id, template_name, version) DO NOTHING
                """,
                    tenant_id,
                    overlay_data['overlay_name'],
                    'OVERLAY',
                    overlay_data['industry_code'],
                    overlay_data['overlay_type'],
                    json.dumps(overlay_data['overlay_definition']),
                    f"Industry overlay for {overlay_data['industry_code']}",
                    [overlay_data['overlay_type']],
                    overlay_data.get('compliance_frameworks', []),
                    1319  # System user
                )
            
            # Update cache
            cache_key = (tenant_id, overlay_id)
            self.overlay_cache[cache_key] = overlay
            self.last_cache_refresh[cache_key] = datetime.now().timestamp()
            
            self.logger.info(f"✅ Created {overlay_data['industry_code']} overlay: {overlay_data['overlay_name']}")
            
            return overlay
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create industry overlay: {e}")
            return None
    
    async def get_overlay_performance_metrics(self, tenant_id: int, industry_code: IndustryCode = None) -> Dict[str, Any]:
        """
        Get overlay performance metrics (Task 13.5)
        
        Args:
            tenant_id: Tenant identifier
            industry_code: Optional industry filter
            
        Returns:
            Performance metrics dictionary
        """
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                # Get overlay usage statistics
                usage_query = """
                    SELECT 
                        industry_overlay,
                        COUNT(*) as usage_count,
                        AVG(execution_time_ms) as avg_execution_time,
                        AVG(trust_score) as avg_trust_score,
                        COUNT(CASE WHEN override_count = 0 THEN 1 END) as compliant_executions,
                        COUNT(CASE WHEN override_count > 0 THEN 1 END) as non_compliant_executions
                    FROM dsl_execution_traces det
                    JOIN dsl_workflows dw ON det.workflow_id = dw.workflow_id
                    WHERE det.tenant_id = $1
                    AND det.created_at >= NOW() - INTERVAL '30 days'
                """
                
                params = [tenant_id]
                
                if industry_code:
                    usage_query += " AND dw.industry_overlay = $2"
                    params.append(industry_code.value)
                
                usage_query += " GROUP BY industry_overlay"
                
                usage_stats = await conn.fetch(usage_query, *params)
                
                # Calculate overall performance metrics
                total_executions = sum(row['usage_count'] for row in usage_stats)
                total_compliant = sum(row['compliant_executions'] for row in usage_stats)
                
                performance_metrics = {
                    'tenant_id': tenant_id,
                    'industry_filter': industry_code.value if industry_code else 'all',
                    'reporting_period': '30_days',
                    'generated_at': datetime.now().isoformat(),
                    'summary': {
                        'total_overlay_executions': total_executions,
                        'compliance_rate': (total_compliant / total_executions * 100) if total_executions > 0 else 100,
                        'industries_active': len(usage_stats)
                    },
                    'by_industry': [dict(row) for row in usage_stats]
                }
                
                return performance_metrics
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get overlay performance metrics: {e}")
            return {}
    
    async def _get_tenant_industry_overlays(self, tenant_id: int, industry_code: IndustryCode) -> List[IndustryOverlay]:
        """Get industry overlays for tenant"""
        try:
            overlays = []
            
            # Get built-in overlays for industry
            if industry_code.value in self.built_in_overlays:
                for overlay_name, overlay_def in self.built_in_overlays[industry_code.value].items():
                    overlay = IndustryOverlay(
                        overlay_id=f"builtin_{industry_code.value}_{overlay_name}",
                        tenant_id=tenant_id,
                        industry_code=industry_code,
                        overlay_name=overlay_name,
                        overlay_type=OverlayType(overlay_def['type']),
                        overlay_definition=overlay_def['definition'],
                        compliance_frameworks=overlay_def.get('compliance_frameworks', []),
                        version="1.0.0",
                        status="active",
                        created_at=datetime.now()
                    )
                    overlays.append(overlay)
            
            # Get custom overlays from database (would query industry_overlays table)
            # For now, using workflow_templates table
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                rows = await conn.fetch("""
                    SELECT template_name, category, template_definition, compliance_frameworks
                    FROM dsl_workflow_templates
                    WHERE tenant_id = $1 
                    AND industry_overlay = $2
                    AND template_type = 'OVERLAY'
                    AND status = 'active'
                """, tenant_id, industry_code.value)
                
                for row in rows:
                    overlay = IndustryOverlay(
                        overlay_id=f"custom_{tenant_id}_{row['template_name']}",
                        tenant_id=tenant_id,
                        industry_code=industry_code,
                        overlay_name=row['template_name'],
                        overlay_type=OverlayType(row['category']),
                        overlay_definition=row['template_definition'],
                        compliance_frameworks=row['compliance_frameworks'] or [],
                        version="1.0.0",
                        status="active",
                        created_at=datetime.now()
                    )
                    overlays.append(overlay)
            
            return overlays
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get tenant industry overlays: {e}")
            return []
    
    async def _apply_workflow_template_overlay(self, workflow_data: Dict[str, Any], overlay: IndustryOverlay) -> Dict[str, Any]:
        """Apply workflow template overlay"""
        enhanced_data = workflow_data.copy()
        
        # Merge overlay definition with workflow data
        overlay_def = overlay.overlay_definition
        
        if 'parameters' in overlay_def:
            enhanced_data.setdefault('parameters', {}).update(overlay_def['parameters'])
        
        if 'agents' in overlay_def:
            enhanced_data.setdefault('agents', []).extend(overlay_def['agents'])
        
        if 'validation_rules' in overlay_def:
            enhanced_data.setdefault('validation_rules', []).extend(overlay_def['validation_rules'])
        
        return enhanced_data
    
    async def _apply_business_rule_overlay(self, workflow_data: Dict[str, Any], overlay: IndustryOverlay) -> Dict[str, Any]:
        """Apply business rule overlay"""
        enhanced_data = workflow_data.copy()
        
        overlay_def = overlay.overlay_definition
        
        if 'thresholds' in overlay_def:
            enhanced_data.setdefault('thresholds', {}).update(overlay_def['thresholds'])
        
        if 'scoring_rules' in overlay_def:
            enhanced_data.setdefault('scoring_rules', []).extend(overlay_def['scoring_rules'])
        
        return enhanced_data
    
    async def _apply_kpi_overlay(self, workflow_data: Dict[str, Any], overlay: IndustryOverlay) -> Dict[str, Any]:
        """Apply KPI definition overlay"""
        enhanced_data = workflow_data.copy()
        
        overlay_def = overlay.overlay_definition
        
        if 'kpi_definitions' in overlay_def:
            enhanced_data.setdefault('kpi_definitions', {}).update(overlay_def['kpi_definitions'])
        
        if 'metric_calculations' in overlay_def:
            enhanced_data.setdefault('metric_calculations', []).extend(overlay_def['metric_calculations'])
        
        return enhanced_data
    
    async def _apply_compliance_overlay(self, workflow_data: Dict[str, Any], overlay: IndustryOverlay) -> Dict[str, Any]:
        """Apply compliance requirement overlay"""
        enhanced_data = workflow_data.copy()
        
        overlay_def = overlay.overlay_definition
        
        if 'compliance_checks' in overlay_def:
            enhanced_data.setdefault('compliance_checks', []).extend(overlay_def['compliance_checks'])
        
        if 'required_approvals' in overlay_def:
            enhanced_data.setdefault('required_approvals', []).extend(overlay_def['required_approvals'])
        
        return enhanced_data
    
    async def _calculate_saas_metrics(self, tenant_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate SaaS-specific metrics"""
        try:
            opportunities = data.get('opportunities', [])
            
            # Calculate basic SaaS metrics
            total_arr = sum(opp.get('amount', 0) for opp in opportunities if opp.get('stage') in ['Closed Won', 'Committed'])
            total_mrr = total_arr / 12 if total_arr > 0 else 0
            
            # Calculate churn rate (simplified)
            closed_lost = len([opp for opp in opportunities if opp.get('stage') == 'Closed Lost'])
            total_deals = len(opportunities)
            churn_rate = (closed_lost / total_deals * 100) if total_deals > 0 else 0
            
            # Revenue per customer
            unique_accounts = len(set(opp.get('account_name', '') for opp in opportunities))
            revenue_per_customer = total_arr / unique_accounts if unique_accounts > 0 else 0
            
            saas_metrics = SaaSMetrics(
                arr=total_arr,
                mrr=total_mrr,
                churn_rate=churn_rate,
                revenue_per_customer=revenue_per_customer
            )
            
            return {
                'industry': 'SaaS',
                'metrics': asdict(saas_metrics),
                'calculated_at': datetime.now().isoformat(),
                'data_points': len(opportunities)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate SaaS metrics: {e}")
            return {}
    
    async def _calculate_banking_metrics(self, tenant_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Banking-specific metrics"""
        try:
            loans = data.get('loans', [])
            
            # Calculate NPA ratio
            total_loans = len(loans)
            npa_loans = len([loan for loan in loans if loan.get('status') == 'NPA'])
            npa_ratio = (npa_loans / total_loans * 100) if total_loans > 0 else 0
            
            # Loan approval rate
            approved_loans = len([loan for loan in loans if loan.get('status') == 'Approved'])
            approval_rate = (approved_loans / total_loans * 100) if total_loans > 0 else 0
            
            banking_metrics = BankingMetrics(
                npa_ratio=npa_ratio,
                loan_approval_rate=approval_rate
            )
            
            return {
                'industry': 'Banking',
                'metrics': asdict(banking_metrics),
                'calculated_at': datetime.now().isoformat(),
                'data_points': len(loans)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate Banking metrics: {e}")
            return {}
    
    async def _calculate_insurance_metrics(self, tenant_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Insurance-specific metrics"""
        try:
            claims = data.get('claims', [])
            
            # Calculate claims ratio
            total_claims_amount = sum(claim.get('amount', 0) for claim in claims)
            total_premium = data.get('total_premium', 1)
            claims_ratio = (total_claims_amount / total_premium * 100) if total_premium > 0 else 0
            
            # Policy renewal rate
            policies = data.get('policies', [])
            renewed_policies = len([policy for policy in policies if policy.get('status') == 'Renewed'])
            total_policies = len(policies)
            renewal_rate = (renewed_policies / total_policies * 100) if total_policies > 0 else 0
            
            insurance_metrics = InsuranceMetrics(
                claims_ratio=claims_ratio,
                policy_renewal_rate=renewal_rate
            )
            
            return {
                'industry': 'Insurance',
                'metrics': asdict(insurance_metrics),
                'calculated_at': datetime.now().isoformat(),
                'data_points': len(claims)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate Insurance metrics: {e}")
            return {}
    
    async def _calculate_generic_metrics(self, tenant_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate generic business metrics"""
        return {
            'industry': 'Generic',
            'metrics': {},
            'calculated_at': datetime.now().isoformat()
        }
    
    async def _create_default_overlays(self) -> None:
        """Create default industry overlays for existing tenants"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Get all active tenants
                tenants = await conn.fetch("""
                    SELECT tenant_id, industry_code FROM tenant_metadata WHERE status = 'active'
                """)
                
                for tenant in tenants:
                    tenant_id = tenant['tenant_id']
                    industry_code = tenant['industry_code']
                    
                    # Set tenant context
                    await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                    
                    # Create default overlays based on industry
                    if industry_code == 'SaaS':
                        await self._create_saas_default_overlays(tenant_id, conn)
                    elif industry_code == 'BANK':
                        await self._create_banking_default_overlays(tenant_id, conn)
                    elif industry_code == 'INSUR':
                        await self._create_insurance_default_overlays(tenant_id, conn)
                
                self.logger.info("✅ Created default industry overlays for all tenants")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create default overlays: {e}")
    
    async def _create_saas_default_overlays(self, tenant_id: int, conn) -> None:
        """Create default SaaS overlays"""
        saas_overlays = [
            {
                'name': 'SaaS Revenue Recognition',
                'type': 'business_rule',
                'definition': {
                    'thresholds': {
                        'high_value_deal': 250000,
                        'enterprise_deal': 500000
                    },
                    'validation_rules': [
                        'revenue_recognition_approval_required',
                        'subscription_terms_validation'
                    ]
                }
            },
            {
                'name': 'SaaS Churn Prevention',
                'type': 'workflow_template',
                'definition': {
                    'agents': ['churn_risk_scoring', 'customer_health_monitoring'],
                    'parameters': {
                        'churn_threshold': 0.7,
                        'engagement_score_weight': 0.4
                    }
                }
            }
        ]
        
        for overlay in saas_overlays:
            await conn.execute("""
                INSERT INTO dsl_workflow_templates (
                    tenant_id, template_name, template_type, industry_overlay, category,
                    template_definition, description, created_by_user_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (tenant_id, template_name, version) DO NOTHING
            """,
                tenant_id,
                overlay['name'],
                'OVERLAY',
                'SaaS',
                overlay['type'],
                json.dumps(overlay['definition']),
                f"Default SaaS overlay: {overlay['name']}",
                1319
            )
    
    async def _create_banking_default_overlays(self, tenant_id: int, conn) -> None:
        """Create default Banking overlays"""
        banking_overlays = [
            {
                'name': 'RBI Credit Risk Assessment',
                'type': 'business_rule',
                'definition': {
                    'thresholds': {
                        'high_risk_loan': 1000000,
                        'npa_threshold': 90
                    },
                    'compliance_checks': [
                        'dual_authorization_required',
                        'credit_bureau_verification'
                    ]
                }
            },
            {
                'name': 'AML Transaction Monitoring',
                'type': 'workflow_template',
                'definition': {
                    'agents': ['aml_screening', 'suspicious_activity_detection'],
                    'parameters': {
                        'transaction_threshold': 1000000,
                        'monitoring_period_days': 30
                    }
                }
            }
        ]
        
        for overlay in banking_overlays:
            await conn.execute("""
                INSERT INTO dsl_workflow_templates (
                    tenant_id, template_name, template_type, industry_overlay, category,
                    template_definition, description, compliance_frameworks, created_by_user_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (tenant_id, template_name, version) DO NOTHING
            """,
                tenant_id,
                overlay['name'],
                'OVERLAY',
                'BANK',
                overlay['type'],
                json.dumps(overlay['definition']),
                f"Default Banking overlay: {overlay['name']}",
                ['RBI', 'DPDP'],
                1319
            )
    
    async def _create_insurance_default_overlays(self, tenant_id: int, conn) -> None:
        """Create default Insurance overlays"""
        insurance_overlays = [
            {
                'name': 'HIPAA Claims Processing',
                'type': 'compliance_requirement',
                'definition': {
                    'compliance_checks': [
                        'phi_access_authorization',
                        'minimum_necessary_standard',
                        'audit_trail_required'
                    ],
                    'required_approvals': [
                        'privacy_officer_approval'
                    ]
                }
            },
            {
                'name': 'Underwriting Risk Assessment',
                'type': 'workflow_template',
                'definition': {
                    'agents': ['risk_scoring', 'fraud_detection', 'policy_validation'],
                    'parameters': {
                        'risk_threshold': 0.8,
                        'fraud_score_weight': 0.3
                    }
                }
            }
        ]
        
        for overlay in insurance_overlays:
            await conn.execute("""
                INSERT INTO dsl_workflow_templates (
                    tenant_id, template_name, template_type, industry_overlay, category,
                    template_definition, description, compliance_frameworks, created_by_user_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (tenant_id, template_name, version) DO NOTHING
            """,
                tenant_id,
                overlay['name'],
                'OVERLAY',
                'INSUR',
                overlay['type'],
                json.dumps(overlay['definition']),
                f"Default Insurance overlay: {overlay['name']}",
                ['HIPAA', 'NAIC'],
                1319
            )
    
    async def _refresh_overlay_cache(self) -> None:
        """Refresh industry overlay cache"""
        # Implementation would refresh cache from database
        pass
    
    def _load_built_in_overlays(self) -> Dict[str, Dict[str, Any]]:
        """Load built-in industry overlay definitions"""
        return {
            'SaaS': {
                'revenue_operations': {
                    'type': 'workflow_template',
                    'definition': {
                        'agents': ['arr_calculation', 'churn_prediction', 'expansion_tracking'],
                        'parameters': {
                            'arr_calculation_method': 'subscription_based',
                            'churn_prediction_model': 'gradient_boosting'
                        }
                    },
                    'compliance_frameworks': ['SOX', 'GDPR']
                },
                'subscription_metrics': {
                    'type': 'kpi_definition',
                    'definition': {
                        'kpi_definitions': {
                            'ARR': 'Annual Recurring Revenue',
                            'MRR': 'Monthly Recurring Revenue',
                            'NDR': 'Net Dollar Retention',
                            'LTV': 'Customer Lifetime Value'
                        }
                    }
                }
            },
            'BANK': {
                'credit_risk_management': {
                    'type': 'business_rule',
                    'definition': {
                        'thresholds': {
                            'high_risk_threshold': 1000000,
                            'npa_classification_days': 90
                        },
                        'scoring_rules': [
                            'credit_bureau_score',
                            'financial_ratio_analysis',
                            'collateral_valuation'
                        ]
                    },
                    'compliance_frameworks': ['RBI', 'DPDP']
                },
                'regulatory_reporting': {
                    'type': 'compliance_requirement',
                    'definition': {
                        'compliance_checks': [
                            'rbi_reporting_requirements',
                            'aml_kyc_compliance',
                            'capital_adequacy_monitoring'
                        ]
                    },
                    'compliance_frameworks': ['RBI']
                }
            },
            'INSUR': {
                'claims_processing': {
                    'type': 'workflow_template',
                    'definition': {
                        'agents': ['claim_validation', 'fraud_detection', 'settlement_calculation'],
                        'parameters': {
                            'fraud_threshold': 0.7,
                            'auto_settlement_limit': 50000
                        }
                    },
                    'compliance_frameworks': ['HIPAA', 'NAIC']
                },
                'underwriting_automation': {
                    'type': 'business_rule',
                    'definition': {
                        'thresholds': {
                            'auto_approval_limit': 100000,
                            'risk_score_threshold': 0.8
                        },
                        'validation_rules': [
                            'medical_history_verification',
                            'financial_capacity_check'
                        ]
                    },
                    'compliance_frameworks': ['NAIC']
                }
            }
        }

# Global instance
_industry_overlay_manager = None

def get_industry_overlay_manager(pool_manager) -> IndustryOverlayManager:
    """Get singleton industry overlay manager instance"""
    global _industry_overlay_manager
    if _industry_overlay_manager is None:
        _industry_overlay_manager = IndustryOverlayManager(pool_manager)
    return _industry_overlay_manager
