"""
RBA-Specific Operators for SaaS Workflows
==========================================

Implements Task 19.3-T03 and 19.3-T04 from RBA Task Sheet:
- SaaS template: Pipeline hygiene + quota compliance  
- SaaS template: Forecast approval governance

Integrates with existing DSL framework, not separate module.
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .base import BaseOperator, OperatorResult, OperatorContext
from .query import QueryOperator
from .decision import DecisionOperator
from .notify import NotifyOperator
from .governance import GovernanceOperator

logger = logging.getLogger(__name__)

@dataclass
class PipelineHygieneResult:
    """Results from pipeline hygiene check"""
    total_opportunities: int
    stale_opportunities: List[Dict[str, Any]]
    hygiene_score: float
    compliance_status: str
    recommended_actions: List[str]

@dataclass
class QuotaComplianceResult:
    """Results from quota compliance check"""
    quota_attainment: float
    pipeline_coverage: float
    forecast_accuracy: float
    compliance_status: str
    risk_factors: List[str]

class PipelineHygieneOperator(BaseOperator):
    """
    Task 19.3-T03: SaaS template - Pipeline hygiene + quota compliance
    
    Checks pipeline health based on configurable rules:
    - Stale opportunities (>15 days no activity)
    - Missing required fields
    - Stage progression anomalies
    - Quota coverage analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.stale_days_threshold = config.get('stale_days_threshold', 15)
        self.required_fields = config.get('required_fields', [
            'amount', 'close_date', 'stage', 'probability'
        ])
        self.min_pipeline_coverage = config.get('min_pipeline_coverage', 3.0)
    
    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate pipeline hygiene operator configuration"""
        errors = []
        
        if config.get('stale_days_threshold', 0) <= 0:
            errors.append("stale_days_threshold must be positive")
        
        required_fields = config.get('required_fields', [])
        if not required_fields or not isinstance(required_fields, list):
            errors.append("required_fields must be a non-empty list")
        
        min_coverage = config.get('min_pipeline_coverage', 0)
        if min_coverage <= 0:
            errors.append("min_pipeline_coverage must be positive")
            
        return errors
    
    async def execute_async(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Execute pipeline hygiene check asynchronously"""
        return await self.execute(context, config)
    
    async def execute(self, context: OperatorContext, config: Dict[str, Any] = None) -> OperatorResult:
        """Execute pipeline hygiene check"""
        try:
            logger.info(f"🔍 Starting pipeline hygiene check for tenant {context.tenant_id}")
            
            # Query opportunities using existing QueryOperator
            query_op = QueryOperator()
            
            query_config = {
                'source': 'salesforce',
                'resource': 'Opportunity',
                'select': ['Id', 'Name', 'Amount', 'CloseDate', 'StageName', 
                          'Probability', 'LastActivityDate', 'OwnerId'],
                'filters': [
                    {'field': 'IsClosed', 'op': 'eq', 'value': False}
                ]
            }
            query_result = await query_op.execute(context, query_config)
            if not query_result.success:
                return OperatorResult(
                    success=False,
                    output_data={},
                    error_message=f"Failed to query opportunities: {query_result.error_message}"
                )
            
            opportunities = query_result.output_data.get('records', [])
            
            # Analyze pipeline hygiene
            hygiene_result = await self._analyze_pipeline_hygiene(opportunities, context)
            
            # Check quota compliance
            quota_result = await self._check_quota_compliance(opportunities, context)
            
            # Determine overall compliance
            overall_status = self._determine_compliance_status(hygiene_result, quota_result)
            
            result_data = {
                'pipeline_hygiene_analysis': {
                    'success': True,
                    'hygiene_score': round(hygiene_result['hygiene_score'], 1),
                    'total_opportunities': hygiene_result['total_opportunities'],
                    'stale_count': hygiene_result['stale_opportunities_count'],
                    'missing_fields_count': hygiene_result['missing_fields_count'],
                    'total_pipeline_value': hygiene_result['total_pipeline_value'],
                    'compliance_status': overall_status,
                    'stale_details': hygiene_result.get('stale_opportunities', []),
                    'recommendations': self._generate_recommendations(hygiene_result, quota_result)
                },
                'quota_compliance': quota_result,
                'execution_metadata': {
                    'tenant_id': context.tenant_id,
                    'execution_id': context.execution_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'data_source': 'salesforce_mock'
                }
            }
            
            # Create governance evidence
            await self._create_evidence_pack(result_data, context)
            
            return OperatorResult(
                success=True,
                output_data=result_data,
                evidence_data={
                    'operator_type': 'pipeline_hygiene',
                    'execution_time_ms': getattr(context, 'execution_time_ms', 0),
                    'compliance_status': overall_status
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Pipeline hygiene check failed: {str(e)}")
            return OperatorResult(
                success=False,
                output_data={},
                error_message=f"Pipeline hygiene execution failed: {str(e)}"
            )
    
    async def _analyze_pipeline_hygiene(self, opportunities: List[Dict], context: OperatorContext) -> Dict:
        """Analyze pipeline for hygiene issues"""
        stale_opportunities = []
        missing_fields_count = 0
        total_pipeline_value = 0
        
        cutoff_date = datetime.now() - timedelta(days=self.stale_days_threshold)
        
        for opp in opportunities:
            # Check for stale opportunities
            last_activity = opp.get('LastActivityDate')
            if last_activity:
                try:
                    # Handle both timezone-aware and naive datetime strings
                    if 'Z' in last_activity:
                        last_activity_date = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
                    elif '+' in last_activity or last_activity.endswith('00:00'):
                        last_activity_date = datetime.fromisoformat(last_activity)
                    else:
                        last_activity_date = datetime.fromisoformat(last_activity)
                    
                    # Make cutoff_date timezone-aware if last_activity_date is timezone-aware
                    if last_activity_date.tzinfo is not None and cutoff_date.tzinfo is None:
                        from datetime import timezone
                        cutoff_date = cutoff_date.replace(tzinfo=timezone.utc)
                    elif last_activity_date.tzinfo is None and cutoff_date.tzinfo is not None:
                        cutoff_date = cutoff_date.replace(tzinfo=None)
                    
                    if last_activity_date < cutoff_date:
                        # Calculate days stale with consistent timezone handling
                        now_for_calc = datetime.now()
                        last_for_calc = last_activity_date
                        if last_activity_date.tzinfo is not None:
                            now_for_calc = datetime.now(last_activity_date.tzinfo)
                        elif last_activity_date.tzinfo is None and now_for_calc.tzinfo is not None:
                            last_for_calc = last_activity_date.replace(tzinfo=now_for_calc.tzinfo)
                        
                        days_stale = (now_for_calc - last_for_calc).days
                        
                        stale_opportunities.append({
                            'id': opp.get('Id'),
                            'name': opp.get('Name'),
                            'amount': opp.get('Amount'),
                            'days_stale': days_stale,
                            'stage': opp.get('StageName')
                        })
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse LastActivityDate '{last_activity}': {str(e)}")
            
            # Check for missing required fields
            missing_fields = []
            for field in self.required_fields:
                if not opp.get(field):
                    missing_fields.append(field)
            
            if missing_fields:
                missing_fields_count += 1
            
            # Calculate total pipeline value
            amount = opp.get('Amount', 0)
            if amount:
                total_pipeline_value += float(amount)
        
        # Calculate hygiene score
        total_opps = len(opportunities)
        hygiene_score = 100.0
        
        if total_opps > 0:
            stale_penalty = (len(stale_opportunities) / total_opps) * 40
            missing_fields_penalty = (missing_fields_count / total_opps) * 30
            hygiene_score = max(0, 100 - stale_penalty - missing_fields_penalty)
        
        return {
            'total_opportunities': total_opps,
            'stale_opportunities': stale_opportunities,
            'stale_opportunities_count': len(stale_opportunities),
            'missing_fields_count': missing_fields_count,
            'total_pipeline_value': total_pipeline_value,
            'hygiene_score': round(hygiene_score, 2),
            'compliance_status': 'COMPLIANT' if hygiene_score >= 80 else 'NON_COMPLIANT',
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    async def _check_quota_compliance(self, opportunities: List[Dict], context: OperatorContext) -> Dict:
        """Check quota compliance and coverage"""
        # This would integrate with quota data from Salesforce/Fabric
        # For now, implementing basic pipeline coverage logic
        
        total_pipeline_value = sum(float(opp.get('Amount', 0)) for opp in opportunities)
        
        # Mock quota data - in real implementation, fetch from Salesforce/Fabric
        quarterly_quota = 1000000  # $1M quarterly quota
        
        pipeline_coverage = total_pipeline_value / quarterly_quota if quarterly_quota > 0 else 0
        quota_attainment = 0.65  # Mock current attainment
        
        compliance_status = 'COMPLIANT' if pipeline_coverage >= self.min_pipeline_coverage else 'AT_RISK'
        
        risk_factors = []
        if pipeline_coverage < self.min_pipeline_coverage:
            risk_factors.append(f"Pipeline coverage below {self.min_pipeline_coverage}x quota")
        
        if quota_attainment < 0.8:
            risk_factors.append("Quota attainment below 80%")
        
        return {
            'quota_attainment': quota_attainment,
            'pipeline_coverage': round(pipeline_coverage, 2),
            'total_pipeline_value': total_pipeline_value,
            'quarterly_quota': quarterly_quota,
            'compliance_status': compliance_status,
            'risk_factors': risk_factors
        }
    
    def _determine_compliance_status(self, hygiene_result: Dict, quota_result: Dict) -> str:
        """Determine overall compliance status"""
        hygiene_compliant = hygiene_result.get('compliance_status') == 'COMPLIANT'
        quota_compliant = quota_result.get('compliance_status') == 'COMPLIANT'
        
        if hygiene_compliant and quota_compliant:
            return 'FULLY_COMPLIANT'
        elif hygiene_compliant or quota_compliant:
            return 'PARTIALLY_COMPLIANT'
        else:
            return 'NON_COMPLIANT'
    
    async def _create_evidence_pack(self, result_data: Dict, context: OperatorContext):
        """Create evidence pack for audit trail"""
        evidence_op = GovernanceOperator({
            'action': 'create_evidence',
            'evidence_type': 'pipeline_hygiene_audit',
            'retention_days': 2555  # 7 years for SOX compliance
        })
        
        await evidence_op.execute(context, {})
    
    def _generate_recommendations(self, hygiene_result: Dict, quota_result: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if hygiene_result['stale_opportunities_count'] > 0:
            recommendations.append(f"Follow up on {hygiene_result['stale_opportunities_count']} stale opportunities (>15 days without activity)")
        
        if hygiene_result['missing_fields_count'] > 0:
            recommendations.append(f"Update {hygiene_result['missing_fields_count']} opportunities with missing critical fields")
        
        if hygiene_result['hygiene_score'] < 70:
            recommendations.append("Pipeline hygiene score is below 70% - immediate attention required")
        elif hygiene_result['hygiene_score'] < 85:
            recommendations.append("Pipeline hygiene score can be improved - review data quality processes")
        
        if not quota_result.get('quota_met', True):
            recommendations.append("Pipeline coverage is below quota requirements - increase prospecting activities")
        
        if hygiene_result['total_opportunities'] < 5:
            recommendations.append("Low opportunity count - consider expanding target market or increasing lead generation")
        
        if len(recommendations) == 0:
            recommendations.append("Pipeline hygiene is good - maintain current processes")
        
        return recommendations

class ForecastApprovalOperator(BaseOperator):
    """
    Task 19.3-T04: SaaS template - Forecast approval governance
    
    Implements forecast approval workflow with governance:
    - Variance analysis
    - Approval routing based on variance thresholds
    - Evidence pack creation for audit
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.variance_thresholds = config.get('variance_thresholds', {
            'low': 0.05,    # 5% variance - auto-approve
            'medium': 0.15, # 15% variance - manager approval
            'high': 0.25    # 25% variance - director approval
        })
        self.auto_approve_threshold = config.get('auto_approve_threshold', 0.05)
    
    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate forecast approval operator configuration"""
        errors = []
        
        thresholds = config.get('variance_thresholds', {})
        if not isinstance(thresholds, dict):
            errors.append("variance_thresholds must be a dictionary")
        else:
            required_keys = ['low', 'medium', 'high']
            for key in required_keys:
                if key not in thresholds:
                    errors.append(f"variance_thresholds missing '{key}' threshold")
                elif not isinstance(thresholds[key], (int, float)) or thresholds[key] < 0:
                    errors.append(f"variance_thresholds['{key}'] must be a non-negative number")
        
        auto_threshold = config.get('auto_approve_threshold', 0)
        if not isinstance(auto_threshold, (int, float)) or auto_threshold < 0:
            errors.append("auto_approve_threshold must be a non-negative number")
            
        return errors
    
    async def execute_async(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Execute forecast approval workflow asynchronously"""
        return await self.execute(context, config)
    
    async def execute(self, context: OperatorContext, config: Dict[str, Any] = None) -> OperatorResult:
        """Execute forecast approval workflow"""
        try:
            logger.info(f"🎯 Starting forecast approval workflow for tenant {context.tenant_id}")
            
            # Get forecast data
            forecast_data = context.input_data.get('forecast', {})
            current_forecast = forecast_data.get('current_amount', 0)
            previous_forecast = forecast_data.get('previous_amount', 0)
            
            # Calculate variance
            variance_analysis = self._calculate_variance(current_forecast, previous_forecast)
            
            # Determine approval requirements
            approval_requirements = self._determine_approval_requirements(variance_analysis)
            
            # Execute approval workflow
            approval_result = await self._execute_approval_workflow(
                forecast_data, variance_analysis, approval_requirements, context
            )
            
            result_data = {
                'forecast_data': forecast_data,
                'variance_analysis': variance_analysis,
                'approval_requirements': approval_requirements,
                'approval_result': approval_result,
                'timestamp': datetime.utcnow().isoformat(),
                'tenant_id': context.tenant_id
            }
            
            # Create governance evidence
            await self._create_approval_evidence(result_data, context)
            
            return OperatorResult(
                success=True,
                output_data=result_data,
                metadata={
                    'operator_type': 'forecast_approval',
                    'approval_status': approval_result.get('status'),
                    'variance_level': variance_analysis.get('level')
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Forecast approval failed: {str(e)}")
            return OperatorResult(
                success=False,
                output_data={},
                error_message=f"Forecast approval execution failed: {str(e)}"
            )
    
    def _calculate_variance(self, current: float, previous: float) -> Dict:
        """Calculate forecast variance"""
        if previous == 0:
            variance_percent = 0
        else:
            variance_percent = abs(current - previous) / previous
        
        variance_amount = current - previous
        
        # Determine variance level
        if variance_percent <= self.variance_thresholds['low']:
            level = 'LOW'
        elif variance_percent <= self.variance_thresholds['medium']:
            level = 'MEDIUM'
        else:
            level = 'HIGH'
        
        return {
            'variance_amount': variance_amount,
            'variance_percent': round(variance_percent * 100, 2),
            'level': level,
            'current_forecast': current,
            'previous_forecast': previous
        }
    
    def _determine_approval_requirements(self, variance_analysis: Dict) -> Dict:
        """Determine what approvals are needed"""
        level = variance_analysis.get('level')
        
        if level == 'LOW':
            return {
                'auto_approve': True,
                'required_approvers': [],
                'approval_reason': 'Low variance - auto-approved'
            }
        elif level == 'MEDIUM':
            return {
                'auto_approve': False,
                'required_approvers': ['MANAGER'],
                'approval_reason': 'Medium variance - manager approval required'
            }
        else:  # HIGH
            return {
                'auto_approve': False,
                'required_approvers': ['MANAGER', 'DIRECTOR'],
                'approval_reason': 'High variance - director approval required'
            }
    
    async def _execute_approval_workflow(self, forecast_data: Dict, variance_analysis: Dict, 
                                       approval_requirements: Dict, context: OperatorContext) -> Dict:
        """Execute the approval workflow"""
        
        if approval_requirements.get('auto_approve'):
            return {
                'status': 'APPROVED',
                'approval_type': 'AUTOMATIC',
                'approved_by': 'SYSTEM',
                'approved_at': datetime.utcnow().isoformat(),
                'comments': 'Auto-approved due to low variance'
            }
        
        # For manual approvals, this would integrate with approval system
        # For now, return pending status
        return {
            'status': 'PENDING_APPROVAL',
            'approval_type': 'MANUAL',
            'required_approvers': approval_requirements.get('required_approvers', []),
            'submitted_at': datetime.utcnow().isoformat(),
            'approval_deadline': (datetime.utcnow() + timedelta(hours=24)).isoformat()
        }
    
    async def _create_approval_evidence(self, result_data: Dict, context: OperatorContext):
        """Create evidence pack for forecast approval audit"""
        evidence_op = GovernanceOperator({
            'action': 'create_evidence',
            'evidence_type': 'forecast_approval_audit',
            'retention_days': 2555  # 7 years for SOX compliance
        })
        
        await evidence_op.execute(context, {})

# Integration with existing DSL framework
RBA_OPERATORS = {
    'pipeline_hygiene': PipelineHygieneOperator,
    'forecast_approval': ForecastApprovalOperator
}

def register_rba_operators():
    """Register RBA operators with the DSL framework"""
    # Return RBA operators for registration
    return RBA_OPERATORS
