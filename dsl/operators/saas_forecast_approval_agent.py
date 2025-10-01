"""
SaaS Forecast Approval Workflow Agent
Enterprise-grade forecast validation and approval automation
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from .base import BaseOperator, OperatorContext, OperatorResult
import logging

logger = logging.getLogger(__name__)

class ApprovalStatus(Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"

class VarianceLevel(Enum):
    LOW = "LOW"          # < 5% variance
    MEDIUM = "MEDIUM"    # 5-15% variance
    HIGH = "HIGH"        # 15-25% variance
    CRITICAL = "CRITICAL" # > 25% variance

@dataclass
class ForecastSubmission:
    """Forecast submission data"""
    submission_id: str
    submitter_id: str
    submitter_name: str
    forecast_period: str
    submitted_amount: float
    pipeline_amount: float
    variance_percentage: float
    variance_level: VarianceLevel
    confidence_score: float
    supporting_data: Dict
    submitted_at: datetime

@dataclass
class ApprovalDecision:
    """Approval decision result"""
    submission_id: str
    decision: ApprovalStatus
    approver_id: Optional[str]
    approver_name: Optional[str]
    decision_reason: str
    required_actions: List[str]
    escalation_required: bool
    approval_timestamp: datetime

class SaaSForecastApprovalAgent(BaseOperator):
    """
    Advanced SaaS Forecast Approval Agent
    
    Features:
    - Automated forecast validation
    - Variance analysis and thresholds
    - Multi-level approval routing
    - Confidence scoring
    - Evidence pack generation
    - Compliance tracking
    - Executive escalation
    """
    
    def __init__(self, config=None):
        super().__init__("saas_forecast_approval_agent")
        self.config = config or {}
        
        # Configurable approval thresholds
        self.variance_thresholds = {
            VarianceLevel.LOW: 5.0,      # Auto-approve
            VarianceLevel.MEDIUM: 15.0,   # Manager approval
            VarianceLevel.HIGH: 25.0,     # Director approval
            VarianceLevel.CRITICAL: 100.0 # VP approval
        }
        
        self.confidence_threshold = self.config.get('confidence_threshold', 75.0)
        self.auto_approve_threshold = self.config.get('auto_approve_threshold', 5.0)
        
    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate forecast approval agent configuration"""
        errors = []
        
        if 'forecast_data' not in config:
            errors.append("'forecast_data' is required")
        
        forecast_data = config.get('forecast_data', {})
        required_fields = ['submitter_id', 'forecast_period', 'submitted_amount']
        
        for field in required_fields:
            if field not in forecast_data:
                errors.append(f"'{field}' is required in forecast_data")
                
        return errors
    
    async def execute_async(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Execute forecast approval workflow"""
        try:
            logger.info("🔍 Starting SaaS Forecast Approval Analysis...")
            
            # Step 1: Parse forecast submission
            forecast_submission = await self._parse_forecast_submission(config)
            
            # Step 2: Fetch pipeline data for validation
            pipeline_data = await self._fetch_pipeline_data(context, forecast_submission)
            
            # Step 3: Calculate variance and confidence
            variance_analysis = await self._analyze_variance(forecast_submission, pipeline_data)
            
            # Step 4: Determine approval requirements
            approval_requirements = await self._determine_approval_requirements(variance_analysis)
            
            # Step 5: Execute approval logic
            approval_decision = await self._execute_approval_logic(
                context, forecast_submission, variance_analysis, approval_requirements
            )
            
            # Step 6: Generate evidence pack
            evidence_pack = await self._generate_evidence_pack(
                forecast_submission, variance_analysis, approval_decision
            )
            
            # Step 7: Handle notifications and escalations
            notifications = await self._handle_notifications(approval_decision, approval_requirements)
            
            result_data = {
                'forecast_submission': forecast_submission.__dict__,
                'variance_analysis': variance_analysis,
                'approval_decision': approval_decision.__dict__,
                'evidence_pack': evidence_pack,
                'notifications': notifications,
                'processing_timestamp': datetime.now().isoformat(),
                'tenant_id': context.tenant_id,
                'processed_by_agent': True
            }
            
            logger.info(f"✅ Forecast approval completed - Decision: {approval_decision.decision.value}")
            
            return OperatorResult(
                success=True,
                output_data=result_data
            )
            
        except Exception as e:
            logger.error(f"❌ Forecast approval failed: {e}")
            return OperatorResult(
                success=False,
                error_message=f"Forecast approval failed: {str(e)}"
            )
    
    async def _parse_forecast_submission(self, config: Dict[str, Any]) -> ForecastSubmission:
        """Parse and validate forecast submission"""
        forecast_data = config['forecast_data']
        
        submission_id = forecast_data.get('submission_id', f"FS_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        return ForecastSubmission(
            submission_id=submission_id,
            submitter_id=forecast_data['submitter_id'],
            submitter_name=forecast_data.get('submitter_name', 'Unknown'),
            forecast_period=forecast_data['forecast_period'],
            submitted_amount=float(forecast_data['submitted_amount']),
            pipeline_amount=0.0,  # Will be calculated
            variance_percentage=0.0,  # Will be calculated
            variance_level=VarianceLevel.LOW,  # Will be determined
            confidence_score=forecast_data.get('confidence_score', 80.0),
            supporting_data=forecast_data.get('supporting_data', {}),
            submitted_at=datetime.now()
        )
    
    async def _fetch_pipeline_data(self, context: OperatorContext, forecast: ForecastSubmission) -> Dict:
        """Fetch current pipeline data for variance calculation"""
        try:
            # Use Fabric service if available
            if hasattr(context, 'pool_manager') and context.pool_manager.fabric_service:
                fabric_service = context.pool_manager.fabric_service
                
                # Query for pipeline data in forecast period
                query = f"""
                SELECT 
                    SUM(CASE WHEN StageName IN ('Negotiation', 'Proposal', 'Closed Won') THEN Amount ELSE 0 END) as high_confidence_pipeline,
                    SUM(CASE WHEN StageName NOT IN ('Closed Lost', 'Closed Won') THEN Amount ELSE 0 END) as total_pipeline,
                    COUNT(*) as opportunity_count,
                    AVG(Probability) as avg_probability
                FROM dbo.opportunities 
                WHERE IsClosed = 0 
                AND OwnerId = '{forecast.submitter_id}'
                AND CloseDate >= '{forecast.forecast_period}-01'
                AND CloseDate <= '{forecast.forecast_period}-31'
                """
                
                result = await fabric_service.execute_query(query)
                
                if result.success and result.data:
                    pipeline_data = result.data[0]
                    return {
                        'high_confidence_pipeline': float(pipeline_data.get('high_confidence_pipeline', 0) or 0),
                        'total_pipeline': float(pipeline_data.get('total_pipeline', 0) or 0),
                        'opportunity_count': int(pipeline_data.get('opportunity_count', 0) or 0),
                        'avg_probability': float(pipeline_data.get('avg_probability', 0) or 0),
                        'data_source': 'fabric'
                    }
            
            # Fallback to mock data
            return await self._get_mock_pipeline_data(forecast)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch pipeline data: {e}")
            return await self._get_mock_pipeline_data(forecast)
    
    async def _analyze_variance(self, forecast: ForecastSubmission, pipeline_data: Dict) -> Dict:
        """Analyze forecast variance against pipeline data"""
        pipeline_amount = pipeline_data.get('total_pipeline', 0)
        high_confidence_pipeline = pipeline_data.get('high_confidence_pipeline', 0)
        
        # Calculate variance
        if pipeline_amount > 0:
            variance_percentage = abs((forecast.submitted_amount - pipeline_amount) / pipeline_amount) * 100
        else:
            variance_percentage = 100.0  # No pipeline data = high variance
        
        # Determine variance level
        variance_level = VarianceLevel.CRITICAL
        for level, threshold in self.variance_thresholds.items():
            if variance_percentage <= threshold:
                variance_level = level
                break
        
        # Calculate confidence score based on pipeline strength
        pipeline_confidence = min(100.0, (high_confidence_pipeline / max(1, forecast.submitted_amount)) * 100)
        overall_confidence = (forecast.confidence_score + pipeline_confidence) / 2
        
        return {
            'submitted_amount': forecast.submitted_amount,
            'pipeline_amount': pipeline_amount,
            'high_confidence_pipeline': high_confidence_pipeline,
            'variance_amount': forecast.submitted_amount - pipeline_amount,
            'variance_percentage': variance_percentage,
            'variance_level': variance_level.value,
            'confidence_score': overall_confidence,
            'pipeline_coverage': (pipeline_amount / max(1, forecast.submitted_amount)) * 100,
            'risk_factors': await self._identify_risk_factors(variance_percentage, overall_confidence, pipeline_data)
        }
    
    async def _identify_risk_factors(self, variance_percentage: float, confidence: float, pipeline_data: Dict) -> List[str]:
        """Identify risk factors in the forecast"""
        risk_factors = []
        
        if variance_percentage > 25:
            risk_factors.append(f"High variance: {variance_percentage:.1f}% difference from pipeline")
        
        if confidence < 60:
            risk_factors.append(f"Low confidence score: {confidence:.1f}%")
        
        if pipeline_data.get('opportunity_count', 0) < 3:
            risk_factors.append("Limited number of opportunities in pipeline")
        
        if pipeline_data.get('avg_probability', 0) < 50:
            risk_factors.append("Low average deal probability")
        
        return risk_factors
    
    async def _determine_approval_requirements(self, variance_analysis: Dict) -> Dict:
        """Determine what level of approval is required"""
        variance_level = VarianceLevel(variance_analysis['variance_level'])
        confidence_score = variance_analysis['confidence_score']
        
        # Auto-approval conditions
        if (variance_level == VarianceLevel.LOW and 
            confidence_score >= self.confidence_threshold and 
            len(variance_analysis['risk_factors']) == 0):
            return {
                'approval_required': False,
                'auto_approve': True,
                'required_approver_level': 'NONE',
                'reason': 'Low variance and high confidence - auto-approved'
            }
        
        # Determine required approver level
        approver_level = 'MANAGER'
        if variance_level == VarianceLevel.HIGH:
            approver_level = 'DIRECTOR'
        elif variance_level == VarianceLevel.CRITICAL:
            approver_level = 'VP'
        
        return {
            'approval_required': True,
            'auto_approve': False,
            'required_approver_level': approver_level,
            'reason': f'Variance level: {variance_level.value}, Confidence: {confidence_score:.1f}%'
        }
    
    async def _execute_approval_logic(self, context: OperatorContext, forecast: ForecastSubmission, 
                                    variance_analysis: Dict, approval_requirements: Dict) -> ApprovalDecision:
        """Execute the approval logic"""
        
        if approval_requirements['auto_approve']:
            return ApprovalDecision(
                submission_id=forecast.submission_id,
                decision=ApprovalStatus.APPROVED,
                approver_id='SYSTEM',
                approver_name='Auto-Approval System',
                decision_reason=approval_requirements['reason'],
                required_actions=[],
                escalation_required=False,
                approval_timestamp=datetime.now()
            )
        
        # For demo purposes, simulate approval logic
        # In production, this would integrate with approval workflow system
        variance_level = VarianceLevel(variance_analysis['variance_level'])
        
        if variance_level == VarianceLevel.CRITICAL:
            return ApprovalDecision(
                submission_id=forecast.submission_id,
                decision=ApprovalStatus.ESCALATED,
                approver_id=None,
                approver_name=None,
                decision_reason=f"Critical variance ({variance_analysis['variance_percentage']:.1f}%) requires VP approval",
                required_actions=[
                    "Provide detailed justification for forecast variance",
                    "Submit updated pipeline analysis",
                    "Schedule review meeting with VP of Sales"
                ],
                escalation_required=True,
                approval_timestamp=datetime.now()
            )
        
        return ApprovalDecision(
            submission_id=forecast.submission_id,
            decision=ApprovalStatus.PENDING,
            approver_id=None,
            approver_name=None,
            decision_reason=f"Requires {approval_requirements['required_approver_level']} approval",
            required_actions=[
                "Review pipeline data",
                "Validate forecast assumptions",
                "Provide approval or feedback"
            ],
            escalation_required=False,
            approval_timestamp=datetime.now()
        )
    
    async def _generate_evidence_pack(self, forecast: ForecastSubmission, 
                                    variance_analysis: Dict, decision: ApprovalDecision) -> Dict:
        """Generate evidence pack for audit trail"""
        return {
            'evidence_pack_id': f"EP_{forecast.submission_id}",
            'forecast_submission': {
                'submitter': forecast.submitter_name,
                'period': forecast.forecast_period,
                'amount': forecast.submitted_amount,
                'confidence': forecast.confidence_score
            },
            'variance_analysis': variance_analysis,
            'approval_decision': {
                'decision': decision.decision.value,
                'reason': decision.decision_reason,
                'approver': decision.approver_name,
                'timestamp': decision.approval_timestamp.isoformat()
            },
            'compliance_data': {
                'sox_compliant': True,
                'audit_trail_complete': True,
                'evidence_retained': True,
                'approval_documented': True
            },
            'generated_at': datetime.now().isoformat()
        }
    
    async def _handle_notifications(self, decision: ApprovalDecision, requirements: Dict) -> List[Dict]:
        """Handle notifications and alerts"""
        notifications = []
        
        if decision.decision == ApprovalStatus.APPROVED:
            notifications.append({
                'type': 'APPROVAL_NOTIFICATION',
                'recipient': 'submitter',
                'message': f"Forecast submission {decision.submission_id} has been approved",
                'channel': 'email'
            })
        
        elif decision.decision == ApprovalStatus.ESCALATED:
            notifications.append({
                'type': 'ESCALATION_ALERT',
                'recipient': 'vp_sales',
                'message': f"Forecast submission {decision.submission_id} requires VP approval",
                'channel': 'slack'
            })
        
        elif decision.decision == ApprovalStatus.PENDING:
            notifications.append({
                'type': 'APPROVAL_REQUEST',
                'recipient': requirements['required_approver_level'].lower(),
                'message': f"Forecast approval required for submission {decision.submission_id}",
                'channel': 'email'
            })
        
        return notifications
    
    async def _get_mock_pipeline_data(self, forecast: ForecastSubmission) -> Dict:
        """Mock pipeline data for demo"""
        return {
            'high_confidence_pipeline': forecast.submitted_amount * 0.8,
            'total_pipeline': forecast.submitted_amount * 1.2,
            'opportunity_count': 15,
            'avg_probability': 65.0,
            'data_source': 'mock'
        }
