#!/usr/bin/env python3
"""
AI Approval Assistant API - Chapter 14.1 Tasks T40-T41
======================================================
Tasks 14.1-T40, T41: AI agent approver suggestions and override reviewers

Features:
- AI-powered approver suggestions based on workload, expertise, and availability
- AI override risk assessment and flagging
- Machine learning-based approval pattern analysis
- Intelligent escalation path recommendations
- Risk-based approval routing
- Historical pattern learning for better suggestions
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
import json
import asyncio

from src.database.connection_pool import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter()

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ApproverSuggestionRequest(BaseModel):
    workflow_id: str = Field(..., description="Workflow ID")
    request_type: str = Field(..., description="Approval request type")
    financial_amount: Optional[float] = Field(None, description="Financial amount involved")
    urgency_level: str = Field("medium", description="Urgency level")
    industry_context: str = Field("SaaS", description="Industry context")
    compliance_frameworks: List[str] = Field([], description="Required compliance frameworks")
    current_approvers: List[int] = Field([], description="Currently assigned approvers")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")

class OverrideReviewRequest(BaseModel):
    override_id: str = Field(..., description="Override ID")
    workflow_id: str = Field(..., description="Associated workflow ID")
    override_reason: str = Field(..., description="Override reason")
    reason_code: str = Field(..., description="Override reason code")
    requested_by: int = Field(..., description="User who requested override")
    financial_impact: Optional[float] = Field(None, description="Financial impact")
    compliance_impact: List[str] = Field([], description="Compliance frameworks affected")
    historical_context: Dict[str, Any] = Field({}, description="Historical context")

class ApproverSuggestion(BaseModel):
    approver_id: int
    approver_name: str
    role: str
    suggestion_score: float
    reasoning: List[str]
    availability_status: str
    current_workload: int
    expertise_match: float
    estimated_response_time_hours: float

class OverrideRiskAssessment(BaseModel):
    risk_level: RiskLevel
    risk_score: float
    risk_factors: List[str]
    compliance_violations: List[str]
    recommended_actions: List[str]
    requires_escalation: bool
    escalation_path: List[str]

class AIApprovalAssistantService:
    """
    AI-powered approval assistant service
    Tasks 14.1-T40, T41: AI approver suggestions and override risk assessment
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        
        # AI model configuration
        self.ai_config = {
            "approver_suggestion_model": "gpt-4",
            "risk_assessment_model": "gpt-4",
            "confidence_threshold": 0.7,
            "max_suggestions": 5,
            "learning_window_days": 90
        }
        
        # Expertise mapping for different domains
        self.expertise_domains = {
            "financial_approvals": ["finance_manager", "cfo", "controller"],
            "compliance_approvals": ["compliance_officer", "legal_counsel", "audit_manager"],
            "technical_approvals": ["cto", "engineering_manager", "security_officer"],
            "sales_approvals": ["sales_manager", "sales_director", "vp_sales", "cro"],
            "hr_approvals": ["hr_manager", "hr_director", "chro"],
            "procurement_approvals": ["procurement_manager", "operations_director", "coo"]
        }
        
        # Risk assessment criteria
        self.risk_criteria = {
            "financial_thresholds": {
                "low": 10000,
                "medium": 100000,
                "high": 500000,
                "critical": 1000000
            },
            "compliance_risk_multipliers": {
                "SOX": 1.5,
                "GDPR": 1.3,
                "HIPAA": 1.4,
                "PCI_DSS": 1.2,
                "RBI": 1.6,
                "IRDAI": 1.4
            },
            "frequency_risk_thresholds": {
                "daily_overrides": 3,
                "weekly_overrides": 10,
                "monthly_overrides": 25
            }
        }
    
    async def suggest_approvers(self, request: ApproverSuggestionRequest, 
                              tenant_id: int) -> List[ApproverSuggestion]:
        """
        AI-powered approver suggestions (Task 14.1-T40)
        """
        try:
            # Get eligible approvers based on request context
            eligible_approvers = await self._get_eligible_approvers(
                request.request_type, request.compliance_frameworks, tenant_id
            )
            
            # Filter out currently assigned approvers
            available_approvers = [
                approver for approver in eligible_approvers 
                if approver['user_id'] not in request.current_approvers
            ]
            
            if not available_approvers:
                return []
            
            # Calculate suggestion scores for each approver
            suggestions = []
            for approver in available_approvers:
                suggestion = await self._calculate_approver_suggestion_score(
                    approver, request, tenant_id
                )
                if suggestion.suggestion_score >= self.ai_config["confidence_threshold"]:
                    suggestions.append(suggestion)
            
            # Sort by suggestion score and return top N
            suggestions.sort(key=lambda x: x.suggestion_score, reverse=True)
            return suggestions[:self.ai_config["max_suggestions"]]
            
        except Exception as e:
            logger.error(f"❌ AI approver suggestion failed: {e}")
            return []
    
    async def assess_override_risk(self, request: OverrideReviewRequest, 
                                 tenant_id: int) -> OverrideRiskAssessment:
        """
        AI-powered override risk assessment (Task 14.1-T41)
        """
        try:
            # Analyze override context and history
            risk_factors = await self._analyze_override_risk_factors(request, tenant_id)
            
            # Calculate base risk score
            base_risk_score = await self._calculate_base_risk_score(request, tenant_id)
            
            # Apply compliance multipliers
            compliance_adjusted_score = self._apply_compliance_risk_multipliers(
                base_risk_score, request.compliance_impact
            )
            
            # Analyze historical patterns
            historical_risk = await self._analyze_historical_override_patterns(
                request.requested_by, request.reason_code, tenant_id
            )
            
            # Calculate final risk score
            final_risk_score = min(1.0, (compliance_adjusted_score + historical_risk) / 2)
            
            # Determine risk level
            risk_level = self._determine_risk_level(final_risk_score)
            
            # Generate recommendations
            recommended_actions = await self._generate_risk_recommendations(
                risk_level, risk_factors, request
            )
            
            # Determine escalation requirements
            requires_escalation = risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            escalation_path = await self._generate_escalation_path(
                risk_level, request.compliance_impact, tenant_id
            ) if requires_escalation else []
            
            # Check for compliance violations
            compliance_violations = await self._check_compliance_violations(request, tenant_id)
            
            return OverrideRiskAssessment(
                risk_level=risk_level,
                risk_score=final_risk_score,
                risk_factors=risk_factors,
                compliance_violations=compliance_violations,
                recommended_actions=recommended_actions,
                requires_escalation=requires_escalation,
                escalation_path=escalation_path
            )
            
        except Exception as e:
            logger.error(f"❌ Override risk assessment failed: {e}")
            return OverrideRiskAssessment(
                risk_level=RiskLevel.HIGH,
                risk_score=0.8,
                risk_factors=["Assessment error occurred"],
                compliance_violations=[],
                recommended_actions=["Manual review required"],
                requires_escalation=True,
                escalation_path=["compliance_officer", "legal_counsel"]
            )
    
    async def learn_from_approval_patterns(self, tenant_id: int, 
                                         lookback_days: int = 90) -> Dict[str, Any]:
        """
        Learn from historical approval patterns to improve suggestions
        """
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                # Analyze successful approval patterns
                success_patterns = await conn.fetch("""
                    SELECT 
                        request_type,
                        approval_chain,
                        AVG(EXTRACT(EPOCH FROM (completed_at - created_at))/3600) as avg_time_hours,
                        COUNT(*) as success_count,
                        AVG(CASE WHEN status = 'approved' THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM approval_ledger
                    WHERE tenant_id = $1 
                    AND created_at >= NOW() - INTERVAL '%s days'
                    AND status IN ('approved', 'rejected')
                    GROUP BY request_type, approval_chain
                    HAVING COUNT(*) >= 5
                    ORDER BY success_rate DESC, avg_time_hours ASC
                """, tenant_id, lookback_days)
                
                # Analyze override patterns
                override_patterns = await conn.fetch("""
                    SELECT 
                        reason_code,
                        requested_by_user_id,
                        COUNT(*) as override_count,
                        AVG(CASE WHEN status = 'approved' THEN 1.0 ELSE 0.0 END) as approval_rate
                    FROM override_ledger
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY reason_code, requested_by_user_id
                    HAVING COUNT(*) >= 3
                    ORDER BY override_count DESC
                """, tenant_id, lookback_days)
                
                # Extract learning insights
                learning_insights = {
                    "optimal_approval_chains": [],
                    "high_risk_override_patterns": [],
                    "efficient_approvers": [],
                    "learning_period_days": lookback_days,
                    "analyzed_at": datetime.utcnow().isoformat()
                }
                
                # Process success patterns
                for pattern in success_patterns:
                    if pattern['success_rate'] > 0.8 and pattern['avg_time_hours'] < 48:
                        learning_insights["optimal_approval_chains"].append({
                            "request_type": pattern['request_type'],
                            "approval_chain": json.loads(pattern['approval_chain']),
                            "success_rate": float(pattern['success_rate']),
                            "avg_time_hours": float(pattern['avg_time_hours']),
                            "sample_size": pattern['success_count']
                        })
                
                # Process override patterns
                for pattern in override_patterns:
                    if pattern['override_count'] > 5 and pattern['approval_rate'] < 0.5:
                        learning_insights["high_risk_override_patterns"].append({
                            "reason_code": pattern['reason_code'],
                            "user_id": pattern['requested_by_user_id'],
                            "override_frequency": pattern['override_count'],
                            "approval_rate": float(pattern['approval_rate'])
                        })
                
                return learning_insights
                
        except Exception as e:
            logger.error(f"❌ Pattern learning failed: {e}")
            return {"error": str(e)}
    
    # Helper methods
    async def _get_eligible_approvers(self, request_type: str, 
                                    compliance_frameworks: List[str], 
                                    tenant_id: int) -> List[Dict[str, Any]]:
        """Get eligible approvers for the request"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                # Get users with approval permissions
                rows = await conn.fetch("""
                    SELECT 
                        u.user_id, u.first_name, u.last_name, u.email,
                        ur.role_name, ur.crux_approve, ur.segment, ur.region
                    FROM users u
                    JOIN users_role ur ON u.user_id = ur.user_id
                    WHERE u.tenant_id = $1 
                    AND u.active = true
                    AND ur.crux_approve = true
                    ORDER BY ur.role_name
                """, tenant_id)
                
                eligible_approvers = []
                for row in rows:
                    approver = {
                        "user_id": row['user_id'],
                        "name": f"{row['first_name']} {row['last_name']}",
                        "email": row['email'],
                        "role": row['role_name'],
                        "segment": row['segment'],
                        "region": row['region']
                    }
                    eligible_approvers.append(approver)
                
                return eligible_approvers
                
        except Exception as e:
            logger.error(f"❌ Failed to get eligible approvers: {e}")
            return []
    
    async def _calculate_approver_suggestion_score(self, approver: Dict[str, Any], 
                                                 request: ApproverSuggestionRequest,
                                                 tenant_id: int) -> ApproverSuggestion:
        """Calculate AI suggestion score for an approver"""
        try:
            # Get approver workload
            workload = await self._get_approver_workload(approver['user_id'], tenant_id)
            
            # Calculate expertise match
            expertise_match = self._calculate_expertise_match(
                approver['role'], request.request_type, request.industry_context
            )
            
            # Calculate availability score
            availability_score = 1.0 - min(1.0, workload / 10.0)  # Normalize workload
            
            # Calculate urgency alignment
            urgency_alignment = self._calculate_urgency_alignment(
                approver['role'], request.urgency_level
            )
            
            # Calculate compliance expertise
            compliance_expertise = self._calculate_compliance_expertise(
                approver['role'], request.compliance_frameworks
            )
            
            # Calculate overall suggestion score
            suggestion_score = (
                (expertise_match * 0.3) +
                (availability_score * 0.25) +
                (urgency_alignment * 0.2) +
                (compliance_expertise * 0.25)
            )
            
            # Generate reasoning
            reasoning = []
            if expertise_match > 0.8:
                reasoning.append(f"High expertise match for {request.request_type}")
            if availability_score > 0.7:
                reasoning.append("Low current workload")
            if compliance_expertise > 0.7:
                reasoning.append("Strong compliance framework knowledge")
            if urgency_alignment > 0.8:
                reasoning.append(f"Well-suited for {request.urgency_level} urgency requests")
            
            # Estimate response time
            estimated_response_time = self._estimate_response_time(
                workload, request.urgency_level, approver['role']
            )
            
            return ApproverSuggestion(
                approver_id=approver['user_id'],
                approver_name=approver['name'],
                role=approver['role'],
                suggestion_score=round(suggestion_score, 3),
                reasoning=reasoning,
                availability_status="available" if availability_score > 0.5 else "busy",
                current_workload=workload,
                expertise_match=round(expertise_match, 3),
                estimated_response_time_hours=estimated_response_time
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate suggestion score: {e}")
            return ApproverSuggestion(
                approver_id=approver['user_id'],
                approver_name=approver['name'],
                role=approver['role'],
                suggestion_score=0.5,
                reasoning=["Default scoring due to calculation error"],
                availability_status="unknown",
                current_workload=0,
                expertise_match=0.5,
                estimated_response_time_hours=24.0
            )
    
    async def _get_approver_workload(self, approver_id: int, tenant_id: int) -> int:
        """Get current workload for an approver"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                workload = await conn.fetchval("""
                    SELECT COUNT(*)
                    FROM approval_ledger
                    WHERE tenant_id = $1 
                    AND status = 'pending'
                    AND $2 = ANY(string_to_array(approval_chain::text, ',')::int[])
                """, tenant_id, approver_id)
                
                return workload or 0
                
        except Exception as e:
            logger.error(f"❌ Failed to get approver workload: {e}")
            return 0
    
    def _calculate_expertise_match(self, role: str, request_type: str, industry_context: str) -> float:
        """Calculate expertise match score"""
        # Simplified expertise matching logic
        role_lower = role.lower()
        request_lower = request_type.lower()
        
        # Direct role matches
        if "finance" in request_lower and "finance" in role_lower:
            return 0.9
        elif "compliance" in request_lower and "compliance" in role_lower:
            return 0.9
        elif "sales" in request_lower and "sales" in role_lower:
            return 0.9
        elif "technical" in request_lower and ("cto" in role_lower or "engineering" in role_lower):
            return 0.9
        
        # Senior role matches
        elif "director" in role_lower or "manager" in role_lower:
            return 0.7
        elif "officer" in role_lower:
            return 0.6
        
        return 0.5  # Default match
    
    def _calculate_urgency_alignment(self, role: str, urgency_level: str) -> float:
        """Calculate how well the role aligns with urgency level"""
        role_lower = role.lower()
        
        if urgency_level == "critical":
            # Senior roles better for critical requests
            if any(title in role_lower for title in ["ceo", "cfo", "cto", "director"]):
                return 0.9
            elif "manager" in role_lower:
                return 0.7
            else:
                return 0.5
        elif urgency_level == "high":
            if any(title in role_lower for title in ["director", "manager", "head"]):
                return 0.8
            else:
                return 0.6
        else:
            return 0.7  # Medium/low urgency - most roles suitable
    
    def _calculate_compliance_expertise(self, role: str, compliance_frameworks: List[str]) -> float:
        """Calculate compliance expertise score"""
        role_lower = role.lower()
        
        if not compliance_frameworks:
            return 0.7  # Default if no specific compliance requirements
        
        # Compliance-specific roles
        if any(term in role_lower for term in ["compliance", "legal", "audit", "risk"]):
            return 0.9
        
        # Finance roles for financial compliance
        if "finance" in role_lower and any(fw in compliance_frameworks for fw in ["SOX", "GAAP"]):
            return 0.8
        
        # Security roles for data compliance
        if "security" in role_lower and any(fw in compliance_frameworks for fw in ["GDPR", "HIPAA", "PCI_DSS"]):
            return 0.8
        
        return 0.5  # Default compliance knowledge
    
    def _estimate_response_time(self, workload: int, urgency_level: str, role: str) -> float:
        """Estimate response time in hours"""
        base_time = 24.0  # Default 24 hours
        
        # Adjust for workload
        workload_multiplier = 1.0 + (workload * 0.1)
        
        # Adjust for urgency
        urgency_multipliers = {
            "critical": 0.25,  # 6 hours
            "high": 0.5,       # 12 hours
            "medium": 1.0,     # 24 hours
            "low": 2.0         # 48 hours
        }
        
        urgency_multiplier = urgency_multipliers.get(urgency_level, 1.0)
        
        # Adjust for role seniority (senior roles may be slower due to schedule)
        role_lower = role.lower()
        if any(title in role_lower for title in ["ceo", "cfo", "cto"]):
            role_multiplier = 1.5
        elif "director" in role_lower:
            role_multiplier = 1.2
        else:
            role_multiplier = 1.0
        
        estimated_time = base_time * urgency_multiplier * workload_multiplier * role_multiplier
        return round(estimated_time, 1)
    
    async def _analyze_override_risk_factors(self, request: OverrideReviewRequest, 
                                           tenant_id: int) -> List[str]:
        """Analyze risk factors for an override request"""
        risk_factors = []
        
        # Financial risk
        if request.financial_impact and request.financial_impact > self.risk_criteria["financial_thresholds"]["high"]:
            risk_factors.append(f"High financial impact: ${request.financial_impact:,.2f}")
        
        # Compliance risk
        if request.compliance_impact:
            high_risk_frameworks = ["SOX", "RBI", "HIPAA", "PCI_DSS"]
            for framework in request.compliance_impact:
                if framework in high_risk_frameworks:
                    risk_factors.append(f"High-risk compliance framework: {framework}")
        
        # Frequency risk
        override_frequency = await self._get_user_override_frequency(request.requested_by, tenant_id)
        if override_frequency > self.risk_criteria["frequency_risk_thresholds"]["weekly_overrides"]:
            risk_factors.append(f"High override frequency: {override_frequency} overrides in last 7 days")
        
        # Reason code risk
        high_risk_reasons = ["emergency", "urgent", "bypass", "exception"]
        if any(risk_reason in request.reason_code.lower() for risk_reason in high_risk_reasons):
            risk_factors.append(f"High-risk reason code: {request.reason_code}")
        
        return risk_factors
    
    async def _calculate_base_risk_score(self, request: OverrideReviewRequest, tenant_id: int) -> float:
        """Calculate base risk score"""
        risk_score = 0.0
        
        # Financial risk component
        if request.financial_impact:
            if request.financial_impact > self.risk_criteria["financial_thresholds"]["critical"]:
                risk_score += 0.4
            elif request.financial_impact > self.risk_criteria["financial_thresholds"]["high"]:
                risk_score += 0.3
            elif request.financial_impact > self.risk_criteria["financial_thresholds"]["medium"]:
                risk_score += 0.2
            else:
                risk_score += 0.1
        
        # User history component
        user_risk = await self._calculate_user_risk_score(request.requested_by, tenant_id)
        risk_score += user_risk * 0.3
        
        # Reason code component
        reason_risk = self._calculate_reason_code_risk(request.reason_code)
        risk_score += reason_risk * 0.2
        
        return min(1.0, risk_score)
    
    def _apply_compliance_risk_multipliers(self, base_score: float, 
                                         compliance_frameworks: List[str]) -> float:
        """Apply compliance risk multipliers"""
        if not compliance_frameworks:
            return base_score
        
        max_multiplier = 1.0
        for framework in compliance_frameworks:
            multiplier = self.risk_criteria["compliance_risk_multipliers"].get(framework, 1.0)
            max_multiplier = max(max_multiplier, multiplier)
        
        return min(1.0, base_score * max_multiplier)
    
    async def _analyze_historical_override_patterns(self, user_id: int, reason_code: str, 
                                                  tenant_id: int) -> float:
        """Analyze historical override patterns"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                # Get user's override history
                history = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_overrides,
                        COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved_overrides,
                        COUNT(CASE WHEN reason_code = $2 THEN 1 END) as same_reason_count
                    FROM override_ledger
                    WHERE tenant_id = $1 
                    AND requested_by_user_id = $3
                    AND created_at >= NOW() - INTERVAL '90 days'
                """, tenant_id, reason_code, user_id)
                
                if not history or history['total_overrides'] == 0:
                    return 0.2  # Low risk for new users
                
                # Calculate pattern-based risk
                approval_rate = history['approved_overrides'] / history['total_overrides']
                frequency_risk = min(1.0, history['total_overrides'] / 30.0)  # Normalize to 30 overrides
                same_reason_risk = min(1.0, history['same_reason_count'] / 10.0)  # Normalize to 10 same reasons
                
                # Lower approval rate = higher risk
                approval_risk = 1.0 - approval_rate
                
                pattern_risk = (approval_risk * 0.4) + (frequency_risk * 0.3) + (same_reason_risk * 0.3)
                return min(1.0, pattern_risk)
                
        except Exception as e:
            logger.error(f"❌ Failed to analyze historical patterns: {e}")
            return 0.5  # Default medium risk
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _generate_risk_recommendations(self, risk_level: RiskLevel, 
                                           risk_factors: List[str],
                                           request: OverrideReviewRequest) -> List[str]:
        """Generate risk-based recommendations"""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "Require board-level approval",
                "Conduct thorough risk assessment",
                "Document detailed justification",
                "Implement additional monitoring"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Require senior management approval",
                "Enhanced documentation required",
                "Additional compliance review",
                "Monitor for pattern escalation"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Standard approval process",
                "Document business justification",
                "Regular monitoring"
            ])
        else:
            recommendations.extend([
                "Standard approval sufficient",
                "Basic documentation required"
            ])
        
        # Add specific recommendations based on risk factors
        for factor in risk_factors:
            if "financial impact" in factor.lower():
                recommendations.append("Financial impact assessment required")
            elif "compliance" in factor.lower():
                recommendations.append("Compliance officer review required")
            elif "frequency" in factor.lower():
                recommendations.append("Pattern analysis and user training recommended")
        
        return recommendations
    
    async def _generate_escalation_path(self, risk_level: RiskLevel, 
                                      compliance_frameworks: List[str],
                                      tenant_id: int) -> List[str]:
        """Generate escalation path based on risk level"""
        escalation_path = []
        
        if risk_level == RiskLevel.CRITICAL:
            escalation_path = ["compliance_officer", "legal_counsel", "cfo", "ceo", "board_member"]
        elif risk_level == RiskLevel.HIGH:
            escalation_path = ["compliance_officer", "legal_counsel", "cfo"]
        
        # Add compliance-specific escalations
        if "SOX" in compliance_frameworks:
            if "audit_committee" not in escalation_path:
                escalation_path.append("audit_committee")
        
        if "RBI" in compliance_frameworks:
            if "risk_head" not in escalation_path:
                escalation_path.insert(0, "risk_head")
        
        return escalation_path
    
    async def _check_compliance_violations(self, request: OverrideReviewRequest, 
                                         tenant_id: int) -> List[str]:
        """Check for potential compliance violations"""
        violations = []
        
        # Check for SoD violations
        if request.compliance_impact:
            for framework in request.compliance_impact:
                if framework == "SOX":
                    violations.append("Potential SOX segregation of duties violation")
                elif framework == "GDPR":
                    violations.append("Potential GDPR data protection violation")
                elif framework == "HIPAA":
                    violations.append("Potential HIPAA privacy violation")
        
        return violations
    
    async def _get_user_override_frequency(self, user_id: int, tenant_id: int) -> int:
        """Get user's override frequency in last 7 days"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                frequency = await conn.fetchval("""
                    SELECT COUNT(*)
                    FROM override_ledger
                    WHERE tenant_id = $1 
                    AND requested_by_user_id = $2
                    AND created_at >= NOW() - INTERVAL '7 days'
                """, tenant_id, user_id)
                
                return frequency or 0
                
        except Exception as e:
            logger.error(f"❌ Failed to get override frequency: {e}")
            return 0
    
    async def _calculate_user_risk_score(self, user_id: int, tenant_id: int) -> float:
        """Calculate user-specific risk score"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                user_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_overrides,
                        COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved_overrides
                    FROM override_ledger
                    WHERE tenant_id = $1 
                    AND requested_by_user_id = $2
                    AND created_at >= NOW() - INTERVAL '90 days'
                """, tenant_id, user_id)
                
                if not user_stats or user_stats['total_overrides'] == 0:
                    return 0.2  # Low risk for users with no history
                
                approval_rate = user_stats['approved_overrides'] / user_stats['total_overrides']
                frequency_factor = min(1.0, user_stats['total_overrides'] / 20.0)
                
                # Higher frequency and lower approval rate = higher risk
                user_risk = (1.0 - approval_rate) * 0.6 + frequency_factor * 0.4
                return min(1.0, user_risk)
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate user risk score: {e}")
            return 0.5
    
    def _calculate_reason_code_risk(self, reason_code: str) -> float:
        """Calculate risk score based on reason code"""
        high_risk_codes = ["emergency", "bypass", "urgent", "exception", "override"]
        medium_risk_codes = ["expedite", "priority", "special"]
        
        reason_lower = reason_code.lower()
        
        if any(code in reason_lower for code in high_risk_codes):
            return 0.8
        elif any(code in reason_lower for code in medium_risk_codes):
            return 0.5
        else:
            return 0.2

# Initialize service
ai_assistant_service = None

def get_ai_assistant_service(pool_manager=Depends(get_pool_manager)) -> AIApprovalAssistantService:
    global ai_assistant_service
    if ai_assistant_service is None:
        ai_assistant_service = AIApprovalAssistantService(pool_manager)
    return ai_assistant_service

# API Endpoints
@router.post("/suggest-approvers", response_model=List[ApproverSuggestion])
async def suggest_approvers(
    request: ApproverSuggestionRequest,
    tenant_id: int = Query(..., description="Tenant ID"),
    service: AIApprovalAssistantService = Depends(get_ai_assistant_service)
):
    """
    AI-powered approver suggestions
    Task 14.1-T40: AI agent approver suggestion
    """
    return await service.suggest_approvers(request, tenant_id)

@router.post("/assess-override-risk", response_model=OverrideRiskAssessment)
async def assess_override_risk(
    request: OverrideReviewRequest,
    tenant_id: int = Query(..., description="Tenant ID"),
    service: AIApprovalAssistantService = Depends(get_ai_assistant_service)
):
    """
    AI-powered override risk assessment
    Task 14.1-T41: AI agent override reviewer
    """
    return await service.assess_override_risk(request, tenant_id)

@router.post("/learn-patterns", response_model=Dict[str, Any])
async def learn_from_patterns(
    tenant_id: int = Query(..., description="Tenant ID"),
    lookback_days: int = Query(90, description="Days to analyze", ge=7, le=365),
    service: AIApprovalAssistantService = Depends(get_ai_assistant_service)
):
    """
    Learn from historical approval and override patterns
    """
    return await service.learn_from_approval_patterns(tenant_id, lookback_days)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ai_approval_assistant_api", "timestamp": datetime.utcnow()}
