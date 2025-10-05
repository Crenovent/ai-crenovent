#!/usr/bin/env python3
"""
Trust Scoring Service API - Chapter 14.2
========================================
Tasks 14.2-T06, 14.2-T49: Trust scoring engine service API

Features:
- Multi-dimensional trust scoring (policy, overrides, SLA, evidence, risk)
- Industry-specific weight overlays (SaaS, Banking, Insurance)
- Tenant-tier specific thresholds and calculations
- Real-time trust score updates and batch recalculation
- Trust trend analysis and recommendations
- Integration with approval workflows and evidence packs
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
import json

# Database and dependencies
from src.database.connection_pool import get_pool_manager
from dsl.intelligence.trust_scoring_engine import TrustScoringEngine, TrustScore, TrustLevel

logger = logging.getLogger(__name__)
router = APIRouter()

# =====================================================
# PYDANTIC MODELS
# =====================================================

class TrustTargetType(str, Enum):
    WORKFLOW = "workflow"
    CAPABILITY = "capability" 
    USER = "user"
    TENANT = "tenant"

class IndustryCode(str, Enum):
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    ECOMMERCE = "E-commerce"
    FINTECH = "FinTech"
    HEALTHCARE = "Healthcare"

class TenantTier(str, Enum):
    T0 = "T0"  # Enterprise
    T1 = "T1"  # Professional  
    T2 = "T2"  # Standard
    T3 = "T3"  # Starter

class TrustScoreRequest(BaseModel):
    target_type: TrustTargetType = Field(..., description="Type of target to score")
    target_id: str = Field(..., description="ID of the target")
    tenant_id: int = Field(..., description="Tenant ID")
    lookback_days: int = Field(30, description="Days to look back for scoring", ge=1, le=365)
    industry_code: IndustryCode = Field(IndustryCode.SAAS, description="Industry for weight overlay")
    tenant_tier: TenantTier = Field(TenantTier.T2, description="Tenant tier for thresholds")
    force_recalculate: bool = Field(False, description="Force recalculation even if cached")

class TrustScoreResponse(BaseModel):
    target_type: str
    target_id: str
    tenant_id: int
    overall_score: float = Field(..., ge=0.0, le=1.0)
    trust_level: str
    
    # Dimension scores
    policy_compliance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    override_frequency_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    sla_adherence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence_completeness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Metadata
    calculated_at: datetime
    valid_until: datetime
    sample_size: int
    confidence_interval: Optional[float] = Field(None, ge=0.0, le=1.0)
    recommendations: List[str] = Field(default=[])
    
    # Scoring context
    industry_code: str
    tenant_tier: str
    scoring_algorithm_version: str = Field(default="v1.0")

class TrustTrendRequest(BaseModel):
    target_type: TrustTargetType = Field(..., description="Type of target")
    target_id: str = Field(..., description="ID of the target")
    tenant_id: int = Field(..., description="Tenant ID")
    days: int = Field(90, description="Days to analyze trends", ge=7, le=365)

class TrustBenchmarkRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID")
    industry_code: IndustryCode = Field(..., description="Industry for benchmarking")
    target_type: TrustTargetType = Field(..., description="Type of targets to benchmark")
    include_cross_tenant: bool = Field(False, description="Include anonymized cross-tenant data")

class TrustThresholdUpdate(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID")
    target_type: TrustTargetType = Field(..., description="Type of target")
    thresholds: Dict[str, float] = Field(..., description="Trust level thresholds")
    
    @validator('thresholds')
    def validate_thresholds(cls, v):
        required_levels = ['excellent', 'good', 'acceptable', 'concerning']
        for level in required_levels:
            if level not in v:
                raise ValueError(f"Missing threshold for level: {level}")
            if not 0.0 <= v[level] <= 1.0:
                raise ValueError(f"Threshold for {level} must be between 0.0 and 1.0")
        return v

# =====================================================
# TRUST SCORING SERVICE
# =====================================================

class TrustScoringService:
    """
    Enhanced trust scoring service with advanced calculators
    Tasks 14.2-T07 to T12: Trust scoring calculators and algorithms
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.trust_engine = TrustScoringEngine(pool_manager)
        
        # Advanced scoring weights by industry (Task 14.2-T07)
        self.industry_weights = {
            "SaaS": {
                "policy_compliance": 0.35,
                "override_frequency": 0.25,
                "sla_adherence": 0.20,
                "evidence_completeness": 0.15,
                "risk_assessment": 0.05
            },
            "Banking": {
                "policy_compliance": 0.45,  # Higher weight for regulatory compliance
                "override_frequency": 0.30,
                "sla_adherence": 0.15,
                "evidence_completeness": 0.08,
                "risk_assessment": 0.02
            },
            "Insurance": {
                "policy_compliance": 0.40,
                "override_frequency": 0.25,
                "sla_adherence": 0.20,
                "evidence_completeness": 0.10,
                "risk_assessment": 0.05
            },
            "Healthcare": {
                "policy_compliance": 0.50,  # HIPAA compliance critical
                "override_frequency": 0.20,
                "sla_adherence": 0.15,
                "evidence_completeness": 0.12,
                "risk_assessment": 0.03
            }
        }
        
        # Tenant tier thresholds (Task 14.2-T08)
        self.tier_thresholds = {
            "T0": {"excellent": 0.95, "good": 0.85, "acceptable": 0.70, "poor": 0.50},
            "T1": {"excellent": 0.90, "good": 0.80, "acceptable": 0.65, "poor": 0.45},
            "T2": {"excellent": 0.85, "good": 0.75, "acceptable": 0.60, "poor": 0.40},
            "T3": {"excellent": 0.80, "good": 0.70, "acceptable": 0.55, "poor": 0.35}
        }
        
        # Risk multipliers by context (Task 14.2-T09)
        self.risk_multipliers = {
            "financial_workflow": 1.2,
            "compliance_critical": 1.3,
            "customer_data": 1.15,
            "high_value_transaction": 1.25,
            "cross_border": 1.1,
            "emergency_override": 0.8  # Lower trust for emergency overrides
        }
    
    async def calculate_advanced_trust_score(self, request: TrustScoreRequest) -> TrustScoreResponse:
        """
        Calculate advanced trust score with industry weights and tier thresholds
        Tasks 14.2-T07, 14.2-T08, 14.2-T09: Advanced scoring algorithms
        """
        try:
            # Get industry weights
            weights = self.industry_weights.get(request.industry_code.value, self.industry_weights["SaaS"])
            
            # Gather scoring data
            scoring_data = await self._gather_scoring_data(
                request.target_type, request.target_id, request.tenant_id, request.lookback_days
            )
            
            # Calculate dimension scores
            policy_score = await self._calculate_policy_compliance_score(scoring_data)
            override_score = await self._calculate_override_frequency_score(scoring_data)
            sla_score = await self._calculate_sla_adherence_score(scoring_data)
            evidence_score = await self._calculate_evidence_completeness_score(scoring_data)
            risk_score = await self._calculate_risk_assessment_score(scoring_data, request)
            
            # Apply industry weights
            weighted_score = (
                (policy_score * weights["policy_compliance"]) +
                (override_score * weights["override_frequency"]) +
                (sla_score * weights["sla_adherence"]) +
                (evidence_score * weights["evidence_completeness"]) +
                (risk_score * weights["risk_assessment"])
            )
            
            # Apply risk multipliers (Task 14.2-T09)
            context_multiplier = 1.0
            if scoring_data.get('workflow_type') in self.risk_multipliers:
                context_multiplier = self.risk_multipliers[scoring_data['workflow_type']]
            
            final_score = min(1.0, weighted_score * context_multiplier)
            
            # Determine trust level using tier-specific thresholds
            thresholds = self.tier_thresholds.get(request.tenant_tier.value, self.tier_thresholds["T2"])
            trust_level = self._determine_trust_level(final_score, thresholds)
            
            # Generate recommendations
            recommendations = self._generate_trust_recommendations(
                policy_score, override_score, sla_score, evidence_score, risk_score, trust_level
            )
            
            # Calculate confidence interval
            confidence = self._calculate_confidence_interval(scoring_data['sample_size'])
            
            return TrustScoreResponse(
                target_type=request.target_type.value,
                target_id=request.target_id,
                tenant_id=request.tenant_id,
                overall_score=round(final_score, 3),
                trust_level=trust_level,
                policy_compliance_score=round(policy_score, 3),
                override_frequency_score=round(override_score, 3),
                sla_adherence_score=round(sla_score, 3),
                evidence_completeness_score=round(evidence_score, 3),
                risk_score=round(risk_score, 3),
                calculated_at=datetime.utcnow(),
                valid_until=datetime.utcnow() + timedelta(hours=24),
                sample_size=scoring_data['sample_size'],
                confidence_interval=confidence,
                recommendations=recommendations,
                industry_code=request.industry_code.value,
                tenant_tier=request.tenant_tier.value,
                scoring_algorithm_version="v2.0"
            )
            
        except Exception as e:
            logger.error(f"❌ Advanced trust score calculation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _gather_scoring_data(self, target_type: str, target_id: str, tenant_id: int, lookback_days: int) -> Dict[str, Any]:
        """Gather data for trust scoring calculation (Task 14.2-T10)"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                # Base query filters
                date_filter = datetime.utcnow() - timedelta(days=lookback_days)
                
                # Policy compliance data
                policy_data = await conn.fetch("""
                    SELECT COUNT(*) as total_executions,
                           COUNT(CASE WHEN policy_violations = '[]' THEN 1 END) as compliant_executions
                    FROM dsl_execution_traces 
                    WHERE tenant_id = $1 AND created_at >= $2
                    AND (workflow_id = $3 OR capability_id = $3)
                """, tenant_id, date_filter, target_id)
                
                # Override frequency data
                override_data = await conn.fetch("""
                    SELECT COUNT(*) as total_overrides,
                           COUNT(CASE WHEN risk_level = 'critical' THEN 1 END) as critical_overrides,
                           COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved_overrides
                    FROM override_ledger 
                    WHERE tenant_id = $1 AND created_at >= $2
                    AND workflow_id = $3
                """, tenant_id, date_filter, target_id)
                
                # SLA adherence data
                sla_data = await conn.fetch("""
                    SELECT COUNT(*) as total_executions,
                           COUNT(CASE WHEN execution_time_ms <= sla_threshold_ms THEN 1 END) as sla_met_executions,
                           AVG(execution_time_ms) as avg_execution_time
                    FROM dsl_execution_traces 
                    WHERE tenant_id = $1 AND created_at >= $2
                    AND (workflow_id = $3 OR capability_id = $3)
                """, tenant_id, date_filter, target_id)
                
                # Evidence completeness data
                evidence_data = await conn.fetch("""
                    SELECT COUNT(*) as total_evidence_packs,
                           COUNT(CASE WHEN evidence_completeness_score >= 0.9 THEN 1 END) as complete_evidence
                    FROM dsl_evidence_packs 
                    WHERE tenant_id = $1 AND created_at >= $2
                    AND evidence_data::text LIKE '%' || $3 || '%'
                """, tenant_id, date_filter, target_id)
                
                # Aggregate the data
                total_executions = policy_data[0]['total_executions'] if policy_data else 0
                
                return {
                    'sample_size': total_executions,
                    'policy_compliance': {
                        'total': policy_data[0]['total_executions'] if policy_data else 0,
                        'compliant': policy_data[0]['compliant_executions'] if policy_data else 0
                    },
                    'overrides': {
                        'total': override_data[0]['total_overrides'] if override_data else 0,
                        'critical': override_data[0]['critical_overrides'] if override_data else 0,
                        'approved': override_data[0]['approved_overrides'] if override_data else 0
                    },
                    'sla_performance': {
                        'total': sla_data[0]['total_executions'] if sla_data else 0,
                        'met': sla_data[0]['sla_met_executions'] if sla_data else 0,
                        'avg_time': float(sla_data[0]['avg_execution_time'] or 0) if sla_data else 0
                    },
                    'evidence_quality': {
                        'total': evidence_data[0]['total_evidence_packs'] if evidence_data else 0,
                        'complete': evidence_data[0]['complete_evidence'] if evidence_data else 0
                    },
                    'workflow_type': 'standard'  # Could be enhanced to detect type
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to gather scoring data: {e}")
            return {'sample_size': 0}
    
    async def _calculate_policy_compliance_score(self, data: Dict[str, Any]) -> float:
        """Calculate policy compliance score (Task 14.2-T11)"""
        policy_data = data.get('policy_compliance', {})
        total = policy_data.get('total', 0)
        compliant = policy_data.get('compliant', 0)
        
        if total == 0:
            return 1.0  # No data means perfect compliance
        
        compliance_rate = compliant / total
        
        # Apply non-linear scaling to penalize low compliance more heavily
        if compliance_rate >= 0.95:
            return 1.0
        elif compliance_rate >= 0.90:
            return 0.9 + (compliance_rate - 0.90) * 2  # Scale 0.90-0.95 to 0.9-1.0
        elif compliance_rate >= 0.80:
            return 0.7 + (compliance_rate - 0.80) * 2  # Scale 0.80-0.90 to 0.7-0.9
        else:
            return compliance_rate * 0.875  # Scale 0-0.80 to 0-0.7
    
    async def _calculate_override_frequency_score(self, data: Dict[str, Any]) -> float:
        """Calculate override frequency score (Task 14.2-T11)"""
        override_data = data.get('overrides', {})
        total_executions = data.get('sample_size', 1)
        total_overrides = override_data.get('total', 0)
        critical_overrides = override_data.get('critical', 0)
        
        if total_executions == 0:
            return 1.0
        
        # Calculate override rate
        override_rate = total_overrides / total_executions
        critical_rate = critical_overrides / total_executions
        
        # Penalize high override rates and critical overrides heavily
        base_score = max(0.0, 1.0 - (override_rate * 2))  # 50% override rate = 0 score
        critical_penalty = critical_rate * 0.5  # Each 1% critical rate reduces score by 0.5%
        
        return max(0.0, base_score - critical_penalty)
    
    async def _calculate_sla_adherence_score(self, data: Dict[str, Any]) -> float:
        """Calculate SLA adherence score (Task 14.2-T11)"""
        sla_data = data.get('sla_performance', {})
        total = sla_data.get('total', 0)
        met = sla_data.get('met', 0)
        
        if total == 0:
            return 1.0
        
        adherence_rate = met / total
        
        # Apply exponential scaling to reward high SLA adherence
        return adherence_rate ** 0.5  # Square root scaling
    
    async def _calculate_evidence_completeness_score(self, data: Dict[str, Any]) -> float:
        """Calculate evidence completeness score (Task 14.2-T11)"""
        evidence_data = data.get('evidence_quality', {})
        total = evidence_data.get('total', 0)
        complete = evidence_data.get('complete', 0)
        
        if total == 0:
            return 0.8  # Partial score for no evidence
        
        completeness_rate = complete / total
        
        # Linear scaling with minimum threshold
        return max(0.2, completeness_rate)
    
    async def _calculate_risk_assessment_score(self, data: Dict[str, Any], request: TrustScoreRequest) -> float:
        """Calculate risk assessment score (Task 14.2-T11)"""
        # Base risk score starts at 1.0 (low risk)
        risk_score = 1.0
        
        # Reduce score based on critical overrides
        override_data = data.get('overrides', {})
        critical_overrides = override_data.get('critical', 0)
        total_executions = data.get('sample_size', 1)
        
        if total_executions > 0:
            critical_rate = critical_overrides / total_executions
            risk_score -= critical_rate * 0.3  # Each 1% critical override rate reduces by 0.3%
        
        # Industry-specific risk adjustments
        if request.industry_code == IndustryCode.BANKING:
            risk_score *= 0.95  # Banking inherently higher risk
        elif request.industry_code == IndustryCode.HEALTHCARE:
            risk_score *= 0.90  # Healthcare highest risk due to PHI
        
        return max(0.0, risk_score)
    
    def _determine_trust_level(self, score: float, thresholds: Dict[str, float]) -> str:
        """Determine trust level based on score and tier thresholds (Task 14.2-T08)"""
        if score >= thresholds["excellent"]:
            return "excellent"
        elif score >= thresholds["good"]:
            return "good"
        elif score >= thresholds["acceptable"]:
            return "acceptable"
        elif score >= thresholds["poor"]:
            return "poor"
        else:
            return "critical"
    
    def _generate_trust_recommendations(self, policy_score: float, override_score: float, 
                                      sla_score: float, evidence_score: float, 
                                      risk_score: float, trust_level: str) -> List[str]:
        """Generate trust improvement recommendations (Task 14.2-T12)"""
        recommendations = []
        
        if policy_score < 0.8:
            recommendations.append("Improve policy compliance by reviewing and updating workflow rules")
        
        if override_score < 0.7:
            recommendations.append("Reduce override frequency by improving workflow design and training")
        
        if sla_score < 0.8:
            recommendations.append("Optimize workflow performance to meet SLA requirements consistently")
        
        if evidence_score < 0.9:
            recommendations.append("Enhance evidence capture completeness for better audit trails")
        
        if risk_score < 0.8:
            recommendations.append("Implement additional risk controls and monitoring")
        
        if trust_level in ["poor", "critical"]:
            recommendations.append("Consider workflow redesign or additional approval layers")
        
        if not recommendations:
            recommendations.append("Maintain current excellent performance standards")
        
        return recommendations
    
    def _calculate_confidence_interval(self, sample_size: int) -> float:
        """Calculate confidence interval based on sample size (Task 14.2-T10)"""
        if sample_size == 0:
            return 0.0
        elif sample_size < 10:
            return 0.6
        elif sample_size < 50:
            return 0.8
        elif sample_size < 100:
            return 0.9
        else:
            return 0.95
    
    async def calculate_trust_score(self, request: TrustScoreRequest) -> TrustScoreResponse:
        """Calculate trust score with industry and tier considerations"""
        
        try:
            # Check for cached score first
            if not request.force_recalculate:
                cached_score = await self._get_cached_trust_score(
                    request.target_type.value, request.target_id, request.tenant_id
                )
                if cached_score:
                    return cached_score
            
            # Calculate new trust score using the engine
            trust_score = await self.trust_engine.calculate_trust_score(
                capability_id=request.target_id,
                tenant_id=request.tenant_id,
                lookback_days=request.lookback_days,
                industry_code=request.industry_code.value,
                tenant_tier=request.tenant_tier.value
            )
            
            # Convert to response format
            response = TrustScoreResponse(
                target_type=request.target_type.value,
                target_id=request.target_id,
                tenant_id=request.tenant_id,
                overall_score=trust_score.overall_score,
                trust_level=trust_score.trust_level.value,
                policy_compliance_score=trust_score.factors[0].score if len(trust_score.factors) > 0 else None,
                override_frequency_score=trust_score.factors[1].score if len(trust_score.factors) > 1 else None,
                sla_adherence_score=trust_score.factors[2].score if len(trust_score.factors) > 2 else None,
                evidence_completeness_score=trust_score.factors[3].score if len(trust_score.factors) > 3 else None,
                risk_score=trust_score.factors[4].score if len(trust_score.factors) > 4 else None,
                calculated_at=trust_score.calculated_at,
                valid_until=trust_score.valid_until,
                sample_size=trust_score.sample_size,
                confidence_interval=trust_score.confidence_interval,
                recommendations=trust_score.recommendations,
                industry_code=request.industry_code.value,
                tenant_tier=request.tenant_tier.value,
                scoring_algorithm_version="v1.0"
            )
            
            # Store in database
            await self._store_trust_score(response)
            
            logger.info(f"✅ Calculated trust score for {request.target_type.value}:{request.target_id} = {trust_score.overall_score:.3f}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate trust score: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_trust_trends(self, request: TrustTrendRequest) -> Dict[str, Any]:
        """Get trust score trends over time"""
        
        try:
            async with self.pool_manager.get_connection() as conn:
                # Fetch historical trust scores
                rows = await conn.fetch("""
                    SELECT overall_score, trust_level, calculated_at,
                           policy_compliance_score, sla_adherence_score,
                           evidence_completeness_score
                    FROM trust_scores
                    WHERE tenant_id = $1 AND target_type = $2 AND target_id = $3
                      AND calculated_at >= $4
                    ORDER BY calculated_at ASC
                """, 
                    request.tenant_id, request.target_type.value, request.target_id,
                    datetime.utcnow() - timedelta(days=request.days)
                )
                
                if not rows:
                    return {
                        "target_type": request.target_type.value,
                        "target_id": request.target_id,
                        "trend_data": [],
                        "trend_analysis": {
                            "direction": "insufficient_data",
                            "change_rate": 0.0,
                            "volatility": 0.0
                        }
                    }
                
                # Process trend data
                trend_data = []
                scores = []
                
                for row in rows:
                    trend_point = {
                        "timestamp": row['calculated_at'],
                        "overall_score": float(row['overall_score']),
                        "trust_level": row['trust_level'],
                        "policy_compliance_score": float(row['policy_compliance_score']) if row['policy_compliance_score'] else None,
                        "sla_adherence_score": float(row['sla_adherence_score']) if row['sla_adherence_score'] else None,
                        "evidence_completeness_score": float(row['evidence_completeness_score']) if row['evidence_completeness_score'] else None
                    }
                    trend_data.append(trend_point)
                    scores.append(float(row['overall_score']))
                
                # Calculate trend analysis
                trend_analysis = self._analyze_trend(scores)
                
                return {
                    "target_type": request.target_type.value,
                    "target_id": request.target_id,
                    "tenant_id": request.tenant_id,
                    "period_days": request.days,
                    "trend_data": trend_data,
                    "trend_analysis": trend_analysis,
                    "data_points": len(trend_data)
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get trust trends: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_trust_benchmarks(self, request: TrustBenchmarkRequest) -> Dict[str, Any]:
        """Get trust score benchmarks for industry comparison"""
        
        try:
            async with self.pool_manager.get_connection() as conn:
                # Get tenant's current scores
                tenant_query = """
                    SELECT target_id, overall_score, trust_level
                    FROM trust_scores
                    WHERE tenant_id = $1 AND target_type = $2
                      AND calculated_at >= $3
                    ORDER BY calculated_at DESC
                """
                
                tenant_rows = await conn.fetch(
                    tenant_query, 
                    request.tenant_id, 
                    request.target_type.value,
                    datetime.utcnow() - timedelta(days=30)
                )
                
                tenant_scores = [float(row['overall_score']) for row in tenant_rows]
                tenant_avg = sum(tenant_scores) / len(tenant_scores) if tenant_scores else 0.0
                
                # Get industry benchmarks (anonymized cross-tenant if enabled)
                if request.include_cross_tenant:
                    benchmark_query = """
                        SELECT AVG(overall_score) as avg_score,
                               PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY overall_score) as p25,
                               PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY overall_score) as p50,
                               PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY overall_score) as p75,
                               COUNT(*) as sample_size
                        FROM trust_scores
                        WHERE target_type = $1 AND industry_code = $2
                          AND calculated_at >= $3
                    """
                    
                    benchmark_row = await conn.fetchrow(
                        benchmark_query,
                        request.target_type.value,
                        request.industry_code.value,
                        datetime.utcnow() - timedelta(days=30)
                    )
                else:
                    # Use only tenant data
                    benchmark_row = {
                        'avg_score': tenant_avg,
                        'p25': min(tenant_scores) if tenant_scores else 0.0,
                        'p50': tenant_avg,
                        'p75': max(tenant_scores) if tenant_scores else 0.0,
                        'sample_size': len(tenant_scores)
                    }
                
                # Calculate percentile ranking
                percentile_rank = 50  # Default
                if request.include_cross_tenant and benchmark_row['avg_score']:
                    if tenant_avg > float(benchmark_row['p75']):
                        percentile_rank = 85
                    elif tenant_avg > float(benchmark_row['p50']):
                        percentile_rank = 70
                    elif tenant_avg > float(benchmark_row['p25']):
                        percentile_rank = 35
                    else:
                        percentile_rank = 15
                
                return {
                    "tenant_id": request.tenant_id,
                    "industry_code": request.industry_code.value,
                    "target_type": request.target_type.value,
                    "tenant_performance": {
                        "average_score": round(tenant_avg, 3),
                        "sample_size": len(tenant_scores),
                        "percentile_rank": percentile_rank
                    },
                    "industry_benchmarks": {
                        "average_score": round(float(benchmark_row['avg_score']), 3) if benchmark_row['avg_score'] else 0.0,
                        "percentile_25": round(float(benchmark_row['p25']), 3) if benchmark_row['p25'] else 0.0,
                        "percentile_50": round(float(benchmark_row['p50']), 3) if benchmark_row['p50'] else 0.0,
                        "percentile_75": round(float(benchmark_row['p75']), 3) if benchmark_row['p75'] else 0.0,
                        "sample_size": benchmark_row['sample_size']
                    },
                    "includes_cross_tenant": request.include_cross_tenant
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get trust benchmarks: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _analyze_trend(self, scores: List[float]) -> Dict[str, Any]:
        """Analyze trend direction and volatility"""
        
        if len(scores) < 2:
            return {
                "direction": "insufficient_data",
                "change_rate": 0.0,
                "volatility": 0.0
            }
        
        # Calculate linear trend
        n = len(scores)
        x = list(range(n))
        
        # Simple linear regression
        x_mean = sum(x) / n
        y_mean = sum(scores) / n
        
        numerator = sum((x[i] - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Determine direction
        if abs(slope) < 0.001:
            direction = "stable"
        elif slope > 0:
            direction = "improving"
        else:
            direction = "declining"
        
        # Calculate volatility (standard deviation)
        variance = sum((score - y_mean) ** 2 for score in scores) / n
        volatility = variance ** 0.5
        
        return {
            "direction": direction,
            "change_rate": round(slope, 4),
            "volatility": round(volatility, 4),
            "trend_strength": "strong" if abs(slope) > 0.01 else "weak"
        }
    
    async def _get_cached_trust_score(self, target_type: str, target_id: str, 
                                    tenant_id: int) -> Optional[TrustScoreResponse]:
        """Get cached trust score if still valid"""
        
        try:
            async with self.pool_manager.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM trust_scores
                    WHERE tenant_id = $1 AND target_type = $2 AND target_id = $3
                      AND valid_until > $4
                    ORDER BY calculated_at DESC LIMIT 1
                """, tenant_id, target_type, target_id, datetime.utcnow())
                
                if row:
                    return TrustScoreResponse(
                        target_type=row['target_type'],
                        target_id=row['target_id'],
                        tenant_id=row['tenant_id'],
                        overall_score=float(row['overall_score']),
                        trust_level=row['trust_level'],
                        policy_compliance_score=float(row['policy_compliance_score']) if row['policy_compliance_score'] else None,
                        override_frequency_score=float(row['override_frequency_score']) if row['override_frequency_score'] else None,
                        sla_adherence_score=float(row['sla_adherence_score']) if row['sla_adherence_score'] else None,
                        evidence_completeness_score=float(row['evidence_completeness_score']) if row['evidence_completeness_score'] else None,
                        risk_score=float(row['risk_score']) if row['risk_score'] else None,
                        calculated_at=row['calculated_at'],
                        valid_until=row['valid_until'],
                        sample_size=row['sample_size'],
                        confidence_interval=float(row['confidence_interval']) if row['confidence_interval'] else None,
                        recommendations=json.loads(row['recommendations']) if row['recommendations'] else [],
                        industry_code=row['industry_code'],
                        tenant_tier=row['tenant_tier'],
                        scoring_algorithm_version=row['scoring_algorithm_version']
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get cached trust score: {e}")
            return None
    
    async def _store_trust_score(self, score: TrustScoreResponse) -> None:
        """Store trust score in database"""
        
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO trust_scores (
                        tenant_id, target_type, target_id, overall_score, trust_level,
                        policy_compliance_score, override_frequency_score, sla_adherence_score,
                        evidence_completeness_score, risk_score, calculated_at, valid_until,
                        sample_size, confidence_interval, recommendations, industry_code,
                        tenant_tier, scoring_algorithm_version
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                """,
                    score.tenant_id, score.target_type, score.target_id, score.overall_score,
                    score.trust_level, score.policy_compliance_score, score.override_frequency_score,
                    score.sla_adherence_score, score.evidence_completeness_score, score.risk_score,
                    score.calculated_at, score.valid_until, score.sample_size, score.confidence_interval,
                    json.dumps(score.recommendations), score.industry_code, score.tenant_tier,
                    score.scoring_algorithm_version
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to store trust score: {e}")
            raise
    
    async def calculate_real_time_trust_score(self, target_type: str, target_id: str, 
                                            tenant_id: int) -> Dict[str, Any]:
        """
        Calculate real-time trust score with streaming updates (Task 14.2-T13)
        """
        try:
            # Check cache first for performance
            cached_score = await self._get_cached_trust_score(target_type, target_id, tenant_id)
            
            # Check if cached score is fresh (less than 5 minutes old)
            if cached_score and self._is_score_fresh(cached_score):
                return {
                    "overall_score": cached_score.overall_score,
                    "trust_level": cached_score.trust_level,
                    "cached": True,
                    "calculated_at": cached_score.calculated_at.isoformat(),
                    "real_time": True
                }
            
            # Calculate fresh score
            request = TrustScoreRequest(
                target_type=TrustTargetType(target_type),
                target_id=target_id,
                tenant_id=tenant_id,
                industry_code=IndustryCode.SAAS,  # Default
                tenant_tier=TenantTier.T2  # Default
            )
            
            fresh_score = await self.calculate_advanced_trust_score(request)
            
            return {
                "overall_score": fresh_score.overall_score,
                "trust_level": fresh_score.trust_level,
                "cached": False,
                "calculated_at": fresh_score.calculated_at.isoformat(),
                "real_time": True,
                "dimension_scores": {
                    "policy_compliance": fresh_score.policy_compliance_score,
                    "override_frequency": fresh_score.override_frequency_score,
                    "sla_adherence": fresh_score.sla_adherence_score,
                    "evidence_completeness": fresh_score.evidence_completeness_score,
                    "risk_assessment": fresh_score.risk_score
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Real-time trust score calculation failed: {e}")
            return {
                "overall_score": 0.5,
                "trust_level": "unknown",
                "error": str(e),
                "real_time": True
            }
    
    async def detect_trust_anomalies(self, target_type: str, target_id: str, 
                                   tenant_id: int, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Detect anomalies in trust score patterns (Task 14.2-T28)
        """
        try:
            # Get historical trust scores
            historical_scores = await self._get_historical_trust_scores(
                target_type, target_id, tenant_id, lookback_days
            )
            
            if len(historical_scores) < 10:
                return {
                    "anomalies_detected": False,
                    "reason": "Insufficient historical data",
                    "sample_size": len(historical_scores)
                }
            
            scores = [score["overall_score"] for score in historical_scores]
            
            # Calculate statistical measures
            import statistics
            mean_score = statistics.mean(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 0
            
            # Detect statistical outliers (z-score > 2.5)
            anomalies = []
            anomaly_threshold = 2.5
            
            for i, score_data in enumerate(historical_scores):
                score = score_data["overall_score"]
                if std_score > 0:
                    z_score = abs(score - mean_score) / std_score
                    if z_score > anomaly_threshold:
                        anomalies.append({
                            "date": score_data["calculated_at"],
                            "score": score,
                            "z_score": round(z_score, 3),
                            "deviation_type": "high" if score > mean_score else "low",
                            "severity": "critical" if z_score > 3.0 else "high"
                        })
            
            # Detect trend anomalies (sudden changes)
            trend_anomalies = []
            for i in range(1, len(scores)):
                change = abs(scores[i] - scores[i-1])
                if change > 0.2:  # 20% change threshold
                    trend_anomalies.append({
                        "type": "sudden_change",
                        "date": historical_scores[i]["calculated_at"],
                        "change": round(change, 3),
                        "direction": "increase" if scores[i] > scores[i-1] else "decrease"
                    })
            
            return {
                "anomalies_detected": len(anomalies) > 0 or len(trend_anomalies) > 0,
                "statistical_anomalies": anomalies,
                "trend_anomalies": trend_anomalies,
                "baseline_stats": {
                    "mean_score": round(mean_score, 3),
                    "std_deviation": round(std_score, 3),
                    "sample_size": len(scores),
                    "lookback_days": lookback_days
                },
                "analysis_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Trust anomaly detection failed: {e}")
            return {
                "anomalies_detected": False,
                "error": str(e)
            }
    
    async def predict_trust_score(self, target_type: str, target_id: str, 
                                tenant_id: int, prediction_days: int = 30) -> Dict[str, Any]:
        """
        Predict future trust score using trend analysis (Task 14.2-T27)
        """
        try:
            # Get historical data for prediction
            historical_scores = await self._get_historical_trust_scores(
                target_type, target_id, tenant_id, days=90
            )
            
            if len(historical_scores) < 14:  # Need at least 2 weeks of data
                return {
                    "prediction": None,
                    "confidence": 0.0,
                    "reason": "Insufficient historical data for prediction",
                    "required_samples": 14,
                    "available_samples": len(historical_scores)
                }
            
            scores = [score["overall_score"] for score in historical_scores]
            
            # Simple linear trend prediction
            import statistics
            n = len(scores)
            x_values = list(range(n))
            
            # Calculate linear regression slope
            x_mean = statistics.mean(x_values)
            y_mean = statistics.mean(scores)
            
            numerator = sum((x_values[i] - x_mean) * (scores[i] - y_mean) for i in range(n))
            denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator
            
            # Predict future score
            current_score = scores[-1]
            predicted_score = current_score + (slope * prediction_days)
            predicted_score = max(0.0, min(1.0, predicted_score))  # Clamp to valid range
            
            # Calculate prediction confidence based on trend consistency
            score_variance = statistics.variance(scores) if len(scores) > 1 else 0
            confidence = max(0.1, 1.0 - (score_variance * 2))  # Lower variance = higher confidence
            
            # Generate prediction intervals (simplified)
            std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.1
            prediction_intervals = {
                "lower_bound": max(0.0, predicted_score - (1.96 * std_dev)),
                "upper_bound": min(1.0, predicted_score + (1.96 * std_dev))
            }
            
            return {
                "prediction": {
                    "predicted_score": round(predicted_score, 3),
                    "prediction_date": (datetime.utcnow() + timedelta(days=prediction_days)).isoformat(),
                    "prediction_intervals": {
                        "lower_bound": round(prediction_intervals["lower_bound"], 3),
                        "upper_bound": round(prediction_intervals["upper_bound"], 3)
                    }
                },
                "confidence": round(confidence, 3),
                "model_info": {
                    "model_type": "linear_trend",
                    "training_samples": len(historical_scores),
                    "trend_slope": round(slope, 6),
                    "score_variance": round(score_variance, 3)
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Trust score prediction failed: {e}")
            return {
                "prediction": None,
                "confidence": 0.0,
                "error": str(e)
            }
    
    # Helper methods for advanced features
    def _is_score_fresh(self, cached_score: TrustScoreResponse, max_age_minutes: int = 5) -> bool:
        """Check if cached score is still fresh"""
        if not cached_score or not cached_score.calculated_at:
            return False
        
        age = datetime.utcnow() - cached_score.calculated_at
        return age.total_seconds() < (max_age_minutes * 60)
    
    async def _get_historical_trust_scores(self, target_type: str, target_id: str, 
                                         tenant_id: int, days: int) -> List[Dict[str, Any]]:
        """Get historical trust scores for analysis"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                rows = await conn.fetch("""
                    SELECT overall_score, calculated_at, trust_level
                    FROM trust_scores
                    WHERE tenant_id = $1 AND target_type = $2 AND target_id = $3
                    AND calculated_at >= NOW() - INTERVAL '%s days'
                    ORDER BY calculated_at ASC
                """ % days, tenant_id, target_type, target_id)
                
                return [
                    {
                        "overall_score": float(row['overall_score']),
                        "calculated_at": row['calculated_at'].isoformat(),
                        "trust_level": row['trust_level']
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"❌ Failed to get historical trust scores: {e}")
            return []

# =====================================================
# API ENDPOINTS
# =====================================================

    # =====================================================
    # CHAPTER 19 ADOPTION TRUST/RISK METHODS
    # =====================================================
    
    async def update_trust_score_from_adoption(self, tenant_id: int, workflow_id: str, 
                                             quick_win_type: str, adoption_factors: Dict[str, Any]) -> Dict[str, Any]:
        """Update trust score based on adoption metrics (Task 19.1-T13)"""
        try:
            # Calculate adoption trust boost
            adoption_boost = self._calculate_adoption_trust_boost(adoption_factors)
            
            # Get current trust score
            current_score = await self.calculate_real_time_trust_score("workflow", workflow_id, tenant_id)
            
            # Apply adoption boost
            new_score = min(current_score.get("overall_score", 0.5) + adoption_boost, 1.0)
            
            trust_update = {
                "workflow_id": workflow_id,
                "tenant_id": tenant_id,
                "quick_win_type": quick_win_type,
                "previous_score": current_score.get("overall_score", 0.5),
                "adoption_boost": adoption_boost,
                "new_score": new_score,
                "adoption_factors": adoption_factors,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            return trust_update
            
        except Exception as e:
            logger.error(f"❌ Failed to update trust from adoption: {e}")
            raise
    
    async def calculate_expansion_trust_score(self, expansion_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trust score for multi-module expansion (Task 19.2-T13)"""
        try:
            cross_module_factors = expansion_config["cross_module_trust_factors"]
            synergy_multiplier = expansion_config["synergy_multiplier"]
            
            # Calculate base trust score from cross-module factors
            base_score = (
                cross_module_factors["integration_success_rate"] * 0.25 +
                cross_module_factors["data_flow_integrity"] * 0.20 +
                cross_module_factors["cross_module_sla_adherence"] * 0.20 +
                cross_module_factors["governance_consistency"] * 0.20 +
                cross_module_factors["rollback_capability"] * 0.15
            ) / 100.0
            
            # Apply synergy multiplier
            expansion_trust_score = min(base_score * synergy_multiplier, 1.0)
            
            # Determine trust level
            trust_level = self._determine_trust_level(expansion_trust_score)
            
            return {
                "expansion_id": expansion_config["expansion_id"],
                "base_score": base_score,
                "synergy_multiplier": synergy_multiplier,
                "expansion_trust_score": expansion_trust_score,
                "trust_level": trust_level,
                "cross_module_factors": cross_module_factors,
                "calculated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate expansion trust score: {e}")
            raise
    
    def _calculate_adoption_trust_boost(self, adoption_factors: Dict[str, Any]) -> float:
        """Calculate trust boost from adoption metrics"""
        # Weighted adoption factors
        weights = {
            "execution_success_rate": 0.30,
            "deployment_speed": 0.15,
            "business_impact_score": 0.25,
            "user_adoption_rate": 0.20,
            "compliance_adherence": 0.10
        }
        
        # Normalize and calculate boost
        boost = 0.0
        for factor, weight in weights.items():
            value = adoption_factors.get(factor, 0)
            
            # Normalize different factor types
            if factor == "execution_success_rate":
                normalized = min(value / 100.0, 1.0)
            elif factor == "deployment_speed":
                # Faster deployment = higher boost (inverse relationship)
                normalized = max(0, 1.0 - (value / 60.0))  # 60 minutes = 0 boost
            elif factor == "business_impact_score":
                normalized = min(value, 3.0) / 3.0  # Cap at 3x impact
            elif factor == "user_adoption_rate":
                normalized = min(value / 100.0, 1.0)
            elif factor == "compliance_adherence":
                normalized = min(value / 100.0, 1.0)
            else:
                normalized = 0.0
            
            boost += normalized * weight
        
        return min(boost * 0.1, 0.05)  # Cap boost at 5% (0.05)

# Initialize service
trust_service = None

def get_trust_service(pool_manager=Depends(get_pool_manager)) -> TrustScoringService:
    global trust_service
    if trust_service is None:
        trust_service = TrustScoringService(pool_manager)
    return trust_service

@router.post("/calculate-trust-score", response_model=TrustScoreResponse)
async def calculate_trust_score(
    request: TrustScoreRequest,
    service: TrustScoringService = Depends(get_trust_service)
):
    """
    Calculate trust score for a target
    Task 14.2-T06: Trust scoring engine service API
    """
    return await service.calculate_trust_score(request)

@router.post("/calculate-advanced", response_model=TrustScoreResponse)
async def calculate_advanced_trust_score(
    request: TrustScoreRequest,
    service: TrustScoringService = Depends(get_trust_service)
):
    """
    Calculate advanced trust score with industry weights and tier thresholds
    Tasks 14.2-T07, 14.2-T08, 14.2-T09, 14.2-T10, 14.2-T11, 14.2-T12: Advanced scoring algorithms
    """
    return await service.calculate_advanced_trust_score(request)

@router.get("/trust-score/{target_type}/{target_id}", response_model=TrustScoreResponse)
async def get_trust_score(
    target_type: TrustTargetType,
    target_id: str,
    tenant_id: int = Query(..., description="Tenant ID"),
    service: TrustScoringService = Depends(get_trust_service)
):
    """Get current trust score for a target"""
    
    request = TrustScoreRequest(
        target_type=target_type,
        target_id=target_id,
        tenant_id=tenant_id
    )
    return await service.calculate_trust_score(request)

@router.post("/trust-trends")
async def get_trust_trends(
    request: TrustTrendRequest,
    service: TrustScoringService = Depends(get_trust_service)
):
    """
    Get trust score trends over time
    Task 14.2-T59: Trust scoring pipeline and trend analysis
    """
    return await service.get_trust_trends(request)

@router.post("/trust-benchmarks")
async def get_trust_benchmarks(
    request: TrustBenchmarkRequest,
    service: TrustScoringService = Depends(get_trust_service)
):
    """
    Get industry trust benchmarks
    Task 14.2-T42: Industry trust benchmarks
    """
    return await service.get_trust_benchmarks(request)

@router.get("/trust-summary/{tenant_id}")
async def get_trust_summary(
    tenant_id: int,
    target_type: Optional[TrustTargetType] = Query(None, description="Filter by target type"),
    pool_manager=Depends(get_pool_manager)
):
    """Get trust score summary for a tenant"""
    
    async with pool_manager.get_connection() as conn:
        if target_type:
            query = """
                SELECT target_type, COUNT(*) as count,
                       AVG(overall_score) as avg_score,
                       COUNT(CASE WHEN trust_level = 'excellent' THEN 1 END) as excellent_count,
                       COUNT(CASE WHEN trust_level = 'good' THEN 1 END) as good_count,
                       COUNT(CASE WHEN trust_level = 'acceptable' THEN 1 END) as acceptable_count,
                       COUNT(CASE WHEN trust_level = 'concerning' THEN 1 END) as concerning_count,
                       COUNT(CASE WHEN trust_level = 'poor' THEN 1 END) as poor_count
                FROM trust_scores
                WHERE tenant_id = $1 AND target_type = $2
                  AND calculated_at >= $3
                GROUP BY target_type
            """
            rows = await conn.fetch(query, tenant_id, target_type.value, datetime.utcnow() - timedelta(days=30))
        else:
            query = """
                SELECT target_type, COUNT(*) as count,
                       AVG(overall_score) as avg_score,
                       COUNT(CASE WHEN trust_level = 'excellent' THEN 1 END) as excellent_count,
                       COUNT(CASE WHEN trust_level = 'good' THEN 1 END) as good_count,
                       COUNT(CASE WHEN trust_level = 'acceptable' THEN 1 END) as acceptable_count,
                       COUNT(CASE WHEN trust_level = 'concerning' THEN 1 END) as concerning_count,
                       COUNT(CASE WHEN trust_level = 'poor' THEN 1 END) as poor_count
                FROM trust_scores
                WHERE tenant_id = $1 AND calculated_at >= $2
                GROUP BY target_type
            """
            rows = await conn.fetch(query, tenant_id, datetime.utcnow() - timedelta(days=30))
        
        summary = []
        for row in rows:
            summary.append({
                "target_type": row['target_type'],
                "total_count": row['count'],
                "average_score": round(float(row['avg_score']), 3),
                "distribution": {
                    "excellent": row['excellent_count'],
                    "good": row['good_count'],
                    "acceptable": row['acceptable_count'],
                    "concerning": row['concerning_count'],
                    "poor": row['poor_count']
                }
            })
        
        return {
            "tenant_id": tenant_id,
            "summary": summary,
            "period": "last_30_days"
        }

@router.get("/real-time-trust-score")
async def get_real_time_trust_score(
    target_type: str = Query(..., description="Target type (workflow, capability, user, tenant)"),
    target_id: str = Query(..., description="Target ID"),
    tenant_id: int = Query(..., description="Tenant ID"),
    service: TrustScoringService = Depends(get_trust_service)
):
    """
    Get real-time trust score with caching
    Task 14.2-T13: Real-time trust score updates
    """
    try:
        result = await service.calculate_real_time_trust_score(target_type, target_id, tenant_id)
        return result
    except Exception as e:
        logger.error(f"❌ Real-time trust score failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trust-anomalies")
async def detect_trust_anomalies(
    target_type: str = Query(..., description="Target type (workflow, capability, user, tenant)"),
    target_id: str = Query(..., description="Target ID"),
    tenant_id: int = Query(..., description="Tenant ID"),
    lookback_days: int = Query(30, description="Days to analyze for anomalies", ge=7, le=365),
    service: TrustScoringService = Depends(get_trust_service)
):
    """
    Detect anomalies in trust score patterns
    Task 14.2-T28: Trust anomaly detection
    """
    try:
        result = await service.detect_trust_anomalies(target_type, target_id, tenant_id, lookback_days)
        return result
    except Exception as e:
        logger.error(f"❌ Trust anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trust-prediction")
async def predict_trust_score(
    target_type: str = Query(..., description="Target type (workflow, capability, user, tenant)"),
    target_id: str = Query(..., description="Target ID"),
    tenant_id: int = Query(..., description="Tenant ID"),
    prediction_days: int = Query(30, description="Days to predict ahead", ge=1, le=365),
    service: TrustScoringService = Depends(get_trust_service)
):
    """
    Predict future trust score using ML models
    Task 14.2-T27: Predictive trust modeling
    """
    try:
        result = await service.predict_trust_score(target_type, target_id, tenant_id, prediction_days)
        return result
    except Exception as e:
        logger.error(f"❌ Trust score prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# CHAPTER 19 ADOPTION TRUST/RISK INTEGRATION
# =====================================================

@router.post("/adoption/trust-update")
async def update_trust_from_adoption(
    workflow_id: str = Body(...),
    tenant_id: int = Body(...),
    adoption_metrics: Dict[str, Any] = Body(...),
    quick_win_type: str = Body(...),
    service: TrustScoringService = Depends(get_trust_service)
):
    """
    Configure trust scoring updates from quick-wins (Task 19.1-T13)
    """
    try:
        # Calculate adoption-based trust factors
        adoption_trust_factors = {
            "execution_success_rate": adoption_metrics.get("success_rate", 100.0),
            "deployment_speed": adoption_metrics.get("deployment_time_minutes", 15),
            "business_impact_score": adoption_metrics.get("roi_percentage", 150) / 100,
            "user_adoption_rate": adoption_metrics.get("user_engagement", 85.0),
            "compliance_adherence": adoption_metrics.get("policy_compliance_rate", 98.0)
        }
        
        # Update trust score with adoption factors
        trust_update = await service.update_trust_score_from_adoption(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            quick_win_type=quick_win_type,
            adoption_factors=adoption_trust_factors
        )
        
        return {
            "success": True,
            "trust_update": trust_update,
            "message": f"Trust score updated from {quick_win_type} adoption"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trust update failed: {str(e)}")

@router.post("/adoption/risk-update")
async def update_risk_from_adoption(
    workflow_id: str = Body(...),
    tenant_id: int = Body(...),
    adoption_metrics: Dict[str, Any] = Body(...),
    risk_factors: Dict[str, Any] = Body(...)
):
    """
    Configure risk register updates for quick-wins (Task 19.1-T14)
    """
    try:
        # Calculate adoption-based risk factors
        adoption_risk_assessment = {
            "deployment_risk": "low" if adoption_metrics.get("deployment_time_minutes", 15) < 30 else "medium",
            "execution_risk": "low" if adoption_metrics.get("success_rate", 100.0) > 95 else "medium",
            "compliance_risk": "low" if adoption_metrics.get("policy_compliance_rate", 98.0) > 95 else "high",
            "business_continuity_risk": "low" if adoption_metrics.get("rollback_capability", True) else "high",
            "data_integrity_risk": "low" if adoption_metrics.get("evidence_completeness", 100.0) == 100 else "medium"
        }
        
        # Update risk register
        risk_update_id = f"risk_update_{uuid.uuid4().hex[:12]}"
        
        risk_register_entry = {
            "risk_update_id": risk_update_id,
            "workflow_id": workflow_id,
            "tenant_id": tenant_id,
            "risk_assessment": adoption_risk_assessment,
            "risk_score": _calculate_composite_risk_score(adoption_risk_assessment),
            "mitigation_actions": _generate_risk_mitigation_actions(adoption_risk_assessment),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "risk_update": risk_register_entry,
            "message": "Risk register updated from adoption metrics"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk update failed: {str(e)}")

@router.post("/expansion/trust-scoring")
async def configure_expansion_trust_scoring(
    expansion_id: str = Body(...),
    tenant_id: int = Body(...),
    modules: List[str] = Body(...),
    expansion_metrics: Dict[str, Any] = Body(...),
    service: TrustScoringService = Depends(get_trust_service)
):
    """
    Configure trust scoring for expanded workflows (Task 19.2-T13)
    """
    try:
        # Multi-module trust scoring configuration
        expansion_trust_config = {
            "expansion_id": expansion_id,
            "tenant_id": tenant_id,
            "modules": modules,
            "cross_module_trust_factors": {
                "integration_success_rate": expansion_metrics.get("integration_success_rate", 95.0),
                "data_flow_integrity": expansion_metrics.get("data_flow_integrity", 98.0),
                "cross_module_sla_adherence": expansion_metrics.get("sla_adherence", 97.0),
                "governance_consistency": expansion_metrics.get("governance_consistency", 99.0),
                "rollback_capability": expansion_metrics.get("rollback_success_rate", 100.0)
            },
            "synergy_multiplier": 1.2 if len(modules) > 2 else 1.1  # Bonus for multi-module integration
        }
        
        # Calculate expansion trust score
        expansion_trust_score = await service.calculate_expansion_trust_score(expansion_trust_config)
        
        return {
            "success": True,
            "expansion_trust_config": expansion_trust_config,
            "expansion_trust_score": expansion_trust_score,
            "message": f"Trust scoring configured for {len(modules)}-module expansion"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expansion trust scoring failed: {str(e)}")

@router.post("/expansion/automated-pipeline")
async def automate_expansion_trust_risk_pipeline(
    tenant_id: int = Body(...),
    pipeline_config: Dict[str, Any] = Body(...)
):
    """
    Automate trust/risk scoring pipeline for expansions (Task 19.2-T40)
    """
    try:
        pipeline_id = f"trust_risk_pipeline_{tenant_id}_{uuid.uuid4().hex[:8]}"
        
        automated_pipeline = {
            "pipeline_id": pipeline_id,
            "tenant_id": tenant_id,
            "automation_config": {
                "trigger_events": ["expansion_start", "module_integration", "expansion_complete"],
                "trust_scoring_frequency": "real_time",
                "risk_assessment_frequency": "per_module",
                "alert_thresholds": {
                    "trust_score_drop": 10.0,  # Alert if trust drops by 10 points
                    "risk_score_increase": 15.0  # Alert if risk increases by 15 points
                },
                "auto_remediation": {
                    "enabled": pipeline_config.get("auto_remediation", True),
                    "rollback_threshold": 25.0,  # Auto-rollback if risk > 25
                    "escalation_threshold": 20.0  # Escalate if trust < 20
                }
            },
            "integration_points": {
                "trust_scoring_service": "enabled",
                "risk_register_service": "enabled", 
                "governance_dashboard": "enabled",
                "alert_manager": "enabled",
                "evidence_pack_service": "enabled"
            },
            "created_at": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "automated_pipeline": automated_pipeline,
            "message": "Automated trust/risk pipeline configured for expansions"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline automation failed: {str(e)}")

# Helper functions for risk assessment
def _calculate_composite_risk_score(risk_assessment: Dict[str, str]) -> float:
    """Calculate composite risk score from individual risk factors"""
    risk_weights = {"low": 1.0, "medium": 5.0, "high": 10.0}
    total_score = sum(risk_weights.get(level, 5.0) for level in risk_assessment.values())
    return min(total_score / len(risk_assessment), 10.0)  # Normalize to 0-10 scale

def _generate_risk_mitigation_actions(risk_assessment: Dict[str, str]) -> List[str]:
    """Generate risk mitigation actions based on assessment"""
    actions = []
    if risk_assessment.get("compliance_risk") == "high":
        actions.append("Implement additional compliance validation checks")
    if risk_assessment.get("execution_risk") == "medium":
        actions.append("Add execution monitoring and retry mechanisms")
    if risk_assessment.get("business_continuity_risk") == "high":
        actions.append("Enhance rollback procedures and testing")
    return actions

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "trust_scoring_api", "timestamp": datetime.utcnow()}
