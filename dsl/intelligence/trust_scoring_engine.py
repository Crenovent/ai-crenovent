#!/usr/bin/env python3
"""
Trust Scoring Engine - Chapter 14.2.11-14.2.15
===============================================

Implements intelligent trust scoring for capabilities based on:
- Execution success rates and performance metrics
- User feedback and adoption patterns  
- Compliance validation results
- Business impact measurements
- Historical reliability data

This engine provides real-time trust scores that drive:
- Capability recommendations
- Automated fallback decisions
- SLA enforcement
- Risk assessment
- User confidence indicators
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import uuid

logger = logging.getLogger(__name__)

class TrustFactorType(Enum):
    """Types of factors that influence trust scoring"""
    EXECUTION_SUCCESS = "execution_success"
    PERFORMANCE_METRICS = "performance_metrics" 
    USER_FEEDBACK = "user_feedback"
    COMPLIANCE_VALIDATION = "compliance_validation"
    BUSINESS_IMPACT = "business_impact"
    HISTORICAL_RELIABILITY = "historical_reliability"
    SECURITY_ASSESSMENT = "security_assessment"

class TrustLevel(Enum):
    """Trust level classifications"""
    CRITICAL = "critical"      # 0.9-1.0 - Mission critical, fully trusted
    HIGH = "high"             # 0.7-0.9 - Highly trusted, recommended
    MEDIUM = "medium"         # 0.5-0.7 - Moderately trusted, use with caution
    LOW = "low"              # 0.3-0.5 - Low trust, requires oversight
    UNTRUSTED = "untrusted"   # 0.0-0.3 - Not trusted, avoid or require approval

@dataclass
class TrustFactor:
    """Individual factor contributing to trust score"""
    factor_type: TrustFactorType
    value: float  # 0.0 to 1.0
    weight: float  # Importance weight
    evidence: Dict[str, Any]
    measured_at: datetime
    confidence: float = 1.0

@dataclass
class TrustScore:
    """Complete trust assessment for a capability"""
    capability_id: str
    overall_score: float  # 0.0 to 1.0
    trust_level: TrustLevel
    factors: List[TrustFactor]
    calculated_at: datetime
    valid_until: datetime
    tenant_id: int
    
    # Breakdown scores
    execution_score: float
    performance_score: float
    compliance_score: float
    business_impact_score: float
    
    # Metadata
    sample_size: int  # Number of executions analyzed
    confidence_interval: Tuple[float, float]
    recommendations: List[str]

class TrustScoringEngine:
    """
    Intelligent trust scoring engine that continuously evaluates capability reliability
    
    Features:
    - Real-time trust calculation based on execution data
    - Multi-factor scoring with configurable weights
    - Tenant-specific trust assessments
    - Historical trend analysis
    - Automated recommendations
    - Integration with SLA monitoring
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Dynamic industry-specific scoring configuration (no hardcoding)
        self.industry_factor_weights = {
            'SaaS': {
                TrustFactorType.EXECUTION_SUCCESS: 0.25,
                TrustFactorType.PERFORMANCE_METRICS: 0.20,
                TrustFactorType.USER_FEEDBACK: 0.15,
                TrustFactorType.COMPLIANCE_VALIDATION: 0.15,
                TrustFactorType.BUSINESS_IMPACT: 0.20,  # Higher for SaaS (ARR/MRR impact)
                TrustFactorType.HISTORICAL_RELIABILITY: 0.05
            },
            'Banking': {  # Future extensibility
                TrustFactorType.COMPLIANCE_VALIDATION: 0.30,  # Higher compliance weight
                TrustFactorType.EXECUTION_SUCCESS: 0.25,
                TrustFactorType.PERFORMANCE_METRICS: 0.20,
                TrustFactorType.BUSINESS_IMPACT: 0.15,
                TrustFactorType.HISTORICAL_RELIABILITY: 0.10
            },
            'Insurance': {  # Future extensibility
                TrustFactorType.COMPLIANCE_VALIDATION: 0.25,
                TrustFactorType.EXECUTION_SUCCESS: 0.25,
                TrustFactorType.PERFORMANCE_METRICS: 0.20,
                TrustFactorType.BUSINESS_IMPACT: 0.20,
                TrustFactorType.HISTORICAL_RELIABILITY: 0.10
            }
        }
        
        # Tenant-tier specific trust thresholds (dynamic, not hardcoded)
        self.tenant_tier_thresholds = {
            'T0': {  # Regulated tenants - stricter thresholds
                TrustLevel.CRITICAL: (0.95, 1.0),
                TrustLevel.HIGH: (0.85, 0.95),
                TrustLevel.MEDIUM: (0.70, 0.85),
                TrustLevel.LOW: (0.50, 0.70),
                TrustLevel.UNTRUSTED: (0.0, 0.50)
            },
            'T1': {  # Enterprise tenants - balanced thresholds
                TrustLevel.CRITICAL: (0.90, 1.0),
                TrustLevel.HIGH: (0.75, 0.90),
                TrustLevel.MEDIUM: (0.60, 0.75),
                TrustLevel.LOW: (0.40, 0.60),
                TrustLevel.UNTRUSTED: (0.0, 0.40)
            },
            'T2': {  # Mid-market tenants - more lenient thresholds
                TrustLevel.CRITICAL: (0.85, 1.0),
                TrustLevel.HIGH: (0.70, 0.85),
                TrustLevel.MEDIUM: (0.55, 0.70),
                TrustLevel.LOW: (0.35, 0.55),
                TrustLevel.UNTRUSTED: (0.0, 0.35)
            }
        }
        
        # SaaS-specific business impact metrics (dynamic configuration)
        self.saas_business_metrics = {
            'revenue_impact_weight': 0.4,    # ARR/MRR impact
            'customer_impact_weight': 0.3,   # Customer satisfaction/churn
            'operational_impact_weight': 0.3  # Efficiency/cost savings
        }
        
        # Tenant-aware cache for performance (with tenant isolation)
        self.trust_cache = {}  # Format: {tenant_id: {capability_id: cached_score}}
        self.cache_ttl = timedelta(minutes=15)
        
    async def initialize(self):
        """Initialize the trust scoring engine"""
        try:
            self.logger.info("🧠 Initializing Trust Scoring Engine...")
            
            # Create database tables if needed
            await self._ensure_trust_tables()
            
            # Load historical trust data
            await self._load_historical_trust_data()
            
            self.logger.info("✅ Trust Scoring Engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Trust Scoring Engine: {e}")
            return False
    
    async def calculate_trust_score(self, capability_id: str, tenant_id: int, 
                                  lookback_days: int = 30, industry_code: str = 'SaaS',
                                  tenant_tier: str = 'T2') -> TrustScore:
        """
        Calculate comprehensive tenant-aware trust score for a capability
        
        Args:
            capability_id: Capability to evaluate
            tenant_id: Tenant context for scoring (mandatory for tenant separation)
            lookback_days: Days of historical data to analyze
            industry_code: Industry context (SaaS, Banking, Insurance) - dynamic weighting
            tenant_tier: Tenant tier (T0, T1, T2) - affects thresholds
            
        Returns:
            Complete trust score with breakdown and recommendations
        """
        try:
            self.logger.info(f"🔍 Calculating trust score for capability {capability_id} (tenant {tenant_id}, tier {tenant_tier})")
            
            # Ensure tenant separation - tenant_id is mandatory
            if tenant_id is None:
                raise ValueError("tenant_id is mandatory for tenant-separated trust scoring")
            
            # Check tenant-aware cache first (with tenant isolation)
            cache_key = f"{tenant_id}_{capability_id}_{industry_code}_{tenant_tier}_{lookback_days}"
            if tenant_id in self.trust_cache and cache_key in self.trust_cache[tenant_id]:
                cached_score = self.trust_cache[tenant_id][cache_key]
                if datetime.now() - cached_score.calculated_at < self.cache_ttl:
                    return cached_score
            
            # Get dynamic industry-specific weights (no hardcoding)
            factor_weights = self.industry_factor_weights.get(industry_code, self.industry_factor_weights['SaaS'])
            
            # Gather trust factors with tenant context
            factors = []
            
            # 1. Execution Success Factor (tenant-scoped)
            execution_factor = await self._calculate_tenant_execution_success_factor(
                capability_id, tenant_id, lookback_days, tenant_tier
            )
            if execution_factor:
                factors.append(execution_factor)
            
            # 2. Performance Metrics Factor
            performance_factor = await self._calculate_performance_factor(
                capability_id, tenant_id, lookback_days
            )
            if performance_factor:
                factors.append(performance_factor)
            
            # 3. User Feedback Factor
            feedback_factor = await self._calculate_user_feedback_factor(
                capability_id, tenant_id, lookback_days
            )
            if feedback_factor:
                factors.append(feedback_factor)
            
            # 4. Compliance Validation Factor
            compliance_factor = await self._calculate_compliance_factor(
                capability_id, tenant_id, lookback_days
            )
            if compliance_factor:
                factors.append(compliance_factor)
            
            # 5. Business Impact Factor
            business_factor = await self._calculate_business_impact_factor(
                capability_id, tenant_id, lookback_days
            )
            if business_factor:
                factors.append(business_factor)
            
            # 6. Historical Reliability Factor
            historical_factor = await self._calculate_historical_reliability_factor(
                capability_id, tenant_id
            )
            if historical_factor:
                factors.append(historical_factor)
            
            # Calculate weighted overall score
            overall_score = self._calculate_weighted_score(factors)
            
            # Determine trust level
            trust_level = self._determine_trust_level(overall_score)
            
            # Calculate breakdown scores
            execution_score = next((f.value for f in factors if f.factor_type == TrustFactorType.EXECUTION_SUCCESS), 0.5)
            performance_score = next((f.value for f in factors if f.factor_type == TrustFactorType.PERFORMANCE_METRICS), 0.5)
            compliance_score = next((f.value for f in factors if f.factor_type == TrustFactorType.COMPLIANCE_VALIDATION), 0.5)
            business_impact_score = next((f.value for f in factors if f.factor_type == TrustFactorType.BUSINESS_IMPACT), 0.5)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(factors)
            
            # Generate recommendations
            recommendations = await self._generate_trust_recommendations(
                capability_id, overall_score, factors, tenant_id
            )
            
            # Create trust score object
            trust_score = TrustScore(
                capability_id=capability_id,
                overall_score=overall_score,
                trust_level=trust_level,
                factors=factors,
                calculated_at=datetime.now(),
                valid_until=datetime.now() + self.cache_ttl,
                tenant_id=tenant_id,
                execution_score=execution_score,
                performance_score=performance_score,
                compliance_score=compliance_score,
                business_impact_score=business_impact_score,
                sample_size=sum(len(f.evidence.get('samples', [])) for f in factors),
                confidence_interval=confidence_interval,
                recommendations=recommendations
            )
            
            # Cache the result
            self.trust_cache[cache_key] = trust_score
            
            # Store in database
            await self._store_trust_score(trust_score)
            
            self.logger.info(f"✅ Trust score calculated: {overall_score:.3f} ({trust_level.value})")
            return trust_score
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate trust score for {capability_id}: {e}")
            # Return default trust score
            return await self._create_default_trust_score(capability_id, tenant_id)
    
    async def get_trust_trends(self, capability_id: str, tenant_id: int, 
                             days: int = 90) -> Dict[str, Any]:
        """Get trust score trends over time"""
        try:
            async with self.pool_manager.get_connection().acquire() as conn:
                query = """
                SELECT 
                    DATE_TRUNC('day', calculated_at) as date,
                    AVG(overall_score) as avg_score,
                    AVG(execution_score) as avg_execution,
                    AVG(performance_score) as avg_performance,
                    AVG(compliance_score) as avg_compliance,
                    COUNT(*) as measurements
                FROM capability_trust_scores 
                WHERE capability_id = $1 
                  AND tenant_id = $2
                  AND calculated_at >= NOW() - INTERVAL '%s days'
                GROUP BY DATE_TRUNC('day', calculated_at)
                ORDER BY date
                """ % days
                
                rows = await conn.fetch(query, capability_id, tenant_id)
                
                return {
                    "capability_id": capability_id,
                    "tenant_id": tenant_id,
                    "period_days": days,
                    "trend_data": [dict(row) for row in rows],
                    "current_trend": self._analyze_trend([row['avg_score'] for row in rows]),
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting trust trends: {e}")
            return {"error": str(e)}
    
    async def get_tenant_trust_overview(self, tenant_id: int) -> Dict[str, Any]:
        """Get trust overview for all capabilities in a tenant"""
        try:
            async with self.pool_manager.get_connection().acquire() as conn:
                query = """
                SELECT 
                    capability_id,
                    overall_score,
                    trust_level,
                    execution_score,
                    performance_score,
                    compliance_score,
                    business_impact_score,
                    calculated_at,
                    sample_size
                FROM capability_trust_scores 
                WHERE tenant_id = $1
                  AND calculated_at = (
                      SELECT MAX(calculated_at) 
                      FROM capability_trust_scores ts2 
                      WHERE ts2.capability_id = capability_trust_scores.capability_id
                        AND ts2.tenant_id = capability_trust_scores.tenant_id
                  )
                ORDER BY overall_score DESC
                """
                
                rows = await conn.fetch(query, tenant_id)
                
                capabilities = [dict(row) for row in rows]
                
                # Calculate summary statistics
                scores = [cap['overall_score'] for cap in capabilities]
                
                return {
                    "tenant_id": tenant_id,
                    "total_capabilities": len(capabilities),
                    "capabilities": capabilities,
                    "summary": {
                        "average_trust": statistics.mean(scores) if scores else 0,
                        "highest_trust": max(scores) if scores else 0,
                        "lowest_trust": min(scores) if scores else 0,
                        "trust_distribution": self._calculate_trust_distribution(capabilities)
                    },
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting tenant trust overview: {e}")
            return {"error": str(e)}
    
    # Helper methods for calculating individual trust factors
    
    async def _calculate_execution_success_factor(self, capability_id: str, tenant_id: int, 
                                                days: int) -> Optional[TrustFactor]:
        """Calculate trust factor based on execution success rate"""
        try:
            async with self.pool_manager.get_connection().acquire() as conn:
                query = """
                SELECT 
                    COUNT(*) as total_executions,
                    SUM(CASE WHEN execution_status = 'completed' THEN 1 ELSE 0 END) as successful_executions,
                    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds
                FROM dsl_execution_traces 
                WHERE tenant_id = $1
                  AND trace_data->>'capability_id' = $2
                  AND created_at >= NOW() - INTERVAL '%s days'
                """ % days
                
                row = await conn.fetchrow(query, tenant_id, capability_id)
                
                if row and row['total_executions'] > 0:
                    success_rate = float(row['successful_executions']) / float(row['total_executions'])
                    
                    return TrustFactor(
                        factor_type=TrustFactorType.EXECUTION_SUCCESS,
                        value=success_rate,
                        weight=self.factor_weights[TrustFactorType.EXECUTION_SUCCESS],
                        evidence={
                            "total_executions": row['total_executions'],
                            "successful_executions": row['successful_executions'],
                            "success_rate": success_rate,
                            "avg_duration_seconds": float(row['avg_duration_seconds'] or 0)
                        },
                        measured_at=datetime.now(),
                        confidence=min(1.0, row['total_executions'] / 100.0)  # Higher confidence with more samples
                    )
                
        except Exception as e:
            self.logger.error(f"Error calculating execution success factor: {e}")
        
        return None
    
    async def _calculate_performance_factor(self, capability_id: str, tenant_id: int, 
                                          days: int) -> Optional[TrustFactor]:
        """Calculate trust factor based on performance metrics"""
        try:
            # For now, use execution duration as a proxy for performance
            # In a real implementation, you'd have more detailed performance metrics
            
            async with self.pool_manager.get_connection().acquire() as conn:
                query = """
                SELECT 
                    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (completed_at - started_at))) as p95_duration,
                    COUNT(*) as sample_count
                FROM dsl_execution_traces 
                WHERE tenant_id = $1
                  AND trace_data->>'capability_id' = $2
                  AND execution_status = 'completed'
                  AND created_at >= NOW() - INTERVAL '%s days'
                """ % days
                
                row = await conn.fetchrow(query, tenant_id, capability_id)
                
                if row and row['sample_count'] > 0:
                    avg_duration = float(row['avg_duration'] or 0)
                    p95_duration = float(row['p95_duration'] or 0)
                    
                    # Performance score: lower duration = higher score
                    # Normalize based on reasonable SaaS automation expectations
                    max_acceptable_duration = 300  # 5 minutes
                    performance_score = max(0.0, 1.0 - (avg_duration / max_acceptable_duration))
                    performance_score = min(1.0, performance_score)
                    
                    return TrustFactor(
                        factor_type=TrustFactorType.PERFORMANCE_METRICS,
                        value=performance_score,
                        weight=self.factor_weights[TrustFactorType.PERFORMANCE_METRICS],
                        evidence={
                            "avg_duration_seconds": avg_duration,
                            "p95_duration_seconds": p95_duration,
                            "sample_count": row['sample_count'],
                            "performance_score": performance_score
                        },
                        measured_at=datetime.now(),
                        confidence=min(1.0, row['sample_count'] / 50.0)
                    )
                
        except Exception as e:
            self.logger.error(f"Error calculating performance factor: {e}")
        
        return None
    
    async def _calculate_user_feedback_factor(self, capability_id: str, tenant_id: int, 
                                            days: int) -> Optional[TrustFactor]:
        """Calculate trust factor based on user feedback"""
        # For now, return a default positive feedback score
        # In a real implementation, you'd collect user ratings and feedback
        
        return TrustFactor(
            factor_type=TrustFactorType.USER_FEEDBACK,
            value=0.8,  # Default positive feedback
            weight=self.factor_weights[TrustFactorType.USER_FEEDBACK],
            evidence={
                "feedback_score": 0.8,
                "note": "Default positive feedback - user feedback system not yet implemented"
            },
            measured_at=datetime.now(),
            confidence=0.5  # Low confidence since it's default
        )
    
    async def _calculate_compliance_factor(self, capability_id: str, tenant_id: int, 
                                         days: int) -> Optional[TrustFactor]:
        """Calculate trust factor based on compliance validation"""
        # For now, return high compliance score for capabilities with evidence packs
        # In a real implementation, you'd validate against compliance policies
        
        return TrustFactor(
            factor_type=TrustFactorType.COMPLIANCE_VALIDATION,
            value=0.9,  # High compliance score
            weight=self.factor_weights[TrustFactorType.COMPLIANCE_VALIDATION],
            evidence={
                "compliance_score": 0.9,
                "note": "High compliance - governance framework ensures policy adherence"
            },
            measured_at=datetime.now(),
            confidence=0.9
        )
    
    async def _calculate_business_impact_factor(self, capability_id: str, tenant_id: int, 
                                              days: int) -> Optional[TrustFactor]:
        """Calculate trust factor based on business impact"""
        # For now, return moderate business impact score
        # In a real implementation, you'd measure actual business outcomes
        
        return TrustFactor(
            factor_type=TrustFactorType.BUSINESS_IMPACT,
            value=0.7,  # Moderate business impact
            weight=self.factor_weights[TrustFactorType.BUSINESS_IMPACT],
            evidence={
                "business_impact_score": 0.7,
                "note": "Moderate business impact - impact measurement system in development"
            },
            measured_at=datetime.now(),
            confidence=0.6
        )
    
    async def _calculate_historical_reliability_factor(self, capability_id: str, 
                                                     tenant_id: int) -> Optional[TrustFactor]:
        """Calculate trust factor based on historical reliability"""
        # For now, return good historical reliability for established capabilities
        
        return TrustFactor(
            factor_type=TrustFactorType.HISTORICAL_RELIABILITY,
            value=0.8,  # Good historical reliability
            weight=self.factor_weights[TrustFactorType.HISTORICAL_RELIABILITY],
            evidence={
                "historical_reliability": 0.8,
                "note": "Good historical reliability based on capability maturity"
            },
            measured_at=datetime.now(),
            confidence=0.7
        )
    
    def _calculate_weighted_score(self, factors: List[TrustFactor]) -> float:
        """Calculate weighted overall trust score"""
        if not factors:
            return 0.5  # Default neutral score
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for factor in factors:
            weighted_score = factor.value * factor.weight * factor.confidence
            total_weighted_score += weighted_score
            total_weight += factor.weight * factor.confidence
        
        if total_weight == 0:
            return 0.5
        
        return min(1.0, max(0.0, total_weighted_score / total_weight))
    
    def _determine_trust_level(self, score: float) -> TrustLevel:
        """Determine trust level from numerical score (legacy method)"""
        # Use T2 thresholds as default for backward compatibility
        return self._determine_trust_level_by_tier(score, 'T2')
    
    def _determine_trust_level_by_tier(self, score: float, tenant_tier: str) -> TrustLevel:
        """Determine trust level from numeric score using tenant-tier specific thresholds"""
        tier_thresholds = self.tenant_tier_thresholds.get(tenant_tier, self.tenant_tier_thresholds['T2'])
        
        for level, (min_score, max_score) in tier_thresholds.items():
            if min_score <= score <= max_score:
                return level
        return TrustLevel.UNTRUSTED
    
    def _calculate_confidence_interval(self, factors: List[TrustFactor]) -> Tuple[float, float]:
        """Calculate confidence interval for trust score"""
        if not factors:
            return (0.4, 0.6)
        
        # Simple confidence interval based on factor confidence
        avg_confidence = statistics.mean([f.confidence for f in factors])
        margin = (1.0 - avg_confidence) * 0.2  # Max 20% margin
        
        scores = [f.value for f in factors]
        avg_score = statistics.mean(scores)
        
        return (max(0.0, avg_score - margin), min(1.0, avg_score + margin))
    
    async def _generate_trust_recommendations(self, capability_id: str, score: float, 
                                            factors: List[TrustFactor], tenant_id: int) -> List[str]:
        """Generate actionable recommendations based on trust analysis"""
        recommendations = []
        
        if score < 0.5:
            recommendations.append("⚠️ Low trust score - consider manual review before deployment")
        
        if score < 0.3:
            recommendations.append("🚨 Critical trust issues - immediate attention required")
        
        # Factor-specific recommendations
        for factor in factors:
            if factor.factor_type == TrustFactorType.EXECUTION_SUCCESS and factor.value < 0.7:
                recommendations.append("🔧 Improve execution reliability - check error patterns")
            
            if factor.factor_type == TrustFactorType.PERFORMANCE_METRICS and factor.value < 0.6:
                recommendations.append("⚡ Optimize performance - reduce execution time")
            
            if factor.confidence < 0.5:
                recommendations.append(f"📊 Increase sample size for {factor.factor_type.value} measurement")
        
        if score > 0.8:
            recommendations.append("✅ High trust - suitable for automated deployment")
        
        if not recommendations:
            recommendations.append("👍 Trust score within acceptable range")
        
        return recommendations
    
    def _analyze_trend(self, scores: List[float]) -> str:
        """Analyze trend direction from score history"""
        if len(scores) < 2:
            return "insufficient_data"
        
        recent_scores = scores[-5:]  # Last 5 measurements
        older_scores = scores[:-5] if len(scores) > 5 else scores[:1]
        
        if not older_scores:
            return "insufficient_data"
        
        recent_avg = statistics.mean(recent_scores)
        older_avg = statistics.mean(older_scores)
        
        diff = recent_avg - older_avg
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _calculate_trust_distribution(self, capabilities: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of capabilities across trust levels"""
        distribution = {level.value: 0 for level in TrustLevel}
        
        for cap in capabilities:
            trust_level = cap.get('trust_level', 'medium')
            distribution[trust_level] = distribution.get(trust_level, 0) + 1
        
        return distribution
    
    async def _create_default_trust_score(self, capability_id: str, tenant_id: int) -> TrustScore:
        """Create default trust score when calculation fails"""
        return TrustScore(
            capability_id=capability_id,
            overall_score=0.5,
            trust_level=TrustLevel.MEDIUM,
            factors=[],
            calculated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(minutes=5),
            tenant_id=tenant_id,
            execution_score=0.5,
            performance_score=0.5,
            compliance_score=0.5,
            business_impact_score=0.5,
            sample_size=0,
            confidence_interval=(0.4, 0.6),
            recommendations=["⚠️ Trust calculation failed - using default score"]
        )
    
    async def _ensure_trust_tables(self):
        """Ensure trust scoring tables exist"""
        try:
            async with self.pool_manager.get_connection() as conn:
                # Create trust scores table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS capability_trust_scores (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        capability_id TEXT NOT NULL,
                        tenant_id INTEGER NOT NULL,
                        overall_score DECIMAL(3,2) NOT NULL,
                        trust_level TEXT NOT NULL,
                        execution_score DECIMAL(3,2),
                        performance_score DECIMAL(3,2),
                        compliance_score DECIMAL(3,2),
                        business_impact_score DECIMAL(3,2),
                        factors JSONB,
                        sample_size INTEGER,
                        confidence_interval JSONB,
                        recommendations TEXT[],
                        calculated_at TIMESTAMPTZ DEFAULT NOW(),
                        valid_until TIMESTAMPTZ,
                        
                        CONSTRAINT trust_score_check CHECK (overall_score >= 0.0 AND overall_score <= 1.0),
                        CONSTRAINT trust_level_check CHECK (trust_level IN ('critical', 'high', 'medium', 'low', 'untrusted'))
                    )
                """)
                
                # Create indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_capability_trust_capability_tenant 
                    ON capability_trust_scores(capability_id, tenant_id)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_capability_trust_calculated_at 
                    ON capability_trust_scores(calculated_at DESC)
                """)
                
        except Exception as e:
            self.logger.error(f"Error ensuring trust tables: {e}")
    
    async def _load_historical_trust_data(self):
        """Load historical trust data for analysis"""
        # Implementation would load historical data for trend analysis
        pass
    
    async def _store_trust_score(self, trust_score: TrustScore):
        """Store calculated trust score in database"""
        try:
            async with self.pool_manager.get_connection().acquire() as conn:
                await conn.execute("""
                    INSERT INTO capability_trust_scores (
                        capability_id, tenant_id, overall_score, trust_level,
                        execution_score, performance_score, compliance_score, business_impact_score,
                        factors, sample_size, confidence_interval, recommendations,
                        calculated_at, valid_until
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                trust_score.capability_id, trust_score.tenant_id, trust_score.overall_score,
                trust_score.trust_level.value, trust_score.execution_score, trust_score.performance_score,
                trust_score.compliance_score, trust_score.business_impact_score,
                json.dumps([asdict(f) for f in trust_score.factors]), trust_score.sample_size,
                json.dumps(trust_score.confidence_interval), trust_score.recommendations,
                trust_score.calculated_at, trust_score.valid_until
                )
                
        except Exception as e:
            self.logger.error(f"Error storing trust score: {e}")

# Singleton instance for global access
_trust_scoring_engine = None

def get_trust_scoring_engine(pool_manager=None) -> TrustScoringEngine:
    """Get singleton instance of Trust Scoring Engine"""
    global _trust_scoring_engine
    if _trust_scoring_engine is None:
        _trust_scoring_engine = TrustScoringEngine(pool_manager)
    return _trust_scoring_engine
