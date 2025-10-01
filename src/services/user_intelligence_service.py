"""
Revolutionary User Intelligence Service
Provides complete user context awareness for strategic planning
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class UserIntelligence:
    user_id: int
    strategic_patterns: Dict[str, Any]
    success_predictors: Dict[str, float]
    stakeholder_network: List[Dict]
    industry_expertise: List[Dict]
    performance_trajectory: Dict[str, float]
    risk_tolerance_profile: Dict[str, Any]
    communication_preferences: Dict[str, str]
    goal_achievement_patterns: Dict[str, float]
    preferred_strategies: List[str]
    
class UserIntelligenceService:
    """Revolutionary user context awareness system"""
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
    async def get_comprehensive_user_intelligence(self, user_id: int) -> UserIntelligence:
        """
        REVOLUTIONARY: Get complete user intelligence
        This makes the AI fully aware of user's history and patterns
        """
        try:
            self.logger.info(f"🧠 Loading comprehensive intelligence for user {user_id}")
            
            # Sequential execution to avoid asyncio loop conflicts
            strategic_patterns = await self._analyze_strategic_patterns(user_id)
            success_factors = await self._identify_success_factors(user_id)
            stakeholder_relationships = await self._map_stakeholder_relationships(user_id)
            industry_knowledge = await self._assess_industry_knowledge(user_id)
            performance_trends = await self._calculate_performance_trends(user_id)
            risk_patterns = await self._analyze_risk_patterns(user_id)
            communication_style = await self._detect_communication_style(user_id)
            goal_success = await self._analyze_goal_success(user_id)
            preferred_strategies = await self._identify_preferred_strategies(user_id)
            
            results = [
                strategic_patterns, success_factors, stakeholder_relationships,
                industry_knowledge, performance_trends, risk_patterns,
                communication_style, goal_success, preferred_strategies
            ]
            
            return UserIntelligence(
                user_id=user_id,
                strategic_patterns=results[0] if not isinstance(results[0], Exception) else {},
                success_predictors=results[1] if not isinstance(results[1], Exception) else {},
                stakeholder_network=results[2] if not isinstance(results[2], Exception) else [],
                industry_expertise=results[3] if not isinstance(results[3], Exception) else [],
                performance_trajectory=results[4] if not isinstance(results[4], Exception) else {},
                risk_tolerance_profile=results[5] if not isinstance(results[5], Exception) else {},
                communication_preferences=results[6] if not isinstance(results[6], Exception) else {},
                goal_achievement_patterns=results[7] if not isinstance(results[7], Exception) else {},
                preferred_strategies=results[8] if not isinstance(results[8], Exception) else []
            )
            
        except Exception as e:
            self.logger.error(f"❌ User intelligence loading failed: {e}")
            return self._get_default_intelligence(user_id)
    
    async def _analyze_strategic_patterns(self, user_id: int) -> Dict[str, Any]:
        """Analyze user's strategic planning patterns"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Analyze historical plans for patterns
                plans = await conn.fetch("""
                    SELECT plan_name, account_tier, annual_revenue, industry,
                           short_term_goals, long_term_goals, revenue_growth_target,
                           communication_cadence, created_at
                    FROM strategic_account_plans 
                    WHERE created_by_user_id = $1 
                    ORDER BY created_at DESC 
                    LIMIT 50
                """, user_id)
                
                if not plans:
                    return {"status": "no_history", "patterns": []}
                
                patterns = {
                    "typical_account_tiers": self._extract_tier_patterns(plans),
                    "revenue_ranges": self._extract_revenue_patterns(plans),
                    "preferred_industries": self._extract_industry_patterns(plans),
                    "goal_themes": self._extract_goal_patterns(plans),
                    "planning_frequency": self._calculate_planning_frequency(plans),
                    "success_indicators": await self._calculate_success_patterns(conn, user_id)
                }
                
                return patterns
                
        except Exception as e:
            self.logger.warning(f"⚠️ Strategic pattern analysis failed: {e}")
            return {"error": str(e)}
    
    async def _identify_success_factors(self, user_id: int) -> Dict[str, float]:
        """Identify what makes this user's plans successful"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Get plans with success metrics
                success_data = await conn.fetch("""
                    SELECT p.*, 
                           CASE 
                               WHEN p.revenue_growth_target::float > 15 THEN 0.8
                               WHEN p.revenue_growth_target::float > 10 THEN 0.7
                               ELSE 0.6
                           END as success_score
                    FROM strategic_account_plans p
                    WHERE p.created_by_user_id = $1
                    ORDER BY p.created_at DESC
                    LIMIT 20
                """, user_id)
                
                if not success_data:
                    return {"revenue_correlation": 0.5, "stakeholder_impact": 0.6}
                
                # Analyze success factors
                factors = {
                    "revenue_correlation": self._calculate_revenue_success_correlation(success_data),
                    "stakeholder_impact": self._calculate_stakeholder_success_impact(success_data),
                    "timing_factor": self._calculate_timing_success_factor(success_data),
                    "industry_expertise_factor": self._calculate_industry_success_factor(success_data)
                }
                
                return factors
                
        except Exception as e:
            self.logger.warning(f"⚠️ Success factor analysis failed: {e}")
            return {"general_success_rate": 0.7}
    
    async def _map_stakeholder_relationships(self, user_id: int) -> List[Dict]:
        """Map user's stakeholder relationship patterns"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Get stakeholder patterns from plan content
                plans = await conn.fetch("""
                    SELECT stakeholder_map
                    FROM strategic_account_plans
                    WHERE created_by_user_id = $1 AND stakeholder_map IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 30
                """, user_id)
                
                stakeholder_patterns = []
                for plan in plans:
                    try:
                        if plan['stakeholder_map']:
                            stakeholders = json.loads(plan['stakeholder_map']) if isinstance(plan['stakeholder_map'], str) else plan['stakeholder_map']
                            if isinstance(stakeholders, list):
                                stakeholder_patterns.extend(stakeholders)
                    except (json.JSONDecodeError, TypeError):
                        continue
                
                # Analyze patterns
                role_frequency = {}
                for stakeholder in stakeholder_patterns:
                    if isinstance(stakeholder, dict) and 'role' in stakeholder:
                        role = stakeholder['role']
                        role_frequency[role] = role_frequency.get(role, 0) + 1
                
                return [{"role": role, "frequency": freq, "preference_score": freq / len(stakeholder_patterns)} 
                       for role, freq in role_frequency.items()]
                
        except Exception as e:
            self.logger.warning(f"⚠️ Stakeholder mapping failed: {e}")
            return []
    
    async def _assess_industry_knowledge(self, user_id: int) -> List[Dict]:
        """Assess user's industry expertise"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                industry_data = await conn.fetch("""
                    SELECT industry, COUNT(*) as plan_count,
                           AVG(CAST(annual_revenue as FLOAT)) as avg_revenue
                    FROM strategic_account_plans
                    WHERE created_by_user_id = $1 AND industry IS NOT NULL
                    GROUP BY industry
                    ORDER BY plan_count DESC
                """, user_id)
                
                expertise = []
                for row in industry_data:
                    expertise.append({
                        "industry": row['industry'],
                        "experience_level": "expert" if row['plan_count'] >= 5 else "experienced" if row['plan_count'] >= 2 else "familiar",
                        "plan_count": row['plan_count'],
                        "avg_account_size": float(row['avg_revenue']) if row['avg_revenue'] else 0
                    })
                
                return expertise
                
        except Exception as e:
            self.logger.warning(f"⚠️ Industry assessment failed: {e}")
            return []
    
    async def _calculate_performance_trends(self, user_id: int) -> Dict[str, float]:
        """Calculate user's performance trends"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                trends = await conn.fetch("""
                    SELECT 
                        DATE_TRUNC('month', created_at) as month,
                        COUNT(*) as plans_created,
                        AVG(CAST(revenue_growth_target as FLOAT)) as avg_growth_target
                    FROM strategic_account_plans
                    WHERE created_by_user_id = $1 
                    AND created_at >= NOW() - INTERVAL '12 months'
                    GROUP BY DATE_TRUNC('month', created_at)
                    ORDER BY month DESC
                """, user_id)
                
                if not trends:
                    return {"planning_velocity": 0.5, "growth_ambition": 0.1}
                
                # Calculate trends
                total_plans = sum(row['plans_created'] for row in trends)
                avg_growth = sum(row['avg_growth_target'] or 0 for row in trends) / len(trends)
                
                return {
                    "planning_velocity": min(total_plans / 12, 2.0),  # Plans per month, capped at 2
                    "growth_ambition": avg_growth / 100,  # Convert percentage to decimal
                    "consistency": len(trends) / 12  # How consistently they plan
                }
                
        except Exception as e:
            self.logger.warning(f"⚠️ Performance trend calculation failed: {e}")
            return {"planning_velocity": 0.5, "growth_ambition": 0.1}
    
    async def _analyze_risk_patterns(self, user_id: int) -> Dict[str, Any]:
        """Analyze user's risk tolerance patterns"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                risk_data = await conn.fetch("""
                    SELECT known_risks, risk_mitigation_strategies, revenue_growth_target
                    FROM strategic_account_plans
                    WHERE created_by_user_id = $1 
                    AND known_risks IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 20
                """, user_id)
                
                if not risk_data:
                    return {"tolerance_level": "moderate", "mitigation_preference": "standard"}
                
                # Analyze risk patterns
                risk_mentions = []
                mitigation_complexity = []
                
                for row in risk_data:
                    if row['known_risks']:
                        risk_mentions.append(len(row['known_risks'].split(',')))
                    if row['risk_mitigation_strategies']:
                        mitigation_complexity.append(len(row['risk_mitigation_strategies'].split(',')))
                
                avg_risks_identified = sum(risk_mentions) / len(risk_mentions) if risk_mentions else 2
                avg_mitigation_complexity = sum(mitigation_complexity) / len(mitigation_complexity) if mitigation_complexity else 2
                
                return {
                    "tolerance_level": "high" if avg_risks_identified > 4 else "moderate" if avg_risks_identified > 2 else "conservative",
                    "mitigation_preference": "comprehensive" if avg_mitigation_complexity > 3 else "standard",
                    "risk_awareness_score": min(avg_risks_identified / 5, 1.0)
                }
                
        except Exception as e:
            self.logger.warning(f"⚠️ Risk pattern analysis failed: {e}")
            return {"tolerance_level": "moderate", "mitigation_preference": "standard"}
    
    async def _detect_communication_style(self, user_id: int) -> Dict[str, str]:
        """Detect user's communication preferences"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                comm_data = await conn.fetch("""
                    SELECT communication_cadence
                    FROM strategic_account_plans
                    WHERE created_by_user_id = $1 
                    AND communication_cadence IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 20
                """, user_id)
                
                if not comm_data:
                    return {"preferred_cadence": "monthly", "style": "formal"}
                
                # Analyze communication patterns
                cadences = [row['communication_cadence'].lower() for row in comm_data if row['communication_cadence']]
                
                # Determine preferred cadence
                if any('weekly' in cadence for cadence in cadences):
                    preferred_cadence = "weekly"
                elif any('monthly' in cadence for cadence in cadences):
                    preferred_cadence = "monthly"
                elif any('quarterly' in cadence for cadence in cadences):
                    preferred_cadence = "quarterly"
                else:
                    preferred_cadence = "monthly"
                
                # Determine style
                formal_indicators = sum(1 for cadence in cadences if any(word in cadence for word in ['strategic', 'business', 'executive']))
                style = "formal" if formal_indicators > len(cadences) / 2 else "collaborative"
                
                return {
                    "preferred_cadence": preferred_cadence,
                    "style": style,
                    "engagement_level": "high" if preferred_cadence == "weekly" else "moderate"
                }
                
        except Exception as e:
            self.logger.warning(f"⚠️ Communication style detection failed: {e}")
            return {"preferred_cadence": "monthly", "style": "formal"}
    
    async def _analyze_goal_success(self, user_id: int) -> Dict[str, float]:
        """Analyze user's goal achievement patterns"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                goal_data = await conn.fetch("""
                    SELECT short_term_goals, long_term_goals, revenue_growth_target
                    FROM strategic_account_plans
                    WHERE created_by_user_id = $1
                    ORDER BY created_at DESC
                    LIMIT 30
                """, user_id)
                
                if not goal_data:
                    return {"achievement_rate": 0.65, "ambition_level": 0.5}
                
                # Analyze goal complexity and ambition
                short_term_complexity = []
                long_term_complexity = []
                growth_targets = []
                
                for row in goal_data:
                    if row['short_term_goals']:
                        short_term_complexity.append(len(row['short_term_goals'].split('.')))
                    if row['long_term_goals']:
                        long_term_complexity.append(len(row['long_term_goals'].split('.')))
                    if row['revenue_growth_target']:
                        try:
                            growth_targets.append(float(row['revenue_growth_target']))
                        except (ValueError, TypeError):
                            pass
                
                avg_short_complexity = sum(short_term_complexity) / len(short_term_complexity) if short_term_complexity else 2
                avg_long_complexity = sum(long_term_complexity) / len(long_term_complexity) if long_term_complexity else 2
                avg_growth_target = sum(growth_targets) / len(growth_targets) if growth_targets else 10
                
                return {
                    "achievement_rate": min(0.9, 0.5 + (avg_short_complexity / 10)),  # Estimated based on complexity
                    "ambition_level": min(1.0, avg_growth_target / 20),  # Normalized growth ambition
                    "goal_setting_sophistication": (avg_short_complexity + avg_long_complexity) / 8
                }
                
        except Exception as e:
            self.logger.warning(f"⚠️ Goal success analysis failed: {e}")
            return {"achievement_rate": 0.65, "ambition_level": 0.5}
    
    async def _identify_preferred_strategies(self, user_id: int) -> List[str]:
        """Identify user's preferred strategic approaches"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                strategy_data = await conn.fetch("""
                    SELECT key_opportunities, cross_sell_upsell_potential, risk_mitigation_strategies
                    FROM strategic_account_plans
                    WHERE created_by_user_id = $1
                    ORDER BY created_at DESC
                    LIMIT 20
                """, user_id)
                
                if not strategy_data:
                    return ["stakeholder_engagement", "revenue_growth", "risk_management"]
                
                # Analyze strategy preferences
                strategy_keywords = {
                    "stakeholder_engagement": ["stakeholder", "relationship", "engagement", "partnership"],
                    "revenue_growth": ["revenue", "growth", "expansion", "upsell", "cross-sell"],
                    "market_expansion": ["market", "expansion", "new", "penetration"],
                    "risk_management": ["risk", "mitigation", "compliance", "security"],
                    "innovation_focus": ["innovation", "technology", "digital", "transformation"],
                    "operational_excellence": ["efficiency", "optimization", "process", "quality"]
                }
                
                strategy_scores = {strategy: 0 for strategy in strategy_keywords}
                
                for row in strategy_data:
                    text_content = " ".join([
                        row['key_opportunities'] or "",
                        row['cross_sell_upsell_potential'] or "",
                        row['risk_mitigation_strategies'] or ""
                    ]).lower()
                    
                    for strategy, keywords in strategy_keywords.items():
                        for keyword in keywords:
                            if keyword in text_content:
                                strategy_scores[strategy] += 1
                
                # Return top strategies
                sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
                return [strategy for strategy, score in sorted_strategies[:5] if score > 0]
                
        except Exception as e:
            self.logger.warning(f"⚠️ Strategy preference analysis failed: {e}")
            return ["stakeholder_engagement", "revenue_growth", "risk_management"]
    
    async def _calculate_success_patterns(self, conn, user_id: int) -> Dict[str, Any]:
        """Calculate success patterns from historical data"""
        try:
            success_metrics = await conn.fetch("""
                SELECT 
                    COUNT(*) as total_plans,
                    AVG(CAST(revenue_growth_target as FLOAT)) as avg_growth_target,
                    COUNT(DISTINCT industry) as industry_diversity
                FROM strategic_account_plans
                WHERE created_by_user_id = $1
            """, user_id)
            
            if success_metrics:
                row = success_metrics[0]
                return {
                    "total_experience": row['total_plans'],
                    "average_growth_ambition": row['avg_growth_target'] or 10,
                    "industry_versatility": row['industry_diversity']
                }
            
            return {"total_experience": 0, "average_growth_ambition": 10, "industry_versatility": 0}
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_default_intelligence(self, user_id: int) -> UserIntelligence:
        """Fallback intelligence for new users"""
        return UserIntelligence(
            user_id=user_id,
            strategic_patterns={"status": "new_user"},
            success_predictors={"general_rate": 0.6},
            stakeholder_network=[],
            industry_expertise=[],
            performance_trajectory={"growth_trend": 0.1},
            risk_tolerance_profile={"level": "moderate"},
            communication_preferences={"cadence": "monthly"},
            goal_achievement_patterns={"success_rate": 0.65},
            preferred_strategies=["stakeholder_engagement", "revenue_growth"]
        )
    
    # Helper methods for pattern analysis
    def _extract_tier_patterns(self, plans) -> Dict:
        """Extract account tier patterns"""
        tiers = [plan['account_tier'] for plan in plans if plan['account_tier']]
        tier_counts = {}
        for tier in tiers:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        return tier_counts
    
    def _extract_revenue_patterns(self, plans) -> Dict:
        """Extract revenue patterns"""
        revenues = []
        for plan in plans:
            if plan['annual_revenue']:
                try:
                    revenues.append(float(plan['annual_revenue']))
                except (ValueError, TypeError):
                    pass
        
        if not revenues:
            return {"typical_range": "unknown"}
        
        avg_revenue = sum(revenues) / len(revenues)
        return {
            "average": avg_revenue,
            "range": f"{min(revenues):.0f}-{max(revenues):.0f}",
            "preferred_size": "enterprise" if avg_revenue > 10000000 else "mid_market"
        }
    
    def _extract_industry_patterns(self, plans) -> Dict:
        """Extract industry patterns"""
        industries = [plan['industry'] for plan in plans if plan['industry']]
        industry_counts = {}
        for industry in industries:
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        return industry_counts
    
    def _extract_goal_patterns(self, plans) -> List[str]:
        """Extract common goal themes"""
        all_goals = []
        for plan in plans:
            if plan['short_term_goals']:
                all_goals.append(plan['short_term_goals'])
            if plan['long_term_goals']:
                all_goals.append(plan['long_term_goals'])
        
        # Simple keyword extraction for common themes
        themes = []
        goal_text = " ".join(all_goals).lower()
        
        if "growth" in goal_text or "expand" in goal_text:
            themes.append("growth_focused")
        if "efficiency" in goal_text or "optimize" in goal_text:
            themes.append("efficiency_focused")
        if "market" in goal_text or "penetration" in goal_text:
            themes.append("market_expansion")
        if "relationship" in goal_text or "stakeholder" in goal_text:
            themes.append("relationship_building")
        
        return themes
    
    def _calculate_planning_frequency(self, plans) -> str:
        """Calculate how frequently user creates plans"""
        if len(plans) < 2:
            return "new_planner"
        
        # Get date range
        dates = [plan['created_at'] for plan in plans if plan['created_at']]
        if len(dates) < 2:
            return "irregular"
        
        date_range = max(dates) - min(dates)
        plans_per_month = len(plans) / max(1, date_range.days / 30)
        
        if plans_per_month >= 2:
            return "frequent"
        elif plans_per_month >= 0.5:
            return "regular"
        else:
            return "occasional"
    
    def _calculate_revenue_success_correlation(self, success_data) -> float:
        """Calculate correlation between revenue size and success"""
        try:
            revenues = []
            success_scores = []
            
            for row in success_data:
                if row['annual_revenue'] and row['success_score']:
                    try:
                        revenues.append(float(row['annual_revenue']))
                        success_scores.append(float(row['success_score']))
                    except (ValueError, TypeError):
                        continue
            
            if len(revenues) < 3:
                return 0.5  # Default correlation
            
            # Simple correlation calculation
            n = len(revenues)
            sum_x = sum(revenues)
            sum_y = sum(success_scores)
            sum_xy = sum(x * y for x, y in zip(revenues, success_scores))
            sum_x2 = sum(x * x for x in revenues)
            sum_y2 = sum(y * y for y in success_scores)
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
            
            if denominator == 0:
                return 0.5
            
            correlation = numerator / denominator
            return abs(correlation)  # Return absolute correlation
            
        except Exception:
            return 0.5
    
    def _calculate_stakeholder_success_impact(self, success_data) -> float:
        """Calculate stakeholder engagement impact on success"""
        # Simplified calculation based on stakeholder mentions
        stakeholder_mentions = 0
        total_plans = len(success_data)
        
        for row in success_data:
            # Check for stakeholder-related content in various fields
            content = " ".join([
                str(row.get('key_opportunities', '')),
                str(row.get('risk_mitigation_strategies', '')),
                str(row.get('short_term_goals', ''))
            ]).lower()
            
            if any(word in content for word in ['stakeholder', 'relationship', 'engagement', 'partnership']):
                stakeholder_mentions += 1
        
        return min(0.9, stakeholder_mentions / max(1, total_plans) + 0.3)
    
    def _calculate_timing_success_factor(self, success_data) -> float:
        """Calculate timing factor for success"""
        # Simplified calculation based on plan creation timing
        return 0.7  # Default timing factor
    
    def _calculate_industry_success_factor(self, success_data) -> float:
        """Calculate industry expertise factor"""
        industries = set()
        for row in success_data:
            if row.get('industry'):
                industries.add(row['industry'])
        
        # More industries = higher expertise factor (up to a point)
        expertise_factor = min(0.9, len(industries) / 5 + 0.4)
        return expertise_factor

# Global instance
user_intelligence_service = None

def get_user_intelligence_service(pool_manager=None):
    """Get or create user intelligence service instance"""
    global user_intelligence_service
    if user_intelligence_service is None:
        user_intelligence_service = UserIntelligenceService(pool_manager)
    return user_intelligence_service
