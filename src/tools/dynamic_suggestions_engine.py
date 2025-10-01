"""
Dynamic Suggestions Engine
Provides real-time, context-aware suggestions that update based on database changes
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class DynamicSuggestionsEngine:
    """Generate dynamic, data-driven suggestions for users"""
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        self.suggestion_cache = {}
        self.last_update = {}
    
    async def get_dynamic_suggestions(self, user_id: int, context: str = "planning") -> List[Dict[str, Any]]:
        """
        Generate dynamic suggestions based on current database state and user patterns
        """
        try:
            # Check if we need to refresh suggestions (every 30 seconds)
            cache_key = f"{user_id}_{context}"
            now = datetime.now()
            
            if (cache_key not in self.last_update or 
                (now - self.last_update[cache_key]).total_seconds() > 30):
                
                suggestions = await self._generate_fresh_suggestions(user_id, context)
                self.suggestion_cache[cache_key] = suggestions
                self.last_update[cache_key] = now
                self.logger.info(f"✅ Generated {len(suggestions)} dynamic suggestions for user {user_id}")
            
            return self.suggestion_cache.get(cache_key, [])
            
        except Exception as e:
            self.logger.error(f"❌ Dynamic suggestions failed: {e}")
            return self._get_fallback_suggestions(context)
    
    async def _generate_fresh_suggestions(self, user_id: int, context: str) -> List[Dict[str, Any]]:
        """Generate fresh suggestions based on current database state"""
        suggestions = []
        
        try:
            if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool') or not self.pool_manager.postgres_pool:
                return self._get_fallback_suggestions(context)
            
            conn = await asyncio.wait_for(self.pool_manager.postgres_pool.acquire(), timeout=3.0)
            try:
                # Get user's recent activity
                recent_plans = await conn.fetch("""
                    SELECT plan_name, account_id, industry, account_tier, created_at,
                           short_term_goals, long_term_goals, revenue_growth_target
                    FROM strategic_account_plans 
                    WHERE created_by_user_id = $1 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """, user_id)
                
                # Get industry trends
                industry_stats = await conn.fetch("""
                    SELECT industry, COUNT(*) as plan_count, 
                           AVG(CAST(revenue_growth_target AS NUMERIC)) as avg_growth
                    FROM strategic_account_plans 
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                      AND industry IS NOT NULL
                    GROUP BY industry 
                    ORDER BY plan_count DESC 
                    LIMIT 3
                """)
                
                # Generate contextual suggestions
                suggestions.extend(await self._create_account_suggestions(recent_plans, user_id))
                suggestions.extend(await self._create_industry_insights(industry_stats))
                suggestions.extend(await self._create_performance_suggestions(recent_plans))
                suggestions.extend(await self._create_action_suggestions(user_id))
                
            finally:
                await self.pool_manager.postgres_pool.release(conn)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Database suggestions failed: {e}")
            suggestions = self._get_fallback_suggestions(context)
        
        # Randomize and limit suggestions
        random.shuffle(suggestions)
        return suggestions[:8]  # Max 8 suggestions
    
    async def _create_account_suggestions(self, recent_plans: List, user_id: int) -> List[Dict[str, Any]]:
        """Create suggestions based on user's recent account planning activity"""
        suggestions = []
        
        if not recent_plans:
            suggestions.append({
                "type": "action",
                "icon": "💡",
                "title": "Create Your First Strategic Plan",
                "description": "Start by creating a strategic account plan to track goals and opportunities",
                "action": "create_plan",
                "priority": "high",
                "timestamp": datetime.now().isoformat()
            })
            return suggestions
        
        # Analysis based on recent activity
        latest_plan = recent_plans[0]
        industries = list(set([p['industry'] for p in recent_plans if p['industry']]))
        avg_growth = sum([float(p['revenue_growth_target'] or 0) for p in recent_plans]) / len(recent_plans)
        
        # Recent activity insights
        suggestions.append({
            "type": "insight",
            "icon": "📊",
            "title": "Your Account Planning Momentum",
            "description": f"You've created {len(recent_plans)} plans across {len(industries)} industries. Average growth target: {avg_growth:.1f}%",
            "action": "view_analytics",
            "priority": "medium",
            "timestamp": datetime.now().isoformat()
        })
        
        # Industry focus suggestion
        if industries:
            most_common_industry = max(set(industries), key=industries.count) if industries else "Technology"
            suggestions.append({
                "type": "opportunity",
                "icon": "🎯",
                "title": f"Expand Your {most_common_industry} Portfolio",
                "description": f"Consider creating more strategic plans for {most_common_industry} accounts to build expertise",
                "action": f"create_plan_{most_common_industry.lower()}",
                "priority": "medium",
                "timestamp": datetime.now().isoformat()
            })
        
        # Growth optimization
        if avg_growth < 15:
            suggestions.append({
                "type": "optimization",
                "icon": "📈",
                "title": "Optimize Growth Targets",
                "description": f"Your average growth target ({avg_growth:.1f}%) could be more ambitious. Industry benchmark is 15-25%",
                "action": "review_targets",
                "priority": "high",
                "timestamp": datetime.now().isoformat()
            })
        
        return suggestions
    
    async def _create_industry_insights(self, industry_stats: List) -> List[Dict[str, Any]]:
        """Create suggestions based on industry trends"""
        suggestions = []
        
        if not industry_stats:
            return suggestions
        
        # Trending industry
        top_industry = industry_stats[0]
        suggestions.append({
            "type": "trend",
            "icon": "🔥",
            "title": f"{top_industry['industry']} is Trending",
            "description": f"{top_industry['plan_count']} new plans created this month. Avg growth: {top_industry['avg_growth']:.1f}%",
            "action": f"explore_{top_industry['industry'].lower().replace(' ', '_')}",
            "priority": "medium",
            "timestamp": datetime.now().isoformat()
        })
        
        # Growth comparison
        if len(industry_stats) >= 2:
            high_growth_industry = max(industry_stats, key=lambda x: x['avg_growth'] or 0)
            suggestions.append({
                "type": "benchmark",
                "icon": "🏆",
                "title": f"High-Growth Opportunity: {high_growth_industry['industry']}",
                "description": f"Plans in {high_growth_industry['industry']} are targeting {high_growth_industry['avg_growth']:.1f}% growth on average",
                "action": f"benchmark_{high_growth_industry['industry'].lower().replace(' ', '_')}",
                "priority": "medium",
                "timestamp": datetime.now().isoformat()
            })
        
        return suggestions
    
    async def _create_performance_suggestions(self, recent_plans: List) -> List[Dict[str, Any]]:
        """Create performance-related suggestions"""
        suggestions = []
        
        if recent_plans:
            # Recent plan optimization
            latest = recent_plans[0]
            days_since = (datetime.now() - latest['created_at']).days
            
            if days_since < 7:
                suggestions.append({
                    "type": "follow_up",
                    "icon": "🎯",
                    "title": "Review Your Latest Plan",
                    "description": f"Your plan for {latest['account_id']} is {days_since} days old. Consider adding activities or updating goals",
                    "action": f"review_plan_{latest.get('plan_id', 'latest')}",
                    "priority": "medium",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Goal completion tracking
            if len(recent_plans) >= 3:
                suggestions.append({
                    "type": "tracking",
                    "icon": "📋",
                    "title": "Track Goal Progress",
                    "description": f"You have {len(recent_plans)} active plans. Set up progress tracking to monitor goal achievement",
                    "action": "setup_tracking",
                    "priority": "high",
                    "timestamp": datetime.now().isoformat()
                })
        
        return suggestions
    
    async def _create_action_suggestions(self, user_id: int) -> List[Dict[str, Any]]:
        """Create actionable suggestions for immediate value"""
        suggestions = []
        
        # Time-sensitive actions
        current_hour = datetime.now().hour
        
        if 9 <= current_hour <= 17:  # Business hours
            suggestions.append({
                "type": "action",
                "icon": "💬",
                "title": "Schedule Stakeholder Check-in",
                "description": "Perfect time to reach out to key stakeholders. Consider scheduling follow-up meetings",
                "action": "schedule_meetings",
                "priority": "medium",
                "timestamp": datetime.now().isoformat()
            })
        
        # Weekly planning
        if datetime.now().weekday() == 0:  # Monday
            suggestions.append({
                "type": "planning",
                "icon": "📅",
                "title": "Weekly Strategic Planning",
                "description": "Start your week by reviewing account plans and setting priorities for maximum impact",
                "action": "weekly_review",
                "priority": "high",
                "timestamp": datetime.now().isoformat()
            })
        
        # Random productivity tips
        productivity_tips = [
            {
                "type": "tip",
                "icon": "💡",
                "title": "AI-Powered Plan Generation",
                "description": "Try natural language: 'Create plan for Microsoft with 500B revenue and weekly check-ins'",
                "action": "try_ai_generation",
                "priority": "low"
            },
            {
                "type": "tip",
                "icon": "🔄",
                "title": "Template Optimization",
                "description": "Create reusable templates from your best-performing plans to save time",
                "action": "create_templates",
                "priority": "low"
            },
            {
                "type": "tip",
                "icon": "📊",
                "title": "Analytics Deep Dive",
                "description": "Analyze which account tiers and industries drive the highest growth rates",
                "action": "view_analytics",
                "priority": "low"
            }
        ]
        
        suggestions.append(random.choice(productivity_tips))
        suggestions[-1]["timestamp"] = datetime.now().isoformat()
        
        return suggestions
    
    def _get_fallback_suggestions(self, context: str) -> List[Dict[str, Any]]:
        """Fallback suggestions when database is unavailable"""
        fallback_suggestions = [
            {
                "type": "action",
                "icon": "🚀",
                "title": "Quick Plan Creation",
                "description": "Create a strategic account plan in under 2 minutes using AI assistance",
                "action": "quick_create",
                "priority": "high",
                "timestamp": datetime.now().isoformat()
            },
            {
                "type": "feature",
                "icon": "🤖",
                "title": "AI Form Filling",
                "description": "Try: 'Create plan for Salesforce with Marc Benioff, 25% growth target'",
                "action": "try_ai",
                "priority": "medium",
                "timestamp": datetime.now().isoformat()
            },
            {
                "type": "tip",
                "icon": "💡",
                "title": "Strategic Planning Best Practice",
                "description": "Include specific revenue targets, stakeholder roles, and quarterly activities for maximum impact",
                "action": "learn_more",
                "priority": "low",
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        return fallback_suggestions

# Global instance
dynamic_suggestions_engine = None

def get_dynamic_suggestions_engine(pool_manager=None):
    """Get or create dynamic suggestions engine instance"""
    global dynamic_suggestions_engine
    if dynamic_suggestions_engine is None:
        dynamic_suggestions_engine = DynamicSuggestionsEngine(pool_manager)
    return dynamic_suggestions_engine






























