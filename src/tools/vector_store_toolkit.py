"""
Vector Store Toolkit for Semantic Search through Strategic Plans
Enables AI agents to perform semantic similarity search across plan embeddings
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorStoreToolkit:
    """Vector store toolkit for semantic search through plan embeddings"""
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
    async def semantic_search_plans(self, query: str, user_id: int, limit: int = 5, similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Perform semantic search through strategic plans using pgvector
        """
        try:
            if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool'):
                self.logger.warning("⚠️ PostgreSQL pool not available for vector search")
                return []
            
            # Generate embedding for the query
            query_embedding = await self._generate_query_embedding(query)
            if not query_embedding:
                return []
            
            conn = await self.pool_manager.postgres_pool.acquire()
            try:
                # Perform semantic search using pgvector
                results = await conn.fetch("""
                    SELECT 
                        sap.plan_id,
                        sap.plan_name,
                        sap.account_id,
                        sap.short_term_goals,
                        sap.long_term_goals,
                        sap.industry,
                        sap.account_tier,
                        sap.annual_revenue,
                        sap.revenue_growth_target,
                        sap.created_at,
                        pe.embedding <-> $1::vector AS similarity_distance,
                        (1 - (pe.embedding <-> $1::vector)) AS similarity_score
                    FROM strategic_account_plans sap
                    JOIN plan_embeddings pe ON sap.plan_id = pe.plan_id
                    WHERE sap.created_by_user_id = $2
                      AND (1 - (pe.embedding <-> $1::vector)) >= $3
                    ORDER BY pe.embedding <-> $1::vector
                    LIMIT $4
                """, query_embedding, user_id, similarity_threshold, limit)
                
                if results:
                    self.logger.info(f"✅ Vector search found {len(results)} similar plans for query: '{query}'")
                    return [dict(row) for row in results]
                else:
                    self.logger.info(f"🔍 No similar plans found for query: '{query}' (threshold: {similarity_threshold})")
                    return []
                    
            finally:
                await self.pool_manager.postgres_pool.release(conn)
                
        except Exception as e:
            self.logger.error(f"❌ Vector search failed: {e}")
            return []
    
    async def find_similar_goals(self, goals: str, user_id: int, goal_type: str = 'short_term') -> List[Dict]:
        """
        Find similar goals from historical plans
        """
        try:
            query = f"Find plans with similar {goal_type} goals: {goals}"
            similar_plans = await self.semantic_search_plans(query, user_id, limit=3, similarity_threshold=0.6)
            
            if similar_plans:
                self.logger.info(f"✅ Found {len(similar_plans)} plans with similar {goal_type} goals")
                return similar_plans
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"❌ Similar goals search failed: {e}")
            return []
    
    async def find_similar_accounts(self, account_context: str, user_id: int) -> List[Dict]:
        """
        Find similar accounts based on context
        """
        try:
            query = f"Find similar accounts: {account_context}"
            similar_plans = await self.semantic_search_plans(query, user_id, limit=3, similarity_threshold=0.65)
            
            if similar_plans:
                self.logger.info(f"✅ Found {len(similar_plans)} similar account plans")
                return similar_plans
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"❌ Similar accounts search failed: {e}")
            return []
    
    async def get_plan_insights(self, query: str, user_id: int) -> Dict[str, Any]:
        """
        Get comprehensive insights from similar plans
        """
        try:
            similar_plans = await self.semantic_search_plans(query, user_id, limit=5, similarity_threshold=0.6)
            
            if not similar_plans:
                return {"insights": [], "patterns": {}, "recommendations": []}
            
            # Analyze patterns
            patterns = self._analyze_plan_patterns(similar_plans)
            insights = self._extract_plan_insights(similar_plans)
            recommendations = self._generate_recommendations(similar_plans)
            
            return {
                "insights": insights,
                "patterns": patterns,
                "recommendations": recommendations,
                "similar_plans_count": len(similar_plans)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Plan insights generation failed: {e}")
            return {"insights": [], "patterns": {}, "recommendations": []}
    
    async def _generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for search query"""
        try:
            if not self.pool_manager or not self.pool_manager.openai_client:
                self.logger.warning("⚠️ OpenAI client not available for embedding generation")
                return None
            
            response = await self.pool_manager.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            
            embedding = response.data[0].embedding
            self.logger.info(f"✅ Generated embedding for query: '{query[:50]}...'")
            return embedding
            
        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed: {e}")
            return None
    
    def _analyze_plan_patterns(self, plans: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in similar plans"""
        if not plans:
            return {}
        
        patterns = {
            "common_industries": {},
            "typical_revenue_ranges": [],
            "common_account_tiers": {},
            "average_growth_targets": 0,
            "goal_themes": []
        }
        
        # Analyze industries
        for plan in plans:
            industry = plan.get('industry', '')
            if industry:
                patterns["common_industries"][industry] = patterns["common_industries"].get(industry, 0) + 1
        
        # Analyze account tiers
        for plan in plans:
            tier = plan.get('account_tier', '')
            if tier:
                patterns["common_account_tiers"][tier] = patterns["common_account_tiers"].get(tier, 0) + 1
        
        # Analyze revenue ranges
        revenues = []
        for plan in plans:
            revenue = plan.get('annual_revenue')
            if revenue:
                try:
                    revenues.append(float(revenue))
                except (ValueError, TypeError):
                    pass
        
        if revenues:
            patterns["typical_revenue_ranges"] = {
                "min": min(revenues),
                "max": max(revenues),
                "average": sum(revenues) / len(revenues)
            }
        
        # Analyze growth targets
        growth_targets = []
        for plan in plans:
            target = plan.get('revenue_growth_target')
            if target:
                try:
                    growth_targets.append(float(target))
                except (ValueError, TypeError):
                    pass
        
        if growth_targets:
            patterns["average_growth_targets"] = sum(growth_targets) / len(growth_targets)
        
        return patterns
    
    def _extract_plan_insights(self, plans: List[Dict]) -> List[str]:
        """Extract key insights from similar plans"""
        insights = []
        
        if len(plans) >= 2:
            insights.append(f"Found {len(plans)} similar strategic plans with comparable characteristics")
        
        # Revenue insights
        revenues = [plan.get('annual_revenue') for plan in plans if plan.get('annual_revenue')]
        if revenues:
            try:
                avg_revenue = sum(float(r) for r in revenues) / len(revenues)
                insights.append(f"Average revenue of similar accounts: ${avg_revenue:,.0f}")
            except (ValueError, TypeError):
                pass
        
        # Industry insights
        industries = [plan.get('industry') for plan in plans if plan.get('industry')]
        if industries:
            most_common_industry = max(set(industries), key=industries.count)
            insights.append(f"Most common industry in similar plans: {most_common_industry}")
        
        # Timeline insights
        creation_dates = [plan.get('created_at') for plan in plans if plan.get('created_at')]
        if creation_dates:
            recent_plans = len([d for d in creation_dates if d and d.year >= datetime.now().year])
            insights.append(f"{recent_plans} of these plans were created this year")
        
        return insights
    
    def _generate_recommendations(self, plans: List[Dict]) -> List[str]:
        """Generate recommendations based on similar plans"""
        recommendations = []
        
        if len(plans) >= 3:
            recommendations.append("Consider reviewing successful strategies from similar account plans")
        
        # Growth target recommendations
        growth_targets = []
        for plan in plans:
            target = plan.get('revenue_growth_target')
            if target:
                try:
                    growth_targets.append(float(target))
                except (ValueError, TypeError):
                    pass
        
        if growth_targets:
            avg_growth = sum(growth_targets) / len(growth_targets)
            recommendations.append(f"Similar accounts typically target {avg_growth:.1f}% growth")
        
        # Account tier recommendations
        tiers = [plan.get('account_tier') for plan in plans if plan.get('account_tier')]
        if tiers:
            most_common_tier = max(set(tiers), key=tiers.count)
            recommendations.append(f"Most similar accounts are classified as {most_common_tier} tier")
        
        return recommendations

# Global instance
vector_store_toolkit = None

def get_vector_store_toolkit(pool_manager=None):
    """Get or create vector store toolkit instance"""
    global vector_store_toolkit
    if vector_store_toolkit is None:
        vector_store_toolkit = VectorStoreToolkit(pool_manager)
    return vector_store_toolkit














































