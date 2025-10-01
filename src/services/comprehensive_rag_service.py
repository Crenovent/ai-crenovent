"""
Comprehensive RAG Service for Strategic Planning Intelligence
Vectorizes and searches all user plans, patterns, and historical data
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class ComprehensiveRAGService:
    """Comprehensive RAG service for intelligent form filling using pgvector"""
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        self.embedding_cache = {}
        
    async def vectorize_all_user_data(self, user_id: int) -> bool:
        """
        Vectorize ALL user-related data for comprehensive RAG
        """
        try:
            # Skip vectorization if pool manager not available to avoid conflicts
            if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool') or not self.pool_manager.postgres_pool:
                self.logger.warning(f"⚠️ PostgreSQL pool not available, skipping vectorization for user {user_id}")
                return False
            
            self.logger.info(f"🔄 Starting comprehensive vectorization for user {user_id}")
            
            # Vectorize all strategic plans
            plans_vectorized = await self._vectorize_strategic_plans(user_id)
            
            # Vectorize user patterns and preferences  
            patterns_vectorized = await self._vectorize_user_patterns(user_id)
            
            # Vectorize stakeholder relationships
            stakeholders_vectorized = await self._vectorize_stakeholder_data(user_id)
            
            # Vectorize activities and engagement patterns
            activities_vectorized = await self._vectorize_activity_patterns(user_id)
            
            self.logger.info(f"✅ Comprehensive vectorization complete for user {user_id}")
            self.logger.info(f"   Plans: {plans_vectorized} | Patterns: {patterns_vectorized} | Stakeholders: {stakeholders_vectorized} | Activities: {activities_vectorized}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive vectorization failed for user {user_id}: {e}")
            return False
    
    async def intelligent_rag_search(self, query: str, user_id: int, search_type: str = "all") -> Dict[str, Any]:
        """
        Perform intelligent RAG search across all vectorized user data
        """
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return {"results": [], "insights": {}}
            
            try:
                conn = await asyncio.wait_for(self.pool_manager.postgres_pool.acquire(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("⚠️ Database connection timeout during RAG search")
                return {"results": [], "insights": {}}
            
            try:
                # Search across different vector types
                search_results = {}
                
                if search_type in ["all", "plans"]:
                    search_results["plans"] = await self._search_plan_vectors(conn, query_embedding, user_id)
                
                if search_type in ["all", "patterns"]:
                    search_results["patterns"] = await self._search_pattern_vectors(conn, query_embedding, user_id)
                    
                if search_type in ["all", "stakeholders"]:
                    search_results["stakeholders"] = await self._search_stakeholder_vectors(conn, query_embedding, user_id)
                    
                if search_type in ["all", "activities"]:
                    search_results["activities"] = await self._search_activity_vectors(conn, query_embedding, user_id)
                
                # Analyze and synthesize results
                insights = await self._synthesize_rag_insights(search_results, query)
                
                return {
                    "results": search_results,
                    "insights": insights,
                    "query": query,
                    "user_id": user_id,
                    "search_timestamp": datetime.now().isoformat()
                }
                
            finally:
                await self.pool_manager.postgres_pool.release(conn)
                
        except Exception as e:
            self.logger.error(f"❌ RAG search failed: {e}")
            return {"results": [], "insights": {}}
    
    async def generate_form_field_with_rag(self, field_name: str, context: Dict, user_id: int) -> Any:
        """
        Generate specific form field using comprehensive RAG
        """
        try:
            # Create field-specific query
            field_queries = {
                "short_term_goals": f"short term goals for {context.get('industry', '')} {context.get('account_tier', '')} account with {context.get('annual_revenue', '')} revenue",
                "long_term_goals": f"long term strategic objectives for {context.get('industry', '')} company market leadership",
                "account_tier": f"account classification tier for {context.get('annual_revenue', '')} revenue {context.get('industry', '')} company",
                "communication_cadence": f"communication frequency patterns for {context.get('account_tier', '')} {context.get('industry', '')} accounts",
                "product_penetration_goals": f"product expansion strategies for {context.get('industry', '')} {context.get('account_tier', '')} clients",
                "customer_success_metrics": f"success measurement metrics for {context.get('industry', '')} strategic accounts",
                "planned_activities": f"strategic activities and engagements for {context.get('account_tier', '')} account planning"
            }
            
            query = field_queries.get(field_name, f"{field_name} for strategic account planning")
            
            # Perform RAG search
            rag_results = await self.intelligent_rag_search(query, user_id, "all")
            
            # Generate field value using RAG insights
            field_value = await self._generate_field_from_rag(field_name, rag_results, context)
            
            self.logger.info(f"✅ Generated {field_name} using RAG: {str(field_value)[:100]}...")
            return field_value
            
        except Exception as e:
            self.logger.error(f"❌ RAG field generation failed for {field_name}: {e}")
            return None
    
    async def _vectorize_strategic_plans(self, user_id: int) -> int:
        """Vectorize all strategic plans for the user"""
        try:
            # Add timeout to prevent hanging
            try:
                conn = await asyncio.wait_for(self.pool_manager.postgres_pool.acquire(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("⚠️ Database connection timeout during plan vectorization")
                return 0
            
            try:
                # Get all plans for the user
                plans = await conn.fetch("""
                    SELECT plan_id, plan_name, account_id, industry, account_tier,
                           short_term_goals, long_term_goals, key_opportunities,
                           known_risks, risk_mitigation_strategies,
                           annual_revenue, revenue_growth_target,
                           created_at, updated_at
                    FROM strategic_account_plans
                    WHERE created_by_user_id = $1
                    ORDER BY created_at DESC
                """, user_id)
                
                vectorized_count = 0
                
                for plan in plans:
                    # Create comprehensive plan text
                    plan_text = self._create_plan_text(plan)
                    
                    # Generate embedding
                    embedding = await self._generate_embedding(plan_text)
                    if embedding:
                        # Store in plan_embeddings
                        await conn.execute("""
                            INSERT INTO plan_embeddings (plan_id, embedding, content_type, content_text, created_at)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (plan_id, content_type) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                content_text = EXCLUDED.content_text,
                                created_at = EXCLUDED.created_at
                        """, plan['plan_id'], embedding, 'strategic_plan', plan_text, datetime.now())
                        
                        vectorized_count += 1
                
                return vectorized_count
                
            finally:
                await self.pool_manager.postgres_pool.release(conn)
                
        except Exception as e:
            self.logger.error(f"❌ Plan vectorization failed: {e}")
            return 0
    
    async def _vectorize_user_patterns(self, user_id: int) -> int:
        """Vectorize user patterns and preferences"""
        try:
            try:
                conn = await asyncio.wait_for(self.pool_manager.postgres_pool.acquire(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("⚠️ Database connection timeout during pattern vectorization")
                return 0
            try:
                # Analyze user patterns from all plans
                pattern_analysis = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_plans,
                        AVG(CAST(annual_revenue AS NUMERIC)) as avg_revenue,
                        AVG(CAST(revenue_growth_target AS NUMERIC)) as avg_growth_target,
                        MODE() WITHIN GROUP (ORDER BY industry) as preferred_industry,
                        MODE() WITHIN GROUP (ORDER BY account_tier) as preferred_tier,
                        MODE() WITHIN GROUP (ORDER BY communication_cadence) as preferred_cadence,
                        STRING_AGG(DISTINCT region_territory, ', ') as regions_covered,
                        MIN(created_at) as first_plan,
                        MAX(created_at) as last_plan
                    FROM strategic_account_plans
                    WHERE created_by_user_id = $1
                """, user_id)
                
                if pattern_analysis and pattern_analysis['total_plans'] > 0:
                    # Create pattern text
                    pattern_text = f"""
                    User {user_id} Planning Patterns:
                    Total strategic plans created: {pattern_analysis['total_plans']}
                    Average account revenue: ${pattern_analysis['avg_revenue']:,.0f} 
                    Average growth target: {pattern_analysis['avg_growth_target']:.1f}%
                    Preferred industry focus: {pattern_analysis['preferred_industry']}
                    Preferred account tier: {pattern_analysis['preferred_tier']}
                    Preferred communication cadence: {pattern_analysis['preferred_cadence']}
                    Geographic coverage: {pattern_analysis['regions_covered']}
                    Planning experience: {pattern_analysis['first_plan']} to {pattern_analysis['last_plan']}
                    """
                    
                    # Generate embedding for patterns
                    embedding = await self._generate_embedding(pattern_text)
                    if embedding:
                        # Store pattern embedding
                        await conn.execute("""
                            INSERT INTO plan_embeddings (plan_id, embedding, content_type, content_text, created_at)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (plan_id, content_type) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                content_text = EXCLUDED.content_text,
                                created_at = EXCLUDED.created_at
                        """, f"user_patterns_{user_id}", embedding, 'user_patterns', pattern_text, datetime.now())
                        
                        return 1
                
                return 0
                
            finally:
                await self.pool_manager.postgres_pool.release(conn)
                
        except Exception as e:
            self.logger.error(f"❌ Pattern vectorization failed: {e}")
            return 0
    
    async def _vectorize_stakeholder_data(self, user_id: int) -> int:
        """Vectorize stakeholder relationship patterns"""
        try:
            try:
                conn = await asyncio.wait_for(self.pool_manager.postgres_pool.acquire(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("⚠️ Database connection timeout during stakeholder vectorization")
                return 0
            try:
                # Get stakeholder patterns
                stakeholders = await conn.fetch("""
                    SELECT ps.name, ps.role, ps.influence_level, ps.relationship_status,
                           sap.industry, sap.account_tier, sap.account_id
                    FROM plan_stakeholders ps
                    JOIN strategic_account_plans sap ON ps.plan_id = sap.plan_id
                    WHERE sap.created_by_user_id = $1
                """, user_id)
                
                if stakeholders:
                    # Create stakeholder relationship text
                    stakeholder_text = f"""
                    User {user_id} Stakeholder Relationship Patterns:
                    Total stakeholder relationships: {len(stakeholders)}
                    """
                    
                    # Analyze patterns
                    role_patterns = {}
                    influence_patterns = {}
                    industry_stakeholders = {}
                    
                    for sh in stakeholders:
                        role_patterns[sh['role']] = role_patterns.get(sh['role'], 0) + 1
                        influence_patterns[sh['influence_level']] = influence_patterns.get(sh['influence_level'], 0) + 1
                        industry_stakeholders[sh['industry']] = industry_stakeholders.get(sh['industry'], []) + [sh['role']]
                    
                    stakeholder_text += f"""
                    Common stakeholder roles: {', '.join([f"{role}({count})" for role, count in role_patterns.items()])}
                    Influence level distribution: {', '.join([f"{level}({count})" for level, count in influence_patterns.items()])}
                    Industry-specific stakeholder patterns: {json.dumps(industry_stakeholders, indent=2)}
                    """
                    
                    # Generate embedding
                    embedding = await self._generate_embedding(stakeholder_text)
                    if embedding:
                        await conn.execute("""
                            INSERT INTO plan_embeddings (plan_id, embedding, content_type, content_text, created_at)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (plan_id, content_type) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                content_text = EXCLUDED.content_text,
                                created_at = EXCLUDED.created_at
                        """, f"stakeholder_patterns_{user_id}", embedding, 'stakeholder_patterns', stakeholder_text, datetime.now())
                        
                        return 1
                
                return 0
                
            finally:
                await self.pool_manager.postgres_pool.release(conn)
                
        except Exception as e:
            self.logger.error(f"❌ Stakeholder vectorization failed: {e}")
            return 0
    
    async def _vectorize_activity_patterns(self, user_id: int) -> int:
        """Vectorize activity and engagement patterns"""
        try:
            try:
                conn = await asyncio.wait_for(self.pool_manager.postgres_pool.acquire(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("⚠️ Database connection timeout during activity vectorization")
                return 0
            try:
                # Get activity patterns
                activities = await conn.fetch("""
                    SELECT pa.activity_title, pa.activity_description, pa.planned_date,
                           sap.industry, sap.account_tier, sap.account_id
                    FROM plan_activities pa
                    JOIN strategic_account_plans sap ON pa.plan_id = sap.plan_id
                    WHERE sap.created_by_user_id = $1
                    ORDER BY pa.planned_date DESC
                """, user_id)
                
                if activities:
                    # Create activity pattern text
                    activity_text = f"""
                    User {user_id} Activity and Engagement Patterns:
                    Total planned activities: {len(activities)}
                    """
                    
                    # Analyze activity patterns
                    activity_types = {}
                    industry_activities = {}
                    
                    for activity in activities:
                        activity_types[activity['activity_title']] = activity_types.get(activity['activity_title'], 0) + 1
                        industry_activities[activity['industry']] = industry_activities.get(activity['industry'], []) + [activity['activity_title']]
                    
                    activity_text += f"""
                    Common activity types: {', '.join([f"{title}({count})" for title, count in activity_types.items()])}
                    Industry-specific activities: {json.dumps(industry_activities, indent=2)}
                    Recent activities: {[a['activity_title'] for a in activities[:5]]}
                    """
                    
                    # Generate embedding
                    embedding = await self._generate_embedding(activity_text)
                    if embedding:
                        await conn.execute("""
                            INSERT INTO plan_embeddings (plan_id, embedding, content_type, content_text, created_at)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (plan_id, content_type) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                content_text = EXCLUDED.content_text,
                                created_at = EXCLUDED.created_at
                        """, f"activity_patterns_{user_id}", embedding, 'activity_patterns', activity_text, datetime.now())
                        
                        return 1
                
                return 0
                
            finally:
                await self.pool_manager.postgres_pool.release(conn)
                
        except Exception as e:
            self.logger.error(f"❌ Activity vectorization failed: {e}")
            return 0
    
    async def _search_plan_vectors(self, conn, query_embedding: List[float], user_id: int) -> List[Dict]:
        """Search plan vectors"""
        try:
            results = await conn.fetch("""
                SELECT pe.content_text, pe.content_type,
                       (1 - (pe.embedding <-> $1::vector)) AS similarity_score
                FROM plan_embeddings pe
                WHERE pe.content_type = 'strategic_plan'
                  AND pe.plan_id IN (
                      SELECT CAST(plan_id AS TEXT) FROM strategic_account_plans 
                      WHERE created_by_user_id = $2
                  )
                  AND (1 - (pe.embedding <-> $1::vector)) >= 0.7
                ORDER BY pe.embedding <-> $1::vector
                LIMIT 5
            """, query_embedding, user_id)
            
            return [{"content": r['content_text'], "type": r['content_type'], "score": r['similarity_score']} for r in results]
            
        except Exception as e:
            self.logger.warning(f"⚠️ Plan vector search failed: {e}")
            return []
    
    async def _search_pattern_vectors(self, conn, query_embedding: List[float], user_id: int) -> List[Dict]:
        """Search pattern vectors"""
        try:
            results = await conn.fetch("""
                SELECT pe.content_text, pe.content_type,
                       (1 - (pe.embedding <-> $1::vector)) AS similarity_score
                FROM plan_embeddings pe
                WHERE pe.content_type = 'user_patterns'
                  AND pe.plan_id = $2
                  AND (1 - (pe.embedding <-> $1::vector)) >= 0.6
                ORDER BY pe.embedding <-> $1::vector
                LIMIT 3
            """, query_embedding, f"user_patterns_{user_id}")
            
            return [{"content": r['content_text'], "type": r['content_type'], "score": r['similarity_score']} for r in results]
            
        except Exception as e:
            self.logger.warning(f"⚠️ Pattern vector search failed: {e}")
            return []
    
    async def _search_stakeholder_vectors(self, conn, query_embedding: List[float], user_id: int) -> List[Dict]:
        """Search stakeholder vectors"""
        try:
            results = await conn.fetch("""
                SELECT pe.content_text, pe.content_type,
                       (1 - (pe.embedding <-> $1::vector)) AS similarity_score
                FROM plan_embeddings pe
                WHERE pe.content_type = 'stakeholder_patterns'
                  AND pe.plan_id = $2
                  AND (1 - (pe.embedding <-> $1::vector)) >= 0.6
                ORDER BY pe.embedding <-> $1::vector
                LIMIT 3
            """, query_embedding, f"stakeholder_patterns_{user_id}")
            
            return [{"content": r['content_text'], "type": r['content_type'], "score": r['similarity_score']} for r in results]
            
        except Exception as e:
            self.logger.warning(f"⚠️ Stakeholder vector search failed: {e}")
            return []
    
    async def _search_activity_vectors(self, conn, query_embedding: List[float], user_id: int) -> List[Dict]:
        """Search activity vectors"""
        try:
            results = await conn.fetch("""
                SELECT pe.content_text, pe.content_type,
                       (1 - (pe.embedding <-> $1::vector)) AS similarity_score
                FROM plan_embeddings pe
                WHERE pe.content_type = 'activity_patterns'
                  AND pe.plan_id = $2
                  AND (1 - (pe.embedding <-> $1::vector)) >= 0.6
                ORDER BY pe.embedding <-> $1::vector
                LIMIT 3
            """, query_embedding, f"activity_patterns_{user_id}")
            
            return [{"content": r['content_text'], "type": r['content_type'], "score": r['similarity_score']} for r in results]
            
        except Exception as e:
            self.logger.warning(f"⚠️ Activity vector search failed: {e}")
            return []
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI"""
        try:
            # Cache check
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
            
            if not self.pool_manager or not self.pool_manager.openai_client:
                return None
                
            response = await self.pool_manager.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000]  # Truncate for embedding model
            )
            
            embedding = response.data[0].embedding
            self.embedding_cache[text_hash] = embedding  # Cache for reuse
            return embedding
            
        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed: {e}")
            return None
    
    def _create_plan_text(self, plan: Dict) -> str:
        """Create comprehensive text representation of a plan"""
        return f"""
        Strategic Account Plan: {plan['plan_name']}
        Account: {plan['account_id']} 
        Industry: {plan['industry']}
        Account Tier: {plan['account_tier']}
        Annual Revenue: ${plan['annual_revenue']}
        Revenue Growth Target: {plan['revenue_growth_target']}%
        
        Short-term Goals: {plan['short_term_goals']}
        Long-term Goals: {plan['long_term_goals']}
        
        Key Opportunities: {plan['key_opportunities']}
        Known Risks: {plan['known_risks']}
        Risk Mitigation: {plan['risk_mitigation_strategies']}
        
        Created: {plan['created_at']}
        Updated: {plan['updated_at']}
        """
    
    async def _synthesize_rag_insights(self, search_results: Dict, query: str) -> Dict[str, Any]:
        """Synthesize insights from RAG search results"""
        insights = {
            "total_matches": 0,
            "strongest_patterns": [],
            "recommendations": [],
            "confidence_score": 0.0
        }
        
        all_results = []
        for result_type, results in search_results.items():
            all_results.extend(results)
            insights["total_matches"] += len(results)
        
        if all_results:
            # Find strongest patterns
            high_confidence = [r for r in all_results if r.get("score", 0) >= 0.8]
            insights["strongest_patterns"] = high_confidence[:3]
            
            # Calculate overall confidence
            if all_results:
                avg_score = sum(r.get("score", 0) for r in all_results) / len(all_results)
                insights["confidence_score"] = round(avg_score, 2)
            
            # Generate recommendations
            if high_confidence:
                insights["recommendations"].append("High-confidence historical patterns found - use for form filling")
            if len(all_results) >= 3:
                insights["recommendations"].append("Multiple relevant examples available - synthesize common themes")
        
        return insights
    
    async def _generate_field_from_rag(self, field_name: str, rag_results: Dict, context: Dict) -> Any:
        """Generate specific field value from RAG results"""
        try:
            insights = rag_results.get("insights", {})
            all_content = []
            
            # Collect all relevant content
            results_data = rag_results.get("results", {})
            if isinstance(results_data, dict):
                for result_type, results in results_data.items():
                    if isinstance(results, list):
                        for result in results:
                            if isinstance(result, dict) and result.get("score", 0) >= 0.7:  # High confidence only
                                all_content.append(result.get("content", ""))
            
            if not all_content:
                return None
            
            # Generate field-specific value using LLM with RAG context
            if self.pool_manager and self.pool_manager.openai_client:
                prompt = f"""
                Based on the following historical data and patterns, generate an intelligent value for the '{field_name}' field:
                
                Context: {json.dumps(context, indent=2)}
                
                Historical Data:
                {chr(10).join(all_content[:3])}  # Top 3 most relevant
                
                Generate a specific, actionable value for '{field_name}' that:
                1. Leverages the historical patterns shown above
                2. Adapts to the current context
                3. Follows best practices for strategic account planning
                4. Is specific and measurable where appropriate
                
                Respond with ONLY the field value, no explanation.
                """
                
                response = await self.pool_manager.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300
                )
                
                generated_value = response.choices[0].message.content.strip()
                
                # Post-process based on field type
                if field_name == "planned_activities":
                    # Try to parse as activities
                    try:
                        import json
                        return json.loads(generated_value)
                    except:
                        # Fallback to intelligent default activities
                        return [
                            {"title": "Strategic Planning Review", "date": "2025-09-15", "description": generated_value},
                            {"title": "Quarterly Business Review", "date": "2025-10-01", "description": "Comprehensive quarterly assessment"}
                        ]
                
                return generated_value
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ RAG field generation failed for {field_name}: {e}")
            return None

# Global instance
comprehensive_rag_service = None

def get_comprehensive_rag_service(pool_manager=None):
    """Get or create comprehensive RAG service instance"""
    global comprehensive_rag_service
    if comprehensive_rag_service is None:
        comprehensive_rag_service = ComprehensiveRAGService(pool_manager)
    return comprehensive_rag_service
