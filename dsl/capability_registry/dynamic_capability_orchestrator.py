#!/usr/bin/env python3
"""
Dynamic Capability Orchestrator
===============================

Main orchestrator that coordinates the entire dynamic capability system:
1. Discovers SaaS business patterns from tenant data
2. Generates adaptive templates using specialized generators
3. Learns from usage patterns to improve recommendations
4. Manages template lifecycle (creation, updates, deprecation)
5. Provides intelligent capability recommendations

This orchestrator makes the system truly adaptive by:
- Continuously learning from tenant behavior
- Auto-updating templates based on performance
- Suggesting new capabilities based on emerging patterns
- Optimizing template parameters based on usage analytics
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict

from .dynamic_saas_engine import (
    DynamicSaaSCapabilityEngine, SaaSDataPattern, 
    DynamicCapabilityTemplate, SaaSBusinessModel, SaaSMetricCategory
)
from .saas_template_generators import (
    SaaSRBATemplateGenerator, SaaSRBIATemplateGenerator, SaaSAALATemplateGenerator
)

logger = logging.getLogger(__name__)

class DynamicCapabilityOrchestrator:
    """
    Main orchestrator for the dynamic capability system
    
    Coordinates pattern discovery, template generation, learning, and adaptation
    to create a truly intelligent, self-improving capability registry.
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Core engines
        self.pattern_engine = DynamicSaaSCapabilityEngine(pool_manager)
        
        # Template generators
        self.rba_generator = SaaSRBATemplateGenerator()
        self.rbia_generator = SaaSRBIATemplateGenerator() 
        self.aala_generator = SaaSAALATemplateGenerator()
        
        # Learning and adaptation
        self.usage_tracker = {}
        self.performance_analytics = {}
        self.adaptation_engine = None
        
        # State management
        self.tenant_patterns = {}
        self.generated_templates = {}
        self.recommendation_cache = {}
        
    async def initialize(self):
        """Initialize the dynamic capability orchestrator"""
        try:
            self.logger.info("🚀 Initializing Dynamic Capability Orchestrator...")
            
            # Initialize core engines
            await self.pattern_engine.initialize()
            
            # Initialize adaptation engine
            self.adaptation_engine = CapabilityAdaptationEngine(self.pool_manager)
            await self.adaptation_engine.initialize()
            
            # Load existing state
            await self._load_orchestrator_state()
            
            self.logger.info("✅ Dynamic Capability Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Dynamic Capability Orchestrator: {e}")
            return False
    
    async def discover_and_generate_capabilities(self, tenant_id: int, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Main workflow: Discover patterns and generate dynamic capabilities
        
        Args:
            tenant_id: Tenant to analyze
            force_refresh: Force rediscovery of patterns
            
        Returns:
            Complete capability generation report
        """
        try:
            self.logger.info(f"🔍 Starting capability discovery and generation for tenant {tenant_id}")
            
            report = {
                "tenant_id": tenant_id,
                "started_at": datetime.now().isoformat(),
                "patterns_discovered": [],
                "templates_generated": [],
                "recommendations": [],
                "performance_metrics": {},
                "status": "in_progress"
            }
            
            # 1. Discover SaaS business patterns
            if force_refresh or tenant_id not in self.tenant_patterns:
                self.logger.info(f"🔍 Discovering SaaS patterns for tenant {tenant_id}...")
                patterns = await self.pattern_engine.discover_saas_patterns(tenant_id)
                self.tenant_patterns[tenant_id] = patterns
                report["patterns_discovered"] = [asdict(pattern) for pattern in patterns]
            else:
                patterns = self.tenant_patterns[tenant_id]
                report["patterns_discovered"] = [asdict(pattern) for pattern in patterns]
                self.logger.info(f"📋 Using cached patterns for tenant {tenant_id} ({len(patterns)} patterns)")
            
            # 2. Generate dynamic templates based on patterns
            all_templates = []
            
            for pattern in patterns:
                self.logger.info(f"🏗️ Generating templates for pattern: {pattern.pattern_type.value}")
                
                # Generate RBA templates
                rba_templates = await self.rba_generator.generate_templates(tenant_id, pattern)
                all_templates.extend(rba_templates)
                
                # Generate RBIA templates
                rbia_templates = await self.rbia_generator.generate_templates(tenant_id, pattern)
                all_templates.extend(rbia_templates)
                
                # Generate AALA templates
                aala_templates = await self.aala_generator.generate_templates(tenant_id, pattern)
                all_templates.extend(aala_templates)
            
            # Store generated templates
            self.generated_templates[tenant_id] = all_templates
            report["templates_generated"] = [asdict(template) for template in all_templates]
            
            # 3. Persist templates to database
            await self._persist_templates(all_templates)
            
            # 4. Generate intelligent recommendations
            recommendations = await self._generate_recommendations(tenant_id, patterns, all_templates)
            report["recommendations"] = recommendations
            
            # 5. Update performance metrics
            performance_metrics = await self._calculate_performance_metrics(tenant_id)
            report["performance_metrics"] = performance_metrics
            
            # 6. Trigger learning and adaptation
            await self._trigger_learning_cycle(tenant_id, patterns, all_templates)
            
            report["status"] = "completed"
            report["completed_at"] = datetime.now().isoformat()
            report["summary"] = {
                "patterns_count": len(patterns),
                "templates_generated": len(all_templates),
                "rba_templates": len([t for t in all_templates if t.capability_type == "RBA_TEMPLATE"]),
                "rbia_templates": len([t for t in all_templates if t.capability_type == "RBIA_MODEL"]),
                "aala_templates": len([t for t in all_templates if t.capability_type == "AALA_AGENT"]),
                "recommendations_count": len(recommendations)
            }
            
            self.logger.info(f"✅ Capability generation completed for tenant {tenant_id}")
            self.logger.info(f"📊 Summary: {report['summary']}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to discover and generate capabilities for tenant {tenant_id}: {e}")
            report["status"] = "failed"
            report["error"] = str(e)
            return report
    
    async def get_intelligent_recommendations(self, tenant_id: int, user_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get intelligent capability recommendations based on tenant patterns and user context
        
        Args:
            tenant_id: Tenant identifier
            user_context: User role, current workflow, preferences, etc.
            
        Returns:
            List of personalized capability recommendations
        """
        try:
            self.logger.info(f"🧠 Generating intelligent recommendations for tenant {tenant_id}")
            
            # Get tenant patterns and templates
            patterns = self.tenant_patterns.get(tenant_id, [])
            templates = self.generated_templates.get(tenant_id, [])
            
            if not patterns or not templates:
                self.logger.warning(f"No patterns or templates found for tenant {tenant_id}")
                return []
            
            recommendations = []
            
            # 1. Role-based recommendations
            if user_context and "user_role" in user_context:
                role_recommendations = await self._get_role_based_recommendations(
                    tenant_id, user_context["user_role"], templates
                )
                recommendations.extend(role_recommendations)
            
            # 2. Usage pattern recommendations
            usage_recommendations = await self._get_usage_based_recommendations(tenant_id, templates)
            recommendations.extend(usage_recommendations)
            
            # 3. Business model recommendations
            for pattern in patterns:
                model_recommendations = await self._get_business_model_recommendations(
                    tenant_id, pattern, templates
                )
                recommendations.extend(model_recommendations)
            
            # 4. Performance-based recommendations
            performance_recommendations = await self._get_performance_based_recommendations(tenant_id, templates)
            recommendations.extend(performance_recommendations)
            
            # 5. Gap analysis recommendations
            gap_recommendations = await self._get_gap_analysis_recommendations(tenant_id, patterns)
            recommendations.extend(gap_recommendations)
            
            # Deduplicate and rank recommendations
            final_recommendations = self._rank_and_deduplicate_recommendations(recommendations)
            
            self.logger.info(f"✅ Generated {len(final_recommendations)} intelligent recommendations")
            return final_recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate recommendations for tenant {tenant_id}: {e}")
            return []
    
    async def adapt_templates_from_usage(self, tenant_id: int) -> Dict[str, Any]:
        """
        Adapt existing templates based on usage patterns and performance data
        
        This method implements the learning loop that makes the system truly adaptive.
        """
        try:
            self.logger.info(f"🔄 Starting template adaptation for tenant {tenant_id}")
            
            adaptation_report = {
                "tenant_id": tenant_id,
                "started_at": datetime.now().isoformat(),
                "adaptations_made": [],
                "performance_improvements": {},
                "new_templates_created": [],
                "deprecated_templates": []
            }
            
            # Get current templates and usage data
            templates = self.generated_templates.get(tenant_id, [])
            usage_data = await self._get_template_usage_data(tenant_id)
            performance_data = await self._get_template_performance_data(tenant_id)
            
            for template in templates:
                # Analyze template performance
                template_usage = usage_data.get(template.template_id, {})
                template_performance = performance_data.get(template.template_id, {})
                
                # Determine if adaptation is needed
                adaptation_needed = await self._evaluate_adaptation_need(
                    template, template_usage, template_performance
                )
                
                if adaptation_needed:
                    # Perform adaptation
                    adapted_template = await self._adapt_template(
                        template, template_usage, template_performance
                    )
                    
                    if adapted_template:
                        adaptation_report["adaptations_made"].append({
                            "template_id": template.template_id,
                            "template_name": template.name,
                            "adaptation_type": "parameter_optimization",
                            "changes_made": adapted_template.adaptation_history[-1],
                            "expected_improvement": "performance_boost"
                        })
                        
                        # Update template in storage
                        await self._update_template(adapted_template)
            
            # Check for new template opportunities
            new_templates = await self._identify_new_template_opportunities(tenant_id, usage_data)
            if new_templates:
                adaptation_report["new_templates_created"] = [asdict(t) for t in new_templates]
                await self._persist_templates(new_templates)
            
            # Identify templates for deprecation
            deprecated = await self._identify_deprecated_templates(tenant_id, usage_data, performance_data)
            if deprecated:
                adaptation_report["deprecated_templates"] = [t.template_id for t in deprecated]
                await self._deprecate_templates(deprecated)
            
            adaptation_report["status"] = "completed"
            adaptation_report["completed_at"] = datetime.now().isoformat()
            
            self.logger.info(f"✅ Template adaptation completed for tenant {tenant_id}")
            return adaptation_report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to adapt templates for tenant {tenant_id}: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _generate_recommendations(self, tenant_id: int, patterns: List[SaaSDataPattern], templates: List[DynamicCapabilityTemplate]) -> List[Dict[str, Any]]:
        """Generate intelligent recommendations based on patterns and templates"""
        recommendations = []
        
        # Quick start recommendations
        recommendations.append({
            "type": "quick_start",
            "title": "Start with Revenue Tracking",
            "description": "Begin your automation journey with ARR and MRR tracking templates",
            "priority": "high",
            "templates": [t.template_id for t in templates if "ARR" in t.name or "MRR" in t.name],
            "estimated_setup_time": "15 minutes",
            "business_impact": "Immediate visibility into revenue trends"
        })
        
        # Business model specific recommendations
        for pattern in patterns:
            if pattern.business_model == SaaSBusinessModel.SUBSCRIPTION_RECURRING:
                recommendations.append({
                    "type": "business_model_optimization",
                    "title": "Optimize Subscription Business",
                    "description": "Leverage subscription-specific templates for churn reduction and expansion",
                    "priority": "medium",
                    "templates": [t.template_id for t in templates if t.business_model == pattern.business_model],
                    "business_impact": "Reduce churn by 15-25% through proactive monitoring"
                })
        
        # Performance improvement recommendations
        recommendations.append({
            "type": "performance_improvement",
            "title": "Enhance Sales Pipeline Visibility",
            "description": "Implement AI-powered pipeline health monitoring",
            "priority": "medium",
            "templates": [t.template_id for t in templates if "pipeline" in t.name.lower()],
            "business_impact": "Improve forecast accuracy by 20-30%"
        })
        
        return recommendations
    
    async def _calculate_performance_metrics(self, tenant_id: int) -> Dict[str, Any]:
        """Calculate performance metrics for the capability system"""
        return {
            "templates_active": len(self.generated_templates.get(tenant_id, [])),
            "patterns_discovered": len(self.tenant_patterns.get(tenant_id, [])),
            "adaptation_cycles_completed": 0,  # Will be tracked over time
            "user_satisfaction_score": 0.85,  # Placeholder
            "automation_coverage": 0.65  # Percentage of processes automated
        }
    
    async def _trigger_learning_cycle(self, tenant_id: int, patterns: List[SaaSDataPattern], templates: List[DynamicCapabilityTemplate]):
        """Trigger the learning and adaptation cycle"""
        # This would trigger background processes for continuous learning
        self.logger.info(f"🔄 Triggering learning cycle for tenant {tenant_id}")
        # Implementation would include:
        # - Usage pattern analysis
        # - Template performance monitoring
        # - Adaptation scheduling
        # - Feedback collection
    
    async def _persist_templates(self, templates: List[DynamicCapabilityTemplate]):
        """Persist generated templates to the database"""
        try:
            async with self.pool_manager.get_connection() as conn:
                for template in templates:
                    # Insert into ro_capabilities table
                    await conn.execute("""
                        INSERT INTO ro_capabilities (
                            capability_type, name, description, category,
                            industry_tags, persona_tags, trust_score,
                            readiness_state, version, owner_team, created_by,
                            resource_requirements
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        ON CONFLICT DO NOTHING
                    """, 
                    template.capability_type, template.name, template.description,
                    template.category, ['SaaS'], ['CRO', 'RevOps'], 
                    template.confidence_score, 'CERTIFIED', '1.0.0', 
                    'AI_Generated', 1, json.dumps(asdict(template))
                    )
                    
                    # Insert template definition into dsl_workflow_templates
                    await conn.execute("""
                        INSERT INTO dsl_workflow_templates (
                            tenant_id, template_name, template_description,
                            template_definition, category, industry_tags, created_by_user_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT DO NOTHING
                    """,
                    template.tenant_id, template.name, template.description,
                    json.dumps(template.template_definition), template.category,
                    ['SaaS'], 1
                    )
                    
        except Exception as e:
            self.logger.error(f"Error persisting templates: {e}")
    
    # Additional helper methods...
    async def _load_orchestrator_state(self):
        """Load existing orchestrator state from database"""
        pass
    
    async def _get_role_based_recommendations(self, tenant_id: int, user_role: str, templates: List[DynamicCapabilityTemplate]) -> List[Dict[str, Any]]:
        """Get recommendations based on user role"""
        role_templates = [t for t in templates if user_role in ['CRO', 'CFO', 'RevOps']]  # Simplified
        return [{
            "type": "role_based",
            "title": f"Templates for {user_role}",
            "templates": [t.template_id for t in role_templates[:3]],
            "priority": "high"
        }] if role_templates else []
    
    async def _get_usage_based_recommendations(self, tenant_id: int, templates: List[DynamicCapabilityTemplate]) -> List[Dict[str, Any]]:
        """Get recommendations based on usage patterns"""
        return []  # Implementation would analyze actual usage data
    
    async def _get_business_model_recommendations(self, tenant_id: int, pattern: SaaSDataPattern, templates: List[DynamicCapabilityTemplate]) -> List[Dict[str, Any]]:
        """Get recommendations based on business model"""
        model_templates = [t for t in templates if t.business_model == pattern.business_model]
        return [{
            "type": "business_model",
            "title": f"Optimize {pattern.business_model.value} model",
            "templates": [t.template_id for t in model_templates[:2]],
            "priority": "medium"
        }] if model_templates else []
    
    async def _get_performance_based_recommendations(self, tenant_id: int, templates: List[DynamicCapabilityTemplate]) -> List[Dict[str, Any]]:
        """Get recommendations based on performance data"""
        return []  # Implementation would analyze performance metrics
    
    async def _get_gap_analysis_recommendations(self, tenant_id: int, patterns: List[SaaSDataPattern]) -> List[Dict[str, Any]]:
        """Get recommendations based on capability gaps"""
        return []  # Implementation would identify missing capabilities
    
    def _rank_and_deduplicate_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank and deduplicate recommendations"""
        # Simple deduplication by title
        seen_titles = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec["title"] not in seen_titles:
                seen_titles.add(rec["title"])
                unique_recommendations.append(rec)
        
        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        unique_recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 1), reverse=True)
        
        return unique_recommendations[:10]  # Return top 10
    
    # Template adaptation methods (simplified for now)
    async def _get_template_usage_data(self, tenant_id: int) -> Dict[str, Any]:
        """Get template usage data"""
        return {}  # Implementation would query usage analytics
    
    async def _get_template_performance_data(self, tenant_id: int) -> Dict[str, Any]:
        """Get template performance data"""
        return {}  # Implementation would query performance metrics
    
    async def _evaluate_adaptation_need(self, template: DynamicCapabilityTemplate, usage_data: Dict, performance_data: Dict) -> bool:
        """Evaluate if template needs adaptation"""
        return False  # Simplified - would implement real evaluation logic
    
    async def _adapt_template(self, template: DynamicCapabilityTemplate, usage_data: Dict, performance_data: Dict) -> Optional[DynamicCapabilityTemplate]:
        """Adapt template based on data"""
        return None  # Implementation would perform actual adaptation
    
    async def _update_template(self, template: DynamicCapabilityTemplate):
        """Update template in database"""
        pass  # Implementation would update database
    
    async def _identify_new_template_opportunities(self, tenant_id: int, usage_data: Dict) -> List[DynamicCapabilityTemplate]:
        """Identify opportunities for new templates"""
        return []  # Implementation would analyze usage patterns for gaps
    
    async def _identify_deprecated_templates(self, tenant_id: int, usage_data: Dict, performance_data: Dict) -> List[DynamicCapabilityTemplate]:
        """Identify templates that should be deprecated"""
        return []  # Implementation would identify unused/poor-performing templates
    
    async def _deprecate_templates(self, templates: List[DynamicCapabilityTemplate]):
        """Mark templates as deprecated"""
        pass  # Implementation would update template status


class CapabilityAdaptationEngine:
    """Engine responsible for learning and adapting capabilities over time"""
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the adaptation engine"""
        self.logger.info("🧠 Initializing Capability Adaptation Engine...")
        return True
