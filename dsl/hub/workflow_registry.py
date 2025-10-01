"""
Workflow Registry Hub
Central registry for all RBA workflows with auto-discovery and management
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from ..parser import DSLWorkflow
from ..storage import WorkflowStorage

@dataclass
class WorkflowSpec:
    """Specification for a workflow in the registry"""
    workflow_id: str
    name: str
    module: str
    automation_type: str
    version: str
    business_value: str
    customer_impact: str
    tags: List[str]
    success_rate: float = 0.0
    avg_execution_time_ms: int = 0
    usage_count: int = 0
    last_executed: Optional[datetime] = None

class WorkflowRegistry:
    """
    Central hub for workflow discovery, management, and analytics
    """
    
    def __init__(self, storage: WorkflowStorage, pool_manager=None):
        self.storage = storage
        self.pool_manager = pool_manager
        self._registry_cache = {}
        self._module_workflows = {
            'Forecast': [],
            'Pipeline': [],
            'Planning': [],
            'Revenue': []
        }
    
    async def register_workflow(self, workflow: DSLWorkflow, business_value: str, customer_impact: str) -> bool:
        """Register a new workflow in the hub"""
        try:
            # Create workflow spec
            spec = WorkflowSpec(
                workflow_id=workflow.workflow_id,
                name=workflow.name,
                module=workflow.module,
                automation_type=workflow.automation_type,
                version=workflow.version,
                business_value=business_value,
                customer_impact=customer_impact,
                tags=workflow.metadata.get('tags', [])
            )
            
            # Store in cache
            self._registry_cache[workflow.workflow_id] = spec
            
            # Add to module registry
            if workflow.module in self._module_workflows:
                self._module_workflows[workflow.module].append(spec)
            
            # Store workflow metadata in database
            await self._store_workflow_metadata(spec)
            
            return True
            
        except Exception as e:
            print(f"Error registering workflow {workflow.workflow_id}: {e}")
            return False
    
    async def discover_workflows(self, module: Optional[str] = None, tags: List[str] = None) -> List[WorkflowSpec]:
        """Discover workflows by module or tags"""
        try:
            workflows = []
            
            if module:
                workflows = self._module_workflows.get(module, [])
            else:
                workflows = list(self._registry_cache.values())
            
            # Filter by tags if provided
            if tags:
                workflows = [w for w in workflows if any(tag in w.tags for tag in tags)]
            
            # Sort by success rate and usage
            workflows.sort(key=lambda w: (w.success_rate, w.usage_count), reverse=True)
            
            return workflows
            
        except Exception as e:
            print(f"Error discovering workflows: {e}")
            return []
    
    async def get_workflow_analytics(self, workflow_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a workflow"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return {"error": "Database not available"}
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Get execution statistics
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_executions,
                        AVG(execution_time_ms) as avg_execution_time,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END)::FLOAT / COUNT(*) as success_rate,
                        MAX(started_at) as last_executed
                    FROM dsl_execution_traces 
                    WHERE workflow_id = $1
                """, workflow_id)
                
                # Get recent execution trends
                trends = await conn.fetch("""
                    SELECT 
                        DATE(started_at) as execution_date,
                        COUNT(*) as executions,
                        AVG(execution_time_ms) as avg_time
                    FROM dsl_execution_traces 
                    WHERE workflow_id = $1 
                    AND started_at >= NOW() - INTERVAL '30 days'
                    GROUP BY DATE(started_at)
                    ORDER BY execution_date DESC
                """, workflow_id)
                
                return {
                    "workflow_id": workflow_id,
                    "total_executions": stats['total_executions'] or 0,
                    "avg_execution_time_ms": int(stats['avg_execution_time'] or 0),
                    "success_rate": float(stats['success_rate'] or 0),
                    "last_executed": stats['last_executed'].isoformat() if stats['last_executed'] else None,
                    "trends": [
                        {
                            "date": row['execution_date'].isoformat(),
                            "executions": row['executions'],
                            "avg_time": int(row['avg_time'])
                        }
                        for row in trends
                    ]
                }
                
        except Exception as e:
            return {"error": f"Analytics error: {e}"}
    
    async def get_module_overview(self, module: str) -> Dict[str, Any]:
        """Get overview of all workflows in a module"""
        try:
            module_workflows = self._module_workflows.get(module, [])
            
            if not module_workflows:
                return {"module": module, "workflows": [], "total": 0}
            
            # Get analytics for each workflow
            workflow_analytics = []
            for workflow_spec in module_workflows:
                analytics = await self.get_workflow_analytics(workflow_spec.workflow_id)
                workflow_analytics.append({
                    "spec": workflow_spec.__dict__,
                    "analytics": analytics
                })
            
            # Calculate module-level metrics
            total_executions = sum(w["analytics"].get("total_executions", 0) for w in workflow_analytics)
            avg_success_rate = sum(w["analytics"].get("success_rate", 0) for w in workflow_analytics) / len(workflow_analytics) if workflow_analytics else 0
            
            return {
                "module": module,
                "total_workflows": len(module_workflows),
                "total_executions": total_executions,
                "avg_success_rate": avg_success_rate,
                "workflows": workflow_analytics
            }
            
        except Exception as e:
            return {"error": f"Module overview error: {e}"}
    
    async def suggest_workflows(self, user_context: Dict[str, Any]) -> List[WorkflowSpec]:
        """Suggest workflows based on user context and patterns"""
        try:
            user_role = user_context.get('role', '')
            user_segment = user_context.get('segment', '')
            
            # Simple rule-based suggestions (can be enhanced with ML)
            suggestions = []
            
            # Role-based suggestions
            if 'sales' in user_role.lower():
                suggestions.extend(self._module_workflows.get('Pipeline', []))
                suggestions.extend(self._module_workflows.get('Forecast', []))
            
            if 'manager' in user_role.lower():
                suggestions.extend(self._module_workflows.get('Planning', []))
            
            # Filter by success rate
            suggestions = [w for w in suggestions if w.success_rate > 0.7]
            
            # Sort by relevance (success rate + usage count)
            suggestions.sort(key=lambda w: w.success_rate * w.usage_count, reverse=True)
            
            return suggestions[:5]  # Top 5 suggestions
            
        except Exception as e:
            print(f"Error suggesting workflows: {e}")
            return []
    
    async def _store_workflow_metadata(self, spec: WorkflowSpec):
        """Store workflow metadata for analytics"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                return
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO workflow_registry_metadata (
                        workflow_id, business_value, customer_impact, 
                        registered_at, module, automation_type
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (workflow_id) 
                    DO UPDATE SET 
                        business_value = EXCLUDED.business_value,
                        customer_impact = EXCLUDED.customer_impact,
                        updated_at = NOW()
                """, 
                spec.workflow_id, spec.business_value, spec.customer_impact,
                datetime.utcnow(), spec.module, spec.automation_type)
                
        except Exception as e:
            print(f"Error storing workflow metadata: {e}")
    
    async def update_execution_stats(self, workflow_id: str, execution_time_ms: int, success: bool):
        """Update workflow execution statistics"""
        try:
            if workflow_id in self._registry_cache:
                spec = self._registry_cache[workflow_id]
                spec.usage_count += 1
                spec.last_executed = datetime.utcnow()
                
                # Update running averages
                if spec.avg_execution_time_ms == 0:
                    spec.avg_execution_time_ms = execution_time_ms
                else:
                    spec.avg_execution_time_ms = int(
                        (spec.avg_execution_time_ms + execution_time_ms) / 2
                    )
                
                # Update success rate
                if success:
                    spec.success_rate = (spec.success_rate * (spec.usage_count - 1) + 1.0) / spec.usage_count
                else:
                    spec.success_rate = (spec.success_rate * (spec.usage_count - 1)) / spec.usage_count
                
        except Exception as e:
            print(f"Error updating execution stats: {e}")
    
    async def load_registry_from_database(self, tenant_id: str):
        """Load existing workflows from database into registry"""
        try:
            workflows = await self.storage.list_workflows(tenant_id)
            
            for workflow_data in workflows:
                # Create basic spec from database data
                tags = workflow_data.get('tags', [])
                if isinstance(tags, str):
                    try:
                        import json
                        tags = json.loads(tags) if tags else []
                    except:
                        tags = []
                
                spec = WorkflowSpec(
                    workflow_id=workflow_data['workflow_id'],
                    name=workflow_data['name'],
                    module=workflow_data['module'],
                    automation_type=workflow_data['automation_type'],
                    version=workflow_data['version'],
                    business_value="Loaded from existing workflow",
                    customer_impact="Improves operational efficiency",
                    tags=tags,
                    success_rate=float(workflow_data.get('success_rate', 0.0)) / 100.0,
                    avg_execution_time_ms=int(workflow_data.get('avg_execution_time_ms', 0)),
                    usage_count=int(workflow_data.get('execution_count', 0))
                )
                
                self._registry_cache[spec.workflow_id] = spec
                
                if spec.module in self._module_workflows:
                    self._module_workflows[spec.module].append(spec)
            
            print(f"Loaded {len(workflows)} workflows into registry")
            
        except Exception as e:
            print(f"Error loading registry from database: {e}")
