"""
Knowledge Connector - Framework for capturing and utilizing execution intelligence
Prepares the foundation for Knowledge Graph integration and pattern learning
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class ExecutionPattern:
    """Represents a discovered execution pattern"""
    pattern_id: str
    workflow_id: str
    module: str
    pattern_type: str  # 'success', 'failure', 'performance', 'user_behavior'
    frequency: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class KnowledgeEntry:
    """Single knowledge entry from workflow execution"""
    entry_id: str
    workflow_id: str
    user_id: str
    tenant_id: str
    execution_time_ms: int
    success: bool
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    user_context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)

class KnowledgeConnector:
    """
    Framework for capturing and analyzing workflow execution knowledge
    Prepares for future Knowledge Graph and ML integration
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        
        # In-memory storage for patterns (will be replaced with Knowledge Graph)
        self.execution_entries = []
        self.discovered_patterns = []
        self.user_behavior_cache = defaultdict(list)
        self.performance_cache = defaultdict(list)
        
        # Configuration
        self.max_entries_in_memory = 10000
        self.pattern_discovery_interval = 3600  # 1 hour
        self.min_pattern_frequency = 5
        
        # Start background tasks
        self._pattern_discovery_task = None
    
    async def start(self):
        """Start the knowledge connector"""
        # Start pattern discovery background task
        self._pattern_discovery_task = asyncio.create_task(self._pattern_discovery_worker())
        print("🧠 Knowledge Connector started")
    
    async def stop(self):
        """Stop the knowledge connector"""
        if self._pattern_discovery_task:
            self._pattern_discovery_task.cancel()
        print("🧠 Knowledge Connector stopped")
    
    # === Knowledge Capture ===
    
    async def capture_execution_knowledge(self, execution_data: Dict[str, Any]) -> bool:
        """Capture knowledge from a workflow execution"""
        try:
            # Create knowledge entry
            entry = KnowledgeEntry(
                entry_id=execution_data.get('request_id', ''),
                workflow_id=execution_data.get('workflow_id', ''),
                user_id=execution_data.get('user_context', {}).get('user_id', ''),
                tenant_id=execution_data.get('user_context', {}).get('tenant_id', ''),
                execution_time_ms=execution_data.get('execution_time_ms', 0),
                success=execution_data.get('success', False),
                input_data=execution_data.get('input_data', {}),
                output_data=execution_data.get('result', {}),
                user_context=execution_data.get('user_context', {})
            )
            
            # Store in memory (will be replaced with Knowledge Graph)
            self.execution_entries.append(entry)
            
            # Update behavior caches
            await self._update_behavior_cache(entry)
            await self._update_performance_cache(entry)
            
            # Limit memory usage
            if len(self.execution_entries) > self.max_entries_in_memory:
                self.execution_entries = self.execution_entries[-self.max_entries_in_memory:]
            
            print(f"📚 Knowledge captured for workflow {entry.workflow_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error capturing knowledge: {e}")
            return False
    
    # === Pattern Discovery ===
    
    async def discover_patterns(self, module: Optional[str] = None) -> List[ExecutionPattern]:
        """Discover patterns in execution data"""
        patterns = []
        
        try:
            # Filter entries by module if specified
            entries = self.execution_entries
            if module:
                entries = [e for e in entries if self._get_workflow_module(e.workflow_id) == module]
            
            # Discover different types of patterns
            patterns.extend(await self._discover_success_patterns(entries))
            patterns.extend(await self._discover_failure_patterns(entries))
            patterns.extend(await self._discover_performance_patterns(entries))
            patterns.extend(await self._discover_user_behavior_patterns(entries))
            
            return patterns
            
        except Exception as e:
            print(f"❌ Error discovering patterns: {e}")
            return patterns
    
    async def get_workflow_insights(self, workflow_id: str) -> Dict[str, Any]:
        """Get insights for a specific workflow"""
        try:
            # Filter entries for this workflow
            workflow_entries = [e for e in self.execution_entries if e.workflow_id == workflow_id]
            
            if not workflow_entries:
                return {'message': 'No execution data available for this workflow'}
            
            # Calculate basic metrics
            total_executions = len(workflow_entries)
            successful_executions = sum(1 for e in workflow_entries if e.success)
            success_rate = successful_executions / total_executions if total_executions > 0 else 0
            
            avg_execution_time = sum(e.execution_time_ms for e in workflow_entries) / total_executions
            
            # Find most common input patterns
            input_patterns = self._analyze_input_patterns(workflow_entries)
            
            # Find optimal execution times
            optimal_times = self._analyze_optimal_execution_times(workflow_entries)
            
            # User adoption insights
            unique_users = len(set(e.user_id for e in workflow_entries))
            repeat_users = self._analyze_repeat_usage(workflow_entries)
            
            return {
                'workflow_id': workflow_id,
                'total_executions': total_executions,
                'success_rate': success_rate,
                'avg_execution_time_ms': int(avg_execution_time),
                'unique_users': unique_users,
                'repeat_users': repeat_users,
                'input_patterns': input_patterns,
                'optimal_execution_times': optimal_times,
                'last_execution': workflow_entries[-1].timestamp.isoformat() if workflow_entries else None
            }
            
        except Exception as e:
            print(f"❌ Error getting workflow insights: {e}")
            return {'error': str(e)}
    
    async def get_user_behavior_insights(self, user_id: str) -> Dict[str, Any]:
        """Get behavior insights for a specific user"""
        try:
            # Filter entries for this user
            user_entries = [e for e in self.execution_entries if e.user_id == user_id]
            
            if not user_entries:
                return {'message': 'No execution data available for this user'}
            
            # Analyze user behavior
            favorite_workflows = self._analyze_favorite_workflows(user_entries)
            execution_patterns = self._analyze_execution_timing(user_entries)
            success_by_workflow = self._analyze_success_by_workflow(user_entries)
            
            return {
                'user_id': user_id,
                'total_executions': len(user_entries),
                'favorite_workflows': favorite_workflows,
                'execution_patterns': execution_patterns,
                'success_by_workflow': success_by_workflow,
                'first_execution': user_entries[0].timestamp.isoformat() if user_entries else None,
                'last_execution': user_entries[-1].timestamp.isoformat() if user_entries else None
            }
            
        except Exception as e:
            print(f"❌ Error getting user insights: {e}")
            return {'error': str(e)}
    
    # === Recommendations ===
    
    async def get_workflow_recommendations(self, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get workflow recommendations based on user behavior and patterns"""
        try:
            user_id = user_context.get('user_id')
            role = user_context.get('role', '')
            
            recommendations = []
            
            # Find similar users
            similar_users = await self._find_similar_users(user_id)
            
            # Get workflows used by similar users
            for similar_user_id in similar_users:
                similar_user_workflows = self._get_user_workflows(similar_user_id)
                current_user_workflows = self._get_user_workflows(user_id)
                
                # Recommend workflows not used by current user
                new_workflows = set(similar_user_workflows) - set(current_user_workflows)
                
                for workflow_id in new_workflows:
                    workflow_success_rate = self._get_workflow_success_rate(workflow_id)
                    if workflow_success_rate > 0.7:  # Only recommend successful workflows
                        recommendations.append({
                            'workflow_id': workflow_id,
                            'reason': 'used_by_similar_users',
                            'success_rate': workflow_success_rate,
                            'confidence': 0.8
                        })
            
            # Role-based recommendations
            role_recommendations = await self._get_role_based_recommendations(role)
            recommendations.extend(role_recommendations)
            
            # Sort by confidence and limit
            recommendations.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            return recommendations[:5]
            
        except Exception as e:
            print(f"❌ Error getting recommendations: {e}")
            return []
    
    # === Performance Optimization ===
    
    async def get_optimization_suggestions(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get optimization suggestions for a workflow"""
        try:
            workflow_entries = [e for e in self.execution_entries if e.workflow_id == workflow_id]
            
            if len(workflow_entries) < 10:
                return [{'message': 'Need more execution data for optimization suggestions'}]
            
            suggestions = []
            
            # Analyze execution times
            execution_times = [e.execution_time_ms for e in workflow_entries]
            avg_time = sum(execution_times) / len(execution_times)
            
            if avg_time > 5000:  # Slower than 5 seconds
                suggestions.append({
                    'type': 'performance',
                    'issue': 'slow_execution',
                    'avg_time_ms': int(avg_time),
                    'recommendation': 'Consider optimizing database queries or adding caching',
                    'priority': 'high'
                })
            
            # Analyze failure patterns
            failed_entries = [e for e in workflow_entries if not e.success]
            if len(failed_entries) > len(workflow_entries) * 0.2:  # More than 20% failure rate
                suggestions.append({
                    'type': 'reliability',
                    'issue': 'high_failure_rate',
                    'failure_rate': len(failed_entries) / len(workflow_entries),
                    'recommendation': 'Review error handling and add input validation',
                    'priority': 'high'
                })
            
            # Analyze input data patterns
            input_suggestions = self._analyze_input_optimization(workflow_entries)
            suggestions.extend(input_suggestions)
            
            return suggestions
            
        except Exception as e:
            print(f"❌ Error getting optimization suggestions: {e}")
            return []
    
    # === Private Methods ===
    
    async def _update_behavior_cache(self, entry: KnowledgeEntry):
        """Update user behavior cache"""
        key = f"{entry.user_id}:{entry.workflow_id}"
        self.user_behavior_cache[key].append({
            'timestamp': entry.timestamp,
            'success': entry.success,
            'execution_time_ms': entry.execution_time_ms
        })
        
        # Limit cache size
        if len(self.user_behavior_cache[key]) > 100:
            self.user_behavior_cache[key] = self.user_behavior_cache[key][-100:]
    
    async def _update_performance_cache(self, entry: KnowledgeEntry):
        """Update performance cache"""
        self.performance_cache[entry.workflow_id].append({
            'timestamp': entry.timestamp,
            'execution_time_ms': entry.execution_time_ms,
            'success': entry.success
        })
        
        # Limit cache size
        if len(self.performance_cache[entry.workflow_id]) > 1000:
            self.performance_cache[entry.workflow_id] = self.performance_cache[entry.workflow_id][-1000:]
    
    async def _pattern_discovery_worker(self):
        """Background worker for pattern discovery"""
        while True:
            try:
                await asyncio.sleep(self.pattern_discovery_interval)
                
                # Discover new patterns
                new_patterns = await self.discover_patterns()
                
                # Add to discovered patterns
                self.discovered_patterns.extend(new_patterns)
                
                # Limit pattern storage
                if len(self.discovered_patterns) > 1000:
                    self.discovered_patterns = self.discovered_patterns[-1000:]
                
                print(f"🔍 Discovered {len(new_patterns)} new patterns")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Pattern discovery error: {e}")
    
    def _get_workflow_module(self, workflow_id: str) -> str:
        """Get module for a workflow (simplified)"""
        # This would be enhanced with actual workflow metadata
        if 'forecast' in workflow_id.lower():
            return 'Forecast'
        elif 'pipeline' in workflow_id.lower():
            return 'Pipeline'
        elif 'planning' in workflow_id.lower():
            return 'Planning'
        return 'Unknown'
    
    async def _discover_success_patterns(self, entries: List[KnowledgeEntry]) -> List[ExecutionPattern]:
        """Discover patterns in successful executions"""
        patterns = []
        
        # Group by workflow and find common success patterns
        workflow_groups = defaultdict(list)
        for entry in entries:
            if entry.success:
                workflow_groups[entry.workflow_id].append(entry)
        
        for workflow_id, workflow_entries in workflow_groups.items():
            if len(workflow_entries) >= self.min_pattern_frequency:
                patterns.append(ExecutionPattern(
                    pattern_id=f"success_{workflow_id}_{len(patterns)}",
                    workflow_id=workflow_id,
                    module=self._get_workflow_module(workflow_id),
                    pattern_type='success',
                    frequency=len(workflow_entries),
                    confidence=0.8,
                    metadata={'avg_execution_time': sum(e.execution_time_ms for e in workflow_entries) / len(workflow_entries)}
                ))
        
        return patterns
    
    async def _discover_failure_patterns(self, entries: List[KnowledgeEntry]) -> List[ExecutionPattern]:
        """Discover patterns in failed executions"""
        patterns = []
        
        # Find common failure patterns
        failed_entries = [e for e in entries if not e.success]
        
        # Group by similar input patterns (simplified)
        failure_groups = defaultdict(list)
        for entry in failed_entries:
            # Simple grouping by workflow_id (would be enhanced with actual pattern matching)
            failure_groups[entry.workflow_id].append(entry)
        
        for workflow_id, workflow_failures in failure_groups.items():
            if len(workflow_failures) >= self.min_pattern_frequency:
                patterns.append(ExecutionPattern(
                    pattern_id=f"failure_{workflow_id}_{len(patterns)}",
                    workflow_id=workflow_id,
                    module=self._get_workflow_module(workflow_id),
                    pattern_type='failure',
                    frequency=len(workflow_failures),
                    confidence=0.7,
                    metadata={'common_failure_points': 'input_validation'}  # Would be enhanced
                ))
        
        return patterns
    
    async def _discover_performance_patterns(self, entries: List[KnowledgeEntry]) -> List[ExecutionPattern]:
        """Discover performance patterns"""
        patterns = []
        
        # Find workflows with consistent performance characteristics
        workflow_performance = defaultdict(list)
        for entry in entries:
            workflow_performance[entry.workflow_id].append(entry.execution_time_ms)
        
        for workflow_id, times in workflow_performance.items():
            if len(times) >= self.min_pattern_frequency:
                avg_time = sum(times) / len(times)
                if avg_time > 5000:  # Slow workflows
                    patterns.append(ExecutionPattern(
                        pattern_id=f"slow_performance_{workflow_id}",
                        workflow_id=workflow_id,
                        module=self._get_workflow_module(workflow_id),
                        pattern_type='performance',
                        frequency=len(times),
                        confidence=0.9,
                        metadata={'avg_execution_time_ms': avg_time, 'performance_category': 'slow'}
                    ))
        
        return patterns
    
    async def _discover_user_behavior_patterns(self, entries: List[KnowledgeEntry]) -> List[ExecutionPattern]:
        """Discover user behavior patterns"""
        patterns = []
        
        # Find users with similar workflow usage patterns
        user_workflows = defaultdict(set)
        for entry in entries:
            user_workflows[entry.user_id].add(entry.workflow_id)
        
        # Find common workflow combinations (simplified)
        workflow_combinations = defaultdict(int)
        for user_id, workflows in user_workflows.items():
            if len(workflows) > 1:
                workflow_list = sorted(list(workflows))
                combination = ','.join(workflow_list[:3])  # Limit to first 3
                workflow_combinations[combination] += 1
        
        for combination, frequency in workflow_combinations.items():
            if frequency >= self.min_pattern_frequency:
                patterns.append(ExecutionPattern(
                    pattern_id=f"user_behavior_{len(patterns)}",
                    workflow_id=combination.split(',')[0],  # Primary workflow
                    module='Multiple',
                    pattern_type='user_behavior',
                    frequency=frequency,
                    confidence=0.6,
                    metadata={'workflow_combination': combination}
                ))
        
        return patterns
    
    # === Helper Methods (Simplified implementations) ===
    
    def _analyze_input_patterns(self, entries: List[KnowledgeEntry]) -> List[str]:
        """Analyze common input patterns"""
        # Simplified - would be enhanced with actual pattern matching
        return ['common_input_pattern_1', 'common_input_pattern_2']
    
    def _analyze_optimal_execution_times(self, entries: List[KnowledgeEntry]) -> Dict[str, int]:
        """Analyze optimal execution times"""
        successful_entries = [e for e in entries if e.success]
        if not successful_entries:
            return {}
        
        times = [e.execution_time_ms for e in successful_entries]
        return {
            'min_time_ms': min(times),
            'avg_time_ms': int(sum(times) / len(times)),
            'max_time_ms': max(times)
        }
    
    def _analyze_repeat_usage(self, entries: List[KnowledgeEntry]) -> int:
        """Analyze repeat usage"""
        user_counts = defaultdict(int)
        for entry in entries:
            user_counts[entry.user_id] += 1
        
        return sum(1 for count in user_counts.values() if count > 1)
    
    def _analyze_favorite_workflows(self, entries: List[KnowledgeEntry]) -> List[Dict[str, Any]]:
        """Analyze user's favorite workflows"""
        workflow_counts = defaultdict(int)
        for entry in entries:
            workflow_counts[entry.workflow_id] += 1
        
        return [
            {'workflow_id': wf_id, 'usage_count': count}
            for wf_id, count in sorted(workflow_counts.items(), key=lambda x: x[1], reverse=True)
        ][:5]
    
    def _analyze_execution_timing(self, entries: List[KnowledgeEntry]) -> Dict[str, Any]:
        """Analyze when user typically executes workflows"""
        hours = [entry.timestamp.hour for entry in entries]
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1
        
        peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else 9
        
        return {
            'peak_hour': peak_hour,
            'total_executions': len(entries)
        }
    
    def _analyze_success_by_workflow(self, entries: List[KnowledgeEntry]) -> Dict[str, float]:
        """Analyze success rate by workflow for user"""
        workflow_success = defaultdict(lambda: {'total': 0, 'success': 0})
        
        for entry in entries:
            workflow_success[entry.workflow_id]['total'] += 1
            if entry.success:
                workflow_success[entry.workflow_id]['success'] += 1
        
        return {
            wf_id: data['success'] / data['total'] if data['total'] > 0 else 0
            for wf_id, data in workflow_success.items()
        }
    
    async def _find_similar_users(self, user_id: str) -> List[str]:
        """Find users with similar behavior patterns"""
        # Simplified implementation
        return []  # Would implement collaborative filtering
    
    def _get_user_workflows(self, user_id: str) -> List[str]:
        """Get workflows used by a user"""
        return list(set(e.workflow_id for e in self.execution_entries if e.user_id == user_id))
    
    def _get_workflow_success_rate(self, workflow_id: str) -> float:
        """Get success rate for a workflow"""
        workflow_entries = [e for e in self.execution_entries if e.workflow_id == workflow_id]
        if not workflow_entries:
            return 0.0
        
        success_count = sum(1 for e in workflow_entries if e.success)
        return success_count / len(workflow_entries)
    
    async def _get_role_based_recommendations(self, role: str) -> List[Dict[str, Any]]:
        """Get recommendations based on user role"""
        # Simplified role-based recommendations
        role_workflows = {
            'sales_manager': ['forecast_calibration', 'deal_risk_alerting'],
            'sales_rep': ['pipeline_hygiene', 'opportunity_scoring'],
            'revenue_ops': ['territory_planning', 'quota_planning']
        }
        
        workflows = role_workflows.get(role.lower(), [])
        return [
            {
                'workflow_id': wf_id,
                'reason': 'role_based',
                'success_rate': 0.8,
                'confidence': 0.7
            }
            for wf_id in workflows
        ]
    
    def _analyze_input_optimization(self, entries: List[KnowledgeEntry]) -> List[Dict[str, Any]]:
        """Analyze input data for optimization opportunities"""
        # Simplified implementation
        return [
            {
                'type': 'input_optimization',
                'issue': 'unnecessary_input_fields',
                'recommendation': 'Remove unused input fields to improve performance',
                'priority': 'medium'
            }
        ]
