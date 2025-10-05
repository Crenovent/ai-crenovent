"""
Block Intelligence Engine
========================

Task 7.3.1: Build block intelligence engine
AI-powered intelligent block recommendations, optimization suggestions, and workflow enhancement
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import uuid

logger = logging.getLogger(__name__)

class RecommendationType(Enum):
    """Types of block recommendations"""
    OPTIMIZATION = "optimization"
    REPLACEMENT = "replacement"
    ADDITION = "addition"
    REMOVAL = "removal"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"

class RecommendationPriority(Enum):
    """Priority levels for recommendations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class BlockRecommendation:
    """Individual block recommendation"""
    recommendation_id: str
    recommendation_type: RecommendationType
    priority: RecommendationPriority
    title: str
    description: str
    target_block_id: Optional[str] = None
    suggested_block_type: Optional[str] = None
    suggested_configuration: Optional[Dict[str, Any]] = None
    estimated_impact: str = "medium"  # low, medium, high
    effort_required: str = "medium"   # low, medium, high
    confidence_score: float = 0.8
    reasoning: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    related_blocks: List[str] = field(default_factory=list)
    compliance_impact: List[str] = field(default_factory=list)
    performance_impact: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class BlockAnalysisResult:
    """Result of block analysis"""
    workflow_id: str
    block_id: str
    block_type: str
    analysis_score: float
    performance_metrics: Dict[str, Any]
    usage_patterns: Dict[str, Any]
    optimization_opportunities: List[str]
    risk_factors: List[str]
    compliance_status: Dict[str, Any]

@dataclass
class WorkflowIntelligenceReport:
    """Comprehensive workflow intelligence report"""
    workflow_id: str
    overall_score: float
    recommendations: List[BlockRecommendation]
    block_analyses: List[BlockAnalysisResult]
    workflow_patterns: Dict[str, Any]
    optimization_summary: Dict[str, Any]
    compliance_summary: Dict[str, Any]
    performance_summary: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.utcnow)

class BlockIntelligenceEngine:
    """
    AI-powered Block Intelligence Engine - Task 7.3.1
    
    Provides intelligent recommendations for DSL blocks including:
    - Block optimization suggestions
    - Performance improvements
    - Security enhancements
    - Compliance recommendations
    - Workflow pattern analysis
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Intelligence configuration
        self.intelligence_config = self._load_intelligence_config()
        
        # Block pattern library
        self.block_patterns = self._load_block_patterns()
        
        # Performance benchmarks
        self.performance_benchmarks = self._load_performance_benchmarks()
        
        # Compliance rules
        self.compliance_rules = self._load_compliance_rules()
        
        # ML models for pattern recognition (placeholder)
        self.pattern_models = {}
        
    def _load_intelligence_config(self) -> Dict[str, Any]:
        """Load intelligence engine configuration"""
        return {
            'analysis_depth': 'comprehensive',  # basic, standard, comprehensive
            'recommendation_threshold': 0.7,   # Minimum confidence for recommendations
            'max_recommendations': 20,         # Maximum recommendations per workflow
            'performance_weight': 0.3,         # Weight for performance factors
            'security_weight': 0.3,           # Weight for security factors
            'compliance_weight': 0.4,         # Weight for compliance factors
            'pattern_matching_enabled': True,
            'ml_recommendations_enabled': True,
            'historical_analysis_enabled': True
        }
    
    def _load_block_patterns(self) -> Dict[str, Any]:
        """Load common block patterns and anti-patterns"""
        return {
            'optimal_patterns': {
                'data_pipeline': [
                    {'type': 'query', 'purpose': 'data_extraction'},
                    {'type': 'decision', 'purpose': 'data_validation'},
                    {'type': 'governance', 'purpose': 'compliance_check'},
                    {'type': 'notify', 'purpose': 'completion_notification'}
                ],
                'approval_workflow': [
                    {'type': 'query', 'purpose': 'request_validation'},
                    {'type': 'decision', 'purpose': 'auto_approval_check'},
                    {'type': 'agent_call', 'purpose': 'human_approval'},
                    {'type': 'governance', 'purpose': 'audit_logging'},
                    {'type': 'notify', 'purpose': 'status_notification'}
                ],
                'ml_pipeline': [
                    {'type': 'query', 'purpose': 'feature_extraction'},
                    {'type': 'ml_decision', 'purpose': 'prediction'},
                    {'type': 'decision', 'purpose': 'confidence_check'},
                    {'type': 'governance', 'purpose': 'model_governance'},
                    {'type': 'notify', 'purpose': 'result_notification'}
                ]
            },
            'anti_patterns': {
                'missing_governance': {
                    'pattern': 'workflow without governance blocks',
                    'risk': 'compliance_violation',
                    'recommendation': 'Add governance blocks for audit trail'
                },
                'excessive_complexity': {
                    'pattern': 'more than 15 blocks in single workflow',
                    'risk': 'performance_degradation',
                    'recommendation': 'Break into smaller sub-workflows'
                },
                'missing_error_handling': {
                    'pattern': 'no fallback or retry logic',
                    'risk': 'workflow_failure',
                    'recommendation': 'Add error handling and retry blocks'
                }
            }
        }
    
    def _load_performance_benchmarks(self) -> Dict[str, Any]:
        """Load performance benchmarks for different block types"""
        return {
            'query': {
                'max_execution_time_ms': 5000,
                'max_memory_usage_mb': 100,
                'max_rows_processed': 10000,
                'optimal_cache_hit_rate': 0.8
            },
            'decision': {
                'max_execution_time_ms': 1000,
                'max_memory_usage_mb': 50,
                'max_conditions': 10,
                'optimal_branch_balance': 0.6
            },
            'ml_decision': {
                'max_execution_time_ms': 10000,
                'max_memory_usage_mb': 500,
                'min_confidence_score': 0.7,
                'max_feature_count': 100
            },
            'agent_call': {
                'max_execution_time_ms': 30000,
                'max_memory_usage_mb': 200,
                'max_retry_attempts': 3,
                'min_success_rate': 0.9
            },
            'notify': {
                'max_execution_time_ms': 2000,
                'max_memory_usage_mb': 25,
                'max_recipients': 50,
                'optimal_delivery_rate': 0.95
            },
            'governance': {
                'max_execution_time_ms': 3000,
                'max_memory_usage_mb': 75,
                'required_audit_fields': 8,
                'min_compliance_score': 0.9
            }
        }
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules for different industries and regions"""
        return {
            'GDPR': {
                'required_blocks': ['governance'],
                'data_processing_consent': True,
                'audit_trail_mandatory': True,
                'data_minimization': True
            },
            'SOX': {
                'required_blocks': ['governance', 'decision'],
                'segregation_of_duties': True,
                'audit_trail_mandatory': True,
                'approval_workflow_required': True
            },
            'HIPAA': {
                'required_blocks': ['governance'],
                'phi_encryption': True,
                'access_logging': True,
                'minimum_necessary_rule': True
            },
            'PCI_DSS': {
                'required_blocks': ['governance', 'decision'],
                'cardholder_data_protection': True,
                'access_control': True,
                'monitoring_required': True
            }
        }
    
    async def analyze_workflow_intelligence(
        self,
        workflow_definition: Dict[str, Any],
        tenant_id: int,
        execution_history: Optional[List[Dict[str, Any]]] = None,
        industry_context: Optional[str] = None
    ) -> WorkflowIntelligenceReport:
        """
        Perform comprehensive intelligence analysis of a workflow
        
        Args:
            workflow_definition: DSL workflow definition
            tenant_id: Tenant identifier
            execution_history: Historical execution data
            industry_context: Industry context for compliance
            
        Returns:
            WorkflowIntelligenceReport: Comprehensive analysis report
        """
        
        workflow_id = workflow_definition.get('workflow_id', str(uuid.uuid4()))
        
        try:
            self.logger.info(f"🧠 Starting workflow intelligence analysis: {workflow_id}")
            
            # 1. Analyze individual blocks
            block_analyses = await self._analyze_blocks(workflow_definition, execution_history)
            
            # 2. Analyze workflow patterns
            workflow_patterns = await self._analyze_workflow_patterns(workflow_definition)
            
            # 3. Generate recommendations
            recommendations = await self._generate_recommendations(
                workflow_definition, block_analyses, workflow_patterns, industry_context
            )
            
            # 4. Calculate overall score
            overall_score = self._calculate_overall_score(block_analyses, recommendations)
            
            # 5. Generate summaries
            optimization_summary = self._generate_optimization_summary(recommendations)
            compliance_summary = self._generate_compliance_summary(recommendations, industry_context)
            performance_summary = self._generate_performance_summary(block_analyses)
            
            report = WorkflowIntelligenceReport(
                workflow_id=workflow_id,
                overall_score=overall_score,
                recommendations=recommendations,
                block_analyses=block_analyses,
                workflow_patterns=workflow_patterns,
                optimization_summary=optimization_summary,
                compliance_summary=compliance_summary,
                performance_summary=performance_summary
            )
            
            self.logger.info(f"✅ Workflow intelligence analysis completed: {overall_score:.2f} score, "
                           f"{len(recommendations)} recommendations")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Workflow intelligence analysis failed: {e}")
            raise
    
    async def _analyze_blocks(
        self,
        workflow_definition: Dict[str, Any],
        execution_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[BlockAnalysisResult]:
        """Analyze individual blocks in the workflow"""
        
        block_analyses = []
        steps = workflow_definition.get('steps', [])
        
        for step in steps:
            block_id = step.get('id', str(uuid.uuid4()))
            block_type = step.get('type', 'unknown')
            
            # Analyze block performance
            performance_metrics = await self._analyze_block_performance(
                step, execution_history
            )
            
            # Analyze usage patterns
            usage_patterns = await self._analyze_block_usage_patterns(
                step, execution_history
            )
            
            # Identify optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(
                step, performance_metrics
            )
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(step, performance_metrics)
            
            # Check compliance status
            compliance_status = self._check_block_compliance(step)
            
            # Calculate analysis score
            analysis_score = self._calculate_block_score(
                performance_metrics, usage_patterns, optimization_opportunities, risk_factors
            )
            
            block_analysis = BlockAnalysisResult(
                workflow_id=workflow_definition.get('workflow_id', 'unknown'),
                block_id=block_id,
                block_type=block_type,
                analysis_score=analysis_score,
                performance_metrics=performance_metrics,
                usage_patterns=usage_patterns,
                optimization_opportunities=optimization_opportunities,
                risk_factors=risk_factors,
                compliance_status=compliance_status
            )
            
            block_analyses.append(block_analysis)
        
        return block_analyses
    
    async def _analyze_block_performance(
        self,
        block_definition: Dict[str, Any],
        execution_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Analyze performance metrics for a specific block"""
        
        block_type = block_definition.get('type', 'unknown')
        benchmarks = self.performance_benchmarks.get(block_type, {})
        
        # Default performance metrics
        metrics = {
            'avg_execution_time_ms': 0,
            'max_execution_time_ms': 0,
            'avg_memory_usage_mb': 0,
            'success_rate': 1.0,
            'error_rate': 0.0,
            'performance_score': 1.0
        }
        
        # Analyze historical execution data if available
        if execution_history:
            block_id = block_definition.get('id')
            block_executions = [
                exec_data for exec_data in execution_history
                if exec_data.get('block_id') == block_id
            ]
            
            if block_executions:
                execution_times = [e.get('execution_time_ms', 0) for e in block_executions]
                memory_usage = [e.get('memory_usage_mb', 0) for e in block_executions]
                success_count = sum(1 for e in block_executions if e.get('status') == 'success')
                
                metrics.update({
                    'avg_execution_time_ms': sum(execution_times) / len(execution_times),
                    'max_execution_time_ms': max(execution_times) if execution_times else 0,
                    'avg_memory_usage_mb': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                    'success_rate': success_count / len(block_executions),
                    'error_rate': 1 - (success_count / len(block_executions)),
                    'total_executions': len(block_executions)
                })
        
        # Calculate performance score based on benchmarks
        performance_score = 1.0
        if benchmarks:
            time_score = min(1.0, benchmarks.get('max_execution_time_ms', 1000) / max(metrics['avg_execution_time_ms'], 1))
            memory_score = min(1.0, benchmarks.get('max_memory_usage_mb', 100) / max(metrics['avg_memory_usage_mb'], 1))
            success_score = metrics['success_rate']
            
            performance_score = (time_score + memory_score + success_score) / 3
        
        metrics['performance_score'] = performance_score
        
        return metrics
    
    async def _analyze_block_usage_patterns(
        self,
        block_definition: Dict[str, Any],
        execution_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Analyze usage patterns for a specific block"""
        
        patterns = {
            'usage_frequency': 'unknown',
            'peak_usage_hours': [],
            'common_parameters': {},
            'failure_patterns': [],
            'dependency_patterns': []
        }
        
        if execution_history:
            block_id = block_definition.get('id')
            block_executions = [
                exec_data for exec_data in execution_history
                if exec_data.get('block_id') == block_id
            ]
            
            if block_executions:
                # Calculate usage frequency
                total_days = 30  # Assume 30-day analysis window
                patterns['usage_frequency'] = len(block_executions) / total_days
                
                # Analyze peak usage hours
                hours = [datetime.fromisoformat(e.get('timestamp', datetime.utcnow().isoformat())).hour 
                        for e in block_executions if e.get('timestamp')]
                if hours:
                    hour_counts = {}
                    for hour in hours:
                        hour_counts[hour] = hour_counts.get(hour, 0) + 1
                    patterns['peak_usage_hours'] = sorted(hour_counts.keys(), 
                                                        key=lambda h: hour_counts[h], reverse=True)[:3]
                
                # Analyze common parameters
                all_params = [e.get('parameters', {}) for e in block_executions]
                param_counts = {}
                for params in all_params:
                    for key, value in params.items():
                        param_key = f"{key}={value}"
                        param_counts[param_key] = param_counts.get(param_key, 0) + 1
                
                patterns['common_parameters'] = dict(sorted(param_counts.items(), 
                                                          key=lambda x: x[1], reverse=True)[:5])
                
                # Analyze failure patterns
                failures = [e for e in block_executions if e.get('status') == 'failed']
                failure_reasons = [f.get('error_message', 'unknown') for f in failures]
                patterns['failure_patterns'] = list(set(failure_reasons))[:5]
        
        return patterns
    
    def _identify_optimization_opportunities(
        self,
        block_definition: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> List[str]:
        """Identify optimization opportunities for a block"""
        
        opportunities = []
        block_type = block_definition.get('type', 'unknown')
        benchmarks = self.performance_benchmarks.get(block_type, {})
        
        # Performance-based opportunities
        if performance_metrics.get('avg_execution_time_ms', 0) > benchmarks.get('max_execution_time_ms', float('inf')):
            opportunities.append("Optimize execution time - consider caching or query optimization")
        
        if performance_metrics.get('avg_memory_usage_mb', 0) > benchmarks.get('max_memory_usage_mb', float('inf')):
            opportunities.append("Reduce memory usage - optimize data structures or processing")
        
        if performance_metrics.get('success_rate', 1.0) < 0.95:
            opportunities.append("Improve reliability - add error handling and retry logic")
        
        # Configuration-based opportunities
        params = block_definition.get('params', {})
        
        if block_type == 'query' and not params.get('cache_enabled', False):
            opportunities.append("Enable caching for better performance")
        
        if block_type in ['decision', 'ml_decision'] and not params.get('timeout'):
            opportunities.append("Add timeout configuration to prevent hanging")
        
        if block_type == 'agent_call' and not params.get('retry_config'):
            opportunities.append("Add retry configuration for better reliability")
        
        # Governance opportunities
        governance = block_definition.get('governance', {})
        if not governance.get('evidence_capture', False):
            opportunities.append("Enable evidence capture for better audit trail")
        
        if not governance.get('trust_threshold'):
            opportunities.append("Set trust threshold for quality assurance")
        
        return opportunities
    
    def _identify_risk_factors(
        self,
        block_definition: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> List[str]:
        """Identify risk factors for a block"""
        
        risk_factors = []
        block_type = block_definition.get('type', 'unknown')
        params = block_definition.get('params', {})
        governance = block_definition.get('governance', {})
        
        # Performance risks
        if performance_metrics.get('error_rate', 0) > 0.1:
            risk_factors.append("High error rate may impact workflow reliability")
        
        if performance_metrics.get('avg_execution_time_ms', 0) > 30000:
            risk_factors.append("Long execution time may cause timeouts")
        
        # Security risks
        if block_type == 'query' and not params.get('sql_injection_protection', True):
            risk_factors.append("Potential SQL injection vulnerability")
        
        if block_type == 'agent_call' and not params.get('input_validation', True):
            risk_factors.append("Missing input validation for agent calls")
        
        # Compliance risks
        if not governance.get('policy_id'):
            risk_factors.append("Missing policy enforcement may violate compliance")
        
        if block_type in ['query', 'decision'] and not governance.get('audit_trail', True):
            risk_factors.append("Missing audit trail for compliance requirements")
        
        # Data handling risks
        if params.get('data_type') == 'pii' and not params.get('encryption_enabled', False):
            risk_factors.append("PII data not encrypted - privacy risk")
        
        if params.get('data_retention_days', 0) > 2555:  # 7 years
            risk_factors.append("Data retention period exceeds regulatory limits")
        
        return risk_factors
    
    def _check_block_compliance(self, block_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance status of a block"""
        
        compliance_status = {
            'overall_compliant': True,
            'framework_compliance': {},
            'missing_requirements': [],
            'compliance_score': 1.0
        }
        
        block_type = block_definition.get('type', 'unknown')
        governance = block_definition.get('governance', {})
        params = block_definition.get('params', {})
        
        # Check against each compliance framework
        for framework, rules in self.compliance_rules.items():
            framework_compliant = True
            missing_reqs = []
            
            # Check required blocks
            if block_type not in rules.get('required_blocks', []) and rules.get('required_blocks'):
                # This check is more relevant at workflow level
                pass
            
            # Check specific requirements
            if rules.get('audit_trail_mandatory', False) and not governance.get('audit_trail', False):
                framework_compliant = False
                missing_reqs.append("audit_trail_required")
            
            if rules.get('segregation_of_duties', False) and not governance.get('approval_required', False):
                framework_compliant = False
                missing_reqs.append("segregation_of_duties_required")
            
            if rules.get('phi_encryption', False) and params.get('data_type') == 'phi' and not params.get('encryption_enabled', False):
                framework_compliant = False
                missing_reqs.append("phi_encryption_required")
            
            compliance_status['framework_compliance'][framework] = {
                'compliant': framework_compliant,
                'missing_requirements': missing_reqs
            }
            
            if not framework_compliant:
                compliance_status['overall_compliant'] = False
                compliance_status['missing_requirements'].extend(missing_reqs)
        
        # Calculate compliance score
        total_frameworks = len(self.compliance_rules)
        compliant_frameworks = sum(1 for f in compliance_status['framework_compliance'].values() if f['compliant'])
        compliance_status['compliance_score'] = compliant_frameworks / max(total_frameworks, 1)
        
        return compliance_status
    
    def _calculate_block_score(
        self,
        performance_metrics: Dict[str, Any],
        usage_patterns: Dict[str, Any],
        optimization_opportunities: List[str],
        risk_factors: List[str]
    ) -> float:
        """Calculate overall analysis score for a block"""
        
        # Base score from performance
        performance_score = performance_metrics.get('performance_score', 1.0)
        
        # Deduct points for optimization opportunities
        optimization_penalty = len(optimization_opportunities) * 0.05
        
        # Deduct points for risk factors
        risk_penalty = len(risk_factors) * 0.1
        
        # Bonus for high usage (indicates importance)
        usage_bonus = min(0.1, usage_patterns.get('usage_frequency', 0) * 0.01)
        
        final_score = max(0.0, min(1.0, performance_score - optimization_penalty - risk_penalty + usage_bonus))
        
        return final_score
    
    async def _analyze_workflow_patterns(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall workflow patterns"""
        
        steps = workflow_definition.get('steps', [])
        
        patterns = {
            'workflow_length': len(steps),
            'block_type_distribution': {},
            'governance_coverage': 0.0,
            'complexity_score': 0.0,
            'pattern_matches': [],
            'anti_pattern_matches': []
        }
        
        # Analyze block type distribution
        for step in steps:
            block_type = step.get('type', 'unknown')
            patterns['block_type_distribution'][block_type] = patterns['block_type_distribution'].get(block_type, 0) + 1
        
        # Calculate governance coverage
        governance_blocks = sum(1 for step in steps if step.get('governance', {}).get('policy_id'))
        patterns['governance_coverage'] = governance_blocks / max(len(steps), 1)
        
        # Calculate complexity score
        complexity_factors = [
            len(steps) * 0.1,  # Number of steps
            len(patterns['block_type_distribution']) * 0.2,  # Variety of block types
            sum(1 for step in steps if step.get('type') == 'decision') * 0.3,  # Decision complexity
            sum(1 for step in steps if step.get('type') == 'agent_call') * 0.4  # Agent complexity
        ]
        patterns['complexity_score'] = min(1.0, sum(complexity_factors))
        
        # Check for pattern matches
        patterns['pattern_matches'] = self._match_optimal_patterns(steps)
        patterns['anti_pattern_matches'] = self._match_anti_patterns(steps)
        
        return patterns
    
    def _match_optimal_patterns(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Match workflow against optimal patterns"""
        
        matches = []
        step_types = [step.get('type') for step in steps]
        
        for pattern_name, pattern_steps in self.block_patterns['optimal_patterns'].items():
            pattern_types = [p['type'] for p in pattern_steps]
            
            # Simple pattern matching (could be enhanced with ML)
            if self._sequence_similarity(step_types, pattern_types) > 0.7:
                matches.append(pattern_name)
        
        return matches
    
    def _match_anti_patterns(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Match workflow against anti-patterns"""
        
        matches = []
        
        # Check for missing governance
        governance_blocks = sum(1 for step in steps if step.get('type') == 'governance')
        if governance_blocks == 0:
            matches.append('missing_governance')
        
        # Check for excessive complexity
        if len(steps) > 15:
            matches.append('excessive_complexity')
        
        # Check for missing error handling
        error_handling_blocks = sum(1 for step in steps 
                                  if step.get('params', {}).get('retry_config') or 
                                     step.get('params', {}).get('fallback_action'))
        if error_handling_blocks == 0:
            matches.append('missing_error_handling')
        
        return matches
    
    def _sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two sequences"""
        
        if not seq1 or not seq2:
            return 0.0
        
        # Simple Jaccard similarity
        set1, set2 = set(seq1), set(seq2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _generate_recommendations(
        self,
        workflow_definition: Dict[str, Any],
        block_analyses: List[BlockAnalysisResult],
        workflow_patterns: Dict[str, Any],
        industry_context: Optional[str] = None
    ) -> List[BlockRecommendation]:
        """Generate intelligent recommendations based on analysis"""
        
        recommendations = []
        
        # Block-level recommendations
        for analysis in block_analyses:
            recommendations.extend(
                self._generate_block_recommendations(analysis, industry_context)
            )
        
        # Workflow-level recommendations
        recommendations.extend(
            self._generate_workflow_recommendations(workflow_patterns, industry_context)
        )
        
        # Sort by priority and confidence
        recommendations.sort(key=lambda r: (r.priority.value, -r.confidence_score))
        
        # Limit to max recommendations
        max_recommendations = self.intelligence_config.get('max_recommendations', 20)
        return recommendations[:max_recommendations]
    
    def _generate_block_recommendations(
        self,
        analysis: BlockAnalysisResult,
        industry_context: Optional[str] = None
    ) -> List[BlockRecommendation]:
        """Generate recommendations for a specific block"""
        
        recommendations = []
        
        # Performance optimization recommendations
        for opportunity in analysis.optimization_opportunities:
            recommendations.append(BlockRecommendation(
                recommendation_id=str(uuid.uuid4()),
                recommendation_type=RecommendationType.OPTIMIZATION,
                priority=RecommendationPriority.MEDIUM,
                title=f"Optimize {analysis.block_id}",
                description=opportunity,
                target_block_id=analysis.block_id,
                confidence_score=0.8,
                reasoning=[f"Block analysis identified optimization opportunity"],
                implementation_steps=[
                    "Review current block configuration",
                    "Apply suggested optimization",
                    "Test performance improvement",
                    "Monitor results"
                ]
            ))
        
        # Risk mitigation recommendations
        for risk in analysis.risk_factors:
            priority = RecommendationPriority.HIGH if 'security' in risk.lower() or 'compliance' in risk.lower() else RecommendationPriority.MEDIUM
            
            recommendations.append(BlockRecommendation(
                recommendation_id=str(uuid.uuid4()),
                recommendation_type=RecommendationType.SECURITY if 'security' in risk.lower() else RecommendationType.COMPLIANCE,
                priority=priority,
                title=f"Mitigate Risk in {analysis.block_id}",
                description=risk,
                target_block_id=analysis.block_id,
                confidence_score=0.9,
                reasoning=[f"Risk factor identified in block analysis"],
                implementation_steps=[
                    "Assess risk impact",
                    "Implement mitigation measures",
                    "Validate security/compliance",
                    "Update documentation"
                ]
            ))
        
        # Compliance recommendations
        if not analysis.compliance_status.get('overall_compliant', True):
            for framework, status in analysis.compliance_status.get('framework_compliance', {}).items():
                if not status.get('compliant', True):
                    recommendations.append(BlockRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        recommendation_type=RecommendationType.COMPLIANCE,
                        priority=RecommendationPriority.HIGH,
                        title=f"Fix {framework} Compliance in {analysis.block_id}",
                        description=f"Block does not meet {framework} compliance requirements",
                        target_block_id=analysis.block_id,
                        confidence_score=0.95,
                        reasoning=[f"{framework} compliance violation detected"],
                        compliance_impact=[framework],
                        implementation_steps=[
                            f"Review {framework} requirements",
                            "Update block configuration",
                            "Add required governance controls",
                            "Validate compliance"
                        ]
                    ))
        
        return recommendations
    
    def _generate_workflow_recommendations(
        self,
        workflow_patterns: Dict[str, Any],
        industry_context: Optional[str] = None
    ) -> List[BlockRecommendation]:
        """Generate workflow-level recommendations"""
        
        recommendations = []
        
        # Governance coverage recommendations
        if workflow_patterns.get('governance_coverage', 0) < 0.5:
            recommendations.append(BlockRecommendation(
                recommendation_id=str(uuid.uuid4()),
                recommendation_type=RecommendationType.ADDITION,
                priority=RecommendationPriority.HIGH,
                title="Improve Governance Coverage",
                description="Workflow has insufficient governance blocks for compliance",
                suggested_block_type="governance",
                confidence_score=0.9,
                reasoning=["Low governance coverage detected"],
                implementation_steps=[
                    "Identify critical decision points",
                    "Add governance blocks",
                    "Configure policy enforcement",
                    "Test compliance validation"
                ]
            ))
        
        # Complexity recommendations
        if workflow_patterns.get('complexity_score', 0) > 0.8:
            recommendations.append(BlockRecommendation(
                recommendation_id=str(uuid.uuid4()),
                recommendation_type=RecommendationType.OPTIMIZATION,
                priority=RecommendationPriority.MEDIUM,
                title="Reduce Workflow Complexity",
                description="Workflow is highly complex and may be difficult to maintain",
                confidence_score=0.8,
                reasoning=["High complexity score detected"],
                implementation_steps=[
                    "Break workflow into smaller sub-workflows",
                    "Simplify decision logic",
                    "Remove redundant blocks",
                    "Improve documentation"
                ]
            ))
        
        # Anti-pattern recommendations
        for anti_pattern in workflow_patterns.get('anti_pattern_matches', []):
            anti_pattern_info = self.block_patterns['anti_patterns'].get(anti_pattern, {})
            
            recommendations.append(BlockRecommendation(
                recommendation_id=str(uuid.uuid4()),
                recommendation_type=RecommendationType.ADDITION,
                priority=RecommendationPriority.HIGH,
                title=f"Fix Anti-Pattern: {anti_pattern.replace('_', ' ').title()}",
                description=anti_pattern_info.get('recommendation', f"Address {anti_pattern} anti-pattern"),
                confidence_score=0.85,
                reasoning=[f"Anti-pattern detected: {anti_pattern}"],
                implementation_steps=[
                    "Review anti-pattern impact",
                    "Implement recommended changes",
                    "Validate improvements",
                    "Update best practices"
                ]
            ))
        
        return recommendations
    
    def _calculate_overall_score(
        self,
        block_analyses: List[BlockAnalysisResult],
        recommendations: List[BlockRecommendation]
    ) -> float:
        """Calculate overall workflow intelligence score"""
        
        if not block_analyses:
            return 0.0
        
        # Average block scores
        avg_block_score = sum(analysis.analysis_score for analysis in block_analyses) / len(block_analyses)
        
        # Penalty for critical recommendations
        critical_penalty = len([r for r in recommendations if r.priority == RecommendationPriority.CRITICAL]) * 0.2
        high_penalty = len([r for r in recommendations if r.priority == RecommendationPriority.HIGH]) * 0.1
        
        overall_score = max(0.0, min(1.0, avg_block_score - critical_penalty - high_penalty))
        
        return overall_score
    
    def _generate_optimization_summary(self, recommendations: List[BlockRecommendation]) -> Dict[str, Any]:
        """Generate optimization summary from recommendations"""
        
        optimization_recs = [r for r in recommendations if r.recommendation_type == RecommendationType.OPTIMIZATION]
        
        return {
            'total_optimizations': len(optimization_recs),
            'high_impact_optimizations': len([r for r in optimization_recs if r.estimated_impact == 'high']),
            'quick_wins': len([r for r in optimization_recs if r.effort_required == 'low']),
            'categories': list(set([r.title.split()[0] for r in optimization_recs])),
            'estimated_improvement': min(0.3, len(optimization_recs) * 0.02)  # Max 30% improvement
        }
    
    def _generate_compliance_summary(
        self,
        recommendations: List[BlockRecommendation],
        industry_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate compliance summary from recommendations"""
        
        compliance_recs = [r for r in recommendations if r.recommendation_type == RecommendationType.COMPLIANCE]
        
        affected_frameworks = set()
        for rec in compliance_recs:
            affected_frameworks.update(rec.compliance_impact)
        
        return {
            'total_compliance_issues': len(compliance_recs),
            'critical_issues': len([r for r in compliance_recs if r.priority == RecommendationPriority.CRITICAL]),
            'affected_frameworks': list(affected_frameworks),
            'industry_context': industry_context,
            'compliance_score': max(0.0, 1.0 - len(compliance_recs) * 0.1)
        }
    
    def _generate_performance_summary(self, block_analyses: List[BlockAnalysisResult]) -> Dict[str, Any]:
        """Generate performance summary from block analyses"""
        
        if not block_analyses:
            return {'avg_performance_score': 0.0, 'performance_issues': 0}
        
        performance_scores = [analysis.performance_metrics.get('performance_score', 1.0) for analysis in block_analyses]
        avg_performance = sum(performance_scores) / len(performance_scores)
        
        performance_issues = sum(1 for score in performance_scores if score < 0.7)
        
        return {
            'avg_performance_score': avg_performance,
            'performance_issues': performance_issues,
            'best_performing_blocks': [a.block_id for a in block_analyses if a.performance_metrics.get('performance_score', 0) > 0.9],
            'worst_performing_blocks': [a.block_id for a in block_analyses if a.performance_metrics.get('performance_score', 1) < 0.5],
            'optimization_potential': max(0.0, 1.0 - avg_performance)
        }

# Global block intelligence engine instance
block_intelligence_engine = BlockIntelligenceEngine()

def get_block_intelligence_engine() -> BlockIntelligenceEngine:
    """Get the global block intelligence engine instance"""
    return block_intelligence_engine
